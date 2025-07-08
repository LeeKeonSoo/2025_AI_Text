import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 하이퍼파라미터 설정 (RTX 3070Ti 최적화)
MAX_LEN = 512
BATCH_SIZE = 16  # RTX 3070Ti에 맞춰 증가
EPOCHS = 2  # 에포크 수 감소로 시간 단축
LEARNING_RATE = 3e-5  # 더 높은 학습률로 빠른 수렴
WARMUP_RATIO = 0.06  # 워밍업 비율 감소
model_name = 'skt/kobert-base-v1'

# 긴 텍스트 처리 전략 (최적화)
CHUNK_SIZE = 300  # 청크 크기 감소
OVERLAP_SIZE = 50   # 오버랩 감소
MAX_CHUNKS = 3      # 최대 청크 수 감소 (계산량 감소)

class LongTextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def _chunk_text(self, text):
        """긴 텍스트를 오버랩이 있는 청크로 분할"""
        # 토큰화해서 길이 확인
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= self.max_len - 2:  # CLS, SEP 토큰 고려
            return [text]
        
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + CHUNK_SIZE, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            start = end - OVERLAP_SIZE
            
        return chunks
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        
        # 텍스트를 청크로 분할
        chunks = self._chunk_text(text)
        
        # 각 청크를 토크나이징
        chunk_encodings = []
        for chunk in chunks:
            encoding = self.tokenizer(
                chunk,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            chunk_encodings.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            })
        
        # 패딩하여 고정 크기로 만들기 (최대 3개 청크로 감소)
        while len(chunk_encodings) < MAX_CHUNKS:
            # 빈 청크 추가 (패딩)
            chunk_encodings.append({
                'input_ids': torch.zeros(self.max_len, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_len, dtype=torch.long)
            })
        
        # 너무 많은 청크는 자르기
        chunk_encodings = chunk_encodings[:MAX_CHUNKS]
        
        result = {
            'input_ids': torch.stack([chunk['input_ids'] for chunk in chunk_encodings]),
            'attention_mask': torch.stack([chunk['attention_mask'] for chunk in chunk_encodings]),
            'num_chunks': torch.tensor(min(len(chunks), MAX_CHUNKS), dtype=torch.long)
        }
        
        if self.labels is not None:
            result['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
            
        return result

class OptimizedKoBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout_rate=0.1):  # 드롭아웃 감소
        super(OptimizedKoBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 일부 BERT 레이어 프리징 (훈련 속도 향상)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:8]:  # 처음 8개 레이어 프리징
            for param in layer.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # 간소화된 어텐션 (계산량 감소)
        self.chunk_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True, dropout=dropout_rate  # 헤드 수 감소
        )
        
        # 더 간단한 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, num_chunks):
        batch_size, num_chunks_max, seq_len = input_ids.shape
        
        # 혼합 정밀도 사용
        with torch.cuda.amp.autocast():
            # 모든 청크를 배치로 처리
            input_ids_flat = input_ids.view(-1, seq_len)
            attention_mask_flat = attention_mask.view(-1, seq_len)
            
            # BERT 인코딩
            outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
            
            # CLS 토큰 추출
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                chunk_embeddings = outputs.pooler_output
            else:
                chunk_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # 청크 임베딩을 다시 배치 형태로 변환
            chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks_max, -1)
            
            # 간단한 평균 풀링 (어텐션 대신 - 더 빠름)
            chunk_mask = torch.zeros(batch_size, num_chunks_max, device=chunk_embeddings.device)
            for i, num_chunk in enumerate(num_chunks):
                chunk_mask[i, :num_chunk] = 1
            
            # 마스크된 평균
            masked_embeddings = chunk_embeddings * chunk_mask.unsqueeze(-1)
            doc_embedding = masked_embeddings.sum(dim=1) / (chunk_mask.sum(dim=1, keepdim=True) + 1e-8)
            
            # 분류
            logits = self.classifier(doc_embedding)
        
        return logits

def train_epoch(model, data_loader, optimizer, criterion, scheduler, device, scaler):
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    for batch in tqdm(data_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        num_chunks = batch['num_chunks'].to(device)
        
        optimizer.zero_grad()
        
        # 혼합 정밀도 사용
        with torch.cuda.amp.autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, num_chunks=num_chunks)
            loss = criterion(logits, labels)
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item()
        
        # 스케일된 역전파
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    
    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset)

def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    predictions = []
    probabilities = []
    real_values = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            num_chunks = batch['num_chunks'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask, num_chunks=num_chunks)
            loss = criterion(logits, labels)
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_loss += loss.item()
            
            # 확률 계산
            probs = torch.softmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())  # 클래스 1의 확률
            real_values.extend(labels.cpu().numpy())
    
    return (total_loss / len(data_loader), 
            correct_predictions.double() / len(data_loader.dataset),
            predictions, probabilities, real_values)

def predict_test(model, data_loader, device):
    model.eval()
    test_predictions = []
    test_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            num_chunks = batch['num_chunks'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask, num_chunks=num_chunks)
            probs = torch.softmax(logits, dim=1)
            
            _, preds = torch.max(logits, dim=1)
            
            test_predictions.extend(preds.cpu().numpy())
            test_probabilities.extend(probs[:, 1].cpu().numpy())  # 클래스 1의 확률
    
    return test_predictions, test_probabilities

def main():
    print("Loading data...")
    # 데이터 로드
    train = pd.read_csv('./train.csv', encoding='utf-8-sig')
    test = pd.read_csv('./test.csv', encoding='utf-8-sig')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # 텍스트 결합 및 전처리
    print("Preprocessing text data...")
    
    train['title'] = train['title'].fillna('').str.strip()
    train['full_text'] = train['full_text'].fillna('').str.replace(r'\n\s*\n', '\n', regex=True).str.strip()
    train['combined_text'] = train['title'] + ' [SEP] ' + train['full_text']
    
    # 텍스트 길이 분석
    text_lengths = train['combined_text'].str.len()
    print(f"Text length stats:")
    print(f"  Mean: {text_lengths.mean():.1f}")
    print(f"  Median: {text_lengths.median():.1f}")
    print(f"  Max: {text_lengths.max()}")
    print(f"  95th percentile: {text_lengths.quantile(0.95):.1f}")
    print(f"  Texts > 1000 chars: {(text_lengths > 1000).sum()}/{len(train)}")
    print(f"  Texts > 2000 chars: {(text_lengths > 2000).sum()}/{len(train)}")
    
    X = train['combined_text']
    y = train['generated']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Label distribution - Train: {y_train.value_counts().to_dict()}")
    print(f"Label distribution - Val: {y_val.value_counts().to_dict()}")
    
    print("Loading tokenizer and model...")
    # KoBERT 모델과 토크나이저 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print(f"Successfully loaded slow tokenizer: {model_name}")
    except Exception as e:
        print(f"Slow tokenizer failed: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            print(f"Successfully loaded fast tokenizer: {model_name}")
        except Exception as e2:
            print(f"Fast tokenizer also failed: {e2}")
            print("Trying alternative KoBERT implementation...")
            
            try:
                model_name = 'monologg/kobert'
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f"Successfully loaded alternative KoBERT: {model_name}")
            except Exception as e3:
                print(f"Alternative KoBERT failed: {e3}")
                model_name = 'klue/bert-base'
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f"Fallback to KLUE BERT: {model_name}")
    
    # 데이터셋 생성 (계층적 처리)
    train_dataset = LongTextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = LongTextDataset(X_val, y_val, tokenizer, MAX_LEN)
    
    # 데이터로더 생성 (더 큰 배치, 더 많은 워커)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # 모델 초기화 (최적화된 모델)
    model = OptimizedKoBERTClassifier(model_name, num_classes=2)
    model = model.to(device)
    
    # 혼합 정밀도 스케일러
    scaler = torch.cuda.amp.GradScaler()
    
    # 옵티마이저 및 스케줄러 설정 (더 적극적인 설정)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Using optimized processing for RTX 3070Ti")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"Estimated training time: ~2-4 hours")
    print("Starting training...")
    
    # 훈련 실행
    best_auc = 0
    best_model = None
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 50)
        
        # 훈련 (스케일러 추가)
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler, device, scaler)
        
        # 검증
        val_loss, val_acc, val_preds, val_probs, val_labels = eval_model(model, val_loader, criterion, device)
        
        # AUC 계산
        val_auc = roc_auc_score(val_labels, val_probs)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
        
        # 최고 모델 저장
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict().copy()
            print(f'New best model! AUC: {best_auc:.4f}')
        
        print()
    
    print(f'Best Validation AUC: {best_auc:.4f}')
    
    # 최고 모델 로드
    model.load_state_dict(best_model)
    model.eval()
    
    print("Preparing test data...")
    # 테스트 데이터 전처리
    test = test.rename(columns={'paragraph_text': 'full_text'})
    test['title'] = test['title'].fillna('').str.strip()
    test['full_text'] = test['full_text'].fillna('').str.replace(r'\n\s*\n', '\n', regex=True).str.strip()
    test['combined_text'] = test['title'] + ' [SEP] ' + test['full_text']
    
    # 테스트 데이터 길이 확인
    test_lengths = test['combined_text'].str.len()
    print(f"Test text length stats:")
    print(f"  Mean: {test_lengths.mean():.1f}")
    print(f"  Max: {test_lengths.max()}")
    print(f"  Texts > 1000 chars: {(test_lengths > 1000).sum()}/{len(test)}")
    
    # 테스트 데이터셋 생성
    test_dataset = LongTextDataset(test['combined_text'], labels=None, tokenizer=tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, 
                            num_workers=4, pin_memory=True)  # 추론 시 더 큰 배치
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("Making predictions...")
    # 테스트 데이터 예측
    test_predictions, test_probabilities = predict_test(model, test_loader, device)
    
    print(f"Generated {len(test_probabilities)} predictions")
    print(f"Probability range: {min(test_probabilities):.4f} - {max(test_probabilities):.4f}")
    print(f"Mean probability: {np.mean(test_probabilities):.4f}")
    
    # 제출 파일 생성
    sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = test_probabilities
    
    # 제출 파일 저장
    sample_submission.to_csv('./baseline_submission.csv', index=False)
    
    print("Submission file saved as 'baseline_submission.csv'")
    print(f"Submission shape: {sample_submission.shape}")
    print(f"Model used: {model_name}")
    print(sample_submission.head())
    print("Training completed!")

if __name__ == "__main__":
    main()