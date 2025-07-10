import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')

# 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 8GB GPU 메모리 최대 활용 하이퍼파라미터
MAX_LEN = 512        # 최대 시퀀스 길이로 설정
BATCH_SIZE = 12      # 큰 배치 크기
GRADIENT_ACCUMULATION_STEPS = 1  # 누적 없이 바로 업데이트
EPOCHS = 5           # 더 많은 에포크
LEARNING_RATE = 2e-5 # 더 높은 학습률
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
model_name = 'skt/kobert-base-v1'

# 고성능 텍스트 청킹 설정
CHUNK_SIZE = 400     # 더 큰 청크
OVERLAP_SIZE = 100   # 더 많은 오버랩
MAX_CHUNKS = 4       # 더 많은 청크로 정보 보존

class LongTextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def _chunk_text(self, text):
        """텍스트를 청크로 분할"""
        # 문장 단위로 분할
        sentences = text.split('. ')
        if len(sentences) == 1:
            sentences = text.split('.')
        
        # 토큰 기반 청킹
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= self.max_len - 2:
            return [text]
        
        chunks = []
        current_chunk_tokens = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence + '.', add_special_tokens=False)
            
            if current_length + len(sentence_tokens) > CHUNK_SIZE:
                if current_chunk_tokens:
                    chunk_text = self.tokenizer.decode(current_chunk_tokens, skip_special_tokens=True)
                    chunks.append(chunk_text)
                    # 오버랩 유지
                    overlap_tokens = current_chunk_tokens[-OVERLAP_SIZE:] if len(current_chunk_tokens) > OVERLAP_SIZE else []
                    current_chunk_tokens = overlap_tokens + sentence_tokens
                    current_length = len(current_chunk_tokens)
                else:
                    current_chunk_tokens = sentence_tokens
                    current_length = len(sentence_tokens)
            else:
                current_chunk_tokens.extend(sentence_tokens)
                current_length += len(sentence_tokens)
        
        # 마지막 청크 추가
        if current_chunk_tokens:
            chunk_text = self.tokenizer.decode(current_chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        
        return chunks
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        
        # 텍스트 청킹
        chunks = self._chunk_text(text)
        
        # 각 청크 토크나이징
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
        
        # 패딩 (고정 크기)
        while len(chunk_encodings) < MAX_CHUNKS:
            chunk_encodings.append({
                'input_ids': torch.zeros(self.max_len, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_len, dtype=torch.long)
            })
        
        chunk_encodings = chunk_encodings[:MAX_CHUNKS]
        
        result = {
            'input_ids': torch.stack([chunk['input_ids'] for chunk in chunk_encodings]),
            'attention_mask': torch.stack([chunk['attention_mask'] for chunk in chunk_encodings]),
            'num_chunks': torch.tensor(min(len(chunks), MAX_CHUNKS), dtype=torch.long)
        }
        
        if self.labels is not None:
            result['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
            
        return result

class HighPerformanceKoBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout_rate=0.15):
        super(HighPerformanceKoBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 적당한 레이어 프리징 (성능과 메모리 균형)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:6]:  # 6개만 프리징 (더 많은 학습)
            for param in layer.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # 고성능 어텐션 메커니즘
        self.chunk_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True, dropout=dropout_rate
        )
        
        # 위치 인코딩
        self.position_embeddings = nn.Embedding(MAX_CHUNKS, hidden_size)
        
        # 고성능 분류기
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, num_chunks):
        batch_size, num_chunks_max, seq_len = input_ids.shape
        
        with torch.cuda.amp.autocast():
            # 배치 처리로 성능 최적화
            input_ids_flat = input_ids.view(-1, seq_len)
            attention_mask_flat = attention_mask.view(-1, seq_len)
            
            # BERT 인코딩
            outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                chunk_embeddings = outputs.pooler_output
            else:
                chunk_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # 청크 임베딩을 배치 형태로 변환
            chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks_max, -1)
            
            # 위치 인코딩 추가
            positions = torch.arange(num_chunks_max, device=chunk_embeddings.device).unsqueeze(0).repeat(batch_size, 1)
            position_embeddings = self.position_embeddings(positions)
            chunk_embeddings = chunk_embeddings + position_embeddings
            
            # 마스크 생성
            chunk_mask = torch.zeros(batch_size, num_chunks_max, device=chunk_embeddings.device)
            for i, num_chunk in enumerate(num_chunks):
                chunk_mask[i, :num_chunk] = 1
            
            # 어텐션 적용
            attended_chunks, attention_weights = self.chunk_attention(
                chunk_embeddings, chunk_embeddings, chunk_embeddings,
                key_padding_mask=(chunk_mask == 0)
            )
            
            # 가중 평균으로 문서 표현 생성
            doc_mask = chunk_mask.unsqueeze(-1)
            weighted_chunks = attended_chunks * doc_mask
            doc_embedding = weighted_chunks.sum(dim=1) / (doc_mask.sum(dim=1) + 1e-8)
            
            # 잔차 연결
            doc_embedding = doc_embedding + self.pre_classifier(doc_embedding)
            
            # 분류
            logits = self.classifier(doc_embedding)
        
        return logits, attention_weights

def train_epoch(model, data_loader, optimizer, criterion, scheduler, device, scaler, epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f'Training Epoch {epoch}')):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        num_chunks = batch['num_chunks'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits, attention_weights = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_chunks=num_chunks
            )
            loss = criterion(logits, labels)
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item()
        
        # 역전파
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 덜 빈번한 메모리 정리 (성능 향상)
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset)

def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    predictions = []
    probabilities = []
    real_values = []
    attention_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Evaluating')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            num_chunks = batch['num_chunks'].to(device)
            
            with torch.cuda.amp.autocast():
                logits, attention_weights = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    num_chunks=num_chunks
                )
                loss = criterion(logits, labels)
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
            real_values.extend(labels.cpu().numpy())
            attention_scores.extend(attention_weights.mean(dim=1).cpu().numpy())
            
            # 덜 빈번한 메모리 정리
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
    
    f1 = f1_score(real_values, predictions)
    
    return (total_loss / len(data_loader), 
            correct_predictions.double() / len(data_loader.dataset),
            predictions, probabilities, real_values, f1, attention_scores)

def predict_test(model, data_loader, device):
    model.eval()
    test_predictions = []
    test_probabilities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Predicting')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            num_chunks = batch['num_chunks'].to(device)
            
            with torch.cuda.amp.autocast():
                logits, _ = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    num_chunks=num_chunks
                )
            
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, dim=1)
            
            test_predictions.extend(preds.cpu().numpy())
            test_probabilities.extend(probs[:, 1].cpu().numpy())
            
            # 덜 빈번한 메모리 정리
            if batch_idx % 15 == 0:
                torch.cuda.empty_cache()
    
    return test_predictions, test_probabilities

def main():
    print("=== Memory-Optimized KoBERT Training ===")
    print("Loading data...")
    
    # 데이터 로드
    train = pd.read_csv('./train.csv', encoding='utf-8-sig')
    test = pd.read_csv('./test.csv', encoding='utf-8-sig')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # 텍스트 전처리
    print("Preprocessing text data...")
    
    train['title'] = train['title'].fillna('').str.strip()
    train['full_text'] = train['full_text'].fillna('').str.replace(r'\n\s*\n', '\n', regex=True).str.strip()
    train['combined_text'] = train['title'] + ' [SEP] ' + train['full_text']
    
    # 텍스트 길이 분석
    text_lengths = train['combined_text'].str.len()
    print(f"Text length analysis:")
    print(f"  Mean: {text_lengths.mean():.1f}")
    print(f"  Median: {text_lengths.median():.1f}")
    print(f"  95th percentile: {text_lengths.quantile(0.95):.1f}")
    print(f"  Max: {text_lengths.max()}")
    print(f"  Texts > 2000 chars: {(text_lengths > 2000).sum()}/{len(train)} ({(text_lengths > 2000).mean()*100:.1f}%)")
    
    X = train['combined_text']
    y = train['generated']
    
    # 클래스 분포
    class_counts = y.value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    print(f"Class imbalance ratio: {class_counts[1] / class_counts[0]:.3f}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # 토크나이저 로드
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print(f"Successfully loaded tokenizer: {model_name}")
    except Exception as e:
        print(f"Failed to load KoBERT: {e}")
        model_name = 'klue/bert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Fallback to KLUE BERT: {model_name}")
    
    # 데이터셋 생성
    train_dataset = LongTextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = LongTextDataset(X_val, y_val, tokenizer, MAX_LEN)
    
    # 데이터로더 (고성능 설정)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True  # 워커 수 증가
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # 고성능 모델 초기화
    model = HighPerformanceKoBERTClassifier(model_name, num_classes=2)
    model = model.to(device)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-6
    )
    
    # 스케줄러 (그래디언트 누적 없음)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\n8GB GPU 최대 활용 설정:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max length: {MAX_LEN}")
    print(f"  Max chunks: {MAX_CHUNKS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    print(f"\n고성능 최적화:")
    print(f"  - Full sequence length: {MAX_LEN}")
    print(f"  - Large batch size: {BATCH_SIZE}")
    print(f"  - Maximum chunks: {MAX_CHUNKS}")
    print(f"  - Parallel processing")
    print(f"  - Advanced attention mechanism")
    print(f"  - Position encoding")
    print(f"  - Frozen layers: 6/12 (더 많은 학습)")
    
    print("\n8GB 메모리를 최대한 활용한 고성능 훈련 시작...")
    torch.cuda.empty_cache()
    
    # 훈련 실행
    best_auc = 0
    best_f1 = 0
    best_model = None
    training_history = []
    
    for epoch in range(EPOCHS):
        print(f'\n=== Epoch {epoch + 1}/{EPOCHS} ===')
        
        # GPU 메모리 상태 (8GB 활용도 확인)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            utilization = (memory_allocated / 8.0) * 100  # 8GB 기준 활용률
            print(f'GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved')
            print(f'8GB 활용률: {utilization:.1f}%')
        
        # 훈련
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, scaler, epoch + 1
        )
        
        # 검증
        val_loss, val_acc, val_preds, val_probs, val_labels, val_f1, attention_scores = eval_model(
            model, val_loader, criterion, device
        )
        
        # 메트릭
        val_auc = roc_auc_score(val_labels, val_probs)
        
        # 기록
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc.item(),
            'val_loss': val_loss,
            'val_acc': val_acc.item(),
            'val_auc': val_auc,
            'val_f1': val_f1
        }
        training_history.append(epoch_metrics)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}')
        
        # 최고 모델 저장
        combined_score = 0.7 * val_auc + 0.3 * val_f1
        best_combined = 0.7 * best_auc + 0.3 * best_f1
        
        if combined_score > best_combined:
            best_auc = val_auc
            best_f1 = val_f1
            best_model = model.state_dict().copy()
            print(f'🎯 New best model! AUC: {best_auc:.4f}, F1: {best_f1:.4f}')
        
        current_lr = scheduler.get_last_lr()[0]
        print(f'Current LR: {current_lr:.2e}')
        
        torch.cuda.empty_cache()
    
    print(f'\n🏆 Best Results: AUC: {best_auc:.4f}, F1: {best_f1:.4f}')
    
    # 최고 모델 로드
    model.load_state_dict(best_model)
    model.eval()
    
    print("\nPreparing test data...")
    # 테스트 데이터 전처리
    test = test.rename(columns={'paragraph_text': 'full_text'})
    test['title'] = test['title'].fillna('').str.strip()
    test['full_text'] = test['full_text'].fillna('').str.replace(r'\n\s*\n', '\n', regex=True).str.strip()
    test['combined_text'] = test['title'] + ' [SEP] ' + test['full_text']
    
    # 테스트 데이터 길이
    test_lengths = test['combined_text'].str.len()
    print(f"Test text length stats:")
    print(f"  Mean: {test_lengths.mean():.1f}")
    print(f"  Max: {test_lengths.max()}")
    print(f"  Long texts (>2000): {(test_lengths > 2000).sum()}/{len(test)}")
    
    # 테스트 데이터셋 (고성능 설정)
    test_dataset = LongTextDataset(test['combined_text'], labels=None, tokenizer=tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Making high-performance predictions...")
    test_predictions, test_probabilities = predict_test(model, test_loader, device)
    
    print(f"\nPrediction statistics:")
    print(f"  Generated {len(test_probabilities)} predictions")
    print(f"  Probability range: {min(test_probabilities):.4f} - {max(test_probabilities):.4f}")
    print(f"  Mean probability: {np.mean(test_probabilities):.4f}")
    print(f"  Std probability: {np.std(test_probabilities):.4f}")
    
    # 예측 분포
    prob_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    prob_hist, _ = np.histogram(test_probabilities, bins=prob_bins)
    print(f"  Prediction distribution:")
    for i in range(len(prob_bins)-1):
        print(f"    {prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}: {prob_hist[i]}")
    
    # 제출 파일 생성
    sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = test_probabilities
    sample_submission.to_csv('./baseline_submission.csv', index=False)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📊 Final Results:")
    print(f"  Best Validation AUC: {best_auc:.4f}")
    print(f"  Best Validation F1: {best_f1:.4f}")
    print(f"  Model used: {model_name}")
    print(f"  Submission file: baseline_submission.csv")
    print(f"  Submission shape: {sample_submission.shape}")
    
    print(f"\n📈 Training History:")
    for history in training_history:
        print(f"  Epoch {history['epoch']}: "
              f"AUC={history['val_auc']:.4f}, "
              f"F1={history['val_f1']:.4f}, "
              f"Acc={history['val_acc']:.4f}")
    
    torch.cuda.empty_cache()
    print(f"\n💾 Final cleanup completed")
    print(sample_submission.head())

if __name__ == "__main__":
    main()