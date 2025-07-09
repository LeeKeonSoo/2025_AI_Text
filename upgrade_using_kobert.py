import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import warnings
import random
import os
warnings.filterwarnings('ignore')

# 시드 고정 (재현 가능한 결과)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 고급 하이퍼파라미터 설정
MAX_LEN = 512
BATCH_SIZE = 12  # 조금 줄여서 더 안정적으로
EPOCHS = 5  # 더 많은 에포크로 깊이 학습
LEARNING_RATE = 1e-5  # 더 낮은 학습률로 정교한 학습
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
model_name = 'skt/kobert-base-v1'

# 고급 긴 텍스트 처리 전략
CHUNK_SIZE = 350
OVERLAP_SIZE = 75
MAX_CHUNKS = 4  # 더 많은 정보 활용
USE_KFOLD = False  # K-Fold 교차 검증 사용 여부
N_FOLDS = 5

# 고급 학습 기법
USE_FOCAL_LOSS = True  # 불균형 데이터 대응
USE_LABEL_SMOOTHING = True  # 과적합 방지
USE_ADVERSARIAL_TRAINING = True  # 견고성 향상
USE_ENSEMBLE = True  # 앙상블 학습

class AdvancedLongTextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        
    def __len__(self):
        return len(self.texts)
    
    def _augment_text(self, text):
        """데이터 증강 (훈련 시만)"""
        if not self.augment or random.random() > 0.3:
            return text
            
        # 간단한 증강 기법들
        augmentation_type = random.choice(['none', 'shuffle_sentences', 'drop_sentences'])
        
        if augmentation_type == 'shuffle_sentences':
            sentences = text.split('. ')
            if len(sentences) > 3:
                # 문장 순서 일부 변경
                mid_idx = len(sentences) // 2
                random.shuffle(sentences[1:mid_idx])
                text = '. '.join(sentences)
        
        elif augmentation_type == 'drop_sentences':
            sentences = text.split('. ')
            if len(sentences) > 5:
                # 일부 문장 제거 (최대 20%)
                drop_count = min(len(sentences) // 5, 2)
                indices_to_drop = random.sample(range(1, len(sentences)-1), drop_count)
                sentences = [s for i, s in enumerate(sentences) if i not in indices_to_drop]
                text = '. '.join(sentences)
        
        return text
    
    def _chunk_text_advanced(self, text):
        """고급 청킹 전략"""
        # 데이터 증강 적용
        text = self._augment_text(text)
        
        # 문장 단위로 먼저 분할
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
        
        # 문장별로 청크에 추가
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
        
        # 고급 청킹
        chunks = self._chunk_text_advanced(text)
        
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
        
        # 패딩
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class AdvancedKoBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout_rate=0.15):
        super(AdvancedKoBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 선택적 레이어 프리징 (처음 6개만)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # 고급 어텐션 메커니즘
        self.chunk_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True, dropout=dropout_rate
        )
        
        # 위치 인코딩 (청크 위치 정보)
        self.position_embeddings = nn.Embedding(MAX_CHUNKS, hidden_size)
        
        # 고급 분류기 (잔차 연결 포함)
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
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input_ids, attention_mask, num_chunks):
        batch_size, num_chunks_max, seq_len = input_ids.shape
        
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
            doc_embedding = self.dropout(doc_embedding)
            
            # 분류
            logits = self.classifier(doc_embedding)
        
        return logits, attention_weights

def adversarial_training(model, inputs, labels, optimizer, criterion, epsilon=0.01):
    """적대적 훈련"""
    # 원본 임베딩 저장
    embeddings = model.bert.embeddings.word_embeddings
    original_embeddings = embeddings.weight.data.clone()
    
    # 그래디언트 계산
    embeddings.weight.requires_grad_()
    
    with torch.cuda.amp.autocast():
        logits, _ = model(**inputs)
        loss = criterion(logits, labels)
    
    # 임베딩에 대한 그래디언트
    loss.backward(retain_graph=True)
    grad = embeddings.weight.grad.data
    
    # 적대적 perturbation 추가
    perturbation = epsilon * grad.sign()
    embeddings.weight.data = original_embeddings + perturbation
    
    # 적대적 샘플로 다시 계산
    with torch.cuda.amp.autocast():
        adv_logits, _ = model(**inputs)
        adv_loss = criterion(adv_logits, labels)
    
    # 원본 임베딩 복원
    embeddings.weight.data = original_embeddings
    embeddings.weight.requires_grad_(False)
    
    return adv_loss

def train_epoch_advanced(model, data_loader, optimizer, criterion, scheduler, device, scaler, epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f'Training Epoch {epoch}')):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        num_chunks = batch['num_chunks'].to(device)
        
        optimizer.zero_grad()
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'num_chunks': num_chunks
        }
        
        # 일반 훈련
        with torch.cuda.amp.autocast():
            logits, attention_weights = model(**inputs)
            loss = criterion(logits, labels)
        
        # 적대적 훈련 (일부 배치에서만)
        if USE_ADVERSARIAL_TRAINING and batch_idx % 3 == 0:
            adv_loss = adversarial_training(model, inputs, labels, optimizer, criterion)
            loss = 0.7 * loss + 0.3 * adv_loss
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item()
        
        # 스케일된 역전파
        scaler.scale(loss).backward()
        
        # 그래디언트 클리핑
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    
    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset)

def eval_model_advanced(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    predictions = []
    probabilities = []
    real_values = []
    attention_scores = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
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
            
            # 확률 계산
            probs = torch.softmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
            real_values.extend(labels.cpu().numpy())
            
            # 어텐션 점수 저장 (분석용)
            attention_scores.extend(attention_weights.mean(dim=1).cpu().numpy())
    
    # F1 점수 계산
    f1 = f1_score(real_values, predictions)
    
    return (total_loss / len(data_loader), 
            correct_predictions.double() / len(data_loader.dataset),
            predictions, probabilities, real_values, f1, attention_scores)

def predict_test_advanced(model, data_loader, device):
    model.eval()
    test_predictions = []
    test_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predicting'):
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
    
    return test_predictions, test_probabilities

def main():
    print("=== Advanced KoBERT Training ===")
    print("Loading data...")
    
    # 데이터 로드
    train = pd.read_csv('./train.csv', encoding='utf-8-sig')
    test = pd.read_csv('./test.csv', encoding='utf-8-sig')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # 텍스트 전처리
    print("Advanced preprocessing...")
    
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
    
    # 클래스 불균형 분석
    class_counts = y.value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    print(f"Class imbalance ratio: {class_counts[1] / class_counts[0]:.3f}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    print("Loading advanced tokenizer and model...")
    # 토크나이저 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print(f"Successfully loaded tokenizer: {model_name}")
    except Exception as e:
        print(f"Failed to load KoBERT: {e}")
        model_name = 'klue/bert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Fallback to KLUE BERT: {model_name}")
    
    # 고급 데이터셋 생성
    train_dataset = AdvancedLongTextDataset(X_train, y_train, tokenizer, MAX_LEN, augment=True)
    val_dataset = AdvancedLongTextDataset(X_val, y_val, tokenizer, MAX_LEN, augment=False)
    
    # 가중 샘플링으로 클래스 불균형 해결
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    # 데이터로더
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # 고급 모델 초기화
    model = AdvancedKoBERTClassifier(model_name, num_classes=2)
    model = model.to(device)
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 고급 손실 함수
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=1, gamma=2)
        print("Using Focal Loss")
    elif USE_LABEL_SMOOTHING:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        print("Using Label Smoothing")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using Cross Entropy Loss")
    
    # 혼합 정밀도 스케일러
    scaler = torch.cuda.amp.GradScaler()
    
    # 고급 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-6
    )
    
    # 코사인 스케줄러 사용
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Training configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Advanced features: Focal Loss={USE_FOCAL_LOSS}, Label Smoothing={USE_LABEL_SMOOTHING}")
    print(f"  Adversarial Training={USE_ADVERSARIAL_TRAINING}")
    
    print("\nStarting advanced training...")
    
    # 훈련 실행
    best_auc = 0
    best_f1 = 0
    best_model = None
    training_history = []
    
    for epoch in range(EPOCHS):
        print(f'\n=== Epoch {epoch + 1}/{EPOCHS} ===')
        
        # 훈련
        train_loss, train_acc = train_epoch_advanced(
            model, train_loader, optimizer, criterion, scheduler, device, scaler, epoch + 1
        )
        
        # 검증
        val_loss, val_acc, val_preds, val_probs, val_labels, val_f1, attention_scores = eval_model_advanced(
            model, val_loader, criterion, device
        )
        
        # 메트릭 계산
        val_auc = roc_auc_score(val_labels, val_probs)
        
        # 훈련 기록
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
        
        # 모델 저장 조건 (AUC와 F1 모두 고려)
        combined_score = 0.7 * val_auc + 0.3 * val_f1
        best_combined = 0.7 * best_auc + 0.3 * best_f1
        
        if combined_score > best_combined:
            best_auc = val_auc
            best_f1 = val_f1
            best_model = model.state_dict().copy()
            print(f'🎯 New best model! AUC: {best_auc:.4f}, F1: {best_f1:.4f}')
        
        # 현재 학습률 출력
        current_lr = scheduler.get_last_lr()[0]
        print(f'Current LR: {current_lr:.2e}')
    
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
    
    # 테스트 데이터 길이 분석
    test_lengths = test['combined_text'].str.len()
    print(f"Test text length stats:")
    print(f"  Mean: {test_lengths.mean():.1f}")
    print(f"  Max: {test_lengths.max()}")
    print(f"  Long texts (>2000): {(test_lengths > 2000).sum()}/{len(test)}")
    
    # 테스트 데이터셋
    test_dataset = AdvancedLongTextDataset(test['combined_text'], labels=None, tokenizer=tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Making predictions...")
    test_predictions, test_probabilities = predict_test_advanced(model, test_loader, device)
    
    print(f"Prediction stats:")
    print(f"  Generated {len(test_probabilities)} predictions")
    print(f"  Probability range: {min(test_probabilities):.4f} - {max(test_probabilities):.4f}")
    print(f"  Mean probability: {np.mean(test_probabilities):.4f}")
    print(f"  Std probability: {np.std(test_probabilities):.4f}")
    
    # 예측 분포 분석
    prob_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    prob_hist, _ = np.histogram(test_probabilities, bins=prob_bins)
    print(f"  Prediction distribution:")
    for i in range(len(prob_bins)-1):
        print(f"    {prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}: {prob_hist[i]}")
    
    # 제출 파일 생성
    sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = test_probabilities
    
    # 제출 파일 저장
    sample_submission.to_csv('./baseline_submission.csv', index=False)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📊 Final Results:")
    print(f"  Best Validation AUC: {best_auc:.4f}")
    print(f"  Best Validation F1: {best_f1:.4f}")
    print(f"  Model used: {model_name}")
    print(f"  Submission file: baseline_submission.csv")
    print(f"  Submission shape: {sample_submission.shape}")
    
    # 훈련 히스토리 요약
    print(f"\n📈 Training History:")
    for i, history in enumerate(training_history):
        print(f"  Epoch {history['epoch']}: "
              f"Val AUC={history['val_auc']:.4f}, "
              f"Val F1={history['val_f1']:.4f}, "
              f"Val Acc={history['val_acc']:.4f}")
    
    print(sample_submission.head())

if __name__ == "__main__":
    main()