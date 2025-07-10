# ================================================================
# 🚀 진화된 고성능 KoBERT (0.83+ 코드 개선 버전)
# ================================================================

!pip install transformers==4.36.0
!pip install torch torchvision torchaudio
!pip install scikit-learn
!pip install pandas numpy tqdm

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import warnings
import random
import os
import gc
from datetime import datetime
import math

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔥 Using device: {device}')

# 파일 경로
DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks'
DATA_PATH = f'{DRIVE_PATH}'
RESULT_PATH = f'{DRIVE_PATH}/results'
os.makedirs(RESULT_PATH, exist_ok=True)

# 🎯 진화된 하이퍼파라미터 (현재 환경 최적화)
MAX_LEN = 512
BATCH_SIZE = 10          # Colab 환경 고려 (원래 12에서 조정)
GRADIENT_ACCUMULATION_STEPS = 2  # 효과적 배치 크기 20
EPOCHS = 6               # 더 충분한 학습
LEARNING_RATE = 1.5e-5   # 약간 낮춘 학습률로 안정성
WARMUP_RATIO = 0.15      # 더 긴 워밍업
WEIGHT_DECAY = 0.01

MODEL_NAME = 'skt/kobert-base-v1'

# 🎯 개선된 청킹 설정
CHUNK_SIZE = 420         # 약간 증가
OVERLAP_SIZE = 80        # 최적화된 오버랩
MAX_CHUNKS = 4           # 검증된 설정 유지

# ================================================================
# 진화된 텍스트 데이터셋
# ================================================================

class EvolvedLongTextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment  # 새로운 증강 기능
        
    def __len__(self):
        return len(self.texts)
    
    def _enhanced_chunk_text(self, text):
        """개선된 텍스트 청킹 (원래 방식 + 안전성 강화)"""
        if not text.strip():
            return [""]
        
        # 원래 방식: 문장 단위 분할
        sentences = text.split('. ')
        if len(sentences) == 1:
            sentences = text.split('.')
        
        # 토큰 기반 청킹 (검증된 방식)
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        except:
            # 토크나이저 에러 시 안전한 처리
            return [text[:self.max_len-50]]
        
        if len(tokens) <= self.max_len - 2:
            return [text]
        
        chunks = []
        current_chunk_tokens = []
        current_length = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            try:
                sentence_tokens = self.tokenizer.encode(sentence + '.', add_special_tokens=False)
            except:
                continue
            
            if current_length + len(sentence_tokens) > CHUNK_SIZE:
                if current_chunk_tokens:
                    chunk_text = self.tokenizer.decode(current_chunk_tokens, skip_special_tokens=True)
                    chunks.append(chunk_text)
                    # 개선된 오버랩 (정보 손실 최소화)
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
        
        return chunks if chunks else [text[:CHUNK_SIZE*2]]
    
    def _simple_augment(self, text):
        """간단하고 효과적인 데이터 증강"""
        if not self.augment or random.random() > 0.2:
            return text
        
        # 1. 공백 패턴 변경 (AI 텍스트 탐지에 중요)
        if random.random() > 0.5:
            text = text.replace('  ', ' ')  # 이중 공백 제거
            
        # 2. 구두점 정규화
        if random.random() > 0.7:
            text = text.replace('...', '.')
            text = text.replace('!!', '!')
            
        return text
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        
        # 데이터 증강 적용
        text = self._simple_augment(text)
        
        # 텍스트 청킹 (검증된 방식)
        chunks = self._enhanced_chunk_text(text)
        
        # 각 청크 토크나이징
        chunk_encodings = []
        for chunk in chunks:
            try:
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
            except:
                # 에러 시 빈 청크 추가
                chunk_encodings.append({
                    'input_ids': torch.zeros(self.max_len, dtype=torch.long),
                    'attention_mask': torch.zeros(self.max_len, dtype=torch.long)
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

# ================================================================
# 진화된 모델 아키텍처
# ================================================================

class EvolvedKoBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout_rate=0.15):
        super(EvolvedKoBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 검증된 레이어 고정 (6개 유지)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # 개선된 어텐션 메커니즘
        self.chunk_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True, dropout=dropout_rate
        )
        
        # 위치 인코딩 (검증된 요소)
        self.position_embeddings = nn.Embedding(MAX_CHUNKS, hidden_size)
        
        # 🆕 청크 중요도 계산 네트워크 (새 기능)
        self.chunk_importance = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 🆕 적응적 풀링 (새 기능)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 기존 검증된 분류기 구조 유지 + 개선
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),  # ReLU → GELU (성능 향상)
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # 🆕 출력 보정 (새 기능)
        self.output_calibration = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.Dropout(dropout_rate // 4)
        )
        
    def forward(self, input_ids, attention_mask, num_chunks):
        batch_size, num_chunks_max, seq_len = input_ids.shape
        
        with torch.cuda.amp.autocast():
            # 기존 검증된 인코딩 방식
            input_ids_flat = input_ids.view(-1, seq_len)
            attention_mask_flat = attention_mask.view(-1, seq_len)
            
            outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                chunk_embeddings = outputs.pooler_output
            else:
                chunk_embeddings = outputs.last_hidden_state[:, 0, :]
            
            chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks_max, -1)
            
            # 검증된 위치 인코딩
            positions = torch.arange(num_chunks_max, device=chunk_embeddings.device).unsqueeze(0).repeat(batch_size, 1)
            position_embeddings = self.position_embeddings(positions)
            chunk_embeddings = chunk_embeddings + position_embeddings
            
            # 청크 마스크 (검증된 방식)
            chunk_mask = torch.zeros(batch_size, num_chunks_max, device=chunk_embeddings.device)
            for i, num_chunk in enumerate(num_chunks):
                chunk_mask[i, :num_chunk] = 1
            
            # 🆕 청크 중요도 계산 (새 기능)
            chunk_importance = self.chunk_importance(chunk_embeddings).squeeze(-1)
            chunk_importance = chunk_importance * chunk_mask  # 마스킹 적용
            
            # 기존 어텐션 + 중요도 결합
            attended_chunks, attention_weights = self.chunk_attention(
                chunk_embeddings, chunk_embeddings, chunk_embeddings,
                key_padding_mask=(chunk_mask == 0)
            )
            
            # 🆕 적응적 가중 평균 (개선된 풀링)
            doc_mask = chunk_mask.unsqueeze(-1)
            
            # 어텐션과 중요도 결합
            combined_weights = (attention_weights.mean(dim=1) + chunk_importance) / 2
            combined_weights = F.softmax(combined_weights.masked_fill(chunk_mask == 0, float('-inf')), dim=1)
            
            weighted_chunks = attended_chunks * combined_weights.unsqueeze(-1)
            doc_embedding = weighted_chunks.sum(dim=1)
            
            # 기존 검증된 분류 과정
            doc_embedding = doc_embedding + self.pre_classifier(doc_embedding)
            logits = self.classifier(doc_embedding)
            
            # 🆕 출력 보정 (성능 미세 조정)
            calibrated_logits = logits + self.output_calibration(logits)
        
        return calibrated_logits, attention_weights

# ================================================================
# 진화된 훈련 함수들
# ================================================================

def evolved_train_epoch(model, data_loader, optimizer, criterion, scheduler, device, scaler, epoch):
    """진화된 훈련 에포크 (gradient accumulation 추가)"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    accumulated_loss = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f'Training Epoch {epoch}')):
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
            loss = criterion(logits, labels) / GRADIENT_ACCUMULATION_STEPS
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        accumulated_loss += loss.item()
        
        # Gradient accumulation
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += accumulated_loss
            accumulated_loss = 0
        
        # 메모리 정리 (더 자주)
        if batch_idx % 15 == 0:
            torch.cuda.empty_cache()
    
    # 남은 gradient 처리
    if accumulated_loss > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += accumulated_loss
    
    avg_loss = total_loss / (len(data_loader) // GRADIENT_ACCUMULATION_STEPS)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    
    return avg_loss, accuracy

def evolved_eval_model(model, data_loader, criterion, device):
    """진화된 평가 함수 (더 상세한 메트릭)"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    predictions = []
    probabilities = []
    real_values = []
    confidence_scores = []
    
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
            
            # 🆕 신뢰도 점수 계산
            max_probs = torch.max(probs, dim=1)[0]
            confidence_scores.extend(max_probs.cpu().numpy())
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
            real_values.extend(labels.cpu().numpy())
            
            if batch_idx % 15 == 0:
                torch.cuda.empty_cache()
    
    f1 = f1_score(real_values, predictions)
    avg_confidence = np.mean(confidence_scores)
    
    return (total_loss / len(data_loader), 
            correct_predictions.double() / len(data_loader.dataset),
            predictions, probabilities, real_values, f1, avg_confidence)

def main():
    print("🚀 진화된 고성능 KoBERT (0.83+ 기반 개선)")
    print("=" * 60)
    
    # 데이터 로드
    train = pd.read_csv(f'{DATA_PATH}/train.csv', encoding='utf-8-sig')
    test = pd.read_csv(f'{DATA_PATH}/test.csv', encoding='utf-8-sig')
    
    print(f"📊 데이터 정보:")
    print(f"   Train: {train.shape}")
    print(f"   Test: {test.shape}")
    
    # 텍스트 전처리 (검증된 방식 유지)
    train['title'] = train['title'].fillna('').str.strip()
    train['full_text'] = train['full_text'].fillna('').str.replace(r'\n\s*\n', '\n', regex=True).str.strip()
    train['combined_text'] = train['title'] + ' [SEP] ' + train['full_text']
    
    # 텍스트 길이 분석
    text_lengths = train['combined_text'].str.len()
    print(f"   텍스트 길이 - 평균: {text_lengths.mean():.0f}, 최대: {text_lengths.max():,}")
    print(f"   긴 텍스트 (>2000자): {(text_lengths > 2000).sum()}/{len(train)}")
    
    X = train['combined_text']
    y = train['generated']
    
    print(f"   클래스 분포: {dict(y.value_counts())}")
    
    # 데이터 분할 (검증된 비율)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # 토크나이저
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        print(f"✅ {MODEL_NAME} 토크나이저 로드 성공")
    except Exception as e:
        print(f"❌ KoBERT 로드 실패: {e}")
        MODEL_NAME = 'klue/bert-base'
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"   대체: {MODEL_NAME}")
    
    # 데이터셋 (훈련용은 증강 적용)
    train_dataset = EvolvedLongTextDataset(X_train, y_train, tokenizer, MAX_LEN, augment=True)
    val_dataset = EvolvedLongTextDataset(X_val, y_val, tokenizer, MAX_LEN, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # 모델 초기화
    model = EvolvedKoBERTClassifier(MODEL_NAME, num_classes=2)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏗️ 모델 파라미터: {trainable_params:,} / {total_params:,}")
    
    # 옵티마이저 및 스케줄러
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-6
    )
    
    # 🆕 코사인 스케줄러 (더 부드러운 학습률 감소)
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\n🎯 진화된 설정:")
    print(f"   배치 크기: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {effective_batch_size}")
    print(f"   에포크: {EPOCHS}")
    print(f"   학습률: {LEARNING_RATE}")
    print(f"   스케줄러: Cosine with warmup")
    print(f"   새 기능: 청크 중요도, 적응적 풀링, 출력 보정")
    
    # 훈련 실행
    best_score = 0
    best_model = None
    
    print("\n🚀 진화된 훈련 시작!")
    
    for epoch in range(EPOCHS):
        print(f'\n=== Epoch {epoch + 1}/{EPOCHS} ===')
        
        # 훈련
        train_loss, train_acc = evolved_train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, scaler, epoch + 1
        )
        
        # 검증
        val_loss, val_acc, val_preds, val_probs, val_labels, val_f1, avg_confidence = evolved_eval_model(
            model, val_loader, criterion, device
        )
        
        val_auc = roc_auc_score(val_labels, val_probs)
        
        print(f'📈 Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        print(f'📈 Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        print(f'🎯 Metrics - AUC: {val_auc:.4f}, F1: {val_f1:.4f}')
        print(f'💡 신뢰도: {avg_confidence:.4f}')
        
        # 🆕 개선된 복합 점수 (신뢰도 추가)
        combined_score = 0.6 * val_auc + 0.3 * val_f1 + 0.1 * avg_confidence
        
        if combined_score > best_score:
            best_score = combined_score
            best_model = model.state_dict().copy()
            print(f'⭐ 새로운 최고 성능! 복합점수: {combined_score:.4f}')
        
        torch.cuda.empty_cache()
    
    print(f'\n🏆 최고 복합 점수: {best_score:.4f}')
    
    # 테스트 예측
    model.load_state_dict(best_model)
    model.eval()
    
    # 테스트 데이터 전처리
    test = test.rename(columns={'paragraph_text': 'full_text'})
    test['title'] = test['title'].fillna('').str.strip()
    test['full_text'] = test['full_text'].fillna('').str.replace(r'\n\s*\n', '\n', regex=True).str.strip()
    test['combined_text'] = test['title'] + ' [SEP] ' + test['full_text']
    
    test_dataset = EvolvedLongTextDataset(test['combined_text'], labels=None, tokenizer=tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 예측
    test_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='🔮 진화된 예측'):
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
            test_probabilities.extend(probs[:, 1].cpu().numpy())
    
    # 예측 통계
    print(f"\n📊 예측 통계:")
    print(f"   평균: {np.mean(test_probabilities):.4f}")
    print(f"   표준편차: {np.std(test_probabilities):.4f}")
    print(f"   범위: {min(test_probabilities):.4f} ~ {max(test_probabilities):.4f}")
    
    # 제출 파일 생성
    sample_submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = test_probabilities
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'{RESULT_PATH}/evolved_submission_{timestamp}.csv'
    sample_submission.to_csv(submission_path, index=False)
    
    baseline_path = f'{RESULT_PATH}/baseline_submission.csv'
    sample_submission.to_csv(baseline_path, index=False)
    
    print(f"\n✅ 제출 파일 저장:")
    print(f"   📁 {submission_path}")
    print(f"   📁 {baseline_path}")
    
    print(f"\n🎉 진화된 KoBERT 완료!")
    print(f"🎯 목표: 0.83+ → 0.85+ 달성")
    print(f"🆕 추가 기능: 청크 중요도, 적응적 풀링, 출력 보정, 코사인 스케줄러")
    
    torch.cuda.empty_cache()
    return model, best_score, test_probabilities

if __name__ == "__main__":
    model, score, preds = main()