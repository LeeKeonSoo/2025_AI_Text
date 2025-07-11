# ================================================================
# 데이터 레벨 개선 KoBERT (best_score.py 기반)
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
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import warnings
import random
import os
import gc
from datetime import datetime
import re

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
print(f'Using device: {device}')

# 파일 경로
DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks'
DATA_PATH = f'{DRIVE_PATH}'
RESULT_PATH = f'{DRIVE_PATH}/results'
os.makedirs(RESULT_PATH, exist_ok=True)

# best_score.py 동일한 하이퍼파라미터
MAX_LEN = 512
BATCH_SIZE = 12
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MODEL_NAME = 'skt/kobert-base-v1'

# best_score.py 동일한 청킹 설정
CHUNK_SIZE = 400
OVERLAP_SIZE = 100
MAX_CHUNKS = 4

# ================================================================
# 데이터 레벨 개선 함수들
# ================================================================

def smart_text_augment(text, prob=0.15):
    """AI 텍스트 탐지에 특화된 데이터 증강"""
    if random.random() > prob:
        return text
    
    # AI가 자주 하는 실수 패턴 정리
    text = re.sub(r'  +', ' ', text)  # 이중 공백
    text = re.sub(r'([.!?])\1+', r'\1', text)  # 연속 구두점
    text = re.sub(r'([가-힣])\s+([가-힣])', r'\1\2', text)  # 불필요한 한글 간 공백
    
    # 간단한 동의어 치환 (신중하게)
    replacements = {
        '그리고': '또한', '하지만': '그러나', '때문에': '이유로', 
        '따라서': '그러므로', '또는': '혹은'
    }
    
    for orig, repl in replacements.items():
        if orig in text and random.random() > 0.8:
            text = text.replace(orig, repl, 1)
    
    return text

def length_stratified_split(X, y, test_size=0.2, random_state=42):
    """텍스트 길이별 계층적 분할"""
    lengths = X.str.len()
    
    # 길이별 구간 나누기
    length_bins = pd.qcut(lengths, q=4, labels=['short', 'medium', 'long', 'very_long'], duplicates='drop')
    
    # 클래스와 길이 구간을 모두 고려한 stratification
    combined_strata = y.astype(str) + '_' + length_bins.astype(str)
    
    try:
        return train_test_split(X, y, test_size=test_size, stratify=combined_strata, random_state=random_state)
    except:
        # 실패시 기본 stratify로 fallback
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def analyze_text_characteristics(texts, labels):
    """텍스트 특성 분석 및 피드백"""
    ai_texts = texts[labels == 1]
    human_texts = texts[labels == 0]
    
    ai_lengths = ai_texts.str.len()
    human_lengths = human_texts.str.len()
    
    print(f"텍스트 특성 분석:")
    print(f"  AI 텍스트 - 평균 길이: {ai_lengths.mean():.0f}, 표준편차: {ai_lengths.std():.0f}")
    print(f"  인간 텍스트 - 평균 길이: {human_lengths.mean():.0f}, 표준편차: {human_lengths.std():.0f}")
    
    # 구두점 사용 패턴
    ai_punct = ai_texts.str.count(r'[.!?]').mean()
    human_punct = human_texts.str.count(r'[.!?]').mean()
    print(f"  구두점 밀도 - AI: {ai_punct:.2f}, 인간: {human_punct:.2f}")

# ================================================================
# 개선된 데이터셋 (best_score.py 기반 + 증강)
# ================================================================

class ImprovedLongTextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        
    def __len__(self):
        return len(self.texts)
    
    def _chunk_text(self, text):
        """best_score.py와 동일한 청킹"""
        sentences = text.split('. ')
        if len(sentences) == 1:
            sentences = text.split('.')
        
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
                    overlap_tokens = current_chunk_tokens[-OVERLAP_SIZE:] if len(current_chunk_tokens) > OVERLAP_SIZE else []
                    current_chunk_tokens = overlap_tokens + sentence_tokens
                    current_length = len(current_chunk_tokens)
                else:
                    current_chunk_tokens = sentence_tokens
                    current_length = len(sentence_tokens)
            else:
                current_chunk_tokens.extend(sentence_tokens)
                current_length += len(sentence_tokens)
        
        if current_chunk_tokens:
            chunk_text = self.tokenizer.decode(current_chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        
        return chunks
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        
        # 데이터 증강 적용
        if self.augment:
            text = smart_text_augment(text)
        
        chunks = self._chunk_text(text)
        
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
# best_score.py 동일한 모델
# ================================================================

class HighPerformanceKoBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout_rate=0.15):
        super(HighPerformanceKoBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        self.chunk_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True, dropout=dropout_rate
        )
        
        self.position_embeddings = nn.Embedding(MAX_CHUNKS, hidden_size)
        
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
            input_ids_flat = input_ids.view(-1, seq_len)
            attention_mask_flat = attention_mask.view(-1, seq_len)
            
            outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                chunk_embeddings = outputs.pooler_output
            else:
                chunk_embeddings = outputs.last_hidden_state[:, 0, :]
            
            chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks_max, -1)
            
            positions = torch.arange(num_chunks_max, device=chunk_embeddings.device).unsqueeze(0).repeat(batch_size, 1)
            position_embeddings = self.position_embeddings(positions)
            chunk_embeddings = chunk_embeddings + position_embeddings
            
            chunk_mask = torch.zeros(batch_size, num_chunks_max, device=chunk_embeddings.device)
            for i, num_chunk in enumerate(num_chunks):
                chunk_mask[i, :num_chunk] = 1
            
            attended_chunks, attention_weights = self.chunk_attention(
                chunk_embeddings, chunk_embeddings, chunk_embeddings,
                key_padding_mask=(chunk_mask == 0)
            )
            
            doc_mask = chunk_mask.unsqueeze(-1)
            weighted_chunks = attended_chunks * doc_mask
            doc_embedding = weighted_chunks.sum(dim=1) / (doc_mask.sum(dim=1) + 1e-8)
            
            doc_embedding = doc_embedding + self.pre_classifier(doc_embedding)
            logits = self.classifier(doc_embedding)
        
        return logits, attention_weights

# ================================================================
# best_score.py 동일한 훈련 함수
# ================================================================

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
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
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
            
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
    
    f1 = f1_score(real_values, predictions)
    
    return (total_loss / len(data_loader), 
            correct_predictions.double() / len(data_loader.dataset),
            predictions, probabilities, real_values, f1)

def predict_test(model, data_loader, device):
    model.eval()
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
            test_probabilities.extend(probs[:, 1].cpu().numpy())
            
            if batch_idx % 15 == 0:
                torch.cuda.empty_cache()
    
    return test_probabilities

def main():
    print("데이터 레벨 개선 KoBERT")
    
    # 데이터 로드
    train = pd.read_csv(f'{DATA_PATH}/train.csv', encoding='utf-8-sig')
    test = pd.read_csv(f'{DATA_PATH}/test.csv', encoding='utf-8-sig')
    
    print(f"Train: {train.shape}, Test: {test.shape}")
    
    # best_score.py 동일한 전처리
    train['title'] = train['title'].fillna('').str.strip()
    train['full_text'] = train['full_text'].fillna('').str.replace(r'\n\s*\n', '\n', regex=True).str.strip()
    train['combined_text'] = train['title'] + ' [SEP] ' + train['full_text']
    
    X = train['combined_text']
    y = train['generated']
    
    # 텍스트 특성 분석
    analyze_text_characteristics(X, y)
    
    # 길이별 계층적 분할
    print("길이별 계층적 데이터 분할 적용")
    X_train, X_val, y_train, y_val = length_stratified_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # KoBERT 토크나이저 로딩
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        print(f"KoBERT 토크나이저 로드 성공")
    except Exception as e:
        print(f"KoBERT 로딩 실패: {e}")
        raise SystemExit("KoBERT 로딩 필수")
    
    # 데이터셋 생성 (훈련용은 증강 적용)
    train_dataset = ImprovedLongTextDataset(X_train, y_train, tokenizer, MAX_LEN, augment=True)
    val_dataset = ImprovedLongTextDataset(X_val, y_val, tokenizer, MAX_LEN, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # 모델 초기화
    model = HighPerformanceKoBERTClassifier(MODEL_NAME, num_classes=2)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터: {total_params:,}, 훈련 가능: {trainable_params:,}")
    
    # best_score.py 동일한 설정
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-6
    )
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 훈련 실행
    best_auc = 0
    best_f1 = 0
    best_model = None
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, scaler, epoch + 1
        )
        
        val_loss, val_acc, val_preds, val_probs, val_labels, val_f1 = eval_model(
            model, val_loader, criterion, device
        )
        
        val_auc = roc_auc_score(val_labels, val_probs)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}')
        
        combined_score = 0.7 * val_auc + 0.3 * val_f1
        best_combined = 0.7 * best_auc + 0.3 * best_f1
        
        if combined_score > best_combined:
            best_auc = val_auc
            best_f1 = val_f1
            best_model = model.state_dict().copy()
            print(f'새로운 최고 모델: AUC {best_auc:.4f}, F1 {best_f1:.4f}')
        
        torch.cuda.empty_cache()
    
    print(f'최고 결과: AUC {best_auc:.4f}, F1 {best_f1:.4f}')
    
    # 최고 모델로 테스트 예측
    model.load_state_dict(best_model)
    model.eval()
    
    # 테스트 데이터 전처리
    test = test.rename(columns={'paragraph_text': 'full_text'})
    test['title'] = test['title'].fillna('').str.strip()
    test['full_text'] = test['full_text'].fillna('').str.replace(r'\n\s*\n', '\n', regex=True).str.strip()
    test['combined_text'] = test['title'] + ' [SEP] ' + test['full_text']
    
    test_dataset = ImprovedLongTextDataset(test['combined_text'], labels=None, tokenizer=tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=4, pin_memory=True)
    
    test_probabilities = predict_test(model, test_loader, device)
    
    print(f"예측 완료: {len(test_probabilities)}개")
    print(f"확률 범위: {min(test_probabilities):.4f} - {max(test_probabilities):.4f}")
    print(f"평균: {np.mean(test_probabilities):.4f}")
    
    # 제출 파일 생성
    sample_submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = test_probabilities
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'{RESULT_PATH}/data_improved_{timestamp}.csv'
    sample_submission.to_csv(submission_path, index=False)
    
    baseline_path = f'{RESULT_PATH}/baseline_submission.csv'
    sample_submission.to_csv(baseline_path, index=False)
    
    print(f"제출 파일 저장: {submission_path}")
    
    torch.cuda.empty_cache()
    return model, best_auc, test_probabilities

if __name__ == "__main__":
    model, auc, preds = main()