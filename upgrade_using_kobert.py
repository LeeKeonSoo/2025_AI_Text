# ================================================================
# 🚀 개선된 KLUE-BERT (과적합 방지 버전)
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import warnings
import random
import os
import re
import gc
from datetime import datetime

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

# 파일 경로 설정
DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks'
DATA_PATH = f'{DRIVE_PATH}'
MODEL_PATH = f'{DRIVE_PATH}/saved_models'
RESULT_PATH = f'{DRIVE_PATH}/results'

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

# 🎯 과적합 방지를 위한 보수적 하이퍼파라미터
MAX_LEN = 512
BATCH_SIZE = 16  # 더 큰 배치로 안정화
EPOCHS = 3  # 에포크 줄임
LEARNING_RATE = 1e-5  # 더 작은 학습률
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.1  # 더 강한 regularization
DROPOUT_RATE = 0.5  # 더 강한 드롭아웃

# 청킹 단순화
CHUNK_SIZE = 450
MAX_CHUNKS = 4  # 청크 수 줄임

MODEL_NAME = 'klue/bert-base'

class SimpleTextDataset(Dataset):
    """단순화된 텍스트 데이터셋"""
    
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def _simple_chunk_text(self, text):
        """단순한 텍스트 청킹"""
        if not text or len(text.strip()) == 0:
            return [""]
        
        # 토큰 길이 확인
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        
        # 짧으면 그대로 반환
        if len(tokens) <= self.max_len:
            return [text]
        
        # 문장 단위 분할
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            test_tokens = self.tokenizer.encode(test_chunk, add_special_tokens=True, truncation=False)
            
            if len(test_tokens) > CHUNK_SIZE and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        if not chunks:
            truncated_tokens = tokens[:CHUNK_SIZE-2]
            chunks = [self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)]
        
        return chunks[:MAX_CHUNKS]
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        
        # 기본 정리만
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = self._simple_chunk_text(text)
        
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
        
        result = {
            'input_ids': torch.stack([chunk['input_ids'] for chunk in chunk_encodings]),
            'attention_mask': torch.stack([chunk['attention_mask'] for chunk in chunk_encodings]),
            'num_chunks': torch.tensor(min(len(chunks), MAX_CHUNKS), dtype=torch.long)
        }
        
        if self.labels is not None:
            label_value = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
            result['labels'] = torch.tensor(label_value, dtype=torch.long)
            
        return result

class SimplifiedKLUEBERT(nn.Module):
    """단순화된 KLUE-BERT 분류기"""
    
    def __init__(self, model_name=MODEL_NAME, num_classes=2):
        super(SimplifiedKLUEBERT, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 더 많은 레이어 고정 (과적합 방지)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:9]:  # 9개 레이어 고정
            for param in layer.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # 단순한 평균 풀링만 사용
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
        # 단순한 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, num_chunks):
        batch_size, num_chunks_max, seq_len = input_ids.shape
        
        # BERT 인코딩
        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len)
        
        outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        
        # [CLS] 토큰 사용
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            chunk_embeddings = outputs.pooler_output
        else:
            chunk_embeddings = outputs.last_hidden_state[:, 0, :]
        
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks_max, -1)
        
        # 단순 평균 풀링
        chunk_mask = torch.zeros(batch_size, num_chunks_max, device=chunk_embeddings.device)
        for i, num_chunk in enumerate(num_chunks):
            chunk_mask[i, :num_chunk] = 1
        
        masked_embeddings = chunk_embeddings * chunk_mask.unsqueeze(-1)
        doc_embedding = masked_embeddings.sum(dim=1) / (chunk_mask.sum(dim=1, keepdim=True) + 1e-8)
        
        # 드롭아웃 추가
        doc_embedding = self.dropout(doc_embedding)
        
        # 분류
        logits = self.classifier(doc_embedding)
        
        return logits

def train_with_validation_split(model, X, y, tokenizer):
    """제대로 된 검증 분할로 훈련"""
    
    # 더 큰 검증 셋 (30%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )
    
    print(f"📊 데이터 분할:")
    print(f"   Train: {len(X_train)} ({y_train.value_counts().to_dict()})")
    print(f"   Val: {len(X_val)} ({y_val.value_counts().to_dict()})")
    
    # 데이터셋 생성
    train_dataset = SimpleTextDataset(X_train, y_train, tokenizer)
    val_dataset = SimpleTextDataset(X_val, y_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 옵티마이저 (단순하게)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 스케줄러
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps
    )
    
    # 기본 CrossEntropy (Focal Loss 제거)
    criterion = nn.CrossEntropyLoss()
    
    best_auc = 0
    best_model_state = None
    
    print("🚀 훈련 시작! (과적합 방지 버전)")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        # 훈련
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            num_chunks = batch['num_chunks'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask, num_chunks=num_chunks)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            train_bar.set_postfix({
                'Loss': f'{total_loss/(train_bar.n+1):.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # 검증
        model.eval()
        val_predictions = []
        val_labels_list = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                num_chunks = batch['num_chunks'].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask, num_chunks=num_chunks)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                val_predictions.extend(probs[:, 1].cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels_list, val_predictions)
        val_acc = accuracy_score(val_labels_list, [1 if p > 0.5 else 0 for p in val_predictions])
        
        print(f"\n📊 Epoch {epoch+1} 결과:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        # 더 보수적인 저장 (AUC가 0.95 이하일 때만)
        if val_auc > best_auc and val_auc <= 0.95:  # 너무 높은 AUC는 의심
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            print(f"   ⭐ 새로운 최고 성능! (AUC: {val_auc:.4f})")
        
        print("-" * 60)
        torch.cuda.empty_cache()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_auc

def predict_model(model, data_loader):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='🔮 예측 중'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            num_chunks = batch['num_chunks'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask, num_chunks=num_chunks)
            probs = torch.softmax(logits, dim=1)
            predictions.extend(probs[:, 1].cpu().numpy())
    
    return np.array(predictions)

def main():
    print("🎯 개선된 KLUE-BERT (과적합 방지 버전)")
    print("=" * 60)
    
    # 데이터 로드
    train = pd.read_csv(f'{DATA_PATH}/train.csv', encoding='utf-8-sig')
    test = pd.read_csv(f'{DATA_PATH}/test.csv', encoding='utf-8-sig')
    
    print(f"📊 데이터 정보:")
    print(f"   Train: {train.shape}")
    print(f"   Test: {test.shape}")
    print(f"   클래스 분포: {dict(train['generated'].value_counts())}")
    
    # 단순한 전처리
    train['title'] = train['title'].fillna('').astype(str)
    train['full_text'] = train['full_text'].fillna('').astype(str)
    train['combined_text'] = train['title'] + ' ' + train['full_text']
    
    test = test.rename(columns={'paragraph_text': 'full_text'})
    test['title'] = test['title'].fillna('').astype(str)
    test['full_text'] = test['full_text'].fillna('').astype(str)
    test['combined_text'] = test['title'] + ' ' + test['full_text']
    
    X = train['combined_text']
    y = train['generated']
    
    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    # 모델 초기화
    model = SimplifiedKLUEBERT(MODEL_NAME)
    model = model.to(device)
    
    # 훈련 가능한 파라미터 확인
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏗️ 모델 정보:")
    print(f"   훈련 파라미터: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 훈련
    trained_model, best_auc = train_with_validation_split(model, X, y, tokenizer)
    
    print(f"\n🏆 훈련 완료! 최고 AUC: {best_auc:.4f}")
    
    # 테스트 예측
    print("\n🔮 테스트 예측...")
    test_dataset = SimpleTextDataset(test['combined_text'], labels=None, tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    predictions = predict_model(trained_model, test_loader)
    
    # 예측 통계
    print(f"\n📊 예측 통계:")
    print(f"   평균: {predictions.mean():.4f}")
    print(f"   표준편차: {predictions.std():.4f}")
    print(f"   범위: {predictions.min():.4f} ~ {predictions.max():.4f}")
    
    # 제출 파일 생성
    sample_submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = predictions
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'{RESULT_PATH}/improved_submission_{timestamp}.csv'
    sample_submission.to_csv(submission_path, index=False)
    
    baseline_path = f'{RESULT_PATH}/baseline_submission.csv'
    sample_submission.to_csv(baseline_path, index=False)
    
    print(f"\n✅ 제출 파일 저장:")
    print(f"   📁 {submission_path}")
    print(f"   📁 {baseline_path}")
    
    print(f"\n🎉 개선된 버전 완료!")
    print(f"🎯 목표: 더 보수적이고 일반화된 모델")
    print(f"🏆 검증 AUC: {best_auc:.4f} (0.85~0.92 범위가 건전함)")
    
    torch.cuda.empty_cache()
    return trained_model, best_auc, predictions

if __name__ == "__main__":
    model, auc, preds = main()