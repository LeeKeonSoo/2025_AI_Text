# ================================================================
# 🚀 KLUE-BERT Fine-tuning for AI Text Detection (Colab Version)
# ================================================================

# 1. 환경 설정 및 라이브러리 설치
!pip install transformers==4.36.0
!pip install torch torchvision torchaudio
!pip install scikit-learn
!pip install pandas numpy tqdm

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from tqdm import tqdm
import warnings
import random
import os
import re
import gc
from datetime import datetime


warnings.filterwarnings('ignore')

# ================================================================
# 2. 설정 및 하이퍼파라미터
# ================================================================

def set_seed(seed=42):
    """시드 고정"""
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
print(f'🔥 Using device: {device}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# 파일 경로 설정 (Google Drive)
DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks'  # 실제 경로
DATA_PATH = f'{DRIVE_PATH}'  # 데이터 파일들이 있는 위치
MODEL_PATH = f'{DRIVE_PATH}/saved_models'  # 모델 저장용 폴더
RESULT_PATH = f'{DRIVE_PATH}/results'  # 결과 저장용 폴더

# 결과 저장용 디렉토리 생성 (데이터 폴더는 이미 존재)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

print(f"📁 파일 경로 설정:")
print(f"   데이터: {DATA_PATH}")
print(f"   모델 저장: {MODEL_PATH}")
print(f"   결과 저장: {RESULT_PATH}")

# 하이퍼파라미터
MAX_LEN = 512
BATCH_SIZE = 8  # Colab 메모리 고려
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0

# 청킹 설정
CHUNK_SIZE = 400
OVERLAP_SIZE = 50
MAX_CHUNKS = 6

# 모델 설정
MODEL_NAME = 'klue/bert-base'

# ================================================================
# 3. 데이터 전처리 및 Dataset 클래스
# ================================================================

class TextPreprocessor:
    """텍스트 전처리 클래스"""
    
    @staticmethod
    def clean_text(text):
        """텍스트 정리"""
        if pd.isna(text) or text == '':
            return ''
        
        # 기본적인 정리
        text = re.sub(r'\s+', ' ', text)  # 연속 공백 제거
        text = re.sub(r'[^\w\s가-힣.,!?;:()\-\"\'\/]', ' ', text)  # 특수문자 정리
        text = re.sub(r'\.{3,}', '...', text)  # 연속 점 정리
        
        return text.strip()
    
    @staticmethod
    def extract_statistics(text):
        """텍스트 통계 추출"""
        stats = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'korean_ratio': len(re.findall(r'[가-힣]', text)) / len(text) if len(text) > 0 else 0,
            'punctuation_ratio': len(re.findall(r'[.,!?;:]', text)) / len(text) if len(text) > 0 else 0
        }
        return stats

class OptimizedTextDataset(Dataset):
    """최적화된 텍스트 데이터셋"""
    
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN, is_test=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.preprocessor = TextPreprocessor()
        
    def __len__(self):
        return len(self.texts)
    
    def _smart_chunk_text(self, text):
        """스마트한 텍스트 청킹"""
        if not text or len(text.strip()) == 0:
            return [""]
        
        # 전체 텍스트 토큰 길이 확인
        full_tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        
        # 이미 짧으면 그대로 반환
        if len(full_tokens) <= self.max_len:
            return [text]
        
        # 문장 단위 분할
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) <= 1:
            sentences = text.split('.')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 테스트 청크 생성
            test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            test_tokens = self.tokenizer.encode(test_chunk, add_special_tokens=True, truncation=False)
            
            # 길이 초과 시 현재 청크 저장
            if len(test_tokens) > CHUNK_SIZE and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        # 마지막 청크 추가
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # 빈 청크 방지
        if not chunks:
            # 강제 자르기
            truncated_tokens = full_tokens[:CHUNK_SIZE-2]
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            chunks = [truncated_text]
        
        return chunks[:MAX_CHUNKS]
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        
        # 텍스트 전처리
        text = self.preprocessor.clean_text(text)
        
        # 청킹
        chunks = self._smart_chunk_text(text)
        
        # 각 청크 인코딩
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
        
        result = {
            'input_ids': torch.stack([chunk['input_ids'] for chunk in chunk_encodings]),
            'attention_mask': torch.stack([chunk['attention_mask'] for chunk in chunk_encodings]),
            'num_chunks': torch.tensor(min(len(chunks), MAX_CHUNKS), dtype=torch.long)
        }
        
        if self.labels is not None:
            label_value = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
            result['labels'] = torch.tensor(label_value, dtype=torch.long)
            
        return result

# ================================================================
# 4. 모델 정의
# ================================================================

class KLUEBERTClassifier(nn.Module):
    """KLUE-BERT 기반 분류기"""
    
    def __init__(self, model_name=MODEL_NAME, num_classes=2, dropout_rate=0.3):
        super(KLUEBERTClassifier, self).__init__()
        
        # BERT 모델 로드
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 일부 레이어 고정 (transfer learning)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:6]:  # 처음 6개 레이어 고정
            for param in layer.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # 위치 인코딩
        self.position_embeddings = nn.Embedding(MAX_CHUNKS, hidden_size)
        
        # 청크 어텐션
        self.chunk_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True, dropout=dropout_rate
        )
        
        # 청크 가중치 계산 네트워크
        self.chunk_weight_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # 가중치 초기화
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화"""
        for module in [self.chunk_weight_net, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
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
        
        # 위치 인코딩 추가
        positions = torch.arange(num_chunks_max, device=chunk_embeddings.device).unsqueeze(0).repeat(batch_size, 1)
        position_embeddings = self.position_embeddings(positions)
        chunk_embeddings = chunk_embeddings + position_embeddings
        
        # 청크 마스크 생성
        chunk_mask = torch.zeros(batch_size, num_chunks_max, device=chunk_embeddings.device)
        for i, num_chunk in enumerate(num_chunks):
            chunk_mask[i, :num_chunk] = 1
        
        # 어텐션 적용
        attended_chunks, attention_weights = self.chunk_attention(
            chunk_embeddings, chunk_embeddings, chunk_embeddings,
            key_padding_mask=(chunk_mask == 0)
        )
        
        # 청크별 가중치 계산
        chunk_weights = self.chunk_weight_net(attended_chunks).squeeze(-1)
        chunk_weights = chunk_weights.masked_fill(chunk_mask == 0, float('-inf'))
        chunk_weights = F.softmax(chunk_weights, dim=1)
        
        # 가중 평균으로 문서 표현 생성
        doc_embedding = torch.sum(attended_chunks * chunk_weights.unsqueeze(-1), dim=1)
        
        # 분류
        logits = self.classifier(doc_embedding)
        
        return logits

# ================================================================
# 5. 손실 함수 및 훈련 함수
# ================================================================

def focal_loss(logits, labels, alpha=1.0, gamma=2.0):
    """Focal Loss - 클래스 불균형에 효과적"""
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

def train_model(model, train_loader, val_loader, save_path):
    """모델 훈련"""
    
    # 옵티마이저 (차별적 학습률)
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': LEARNING_RATE * 0.1},  # BERT는 작은 학습률
        {'params': [p for n, p in model.named_parameters() if 'bert' not in n], 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
    # 스케줄러
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # 기록용
    train_losses = []
    val_aucs = []
    best_auc = 0
    best_model_state = None
    patience = 0
    max_patience = 2
    
    print("🚀 훈련 시작!")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        # === 훈련 ===
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            num_chunks = batch['num_chunks'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_chunks=num_chunks
                )
                loss = focal_loss(logits, labels)
            
            # Predictions
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Progress bar 업데이트
            current_loss = total_loss / (train_bar.n + 1)
            current_acc = correct_predictions.double() / total_samples
            train_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct_predictions.double() / total_samples
        
        # === 검증 ===
        model.eval()
        val_predictions = []
        val_labels_list = []
        val_loss = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')
            
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                num_chunks = batch['num_chunks'].to(device)
                
                with torch.cuda.amp.autocast():
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_chunks=num_chunks
                    )
                    loss = focal_loss(logits, labels)
                
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                val_predictions.extend(probs[:, 1].cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels_list, val_predictions)
        val_acc = accuracy_score(val_labels_list, [1 if p > 0.5 else 0 for p in val_predictions])
        
        # 기록
        train_losses.append(train_loss)
        val_aucs.append(val_auc)
        
        # 결과 출력
        print(f"\n📊 Epoch {epoch+1} 결과:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        # 최고 성능 모델 저장
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience = 0
            
            # 모델 저장
            torch.save({
                'model_state_dict': best_model_state,
                'val_auc': best_auc,
                'epoch': epoch,
                'hyperparameters': {
                    'max_len': MAX_LEN,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'model_name': MODEL_NAME
                }
            }, save_path)
            print(f"   ⭐ 새로운 최고 성능! 모델 저장됨: {save_path}")
            
        else:
            patience += 1
            if patience >= max_patience:
                print(f"   ⏰ Early stopping (patience: {max_patience})")
                break
        
        print("-" * 60)
        
        # 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
    
    # 최고 성능 모델 로드
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_auc, train_losses, val_aucs

# ================================================================
# 6. 예측 함수
# ================================================================

def predict_model(model, data_loader):
    """모델 예측"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='🔮 예측 중'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            num_chunks = batch['num_chunks'].to(device)
            
            with torch.cuda.amp.autocast():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_chunks=num_chunks
                )
            
            probs = torch.softmax(logits, dim=1)
            predictions.extend(probs[:, 1].cpu().numpy())
    
    return np.array(predictions)

# ================================================================
# 7. 메인 실행 함수
# ================================================================

def main():
    """메인 실행 함수"""
    
    print("🎯 KLUE-BERT AI 텍스트 탐지 시스템")
    print("=" * 60)
    
    # 데이터 로드
    print("📁 데이터 로딩...")
    try:
        train = pd.read_csv(f'{DATA_PATH}/train.csv', encoding='utf-8-sig')
        test = pd.read_csv(f'{DATA_PATH}/test.csv', encoding='utf-8-sig')
        print(f"   ✅ Train: {train.shape}, Test: {test.shape}")
    except FileNotFoundError:
        print("❌ 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        print(f"   예상 경로: {DATA_PATH}")
        return
    
    # 데이터 정보
    print(f"\n📊 데이터 정보:")
    print(f"   클래스 분포: {dict(train['generated'].value_counts())}")
    print(f"   텍스트 길이 통계 (문자 수):")
    text_lengths = train['full_text'].astype(str).apply(len)
    print(f"     평균: {text_lengths.mean():.0f}")
    print(f"     중앙값: {text_lengths.median():.0f}")
    print(f"     최대: {text_lengths.max():,}")
    
    # 텍스트 전처리
    print("\n🔧 텍스트 전처리...")
    preprocessor = TextPreprocessor()
    
    # Train 데이터
    train['title'] = train['title'].fillna('').astype(str).apply(preprocessor.clean_text)
    train['full_text'] = train['full_text'].fillna('').astype(str).apply(preprocessor.clean_text)
    train['combined_text'] = train.apply(
        lambda x: f"{x['title']}\\n\\n{x['full_text']}" if x['title'] and x['full_text'] 
        else x['title'] or x['full_text'], axis=1
    )
    
    # Test 데이터
    test = test.rename(columns={'paragraph_text': 'full_text'})
    test['title'] = test['title'].fillna('').astype(str).apply(preprocessor.clean_text)
    test['full_text'] = test['full_text'].fillna('').astype(str).apply(preprocessor.clean_text)
    test['combined_text'] = test.apply(
        lambda x: f"{x['title']}\\n\\n{x['full_text']}" if x['title'] and x['full_text'] 
        else x['title'] or x['full_text'], axis=1
    )
    
    X = train['combined_text']
    y = train['generated']
    
    # 토크나이저 로드
    print(f"🤖 토크나이저 로딩: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    # 데이터 분할
    print("✂️ 데이터 분할...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.15, random_state=42
    )
    
    print(f"   Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # 데이터셋 생성
    print("📦 데이터셋 생성...")
    train_dataset = OptimizedTextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = OptimizedTextDataset(X_val, y_val, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    # 모델 초기화
    print("🏗️ 모델 초기화...")
    model = KLUEBERTClassifier(MODEL_NAME, num_classes=2)
    model = model.to(device)
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   전체 파라미터: {total_params:,}")
    print(f"   훈련 파라미터: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 모델 훈련
    model_save_path = f'{MODEL_PATH}/klue_bert_best.pth'
    trained_model, best_auc, train_losses, val_aucs = train_model(
        model, train_loader, val_loader, model_save_path
    )
    
    print(f"\n🏆 훈련 완료! 최고 성능: {best_auc:.4f}")
    
    # 테스트 예측
    print("\n🔮 테스트 데이터 예측...")
    test_dataset = OptimizedTextDataset(
        test['combined_text'], labels=None, tokenizer=tokenizer, max_len=MAX_LEN, is_test=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    predictions = predict_model(trained_model, test_loader)
    
    # 예측 결과 통계
    print(f"\n📊 예측 통계:")
    print(f"   예측 개수: {len(predictions):,}")
    print(f"   평균 확률: {predictions.mean():.4f}")
    print(f"   표준편차: {predictions.std():.4f}")
    print(f"   최소값: {predictions.min():.4f}")
    print(f"   최대값: {predictions.max():.4f}")
    print(f"   중앙값: {np.median(predictions):.4f}")
    
    # 제출 파일 생성
    print("\n💾 제출 파일 생성...")
    sample_submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = predictions
    
    # 현재 시간으로 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'{RESULT_PATH}/submission_klue_bert_{timestamp}.csv'
    sample_submission.to_csv(submission_path, index=False)
    
    # 기본 submission 파일도 저장 (덮어쓰기)
    baseline_path = f'{RESULT_PATH}/baseline_submission.csv'
    sample_submission.to_csv(baseline_path, index=False)
    
    print(f"✅ 제출 파일 저장 완료:")
    print(f"   📁 {submission_path}")
    print(f"   📁 {baseline_path}")
    
    # 최종 결과 요약
    print("\n" + "="*60)
    print("🎉 훈련 및 예측 완료!")
    print("="*60)
    print(f"🏆 최고 검증 AUC: {best_auc:.4f}")
    print(f"📊 예측 통계: {predictions.mean():.4f} ± {predictions.std():.4f}")
    print(f"💾 모델 저장: {model_save_path}")
    print(f"📝 제출 파일: {submission_path}")
    print("="*60)
    
    # 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    return trained_model, best_auc, predictions

# ================================================================
# 실행
# ================================================================

if __name__ == "__main__":
    model, auc, preds = main()