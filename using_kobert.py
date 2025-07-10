# ================================================================
# 🚀 정교한 KoBERT Fine-tuning (한국어 특화 최적화)
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import warnings
import random
import os
import re
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
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# 파일 경로 설정
DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks'
DATA_PATH = f'{DRIVE_PATH}'
MODEL_PATH = f'{DRIVE_PATH}/saved_models'
RESULT_PATH = f'{DRIVE_PATH}/results'

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

# 🎯 KoBERT 최적화 하이퍼파라미터
MAX_LEN = 512
BATCH_SIZE = 12  # KoBERT에 최적화된 배치 크기
EPOCHS = 5  # 충분한 학습
LEARNING_RATE = 3e-5  # KoBERT에 적합한 학습률
WARMUP_RATIO = 0.15  # 더 긴 워밍업
WEIGHT_DECAY = 0.05  # 적절한 regularization
DROPOUT_RATE = 0.3  # 균형잡힌 드롭아웃

# 청킹 설정 (KoBERT 특성 고려)
CHUNK_SIZE = 400
OVERLAP_SIZE = 50
MAX_CHUNKS = 5

# KoBERT 모델
MODEL_NAME = 'skt/kobert-base-v1'

# ================================================================
# 한국어 텍스트 전처리기
# ================================================================

class KoreanTextPreprocessor:
    """한국어 특화 텍스트 전처리"""
    
    @staticmethod
    def clean_korean_text(text):
        """한국어 텍스트 정리"""
        if pd.isna(text) or text == '':
            return ''
        
        # 한국어 특성 고려한 정리
        text = re.sub(r'\s+', ' ', text)  # 연속 공백
        text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?;:()\-\"\'\/]', ' ', text)  # 한국어 보존
        text = re.sub(r'\.{3,}', '...', text)  # 연속 점
        text = re.sub(r'!{2,}', '!', text)  # 연속 느낌표
        text = re.sub(r'\?{2,}', '?', text)  # 연속 물음표
        
        # 한국어 문장 구조 고려
        text = re.sub(r'([가-힣])\s+([가-힣])', r'\1\2', text)  # 불필요한 한글 간 공백
        
        return text.strip()
    
    @staticmethod
    def extract_korean_features(text):
        """한국어 텍스트 특징 추출"""
        if not text:
            return {}
        
        # 한국어 특성 분석
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(text)
        
        # 문장 종결어미 패턴 (AI가 자주 실수하는 부분)
        ending_patterns = {
            'formal_endings': len(re.findall(r'[가-힣]+(습니다|입니다|였습니다|있습니다)\.', text)),
            'informal_endings': len(re.findall(r'[가-힣]+(이야|야|다|어|아)\.', text)),
            'question_endings': len(re.findall(r'[가-힣]+(까|니|나)\?', text)),
        }
        
        # 조사 사용 패턴
        particles = len(re.findall(r'[가-힣]+(은|는|이|가|을|를|에|의|로|와|과|도)', text))
        
        # 어순 및 문체 특징
        features = {
            'korean_ratio': korean_chars / total_chars if total_chars > 0 else 0,
            'avg_sentence_length': len(text.split('.')) if '.' in text else 1,
            'particle_density': particles / korean_chars if korean_chars > 0 else 0,
            'formal_ratio': ending_patterns['formal_endings'] / len(text.split('.')) if '.' in text else 0,
            'punctuation_variety': len(set(re.findall(r'[.,!?;:]', text))) / 6,  # 구두점 다양성
        }
        
        return features

class OptimizedKoreanDataset(Dataset):
    """한국어 최적화 데이터셋"""
    
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.preprocessor = KoreanTextPreprocessor()
        
    def __len__(self):
        return len(self.texts)
    
    def _smart_korean_chunk(self, text):
        """한국어 문장 구조를 고려한 청킹"""
        if not text or len(text.strip()) == 0:
            return [""]
        
        # 토큰 길이 확인
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        
        if len(tokens) <= self.max_len:
            return [text]
        
        # 한국어 문장 경계 인식 (더 정교하게)
        # 마침표, 느낌표, 물음표 + 공백/줄바꿈으로 분할
        sentences = re.split(r'([.!?])\s+', text)
        
        # 분할된 결과를 재결합 (구두점 포함)
        proper_sentences = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                sentence = sentences[i] + sentences[i+1]
                if sentence.strip():
                    proper_sentences.append(sentence.strip())
        
        if not proper_sentences:
            proper_sentences = [text]
        
        chunks = []
        current_chunk = ""
        
        for sentence in proper_sentences:
            test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            test_tokens = self.tokenizer.encode(test_chunk, add_special_tokens=True, truncation=False)
            
            if len(test_tokens) > CHUNK_SIZE and current_chunk:
                chunks.append(current_chunk.strip())
                # 오버랩 추가 (한국어 문맥 보존)
                words = current_chunk.split()
                overlap_words = words[-OVERLAP_SIZE//10:] if len(words) > OVERLAP_SIZE//10 else []
                current_chunk = " ".join(overlap_words) + " " + sentence if overlap_words else sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        if not chunks:
            # 강제 분할
            truncated_tokens = tokens[:CHUNK_SIZE-2]
            chunks = [self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)]
        
        return chunks[:MAX_CHUNKS]
    
    def _simple_augment(self, text):
        """간단한 데이터 증강 (한국어 특성 고려)"""
        if not self.augment or random.random() > 0.3:
            return text
        
        # 조사 변경 (은/는, 이/가 등)
        augmented = text
        
        # 간단한 동의어 대체 (조심스럽게)
        replacements = {
            '그리고': '또한',
            '하지만': '그러나',
            '때문에': '이유로',
            '따라서': '그러므로'
        }
        
        for original, replacement in replacements.items():
            if original in augmented and random.random() > 0.7:
                augmented = augmented.replace(original, replacement, 1)
        
        return augmented
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        
        # 한국어 전처리
        text = self.preprocessor.clean_korean_text(text)
        
        # 데이터 증강 (훈련시에만)
        if self.augment:
            text = self._simple_augment(text)
        
        # 한국어 특징 추출
        korean_features = self.preprocessor.extract_korean_features(text)
        
        # 청킹
        chunks = self._smart_korean_chunk(text)
        
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
        
        # 한국어 특징을 텐서로 변환
        feature_tensor = torch.tensor([
            korean_features.get('korean_ratio', 0),
            korean_features.get('particle_density', 0),
            korean_features.get('formal_ratio', 0),
            korean_features.get('punctuation_variety', 0),
            len(text) / 1000,  # 정규화된 길이
        ], dtype=torch.float32)
        
        result = {
            'input_ids': torch.stack([chunk['input_ids'] for chunk in chunk_encodings]),
            'attention_mask': torch.stack([chunk['attention_mask'] for chunk in chunk_encodings]),
            'num_chunks': torch.tensor(min(len(chunks), MAX_CHUNKS), dtype=torch.long),
            'korean_features': feature_tensor
        }
        
        if self.labels is not None:
            label_value = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
            result['labels'] = torch.tensor(label_value, dtype=torch.long)
            
        return result

# ================================================================
# 한국어 특화 KoBERT 모델
# ================================================================

class EnhancedKoBERTClassifier(nn.Module):
    """한국어 특화 향상된 KoBERT 분류기"""
    
    def __init__(self, model_name=MODEL_NAME, num_classes=2):
        super(EnhancedKoBERTClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        
        # KoBERT 특성에 맞는 레이어 고정 (덜 공격적으로)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:4]:  # 처음 4개만 고정
            for param in layer.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # 위치 인코딩 (한국어 어순 고려)
        self.position_embeddings = nn.Embedding(MAX_CHUNKS, hidden_size)
        
        # 청크 간 어텐션 (한국어 문맥 고려)
        self.chunk_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True, dropout=DROPOUT_RATE
        )
        
        # 한국어 특징 융합 네트워크
        self.feature_fusion = nn.Sequential(
            nn.Linear(5, hidden_size // 8),  # 한국어 특징 5개
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE // 2)
        )
        
        # 청크 가중치 계산 (한국어 특성 반영)
        self.chunk_weight_net = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 8, hidden_size // 4),
            nn.GELU(),  # KoBERT와 호환성 좋은 활성화 함수
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 향상된 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size + hidden_size // 8, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE // 2),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # 가중치 초기화
        self._init_weights()
        
    def _init_weights(self):
        """KoBERT에 최적화된 가중치 초기화"""
        for module in [self.feature_fusion, self.chunk_weight_net, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_ids, attention_mask, num_chunks, korean_features):
        batch_size, num_chunks_max, seq_len = input_ids.shape
        
        # KoBERT 인코딩
        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len)
        
        outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        
        # KoBERT는 pooler_output이 없을 수 있음
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            chunk_embeddings = outputs.pooler_output
        else:
            chunk_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
        
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks_max, -1)
        
        # 위치 인코딩 추가
        positions = torch.arange(num_chunks_max, device=chunk_embeddings.device).unsqueeze(0).repeat(batch_size, 1)
        position_embeddings = self.position_embeddings(positions)
        chunk_embeddings = chunk_embeddings + position_embeddings
        
        # 한국어 특징 처리
        korean_feat_embedding = self.feature_fusion(korean_features)
        korean_feat_expanded = korean_feat_embedding.unsqueeze(1).repeat(1, num_chunks_max, 1)
        
        # 특징 융합
        enhanced_chunks = torch.cat([chunk_embeddings, korean_feat_expanded], dim=-1)
        
        # 청크 마스크 생성
        chunk_mask = torch.zeros(batch_size, num_chunks_max, device=chunk_embeddings.device)
        for i, num_chunk in enumerate(num_chunks):
            chunk_mask[i, :num_chunk] = 1
        
        # 어텐션 적용
        attended_chunks, _ = self.chunk_attention(
            chunk_embeddings, chunk_embeddings, chunk_embeddings,
            key_padding_mask=(chunk_mask == 0)
        )
        
        # 어텐션 결과와 한국어 특징 결합
        attended_enhanced = torch.cat([attended_chunks, korean_feat_expanded], dim=-1)
        
        # 청크별 가중치 계산
        chunk_weights = self.chunk_weight_net(attended_enhanced).squeeze(-1)
        chunk_weights = chunk_weights.masked_fill(chunk_mask == 0, float('-inf'))
        chunk_weights = F.softmax(chunk_weights, dim=1)
        
        # 가중 평균으로 문서 표현 생성
        doc_embedding = torch.sum(attended_enhanced * chunk_weights.unsqueeze(-1), dim=1)
        
        # 분류
        logits = self.classifier(doc_embedding)
        
        return logits

# ================================================================
# 훈련 함수
# ================================================================

def train_enhanced_kobert(model, X, y, tokenizer):
    """향상된 KoBERT 훈련"""
    
    # 데이터 분할 (더 신중하게)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )
    
    print(f"📊 데이터 분할:")
    print(f"   Train: {len(X_train)} ({dict(y_train.value_counts())})")
    print(f"   Val: {len(X_val)} ({dict(y_val.value_counts())})")
    
    # 데이터셋 생성 (훈련용은 증강 적용)
    train_dataset = OptimizedKoreanDataset(X_train, y_train, tokenizer, augment=True)
    val_dataset = OptimizedKoreanDataset(X_val, y_val, tokenizer, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 차별적 학습률 적용
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': WEIGHT_DECAY,
            'lr': LEARNING_RATE * 0.1  # BERT는 작은 학습률
        },
        {
            'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': LEARNING_RATE * 0.1
        },
        {
            'params': [p for n, p in model.named_parameters() if 'bert' not in n and not any(nd in n for nd in no_decay)],
            'weight_decay': WEIGHT_DECAY,
            'lr': LEARNING_RATE
        },
        {
            'params': [p for n, p in model.named_parameters() if 'bert' not in n and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': LEARNING_RATE
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    # 코사인 스케줄러 (더 부드러운 학습률 감소)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Label smoothing이 적용된 손실 함수
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
            
        def forward(self, pred, target):
            confidence = 1.0 - self.smoothing
            log_probs = F.log_softmax(pred, dim=1)
            smooth_target = target * confidence + (1 - target) * self.smoothing / (pred.size(1) - 1)
            return F.nll_loss(log_probs, target)
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    best_auc = 0
    best_model_state = None
    patience = 0
    max_patience = 2
    
    print("🚀 향상된 KoBERT 훈련 시작!")
    print("=" * 70)
    
    for epoch in range(EPOCHS):
        # === 훈련 ===
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
            korean_features = batch['korean_features'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_chunks=num_chunks,
                    korean_features=korean_features
                )
                loss = criterion(logits, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # 통계
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Progress bar 업데이트
            train_bar.set_postfix({
                'Loss': f'{total_loss/(train_bar.n+1):.4f}',
                'Acc': f'{100*correct/total:.1f}%',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # === 검증 ===
        model.eval()
        val_predictions = []
        val_labels_list = []
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')
            
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                num_chunks = batch['num_chunks'].to(device)
                korean_features = batch['korean_features'].to(device)
                
                with torch.cuda.amp.autocast():
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_chunks=num_chunks,
                        korean_features=korean_features
                    )
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                probs = torch.softmax(logits, dim=1)
                val_predictions.extend(probs[:, 1].cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_labels_list, val_predictions)
        val_f1 = f1_score(val_labels_list, [1 if p > 0.5 else 0 for p in val_predictions])
        val_precision = precision_score(val_labels_list, [1 if p > 0.5 else 0 for p in val_predictions])
        val_recall = recall_score(val_labels_list, [1 if p > 0.5 else 0 for p in val_predictions])
        
        # 상세한 결과 출력
        print(f"\n📊 Epoch {epoch+1} 상세 결과:")
        print(f"   🎯 Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"   🎯 Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"   📈 Metrics - AUC: {val_auc:.4f} | F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
        
        # 최고 성능 모델 저장
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience = 0
            
            # 모델 저장
            model_save_path = f'{MODEL_PATH}/enhanced_kobert_best.pth'
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
            }, model_save_path)
            
            print(f"   ⭐ 새로운 최고 성능! AUC: {val_auc:.4f} (모델 저장됨)")
        else:
            patience += 1
            print(f"   ⏳ Patience: {patience}/{max_patience}")
            
            if patience >= max_patience:
                print(f"   ⏰ Early stopping!")
                break
        
        print("-" * 70)
        torch.cuda.empty_cache()
        gc.collect()
    
    # 최고 성능 모델 로드
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_auc

def predict_enhanced_model(model, data_loader):
    """향상된 모델로 예측"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='🔮 정교한 예측 중'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            num_chunks = batch['num_chunks'].to(device)
            korean_features = batch['korean_features'].to(device)
            
            with torch.cuda.amp.autocast():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_chunks=num_chunks,
                    korean_features=korean_features
                )
            
            probs = torch.softmax(logits, dim=1)
            predictions.extend(probs[:, 1].cpu().numpy())
    
    return np.array(predictions)

# ================================================================
# 메인 실행 함수
# ================================================================

def main():
    print("🎯 정교한 KoBERT Fine-tuning (한국어 특화)")
    print("=" * 70)
    
    # 데이터 로드
    print("📁 데이터 로딩...")
    train = pd.read_csv(f'{DATA_PATH}/train.csv', encoding='utf-8-sig')
    test = pd.read_csv(f'{DATA_PATH}/test.csv', encoding='utf-8-sig')
    
    print(f"📊 데이터 정보:")
    print(f"   Train: {train.shape}")
    print(f"   Test: {test.shape}")
    print(f"   클래스 분포: {dict(train['generated'].value_counts())}")
    
    # 텍스트 길이 분석
    train_lengths = train['full_text'].astype(str).apply(len)
    print(f"   텍스트 길이 - 평균: {train_lengths.mean():.0f}, 중앙값: {train_lengths.median():.0f}, 최대: {train_lengths.max():,}")
    
    # 한국어 특화 전처리
    print("\n🔧 한국어 특화 전처리...")
    preprocessor = KoreanTextPreprocessor()
    
    # Train 데이터
    train['title'] = train['title'].fillna('').astype(str).apply(preprocessor.clean_korean_text)
    train['full_text'] = train['full_text'].fillna('').astype(str).apply(preprocessor.clean_korean_text)
    
    # 제목과 본문 결합 (한국어 특성 고려)
    train['combined_text'] = train.apply(
        lambda x: f"{x['title']}\n\n{x['full_text']}" if x['title'] and x['full_text'] 
        else x['title'] or x['full_text'], axis=1
    )
    
    # Test 데이터
    test = test.rename(columns={'paragraph_text': 'full_text'})
    test['title'] = test['title'].fillna('').astype(str).apply(preprocessor.clean_korean_text)
    test['full_text'] = test['full_text'].fillna('').astype(str).apply(preprocessor.clean_korean_text)
    test['combined_text'] = test.apply(
        lambda x: f"{x['title']}\n\n{x['full_text']}" if x['title'] and x['full_text'] 
        else x['title'] or x['full_text'], axis=1
    )
    
    X = train['combined_text']
    y = train['generated']
    
    # 한국어 특성 분석
    korean_stats = X.apply(preprocessor.extract_korean_features)
    avg_korean_ratio = np.mean([stats.get('korean_ratio', 0) for stats in korean_stats])
    print(f"   평균 한국어 비율: {avg_korean_ratio:.2f}")
    
    # KoBERT 토크나이저 로드
    print(f"🤖 KoBERT 토크나이저 로딩: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        print("   ✅ KoBERT 토크나이저 로드 성공")
    except Exception as e:
        print(f"   ❌ 토크나이저 로드 실패: {e}")
        return
    
    # 토크나이저 테스트
    test_text = "안녕하세요. 이것은 테스트입니다."
    test_tokens = tokenizer.encode(test_text)
    print(f"   토크나이저 테스트: {len(test_tokens)}개 토큰")
    
    # 모델 초기화
    print("🏗️ 향상된 KoBERT 모델 초기화...")
    model = EnhancedKoBERTClassifier(MODEL_NAME)
    model = model.to(device)
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   📊 모델 파라미터:")
    print(f"      전체: {total_params:,}")
    print(f"      훈련가능: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 훈련
    print("\n🚀 정교한 훈련 시작...")
    trained_model, best_auc = train_enhanced_kobert(model, X, y, tokenizer)
    
    print(f"\n🏆 훈련 완료!")
    print(f"   최고 검증 AUC: {best_auc:.4f}")
    
    # 테스트 예측
    print("\n🔮 테스트 데이터 예측...")
    test_dataset = OptimizedKoreanDataset(
        test['combined_text'], labels=None, tokenizer=tokenizer, augment=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    predictions = predict_enhanced_model(trained_model, test_loader)
    
    # 예측 결과 상세 분석
    print(f"\n📊 예측 결과 상세 분석:")
    print(f"   예측 개수: {len(predictions):,}")
    print(f"   평균 확률: {predictions.mean():.4f}")
    print(f"   표준편차: {predictions.std():.4f}")
    print(f"   최소값: {predictions.min():.4f}")
    print(f"   최대값: {predictions.max():.4f}")
    print(f"   중앙값: {np.median(predictions):.4f}")
    
    # 분포 분석
    high_conf = np.sum(predictions > 0.8)
    medium_conf = np.sum((predictions >= 0.2) & (predictions <= 0.8))
    low_conf = np.sum(predictions < 0.2)
    
    print(f"   신뢰도 분포:")
    print(f"      고신뢰도 (>0.8): {high_conf} ({high_conf/len(predictions)*100:.1f}%)")
    print(f"      중간신뢰도 (0.2-0.8): {medium_conf} ({medium_conf/len(predictions)*100:.1f}%)")
    print(f"      저신뢰도 (<0.2): {low_conf} ({low_conf/len(predictions)*100:.1f}%)")
    
    # 제출 파일 생성
    print("\n💾 제출 파일 생성...")
    sample_submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = predictions
    
    # 타임스탬프가 포함된 파일명
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'{RESULT_PATH}/enhanced_kobert_submission_{timestamp}.csv'
    sample_submission.to_csv(submission_path, index=False)
    
    # 기본 submission 파일
    baseline_path = f'{RESULT_PATH}/baseline_submission.csv'
    sample_submission.to_csv(baseline_path, index=False)
    
    print(f"✅ 제출 파일 저장 완료:")
    print(f"   📁 상세버전: {submission_path}")
    print(f"   📁 기본버전: {baseline_path}")
    
    # 최종 요약
    print("\n" + "="*70)
    print("🎉 정교한 KoBERT Fine-tuning 완료!")
    print("="*70)
    print(f"🏆 성능 지표:")
    print(f"   최고 검증 AUC: {best_auc:.4f}")
    print(f"   예측 평균 확률: {predictions.mean():.4f}")
    print(f"   예측 신뢰도: {predictions.std():.4f} (낮을수록 좋음)")
    
    print(f"\n🎯 한국어 특화 개선사항:")
    print(f"   ✅ 한국어 문장 구조 인식")
    print(f"   ✅ 조사/어미 패턴 분석")
    print(f"   ✅ 한국어 특징 융합")
    print(f"   ✅ 데이터 증강 적용")
    print(f"   ✅ 차별적 학습률")
    print(f"   ✅ Label smoothing")
    
    print(f"\n📁 저장된 파일:")
    print(f"   🤖 모델: {MODEL_PATH}/enhanced_kobert_best.pth")
    print(f"   📝 제출: {submission_path}")
    print("="*70)
    
    # 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    return trained_model, best_auc, predictions

# ================================================================
# 실행
# ================================================================

if __name__ == "__main__":
    try:
        model, auc, preds = main()
        print(f"\n🎊 모든 작업이 성공적으로 완료되었습니다!")
        print(f"   최종 AUC: {auc:.4f}")
        print(f"   예측 파일이 Google Drive에 저장되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()