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
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# A100 최적화 하이퍼파라미터
MAX_LEN = 1024  # 긴 시퀀스 처리 (1024 시작, 메모리 여유 시 1536까지)
BATCH_SIZE = 8  # A100에 최적화된 배치 크기
EPOCHS = 4
LEARNING_RATE = 1e-5  # 긴 시퀀스에서는 더 낮은 학습률
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# 앙상블 설정 (시간 절약)
ENSEMBLE_MODELS = [
    'skt/kobert-base-v1',
    'klue/bert-base'
]
USE_KFOLD = True
N_FOLDS = 2  # 2-Fold로 시간 절약

class LongSequenceDataset(Dataset):
    """긴 시퀀스 전용 데이터셋 (청킹 없음)"""
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        
        # 긴 시퀀스 직접 토크나이징 (청킹 없음)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,  # 너무 긴 경우에만 자름
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            result['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
            
        return result

class LongSequenceBERTClassifier(nn.Module):
    """긴 시퀀스 처리용 BERT 분류기"""
    def __init__(self, model_name, num_classes=2, dropout_rate=0.15, model_idx=0):
        super(LongSequenceBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self.model_idx = model_idx
        
        # A100에서는 더 적게 프리징 (메모리 충분)
        freeze_layers = [4, 5][model_idx % 2]  # 더 많은 레이어 학습
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # 긴 시퀀스용 고급 분류기
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 모델별 다른 분류기
        if model_idx == 0:  # KoBERT
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 4, num_classes)
            )
        else:  # KLUE-BERT
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, num_classes)
            )
        
        # 어텐션 풀링 (긴 시퀀스에서 효과적)
        self.attention_pooling = nn.MultiheadAttention(
            hidden_size, num_heads=16, batch_first=True, dropout=dropout_rate
        )
        
    def forward(self, input_ids, attention_mask):
        with torch.cuda.amp.autocast():
            # BERT 인코딩
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            
            # 전체 시퀀스 사용
            sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
            
            # 어텐션 풀링으로 중요한 부분에 집중
            pooled_output, attention_weights = self.attention_pooling(
                sequence_output, sequence_output, sequence_output,
                key_padding_mask=(attention_mask == 0)
            )
            
            # 가중 평균
            mask_expanded = attention_mask.unsqueeze(-1).float()
            weighted_output = (pooled_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            
            # 분류
            enhanced_output = self.pre_classifier(weighted_output)
            enhanced_output = enhanced_output + weighted_output  # 잔차 연결
            logits = self.classifier(enhanced_output)
        
        return logits, attention_weights

def analyze_text_lengths(texts):
    """텍스트 길이 분석"""
    lengths = texts.str.len()
    print(f"\n📊 텍스트 길이 분석:")
    print(f"  평균: {lengths.mean():.1f} 문자")
    print(f"  중간값: {lengths.median():.1f} 문자")
    print(f"  95th percentile: {lengths.quantile(0.95):.1f} 문자")
    print(f"  99th percentile: {lengths.quantile(0.99):.1f} 문자")
    print(f"  최대: {lengths.max()} 문자")
    print(f"  1000자 이상: {(lengths > 1000).sum()}/{len(texts)} ({(lengths > 1000).mean()*100:.1f}%)")
    print(f"  2000자 이상: {(lengths > 2000).sum()}/{len(texts)} ({(lengths > 2000).mean()*100:.1f}%)")
    print(f"  3000자 이상: {(lengths > 3000).sum()}/{len(texts)} ({(lengths > 3000).mean()*100:.1f}%)")
    
    # MAX_LEN 추천
    tokenizer_temp = AutoTokenizer.from_pretrained('klue/bert-base')
    sample_tokens = []
    for text in texts.sample(min(1000, len(texts))):
        tokens = tokenizer_temp.encode(text, add_special_tokens=True)
        sample_tokens.append(len(tokens))
    
    token_95th = np.percentile(sample_tokens, 95)
    token_99th = np.percentile(sample_tokens, 99)
    
    print(f"\n🔤 토큰 길이 분석:")
    print(f"  95th percentile: {token_95th:.0f} 토큰")
    print(f"  99th percentile: {token_99th:.0f} 토큰")
    
    if token_95th <= 1024:
        print(f"  ✅ MAX_LEN=1024 추천 (95% 커버)")
    elif token_95th <= 1536:
        print(f"  ⚠️ MAX_LEN=1536 권장 (95% 커버)")
    else:
        print(f"  🚨 MAX_LEN=2048 고려 필요")

def train_single_model(model, train_loader, val_loader, model_name, fold_idx=None):
    """단일 모델 훈련 (긴 시퀀스 최적화)"""
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # A100 최적화 옵티마이저
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
    
    best_auc = 0
    best_model_state = None
    
    fold_str = f"Fold {fold_idx+1}" if fold_idx is not None else "Single"
    model_name_short = model_name.split('/')[-1]
    
    for epoch in range(EPOCHS):
        # 훈련
        model.train()
        total_loss = 0
        correct_predictions = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'{fold_str} {model_name_short} Epoch {epoch+1}')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits, attention_weights = model(input_ids=input_ids, attention_mask=attention_mask)
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
            
            # 긴 시퀀스로 인한 메모리 관리
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct_predictions.double() / len(train_loader.dataset)
        
        # 검증
        model.eval()
        val_loss = 0
        val_predictions = []
        val_probabilities = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.cuda.amp.autocast():
                    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                val_probabilities.extend(probs[:, 1].cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_probabilities)
        val_f1 = f1_score(val_labels, [1 if p > 0.5 else 0 for p in val_probabilities])
        
        print(f'  Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'             Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}')
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            print(f'  🎯 New best AUC: {best_auc:.4f}')
        
        # GPU 메모리 상태
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f'  GPU Memory: {memory_allocated:.2f}GB')
    
    model.load_state_dict(best_model_state)
    return model, best_auc

def predict_with_model(model, data_loader):
    """단일 모델로 예측"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with torch.cuda.amp.autocast():
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            probs = torch.softmax(logits, dim=1)
            predictions.extend(probs[:, 1].cpu().numpy())
    
    return np.array(predictions)

def ensemble_predict(models, tokenizers, test_data, weights=None):
    """앙상블 예측"""
    if weights is None:
        weights = [1.0] * len(models)
    
    all_predictions = []
    
    for i, (model, tokenizer) in enumerate(zip(models, tokenizers)):
        print(f"\n예측 중: {ENSEMBLE_MODELS[i].split('/')[-1]}")
        
        test_dataset = LongSequenceDataset(test_data, labels=None, tokenizer=tokenizer, max_len=MAX_LEN)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        predictions = predict_with_model(model, test_loader)
        all_predictions.append(predictions * weights[i])
    
    # 가중 평균
    ensemble_predictions = np.mean(all_predictions, axis=0)
    return ensemble_predictions

def main():
    print("=== 🚀 A100 Long Sequence Ensemble Training ===")
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
    analyze_text_lengths(train['combined_text'])
    
    X = train['combined_text']
    y = train['generated']
    
    print(f"\nClass distribution: {y.value_counts().to_dict()}")
    
    # 앙상블 모델들과 토크나이저들
    ensemble_models = []
    ensemble_tokenizers = []
    model_weights = []
    
    print(f"\n🎯 A100 Long Sequence 앙상블 훈련 시작")
    print(f"설정: MAX_LEN={MAX_LEN}, BATCH_SIZE={BATCH_SIZE}, {N_FOLDS}-Fold")
    
    for model_idx, model_name in enumerate(ENSEMBLE_MODELS):
        print(f"\n{'='*60}")
        print(f"모델 {model_idx+1}/{len(ENSEMBLE_MODELS)}: {model_name}")
        print(f"{'='*60}")
        
        # 토크나이저 로드
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            print(f"Successfully loaded tokenizer: {model_name}")
        except:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                print(f"Loaded fast tokenizer: {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue
        
        if USE_KFOLD:
            # K-Fold 교차 검증
            fold_aucs = []
            
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                print(f"\n--- Fold {fold_idx+1}/{N_FOLDS} ---")
                
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # 긴 시퀀스 데이터셋 생성
                train_dataset = LongSequenceDataset(X_train_fold, y_train_fold, tokenizer, MAX_LEN)
                val_dataset = LongSequenceDataset(X_val_fold, y_val_fold, tokenizer, MAX_LEN)
                
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
                
                # 긴 시퀀스 모델 초기화
                model = LongSequenceBERTClassifier(model_name, num_classes=2, model_idx=model_idx)
                model = model.to(device)
                
                # 훈련
                trained_model, fold_auc = train_single_model(
                    model, train_loader, val_loader, model_name, fold_idx
                )
                
                fold_aucs.append(fold_auc)
                
                # 첫 번째 폴드 모델만 저장 (앙상블용)
                if fold_idx == 0:
                    ensemble_models.append(trained_model)
                    ensemble_tokenizers.append(tokenizer)
                
                torch.cuda.empty_cache()
            
            avg_auc = np.mean(fold_aucs)
            model_weights.append(avg_auc)
            print(f"\n{model_name} K-Fold 평균 AUC: {avg_auc:.4f}")
            
        else:
            # 단일 학습
            X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
            
            train_dataset = LongSequenceDataset(X_train, y_train, tokenizer, MAX_LEN)
            val_dataset = LongSequenceDataset(X_val, y_val, tokenizer, MAX_LEN)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            
            model = LongSequenceBERTClassifier(model_name, num_classes=2, model_idx=model_idx)
            model = model.to(device)
            
            trained_model, auc = train_single_model(model, train_loader, val_loader, model_name)
            
            ensemble_models.append(trained_model)
            ensemble_tokenizers.append(tokenizer)
            model_weights.append(auc)
            
            print(f"\n{model_name} AUC: {auc:.4f}")
    
    # 가중치 정규화
    total_weight = sum(model_weights)
    normalized_weights = [w / total_weight for w in model_weights]
    
    print(f"\n🎯 앙상블 가중치:")
    for i, (model_name, weight) in enumerate(zip(ENSEMBLE_MODELS, normalized_weights)):
        print(f"  {model_name.split('/')[-1]}: {weight:.3f}")
    
    # 테스트 데이터 전처리
    print("\n테스트 데이터 전처리...")
    test = test.rename(columns={'paragraph_text': 'full_text'})
    test['title'] = test['title'].fillna('').str.strip()
    test['full_text'] = test['full_text'].fillna('').str.replace(r'\n\s*\n', '\n', regex=True).str.strip()
    test['combined_text'] = test['title'] + ' [SEP] ' + test['full_text']
    
    # 테스트 데이터 길이 분석
    analyze_text_lengths(test['combined_text'])
    
    # 앙상블 예측
    print("\n🚀 A100 긴 시퀀스 앙상블 예측 시작...")
    ensemble_predictions = ensemble_predict(
        ensemble_models, ensemble_tokenizers, test['combined_text'], normalized_weights
    )
    
    print(f"\n📊 앙상블 예측 통계:")
    print(f"  예측 개수: {len(ensemble_predictions)}")
    print(f"  확률 범위: {ensemble_predictions.min():.4f} - {ensemble_predictions.max():.4f}")
    print(f"  평균 확률: {ensemble_predictions.mean():.4f}")
    print(f"  표준편차: {ensemble_predictions.std():.4f}")
    
    # 제출 파일 생성
    sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = ensemble_predictions
    sample_submission.to_csv('./long_sequence_submission.csv', index=False)
    
    print(f"\n🎉 A100 긴 시퀀스 앙상블 훈련 완료!")
    print(f"📊 최종 결과:")
    print(f"  시퀀스 길이: {MAX_LEN} (청킹 없음)")
    print(f"  앙상블 모델 수: {len(ensemble_models)}")
    print(f"  사용된 모델들:")
    for i, model_name in enumerate(ENSEMBLE_MODELS):
        print(f"    {i+1}. {model_name} (가중치: {normalized_weights[i]:.3f})")
    print(f"  K-Fold: {N_FOLDS}-Fold")
    print(f"  제출 파일: long_sequence_submission.csv")
    print(f"  예상 성능: 기존 대비 +3~6% (정보 손실 최소화)")
    
    torch.cuda.empty_cache()
    print(sample_submission.head())

if __name__ == "__main__":
    main()