import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import warnings
import random
import re
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
device = torch.device('cuda')

# KoBERT 극한 최적화 설정
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 6                    # 충분한 학습
LEARNING_RATE = 8e-6         # 정교한 학습률
WARMUP_RATIO = 0.15          # 긴 워밍업
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.15

# 고급 기법 활용
USE_FOCAL_LOSS = True
USE_LABEL_SMOOTHING = True
USE_ADVERSARIAL = True
USE_MIXUP = True
USE_KFOLD = True
N_FOLDS = 5

print("🔥 KoBERT 극한 최적화 모드")

def advanced_text_preprocessing(text):
    """고급 텍스트 전처리"""
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    # 연속된 문장부호 정리
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    # 불필요한 기호 정리
    text = re.sub(r'[^\w\s가-힣.,!?;:()\[\]{}"\'-]', ' ', text)
    
    # 다시 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class AdvancedDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, is_train=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.is_train = is_train
        
    def __len__(self):
        return len(self.texts)
    
    def text_augmentation(self, text):
        """데이터 증강"""
        if not self.is_train or random.random() > 0.3:
            return text
        
        sentences = text.split('. ')
        if len(sentences) > 3:
            # 문장 순서 일부 섞기
            if random.random() < 0.5:
                mid = len(sentences) // 2
                random.shuffle(sentences[1:mid])
            
            # 일부 문장 제거 (최대 20%)
            if random.random() < 0.3:
                remove_count = min(len(sentences) // 5, 2)
                if remove_count > 0:
                    indices = random.sample(range(1, len(sentences)-1), remove_count)
                    sentences = [s for i, s in enumerate(sentences) if i not in indices]
            
            text = '. '.join(sentences)
        
        return text
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        text = self.text_augmentation(text)
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class ExtremeKoBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('skt/kobert-base-v1')
        
        # 최소한의 프리징 (성능 최우선)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:2]:  # 처음 2개만 프리징
            for param in layer.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # 고성능 분류 헤드
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Multi-layer 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE // 2),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE // 2),
            
            nn.Linear(hidden_size // 4, 2)
        )
        
        # 어텐션 풀링
        self.attention_pool = nn.MultiheadAttention(
            hidden_size, num_heads=12, batch_first=True, dropout=DROPOUT_RATE
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 고급 풀링: 어텐션 + CLS
        sequence_output = outputs.last_hidden_state
        
        # Self-attention pooling
        attended_output, _ = self.attention_pool(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=(attention_mask == 0)
        )
        
        # 가중 평균
        mask_expanded = attention_mask.unsqueeze(-1).float()
        weighted_output = (attended_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # CLS 토큰과 결합
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            cls_output = outputs.pooler_output
        else:
            cls_output = sequence_output[:, 0, :]
        
        # 두 표현 결합
        combined = weighted_output + cls_output
        enhanced = self.pre_classifier(combined)
        enhanced = enhanced + combined  # 잔차 연결
        
        logits = self.classifier(enhanced)
        
        return logits

def mixup_data(x, y, alpha=0.2):
    """Mixup 데이터 증강"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def adversarial_training(model, inputs, labels, optimizer, criterion, epsilon=0.01):
    """적대적 훈련"""
    embeddings = model.bert.embeddings.word_embeddings
    
    # 원본 임베딩 저장
    original_embeddings = embeddings.weight.data.clone()
    
    # 그래디언트 계산을 위해 requires_grad 설정
    embeddings.weight.requires_grad_()
    
    # Forward pass
    logits = model(inputs['input_ids'], inputs['attention_mask'])
    loss = criterion(logits, labels)
    
    # 임베딩에 대한 그래디언트 계산
    loss.backward(retain_graph=True)
    grad = embeddings.weight.grad.data
    
    # 적대적 perturbation 생성
    perturbation = epsilon * grad.sign()
    embeddings.weight.data = original_embeddings + perturbation
    
    # 적대적 샘플로 다시 계산
    adv_logits = model(inputs['input_ids'], inputs['attention_mask'])
    adv_loss = criterion(adv_logits, labels)
    
    # 원본 임베딩 복원
    embeddings.weight.data = original_embeddings
    embeddings.weight.requires_grad_(False)
    
    return adv_loss

def train_extreme_kobert(model, train_loader, val_loader, fold=None):
    # 고급 손실함수 조합
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=1, gamma=2)
    elif USE_LABEL_SMOOTHING:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 고급 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-6
    )
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    # 코사인 스케줄러 사용
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_auc = 0
    best_state = None
    patience = 0
    max_patience = 3
    
    fold_str = f"Fold{fold}" if fold is not None else "Single"
    
    for epoch in range(EPOCHS):
        # 훈련
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'{fold_str}-E{epoch+1}')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            
            # Mixup 적용
            if USE_MIXUP and random.random() < 0.3:
                mixed_inputs, y_a, y_b, lam = mixup_data(input_ids, labels)
                inputs['input_ids'] = mixed_inputs
                logits = model(**inputs)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(**inputs)
                loss = criterion(logits, labels)
            
            # 적대적 훈련 (일부 배치에서)
            if USE_ADVERSARIAL and batch_idx % 3 == 0:
                adv_loss = adversarial_training(model, inputs, labels, optimizer, criterion)
                loss = 0.7 * loss + 0.3 * adv_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # 검증
        model.eval()
        val_probs, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # 메트릭 계산
        auc = roc_auc_score(val_labels, val_probs)
        f1 = f1_score(val_labels, [1 if p > 0.5 else 0 for p in val_probs])
        acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_probs])
        
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'  E{epoch+1}: Loss={avg_loss:.4f}, AUC={auc:.4f}, F1={f1:.4f}, Acc={acc:.4f}, LR={current_lr:.2e}')
        
        # Early stopping with patience
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict().copy()
            patience = 0
            print(f'    🚀 New best AUC: {best_auc:.4f}')
        else:
            patience += 1
            if patience >= max_patience:
                print(f'    Early stopping at epoch {epoch+1}')
                break
    
    model.load_state_dict(best_state)
    return model, best_auc

def predict_extreme(model, test_loader):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Extreme Prediction'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            predictions.extend(probs[:, 1].cpu().numpy())
    
    return np.array(predictions)

def main():
    print("🔥 KoBERT 극한 최적화 시작!")
    
    # 데이터 로드
    train = pd.read_csv('./train.csv', encoding='utf-8-sig')
    test = pd.read_csv('./test.csv', encoding='utf-8-sig')
    
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    # 고급 전처리
    print("고급 텍스트 전처리...")
    train['title'] = train['title'].apply(advanced_text_preprocessing)
    train['full_text'] = train['full_text'].apply(advanced_text_preprocessing)
    train['text'] = train['title'] + ' [SEP] ' + train['full_text']
    
    test = test.rename(columns={'paragraph_text': 'full_text'})
    test['title'] = test['title'].apply(advanced_text_preprocessing)
    test['full_text'] = test['full_text'].apply(advanced_text_preprocessing)
    test['text'] = test['title'] + ' [SEP] ' + test['full_text']
    
    # 텍스트 길이 분석
    lengths = train['text'].str.len()
    print(f"텍스트 길이 - 평균: {lengths.mean():.0f}, 최대: {lengths.max()}, 95%ile: {lengths.quantile(0.95):.0f}")
    
    X = train['text']
    y = train['generated']
    print(f"클래스 분포: {y.value_counts().to_dict()}")
    
    # KoBERT 토크나이저
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    
    if USE_KFOLD:
        # K-Fold 교차 검증
        print(f"\n{N_FOLDS}-Fold 교차 검증 시작")
        fold_predictions = []
        fold_aucs = []
        
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'='*50}")
            print(f"Fold {fold+1}/{N_FOLDS}")
            print(f"{'='*50}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 가중 샘플링
            class_counts = y_train.value_counts()
            class_weights = {0: 1.0, 1: class_counts[0] / class_counts[1]}
            sample_weights = y_train.map(class_weights)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            
            # 데이터셋
            train_dataset = AdvancedDataset(X_train, y_train, tokenizer, is_train=True)
            val_dataset = AdvancedDataset(X_val, y_val, tokenizer, is_train=False)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=4, pin_memory=True)
            
            # 모델 훈련
            model = ExtremeKoBERT().to(device)
            model, fold_auc = train_extreme_kobert(model, train_loader, val_loader, fold+1)
            
            fold_aucs.append(fold_auc)
            
            # 첫 번째 폴드 모델 저장 (테스트용)
            if fold == 0:
                best_model = model
        
        avg_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"\n🎯 K-Fold 결과: AUC = {avg_auc:.4f} ± {std_auc:.4f}")
        print(f"Fold AUCs: {[f'{auc:.4f}' for auc in fold_aucs]}")
        
    else:
        # 단일 훈련
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        
        train_dataset = AdvancedDataset(X_train, y_train, tokenizer, is_train=True)
        val_dataset = AdvancedDataset(X_val, y_val, tokenizer, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=4, pin_memory=True)
        
        best_model = ExtremeKoBERT().to(device)
        best_model, best_auc = train_extreme_kobert(best_model, train_loader, val_loader)
        print(f"\n🎯 최종 AUC: {best_auc:.4f}")
    
    # 테스트 예측
    print("\n🚀 극한 성능 예측 시작...")
    test_dataset = AdvancedDataset(test['text'], labels=None, tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=4, pin_memory=True)
    
    predictions = predict_extreme(best_model, test_loader)
    
    print(f"\n📊 예측 결과:")
    print(f"  범위: {predictions.min():.4f} - {predictions.max():.4f}")
    print(f"  평균: {predictions.mean():.4f}")
    print(f"  표준편차: {predictions.std():.4f}")
    
    # 제출 파일
    submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    submission['generated'] = predictions
    submission.to_csv('./extreme_kobert_submission.csv', index=False)
    
    print(f"\n🔥 KoBERT 극한 최적화 완료!")
    print(f"📁 제출 파일: extreme_kobert_submission.csv")
    print(f"🎯 목표: AUC 0.90+ 달성!")
    print(submission.head())

if __name__ == "__main__":
    main()