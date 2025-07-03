import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import roc_auc_score

from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

from scipy.sparse import hstack
from scipy.stats import entropy
from collections import Counter
import gc
import os

# 단일 실행 체크 (중복 메시지 방지)
if 'STRATEGY_INITIALIZED' not in os.environ:
    print("🚀 VRAM 8GB + RAM 64GB 최대 활용 전략 시작!")
    print(f"🔥 CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"💻 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎯 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        print("💪 메모리 제한 해제 - 최대 활용 모드!")
    os.environ['STRATEGY_INITIALIZED'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU 메모리 최대 활용 설정
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    # 메모리 프래그멘테이션 방지
    torch.cuda.set_per_process_memory_fraction(0.95)  # VRAM 95% 사용 허용

# ============ 고성능 KoBERT 모델 (메모리 제한 해제) ============
class HighPerformanceKoBERTClassifier(nn.Module):
    def __init__(self, model_name='klue/bert-base', dropout=0.3):
        super(HighPerformanceKoBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 더 강력한 헤드 (RAM 64GB 활용)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout * 0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout * 0.5)
        self.classifier = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 더 풍부한 특징 추출 (마지막 4개 레이어 평균)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)  # Global average pooling
        
        # 멀티 레이어 헤드
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        logits = self.classifier(x)
        return logits

class HighCapacityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=384):  # 더 긴 시퀀스
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])[:2000]  # 더 긴 텍스트 허용
        label = self.labels[idx] if self.labels is not None else 0
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float32)
        }

def train_high_performance_kobert(train_texts, train_labels, val_texts, val_labels, epochs=4, batch_size=32):
    """고성능 KoBERT 훈련 - 메모리 최대 활용"""
    print("🤖 고성능 KoBERT 모델 초기화 중...")
    
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    model = HighPerformanceKoBERTClassifier('klue/bert-base').to(device)
    
    # 그래디언트 체크포인팅 비활성화 (RAM 64GB 활용)
    # model.bert.gradient_checkpointing_enable()  # 주석 처리
    
    # 데이터셋 생성
    train_dataset = HighCapacityDataset(train_texts, train_labels, tokenizer)
    val_dataset = HighCapacityDataset(val_texts, val_labels, tokenizer)
    
    # 더 큰 배치와 더 많은 워커 (RAM 64GB 활용)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=8, pin_memory=True, persistent_workers=True)
    
    # 고성능 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, eps=1e-8)
    criterion = nn.BCEWithLogitsLoss()
    
    # 더 적극적인 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), eta_min=1e-6)
    
    # GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_auc = 0
    best_model = None
    
    print(f"🏋️ 고성능 KoBERT 훈련 시작 (에포크: {epochs}, 배치: {batch_size})...")
    
    for epoch in range(epochs):
        # 훈련
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # 혼합 정밀도 사용
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    logits = logits.view(-1)
                    labels = labels.view(-1)
                    loss = criterion(logits, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                logits = logits.view(-1)
                labels = labels.view(-1)
                loss = criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            
            # 진행상황 출력 (매 100 배치마다로 줄임)
            if batch_idx % 100 == 0:
                print(f"  배치 {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 검증
        model.eval()
        val_predictions = []
        val_true = []
        
        print("검증 중...")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        logits = model(input_ids, attention_mask)
                else:
                    logits = model(input_ids, attention_mask)
                
                probs = torch.sigmoid(logits).view(-1)
                val_predictions.extend(probs.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_auc = roc_auc_score(val_true, val_predictions)
        avg_loss = total_loss / batch_count
        
        print(f"🎯 에포크 {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict().copy()
        
        # 메모리 정리 (덜 자주)
        if epoch % 2 == 0:  # 2 에포크마다만
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    print(f"✅ 고성능 KoBERT 훈련 완료! 최고 AUC: {best_auc:.4f}")
    
    model.load_state_dict(best_model)
    return model, tokenizer, best_auc

def predict_high_performance_kobert(model, tokenizer, texts, batch_size=48):
    """고성능 KoBERT 예측"""
    model.eval()
    dataset = HighCapacityDataset(texts, None, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=8, pin_memory=True, persistent_workers=True)
    
    predictions = []
    
    print("KoBERT 예측 중...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)
            
            probs = torch.sigmoid(logits).view(-1)
            predictions.extend(probs.cpu().numpy())
            
            # 진행상황 (매 200 배치마다)
            if batch_idx % 200 == 0:
                print(f"  예측 배치 {batch_idx}/{len(loader)}")
    
    return np.array(predictions)

def extract_efficient_features(text):
    """효율적이고 안전한 특징 추출"""
    if pd.isna(text) or text == "":
        return np.zeros(15)  # 특징 수 줄임
    
    text = str(text)
    words = text.split()
    
    if len(words) == 0:
        return np.zeros(15)
    
    features = []
    
    # 핵심 통계만 (1-6)
    features.extend([
        len(text),  # 문자 수
        len(words),  # 단어 수
        len(text) / len(words),  # 평균 단어 길이
        len(set(words)) / len(words),  # 어휘 다양성
        np.mean([len(w) for w in words]),  # 평균 단어 길이
        np.std([len(w) for w in words]) if len(words) > 1 else 0,  # 단어 길이 표준편차
    ])
    
    # 안전한 문장 특징 (7-10)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)
    
    if sentence_count > 0:
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:  # 안전성 체크
            features.extend([
                sentence_count,
                len(text) / sentence_count,  # 평균 문장 길이
                np.mean(sentence_lengths),  # 평균 문장 단어 수
                np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0,
            ])
        else:
            features.extend([1, len(text), len(words), 0])
    else:
        features.extend([1, len(text), len(words), 0])
    
    # 핵심 구두점만 (11-13)
    features.extend([
        text.count('.') / len(text),
        text.count(',') / len(text),
        text.count('?') / len(text),
    ])
    
    # 한국어 특성 (14-15)
    korean_chars = len(re.findall(r'[가-힣]', text))
    features.extend([
        korean_chars / len(text),  # 한글 비율
        len(re.findall(r'[a-zA-Z]', text)) / len(text),  # 영문 비율
    ])
    
    return np.array(features)

def create_efficient_context_features(test_df):
    """효율적 문단 맥락 특징 - 연산량 최적화"""
    features_list = []
    grouped = test_df.groupby('title')
    
    print("효율적 문단 맥락 특징 생성 중...")
    for title, group in grouped:
        group = group.sort_values('paragraph_index').reset_index(drop=True)
        
        # 미리 계산 (반복 연산 줄임)
        all_lengths = [len(str(r['full_text'])) for _, r in group.iterrows()]
        avg_length = np.mean(all_lengths)
        median_length = np.median(all_lengths)
        
        for idx, row in group.iterrows():
            features = {}
            
            # 핵심 위치 정보만
            features['paragraph_index'] = row['paragraph_index']
            features['total_paragraphs'] = len(group)
            features['relative_position'] = row['paragraph_index'] / len(group)
            features['is_first'] = 1 if row['paragraph_index'] == 1 else 0
            features['is_last'] = 1 if row['paragraph_index'] == len(group) else 0
            
            # 핵심 길이 특성만
            current_length = len(str(row['full_text']))
            features.update({
                'current_length': current_length,
                'length_vs_avg': current_length / (avg_length + 1),
                'length_vs_median': current_length / (median_length + 1),
            })
            
            # 간단한 인접 관계만
            if idx > 0:
                prev_length = len(str(group.iloc[idx-1]['full_text']))
                features['prev_length_ratio'] = current_length / (prev_length + 1)
            else:
                features['prev_length_ratio'] = 1.0
                
            if idx < len(group) - 1:
                next_length = len(str(group.iloc[idx+1]['full_text']))
                features['next_length_ratio'] = current_length / (next_length + 1)
            else:
                features['next_length_ratio'] = 1.0
            
            features_list.append(features)
    
    return pd.DataFrame(features_list)

def main():
    # ============ 데이터 로딩 ============
    print("📥 데이터 로딩 중...")
    train = pd.read_csv('./train.csv', encoding='utf-8-sig')
    test = pd.read_csv('./test.csv', encoding='utf-8-sig')
    
    # 결측치 처리
    train['title'] = train['title'].fillna('')
    train['full_text'] = train['full_text'].fillna('')
    
    test = test.rename(columns={'paragraph_text': 'full_text'})
    test['title'] = test['title'].fillna('')
    test['full_text'] = test['full_text'].fillna('')
    
    print(f"📚 학습 데이터: {train.shape}")
    print(f"🎯 테스트 데이터: {test.shape}")
    print(f"⚖️ 클래스 분포: {train['generated'].value_counts().to_dict()}")
    
    # 전체 텍스트 생성 (제목 + 본문)
    train['combined_text'] = train['title'] + ' ' + train['full_text']
    test['combined_text'] = test['title'] + ' ' + test['full_text']
    
    # ============ 데이터 분할 ============
    X = train[['title', 'full_text', 'combined_text']]
    y = train['generated']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    
    # ============ 고성능 KoBERT 특징 추출 ============
    print("\\n🤖 === 고성능 KoBERT 특징 추출 ===")
    
    # KoBERT 모델 훈련 (더 큰 배치, 더 긴 시퀀스)
    kobert_model, tokenizer, kobert_auc = train_high_performance_kobert(
        X_train['combined_text'].values,
        y_train.values,
        X_val['combined_text'].values,
        y_val.values,
        epochs=4,  # 더 많은 에포크
        batch_size=32  # 더 큰 배치
    )
    
    # KoBERT 특징 생성
    print("🔮 KoBERT 특징 생성 중...")
    kobert_train_features = predict_high_performance_kobert(kobert_model, tokenizer, X_train['combined_text'].values, batch_size=48).reshape(-1, 1)
    kobert_val_features = predict_high_performance_kobert(kobert_model, tokenizer, X_val['combined_text'].values, batch_size=48).reshape(-1, 1)
    kobert_test_features = predict_high_performance_kobert(kobert_model, tokenizer, test['combined_text'].values, batch_size=48).reshape(-1, 1)
    
    print(f"✅ KoBERT 특징 생성 완료! AUC: {kobert_auc:.4f}")
    
    # 메모리 정리
    del kobert_model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # ============ 고용량 TF-IDF 특징 ============
    print("\\n🔤 === 고용량 TF-IDF 특징 추출 ===")
    
    get_title = FunctionTransformer(lambda x: x['title'], validate=False)
    get_text = FunctionTransformer(lambda x: x['full_text'], validate=False)
    
    # 더 많은 특징 (RAM 64GB 활용)
    tfidf_vectorizer = FeatureUnion([
        ('title', Pipeline([('selector', get_title),
                            ('tfidf', TfidfVectorizer(
                                ngram_range=(1,3),  # 더 긴 n-gram
                                max_features=8000,  # 더 많은 특징
                                min_df=2,
                                max_df=0.95,
                                sublinear_tf=True
                            ))])),
        ('full_text', Pipeline([('selector', get_text), 
                                ('tfidf', TfidfVectorizer(
                                    ngram_range=(1,3),  # 더 긴 n-gram
                                    max_features=20000,  # 더 많은 특징
                                    min_df=2,
                                    max_df=0.95,
                                    sublinear_tf=True
                                ))])),
    ])
    
    # TF-IDF 변환
    print("🔄 고용량 TF-IDF 벡터화 중...")
    tfidf_train = tfidf_vectorizer.fit_transform(X_train[['title', 'full_text']])
    tfidf_val = tfidf_vectorizer.transform(X_val[['title', 'full_text']])
    tfidf_test = tfidf_vectorizer.transform(test[['title', 'full_text']])
    
    print(f"📝 TF-IDF 특징 수: {tfidf_train.shape[1]:,}")
    
    # ============ 효율적 수작업 특징 ============
    print("\\n🧠 === 효율적 특징 추출 ===")
    
    # 제목과 본문 특징 (연산량 최적화)
    print("🔍 제목 특징 추출 중...")
    train_title_features = np.array([extract_efficient_features(text) for text in X_train['title']])
    val_title_features = np.array([extract_efficient_features(text) for text in X_val['title']])
    test_title_features = np.array([extract_efficient_features(text) for text in test['title']])
    
    print("🔍 본문 특징 추출 중...")
    train_text_features = np.array([extract_efficient_features(text) for text in X_train['full_text']])
    val_text_features = np.array([extract_efficient_features(text) for text in X_val['full_text']])
    test_text_features = np.array([extract_efficient_features(text) for text in test['full_text']])
    
    # 특징 결합 및 스케일링
    efficient_train = np.hstack([train_title_features, train_text_features])
    efficient_val = np.hstack([val_title_features, val_text_features])
    efficient_test = np.hstack([test_title_features, test_text_features])
    
    scaler = StandardScaler()
    efficient_train = scaler.fit_transform(efficient_train)
    efficient_val = scaler.transform(efficient_val)
    efficient_test = scaler.transform(efficient_test)
    
    print(f"🧠 효율적 특징 수: {efficient_train.shape[1]}")
    
    # ============ 모든 특징 결합 ============
    print("\\n🔗 === 특징 통합 ===")
    
    X_train_combined = hstack([
        tfidf_train,
        kobert_train_features,
        efficient_train
    ])
    
    X_val_combined = hstack([
        tfidf_val,
        kobert_val_features,
        efficient_val
    ])
    
    X_test_combined = hstack([
        tfidf_test,
        kobert_test_features,
        efficient_test
    ])
    
    print(f"🎯 최종 특징 수: {X_train_combined.shape[1]:,}")
    
    # ============ 고성능 모델 앙상블 ============
    print("\\n🚀 === 고성능 모델 앙상블 ===")
    
    models = {}
    val_predictions = {}
    
    # XGBoost (고성능 설정)
    print("🌳 고성능 XGBoost 훈련 중...")
    xgb_model = XGBClassifier(
        n_estimators=500,  # 더 많은 트리
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        tree_method='hist',
        n_jobs=-1
    )
    xgb_model.fit(X_train_combined, y_train)
    models['xgb'] = xgb_model
    val_predictions['xgb'] = xgb_model.predict_proba(X_val_combined)[:, 1]
    print(f"✅ XGBoost AUC: {roc_auc_score(y_val, val_predictions['xgb']):.4f}")
    
    # LightGBM (고성능 설정)
    print("⚡ 고성능 LightGBM 훈련 중...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='binary',
        metric='auc',
        device='cpu',
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train_combined, y_train)
    models['lgb'] = lgb_model
    val_predictions['lgb'] = lgb_model.predict_proba(X_val_combined)[:, 1]
    print(f"✅ LightGBM AUC: {roc_auc_score(y_val, val_predictions['lgb']):.4f}")
    
    # CatBoost (고성능 설정)
    print("🐱 고성능 CatBoost 훈련 중...")
    catb_model = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.03,
        random_seed=42,
        task_type='CPU',
        thread_count=-1,
        verbose=False
    )
    catb_model.fit(X_train_combined, y_train)
    models['catb'] = catb_model
    val_predictions['catb'] = catb_model.predict_proba(X_val_combined)[:, 1]
    print(f"✅ CatBoost AUC: {roc_auc_score(y_val, val_predictions['catb']):.4f}")
    
    # KoBERT 단독 성능
    val_predictions['kobert'] = kobert_val_features.flatten()
    kobert_val_auc = roc_auc_score(y_val, val_predictions['kobert'])
    print(f"🤖 KoBERT AUC: {kobert_val_auc:.4f}")
    
    # ============ 고급 앙상블 ============
    print("\\n🎯 === 고급 앙상블 ===")
    
    # 성능 기반 가중치
    aucs = {}
    for name, pred in val_predictions.items():
        auc = roc_auc_score(y_val, pred)
        aucs[name] = auc
    
    # 더 적극적인 가중치 (성능 차이 극대화)
    total_weight = sum(np.exp(auc * 8) for auc in aucs.values())  # 더 큰 지수
    weights = {name: np.exp(auc * 8) / total_weight for name, auc in aucs.items()}
    
    print("🏆 모델별 성능 및 가중치:")
    for name in sorted(weights.keys(), key=lambda x: aucs[x], reverse=True):
        print(f"  {name}: AUC={aucs[name]:.4f}, Weight={weights[name]:.3f}")
    
    # 가중 앙상블
    ensemble_pred = np.zeros(len(y_val))
    for name, pred in val_predictions.items():
        ensemble_pred += weights[name] * pred
    
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    print(f"\\n🚀 고급 앙상블 AUC: {ensemble_auc:.4f}")
    
    # ============ 최종 예측 ============
    print("\\n🔮 === 최종 예측 ===")
    
    # 테스트 예측
    print("📊 각 모델로 테스트 예측 중...")
    test_predictions = {}
    test_predictions['xgb'] = models['xgb'].predict_proba(X_test_combined)[:, 1]
    test_predictions['lgb'] = models['lgb'].predict_proba(X_test_combined)[:, 1]
    test_predictions['catb'] = models['catb'].predict_proba(X_test_combined)[:, 1]
    test_predictions['kobert'] = kobert_test_features.flatten()
    
    # 가중 앙상블 적용
    final_probs = np.zeros(len(test_predictions['xgb']))
    for name, pred in test_predictions.items():
        final_probs += weights[name] * pred
    
    # ============ 효율적 문단 맥락 후처리 ============
    print("\\n🎯 === 효율적 문단 맥락 후처리 ===")
    
    # 효율적 문단 맥락 특징
    test_context = create_efficient_context_features(test)
    
    adjusted_probs = final_probs.copy()
    
    # 간단하고 효율적인 title별 조정
    adjustment_count = 0
    for title in test['title'].unique():
        mask = test['title'] == title
        title_indices = test[mask].index
        title_probs = final_probs[mask]
        title_context = test_context[mask]
        
        if len(title_probs) > 1:
            # 간단한 스무딩
            avg_prob = np.mean(title_probs)
            smoothing_factor = 0.15
            
            for i, (idx, row) in enumerate(title_context.iterrows()):
                original_prob = title_probs[i]
                
                # 핵심 조정만
                adjustment = 0
                if row['is_first'] == 1:
                    adjustment -= 0.03  # 첫 문단
                if row['relative_position'] > 0.8:
                    adjustment += 0.02  # 마지막 부분
                
                # 길이 기반 간단 조정
                if row['length_vs_avg'] > 2.0:
                    adjustment += 0.02  # 너무 긴 문단
                elif row['length_vs_avg'] < 0.3:
                    adjustment += 0.025  # 너무 짧은 문단
                
                # 스무딩 + 조정
                smoothed_prob = original_prob * (1 - smoothing_factor) + avg_prob * smoothing_factor
                final_prob = np.clip(smoothed_prob + adjustment, 0.01, 0.99)
                
                adjusted_probs[title_indices[i]] = final_prob
                
                if abs(final_prob - original_prob) > 0.001:
                    adjustment_count += 1
    
    avg_adjustment = np.mean(np.abs(adjusted_probs - final_probs))
    print(f"📈 평균 조정 정도: {avg_adjustment:.4f}")
    print(f"📊 조정된 문단 수: {adjustment_count}/{len(test)} ({adjustment_count/len(test)*100:.1f}%)")
    
    # ============ 품질 검증 및 제출 ============
    print(f"\\n📊 최종 예측 분포:")
    print(f"  최소값: {adjusted_probs.min():.4f}")
    print(f"  최대값: {adjusted_probs.max():.4f}")
    print(f"  평균값: {adjusted_probs.mean():.4f}")
    print(f"  표준편차: {adjusted_probs.std():.4f}")
    print(f"  25% 분위: {np.percentile(adjusted_probs, 25):.4f}")
    print(f"  75% 분위: {np.percentile(adjusted_probs, 75):.4f}")
    
    sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = adjusted_probs
    
    sample_submission.to_csv('./without_knowledge.csv', index=False)
    
    print(f"\\n🎉 최고 성능 제출 파일 생성: without_knowledge.csv")
    print(f"🚀 최종 검증 AUC: {ensemble_auc:.4f}")
    print(f"💻 효율적이면서 고성능 달성:")
    print(f"  - 🤖 고성능 KoBERT (배치32, 길이384, 멀티레이어헤드)")
    print(f"  - 🔤 고용량 TF-IDF ({tfidf_train.shape[1]:,} 특징)")
    print(f"  - 🧠 효율적 특징 ({efficient_train.shape[1]} 특징)")
    print(f"  - 🎯 4개 모델 고성능 앙상블 (500 트리)")
    print(f"  - 🔧 효율적 문단 맥락 후처리")
    print(f"  - ⚡ 연산량 최적화로 빠른 실행")
    print(f"💪 효율성과 성능의 완벽한 균형!")

if __name__ == "__main__":
    main()