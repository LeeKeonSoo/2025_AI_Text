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

print("🚀 KoBERT + CPU 최적화 전략 시작!")
print(f"🔥 CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"💻 GPU: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ KoBERT 모델 정의 ============
class KoBERTClassifier(nn.Module):
    def __init__(self, model_name='klue/bert-base', dropout=0.3):
        super(KoBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            'label': torch.tensor(label, dtype=torch.float)
        }

def train_kobert_model(train_texts, train_labels, val_texts, val_labels, epochs=3, batch_size=16):
    """KoBERT 모델 훈련"""
    print("🤖 KoBERT 모델 초기화 중...")
    
    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    model = KoBERTClassifier('klue/bert-base').to(device)
    
    # 데이터셋 생성
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 옵티마이저와 손실함수
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCELoss()
    
    best_auc = 0
    best_model = None
    
    print(f"🏋️ KoBERT 훈련 시작 (에포크: {epochs}, 배치: {batch_size})...")
    
    for epoch in range(epochs):
        # 훈련
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 진행상황 출력 (50 배치마다)
            if batch_idx % 50 == 0:
                print(f"  배치 {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # 검증
        model.eval()
        val_predictions = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                
                val_predictions.extend(outputs.cpu().numpy().flatten())
                val_true.extend(labels.cpu().numpy())
        
        val_auc = roc_auc_score(val_true, val_predictions)
        avg_loss = total_loss / len(train_loader)
        
        print(f"🎯 에포크 {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict().copy()
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    print(f"✅ KoBERT 훈련 완료! 최고 AUC: {best_auc:.4f}")
    
    # 최고 성능 모델 로드
    model.load_state_dict(best_model)
    return model, tokenizer, best_auc

def predict_kobert(model, tokenizer, texts, batch_size=16):
    """KoBERT 예측"""
    model.eval()
    dataset = TextDataset(texts, None, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            predictions.extend(outputs.cpu().numpy().flatten())
            
            # 진행상황 출력
            if batch_idx % 50 == 0:
                print(f"  예측 배치 {batch_idx}/{len(loader)}")
    
    return np.array(predictions)

def extract_advanced_features(text):
    """고급 텍스트 특징 추출"""
    if pd.isna(text) or text == "":
        return np.zeros(15)
    
    text = str(text)
    words = text.split()
    
    if len(words) == 0:
        return np.zeros(15)
    
    features = []
    
    # 기본 통계
    features.extend([
        len(text),  # 문자 수
        len(words),  # 단어 수
        len(text) / len(words),  # 평균 단어 길이
        len(set(words)) / len(words),  # 어휘 다양성
    ])
    
    # 문장 분석
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)
    
    features.extend([
        sentence_count,
        len(text) / sentence_count,  # 평균 문장 길이
    ])
    
    # 구두점 분석
    features.extend([
        text.count('.') / len(text),
        text.count(',') / len(text),
        text.count('!') / len(text),
        text.count('?') / len(text),
    ])
    
    # 한국어 특성
    korean_chars = len(re.findall(r'[가-힣]', text))
    features.extend([
        korean_chars / len(text),  # 한글 비율
        len(re.findall(r'[a-zA-Z]', text)) / len(text),  # 영문 비율
        len(re.findall(r'\\d', text)) / len(text),  # 숫자 비율
    ])
    
    # 고급 패턴
    # 연결어 사용
    connectors = ['그리고', '하지만', '따라서', '그러나', '또한']
    connector_count = sum(text.count(conn) for conn in connectors)
    features.append(connector_count / len(words))
    
    # 반복 패턴
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    if len(bigrams) > 0:
        bigram_entropy = entropy(list(Counter(bigrams).values()))
        features.append(bigram_entropy)
    else:
        features.append(0)
    
    return np.array(features)

def create_paragraph_context_features(test_df):
    """문단 맥락 특징 생성"""
    features_list = []
    grouped = test_df.groupby('title')
    
    for title, group in grouped:
        group = group.sort_values('paragraph_index').reset_index(drop=True)
        
        for idx, row in group.iterrows():
            features = {}
            
            # 위치 정보
            features['paragraph_index'] = row['paragraph_index']
            features['total_paragraphs'] = len(group)
            features['relative_position'] = row['paragraph_index'] / len(group)
            features['is_first'] = 1 if row['paragraph_index'] == 1 else 0
            features['is_last'] = 1 if row['paragraph_index'] == len(group) else 0
            
            # 길이 특성
            current_length = len(str(row['full_text']))
            all_lengths = [len(str(r['full_text'])) for _, r in group.iterrows()]
            features['length_vs_avg'] = current_length / (np.mean(all_lengths) + 1)
            features['length_vs_median'] = current_length / (np.median(all_lengths) + 1)
            
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
    
    # ============ 1. KoBERT 특징 추출 ============
    print("\\n🤖 === KoBERT 특징 추출 ===")
    
    # KoBERT 모델 훈련
    kobert_model, tokenizer, kobert_auc = train_kobert_model(
        X_train['combined_text'].values,
        y_train.values,
        X_val['combined_text'].values,
        y_val.values,
        epochs=3,
        batch_size=16
    )
    
    # KoBERT 특징 생성 (확률값)
    print("🔮 KoBERT 훈련 데이터 특징 생성 중...")
    kobert_train_features = predict_kobert(kobert_model, tokenizer, X_train['combined_text'].values).reshape(-1, 1)
    
    print("🔮 KoBERT 검증 데이터 특징 생성 중...")
    kobert_val_features = predict_kobert(kobert_model, tokenizer, X_val['combined_text'].values).reshape(-1, 1)
    
    print("🔮 KoBERT 테스트 데이터 특징 생성 중...")
    kobert_test_features = predict_kobert(kobert_model, tokenizer, test['combined_text'].values).reshape(-1, 1)
    
    print(f"✅ KoBERT 특징 생성 완료!")
    
    # ============ 2. TF-IDF 특징 ============
    print("\\n🔤 === TF-IDF 특징 추출 ===")
    
    get_title = FunctionTransformer(lambda x: x['title'], validate=False)
    get_text = FunctionTransformer(lambda x: x['full_text'], validate=False)
    
    tfidf_vectorizer = FeatureUnion([
        ('title', Pipeline([('selector', get_title),
                            ('tfidf', TfidfVectorizer(
                                ngram_range=(1,2), 
                                max_features=5000,
                                min_df=3,
                                max_df=0.95,
                                sublinear_tf=True
                            ))])),
        ('full_text', Pipeline([('selector', get_text), 
                                ('tfidf', TfidfVectorizer(
                                    ngram_range=(1,3), 
                                    max_features=15000,
                                    min_df=3,
                                    max_df=0.95,
                                    sublinear_tf=True
                                ))])),
    ])
    
    # TF-IDF 변환
    print("🔄 TF-IDF 벡터화 중...")
    tfidf_train = tfidf_vectorizer.fit_transform(X_train[['title', 'full_text']])
    tfidf_val = tfidf_vectorizer.transform(X_val[['title', 'full_text']])
    tfidf_test = tfidf_vectorizer.transform(test[['title', 'full_text']])
    
    print(f"📝 TF-IDF 특징 수: {tfidf_train.shape[1]}")
    
    # ============ 3. 고급 수작업 특징 ============
    print("\\n🧠 === 고급 특징 추출 ===")
    
    # 제목과 본문의 고급 특징
    print("🔍 제목 특징 추출 중...")
    train_title_features = np.array([extract_advanced_features(text) for text in X_train['title']])
    val_title_features = np.array([extract_advanced_features(text) for text in X_val['title']])
    test_title_features = np.array([extract_advanced_features(text) for text in test['title']])
    
    print("🔍 본문 특징 추출 중...")
    train_text_features = np.array([extract_advanced_features(text) for text in X_train['full_text']])
    val_text_features = np.array([extract_advanced_features(text) for text in X_val['full_text']])
    test_text_features = np.array([extract_advanced_features(text) for text in test['full_text']])
    
    # 특징 결합
    advanced_train = np.hstack([train_title_features, train_text_features])
    advanced_val = np.hstack([val_title_features, val_text_features])
    advanced_test = np.hstack([test_title_features, test_text_features])
    
    # 스케일링
    scaler = StandardScaler()
    advanced_train = scaler.fit_transform(advanced_train)
    advanced_val = scaler.transform(advanced_val)
    advanced_test = scaler.transform(advanced_test)
    
    print(f"🧠 고급 특징 수: {advanced_train.shape[1]}")
    
    # ============ 4. 모든 특징 결합 ============
    print("\\n🔗 === 특징 통합 ===")
    
    # 모든 특징 결합
    X_train_combined = hstack([
        tfidf_train,
        kobert_train_features,
        advanced_train
    ])
    
    X_val_combined = hstack([
        tfidf_val,
        kobert_val_features,
        advanced_val
    ])
    
    X_test_combined = hstack([
        tfidf_test,
        kobert_test_features,
        advanced_test
    ])
    
    print(f"🎯 최종 특징 수: {X_train_combined.shape[1]:,}")
    
    # ============ 5. CPU 최적화된 모델 훈련 ============
    print("\\n🚀 === CPU 최적화 모델 훈련 ===")
    
    models = {}
    val_predictions = {}
    
    # XGBoost (CPU 전용)
    print("🌳 XGBoost (CPU) 훈련 중...")
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        tree_method='hist',  # CPU 최적화
        n_jobs=-1  # 모든 CPU 코어 사용
    )
    xgb_model.fit(X_train_combined, y_train)
    models['xgb'] = xgb_model
    val_predictions['xgb'] = xgb_model.predict_proba(X_val_combined)[:, 1]
    print(f"✅ XGBoost AUC: {roc_auc_score(y_val, val_predictions['xgb']):.4f}")
    
    # LightGBM (CPU)
    print("⚡ LightGBM (CPU) 훈련 중...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
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
    
    # CatBoost (CPU)
    print("🐱 CatBoost (CPU) 훈련 중...")
    catb_model = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
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
    print(f"🤖 KoBERT AUC: {kobert_auc:.4f}")
    
    # ============ 6. 최적 앙상블 ============
    print("\\n🎯 === 최적 앙상블 ===")
    
    # 성능 기반 가중치
    aucs = {}
    for name, pred in val_predictions.items():
        auc = roc_auc_score(y_val, pred)
        aucs[name] = auc
    
    # 지수적 가중치
    total_weight = sum(np.exp(auc * 5) for auc in aucs.values())
    weights = {name: np.exp(auc * 5) / total_weight for name, auc in aucs.items()}
    
    print("🏆 모델별 성능 및 가중치:")
    for name in weights:
        print(f"  {name}: AUC={aucs[name]:.4f}, Weight={weights[name]:.3f}")
    
    # 가중 앙상블
    ensemble_pred = np.zeros(len(y_val))
    for name, pred in val_predictions.items():
        ensemble_pred += weights[name] * pred
    
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    print(f"\\n🚀 가중 앙상블 AUC: {ensemble_auc:.4f}")
    
    # ============ 7. 최종 예측 ============
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
    
    # ============ 8. 문단 맥락 후처리 ============
    print("\\n🎯 === 문단 맥락 후처리 ===")
    
    # 문단 맥락 특징
    test_context = create_paragraph_context_features(test)
    
    adjusted_probs = final_probs.copy()
    
    # title별 조정
    for title in test['title'].unique():
        mask = test['title'] == title
        title_indices = test[mask].index
        title_probs = final_probs[mask]
        title_context = test_context[mask]
        
        if len(title_probs) > 1:
            avg_prob = np.mean(title_probs)
            smoothing_factor = 0.15
            
            for i, (idx, row) in enumerate(title_context.iterrows()):
                original_prob = title_probs[i]
                
                # 위치 기반 조정
                adjustment = 0
                if row['is_first'] == 1:
                    adjustment -= 0.03  # 첫 문단
                if row['relative_position'] > 0.8:
                    adjustment += 0.02  # 마지막 부분
                
                # 스무딩 + 조정
                smoothed_prob = original_prob * (1 - smoothing_factor) + avg_prob * smoothing_factor
                final_prob = np.clip(smoothed_prob + adjustment, 0, 1)
                
                adjusted_probs[title_indices[i]] = final_prob
    
    print(f"📈 조정 정도: {np.mean(np.abs(adjusted_probs - final_probs)):.4f}")
    
    # ============ 9. 제출 파일 생성 ============
    print(f"\\n📊 최종 예측 분포:")
    print(f"  최소값: {adjusted_probs.min():.4f}")
    print(f"  최대값: {adjusted_probs.max():.4f}")
    print(f"  평균값: {adjusted_probs.mean():.4f}")
    
    sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = adjusted_probs
    
    sample_submission.to_csv('./without_knowledge.csv', index=False)
    
    print(f"\\n🎉 최고 성능 제출 파일 생성: without_knowledge.csv")
    print(f"🚀 최종 검증 AUC: {ensemble_auc:.4f}")
    print(f"🏆 사용된 기술:")
    print(f"  - 🤖 KoBERT (한국어 사전훈련 모델)")
    print(f"  - 🔤 고급 TF-IDF (1-3gram)")
    print(f"  - 🧠 수작업 언어학적 특징")
    print(f"  - 🎯 4개 모델 가중 앙상블")
    print(f"  - 🔧 문단 맥락 후처리")
    print(f"  - 💻 CPU 최적화 훈련")
    print(f"💪 CPU 환경에서도 최고 성능 달성!")

if __name__ == "__main__":
    main()