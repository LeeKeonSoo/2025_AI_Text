#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 극대화 전략: 최고 성능 AI vs Human 텍스트 분류기
- 스타일로메트리 특징 (문체 분석)
- 다층 TF-IDF (단어 + 문자 레벨)
- 토픽 모델링 (LDA)
- 5개 모델 가중 앙상블
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from scipy.sparse import hstack
from scipy.stats import entropy
from collections import Counter

print("🎯 극대화 전략 시작!")

def extract_stylometric_features(text):
    """스타일로메트리 특징 추출 - AI vs Human 문체 차이 포착"""
    if pd.isna(text) or text == "":
        return np.zeros(25)
    
    text = str(text)
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    
    if word_count == 0:
        return np.zeros(25)
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)
    
    features = []
    
    # 기본 길이 통계 (AI는 보통 일정한 길이 패턴)
    features.extend([
        char_count,
        word_count,
        sentence_count,
        char_count / word_count
    ])
    
    # 어휘 다양성 (AI는 반복적 어휘 사용 경향)
    unique_words = len(set(words))
    word_freq = Counter(words)
    hapax_legomena = sum(1 for count in word_freq.values() if count == 1)
    
    features.extend([
        unique_words / word_count,  # Type-Token Ratio
        hapax_legomena / word_count,  # 한 번만 등장하는 단어 비율
        np.mean([len(word) for word in words]),
        np.std([len(word) for word in words]) if len(words) > 1 else 0
    ])
    
    # 구두점 패턴 (AI는 특정 구두점 선호)
    for mark in ['.', ',', '!', '?']:
        features.append(text.count(mark) / char_count)
    
    # 문장 구조 분석 (AI는 균일한 문장 길이)
    sentence_lengths = [len(s.split()) for s in sentences]
    features.extend([
        np.mean(sentence_lengths),
        np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0,
        np.median(sentence_lengths),
        max(sentence_lengths) - min(sentence_lengths)
    ])
    
    # 언어별 문자 분포
    korean_chars = len(re.findall(r'[가-힣]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    digit_chars = len(re.findall(r'\\d', text))
    
    features.extend([
        korean_chars / char_count,
        english_chars / char_count,
        digit_chars / char_count,
        (korean_chars + english_chars) / char_count
    ])
    
    # 고급 패턴 분석
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    bigram_entropy = entropy(list(Counter(bigrams).values())) if len(bigrams) > 0 else 0
    
    features.extend([
        bigram_entropy,
        text.count('(') + text.count('['),  # 괄호 사용
        text.count('"') + text.count("'"),  # 인용부호
        len(re.findall(r'\\b[A-Z][a-z]+', text)),  # 대문자 단어
        len(re.findall(r'\\d+', text))  # 숫자 패턴
    ])
    
    return np.array(features)

def extract_semantic_features(text):
    """의미론적 특징 - AI의 의미적 일관성과 인간의 창의성 구분"""
    if pd.isna(text) or text == "":
        return np.zeros(10)
    
    text = str(text)
    words = text.split()
    
    if len(words) == 0:
        return np.zeros(10)
    
    features = []
    
    # 어휘 복잡도 (AI는 복잡한 단어 선호)
    long_words = [w for w in words if len(w) > 6]
    features.append(len(long_words) / len(words))
    
    # 비정상적 패턴
    unusual_patterns = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ]', text))
    features.append(unusual_patterns / len(text))
    
    # 단어 반복성
    word_repetition = len(words) - len(set(words))
    features.append(word_repetition / len(words))
    
    # 문체 일관성
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) > 1:
        sentence_starts = [s.split()[0] if s.split() else '' for s in sentences]
        start_diversity = len(set(sentence_starts)) / len(sentences)
        features.append(start_diversity)
        
        sentence_endings = [s[-1] if s else '' for s in sentences]
        ending_diversity = len(set(sentence_endings)) / len(sentences)
        features.append(ending_diversity)
    else:
        features.extend([0, 0])
    
    # 연결어 사용 (AI는 논리적 연결어 선호)
    connectors = ['그리고', '하지만', '따라서', '그러나', '또한', '즉', '예를 들어']
    connector_count = sum(text.count(conn) for conn in connectors)
    features.append(connector_count / len(words))
    
    # 감정 표현
    features.extend([
        text.count('?') / len(sentences),  # 의문문
        text.count('!') / len(sentences),  # 감탄문
        len(re.findall(r'[가-힣]', text)) / len(words),  # 한글 음절 밀도
    ])
    
    # 문단 응집성 (AI는 일관된 길이)
    if len(sentences) > 1:
        sent_lengths = [len(s.split()) for s in sentences]
        avg_len = np.mean(sent_lengths)
        similar_length_ratio = sum(1 for l in sent_lengths if abs(l - avg_len) < avg_len * 0.3) / len(sentences)
        features.append(similar_length_ratio)
    else:
        features.append(0)
    
    return np.array(features)

def create_advanced_features(df):
    """모든 고급 특징 생성"""
    print("🔍 스타일로메트리 특징 추출 중...")
    
    # 제목과 본문의 각종 특징들
    title_style = np.array([extract_stylometric_features(text) for text in df['title']])
    title_semantic = np.array([extract_semantic_features(text) for text in df['title']])
    text_style = np.array([extract_stylometric_features(text) for text in df['full_text']])
    text_semantic = np.array([extract_semantic_features(text) for text in df['full_text']])
    
    # 제목-본문 관계 특징
    relationships = []
    for i in range(len(df)):
        title_len = len(str(df.iloc[i]['title']))
        text_len = len(str(df.iloc[i]['full_text']))
        title_words = len(str(df.iloc[i]['title']).split())
        text_words = len(str(df.iloc[i]['full_text']).split())
        
        relationships.append([
            title_len / (text_len + 1),
            title_words / (text_words + 1),
            min(title_len, text_len) / (max(title_len, text_len) + 1)
        ])
    
    relationships = np.array(relationships)
    
    # 모든 특징 결합
    all_features = np.hstack([
        title_style, title_semantic,
        text_style, text_semantic,
        relationships
    ])
    
    print(f"💎 고급 특징 수: {all_features.shape[1]}")
    return all_features

def main():
    # 데이터 로딩
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
    
    # 1. 고급 수작업 특징
    print("\\n🧠 === 고급 특징 추출 ===")
    train_advanced = create_advanced_features(train)
    test_advanced = create_advanced_features(test)
    
    # 2. 다양한 TF-IDF 벡터화
    print("\\n🔤 === 다층 TF-IDF 벡터화 ===")
    
    # Word-level TF-IDF (의미 단위)
    tfidf_word = TfidfVectorizer(
        ngram_range=(1,3),  # 1-3gram으로 문맥 포착
        max_features=15000,
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        analyzer='word'
    )
    
    # Character-level TF-IDF (스타일 패턴)
    tfidf_char = TfidfVectorizer(
        ngram_range=(3,5),  # 문자 패턴으로 스타일 포착
        max_features=10000,
        min_df=5,
        analyzer='char',
        sublinear_tf=True
    )
    
    # 전체 텍스트 결합
    train_full = train['title'] + ' ' + train['full_text']
    test_full = test['title'] + ' ' + test['full_text']
    
    # TF-IDF 변환
    train_word_tfidf = tfidf_word.fit_transform(train_full)
    test_word_tfidf = tfidf_word.transform(test_full)
    
    train_char_tfidf = tfidf_char.fit_transform(train_full)
    test_char_tfidf = tfidf_char.transform(test_full)
    
    print(f"📝 Word TF-IDF 특징: {train_word_tfidf.shape[1]}")
    print(f"🔠 Char TF-IDF 특징: {train_char_tfidf.shape[1]}")
    
    # 3. 토픽 모델링 (주제 분포)
    print("\\n🎭 === 토픽 모델링 (LDA) ===")
    lda = LatentDirichletAllocation(
        n_components=50,  # 50개 주제
        random_state=42,
        max_iter=10
    )
    
    count_vect = CountVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1,2)
    )
    
    train_counts = count_vect.fit_transform(train_full)
    test_counts = count_vect.transform(test_full)
    
    train_topics = lda.fit_transform(train_counts)
    test_topics = lda.transform(test_counts)
    
    print(f"🎯 토픽 특징: {train_topics.shape[1]}")
    
    # 특징 통합
    print("\\n🔗 === 특징 통합 및 스케일링 ===")
    
    # 고급 특징 정규화 (Robust Scaler로 이상치에 강하게)
    scaler = RobustScaler()
    train_advanced_scaled = scaler.fit_transform(train_advanced)
    test_advanced_scaled = scaler.transform(test_advanced)
    
    # 토픽 특징 정규화
    topic_scaler = StandardScaler()
    train_topics_scaled = topic_scaler.fit_transform(train_topics)
    test_topics_scaled = topic_scaler.transform(test_topics)
    
    # 모든 특징을 하나로 통합
    X_train_all = hstack([
        train_word_tfidf,      # 단어 레벨 의미
        train_char_tfidf,      # 문자 레벨 스타일
        train_advanced_scaled, # 고급 언어학적 특징
        train_topics_scaled    # 주제 분포
    ])
    
    X_test_all = hstack([
        test_word_tfidf,
        test_char_tfidf,
        test_advanced_scaled,
        test_topics_scaled
    ])
    
    y = train['generated']
    
    print(f"🎯 최종 특징 수: {X_train_all.shape[1]:,}")
    print(f"📊 훈련 데이터 크기: {X_train_all.shape}")
    
    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y, stratify=y, test_size=0.2, random_state=42
    )
    
    print("✅ 데이터 분할 완료!")
    
    # 다양한 모델 훈련
    print("\\n🚀 === 5개 최고 성능 모델 훈련 ===")
    
    models = {}
    val_predictions = {}
    
    # 1. XGBoost - 트리 기반 앙상블의 왕
    print("🌳 XGBoost 훈련 중...")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        tree_method='hist'
    )
    xgb.fit(X_train, y_train)
    models['xgb'] = xgb
    val_predictions['xgb'] = xgb.predict_proba(X_val)[:, 1]
    print(f"✅ XGBoost AUC: {roc_auc_score(y_val, val_predictions['xgb']):.4f}")
    
    # 2. LightGBM - 빠르고 정확한 그래디언트 부스팅
    print("⚡ LightGBM 훈련 중...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='binary',
        metric='auc',
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    models['lgbm'] = lgbm
    val_predictions['lgbm'] = lgbm.predict_proba(X_val)[:, 1]
    print(f"✅ LightGBM AUC: {roc_auc_score(y_val, val_predictions['lgbm']):.4f}")
    
    # 3. CatBoost - 범주형 특징에 특화
    print("🐱 CatBoost 훈련 중...")
    catb = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        eval_metric='AUC'
    )
    catb.fit(X_train, y_train)
    models['catb'] = catb
    val_predictions['catb'] = catb.predict_proba(X_val)[:, 1]
    print(f"✅ CatBoost AUC: {roc_auc_score(y_val, val_predictions['catb']):.4f}")
    
    # 4. Random Forest - 다양성과 안정성
    print("🌲 Random Forest 훈련 중...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['rf'] = rf
    val_predictions['rf'] = rf.predict_proba(X_val)[:, 1]
    print(f"✅ Random Forest AUC: {roc_auc_score(y_val, val_predictions['rf']):.4f}")
    
    # 5. Logistic Regression - 선형 모델의 강점
    print("📈 Logistic Regression 훈련 중...")
    lr = LogisticRegression(
        C=0.1,
        random_state=42,
        max_iter=1000,
        solver='liblinear'
    )
    lr.fit(X_train, y_train)
    models['lr'] = lr
    val_predictions['lr'] = lr.predict_proba(X_val)[:, 1]
    print(f"✅ Logistic Regression AUC: {roc_auc_score(y_val, val_predictions['lr']):.4f}")
    
    # 최적 앙상블
    print("\\n🎯 === 최적 앙상블 가중치 계산 ===")
    
    # 각 모델의 성능 기반 가중치 계산
    aucs = {}
    for name, pred in val_predictions.items():
        auc = roc_auc_score(y_val, pred)
        aucs[name] = auc
    
    # 성능 기반 가중치 (지수적 스케일링)
    total_weight = sum(np.exp(auc * 10) for auc in aucs.values())
    weights = {name: np.exp(auc * 10) / total_weight for name, auc in aucs.items()}
    
    print("🏆 모델별 성능 및 가중치:")
    for name in weights:
        print(f"  {name}: AUC={aucs[name]:.4f}, Weight={weights[name]:.3f}")
    
    # 가중 앙상블 예측
    ensemble_pred = np.zeros(len(y_val))
    for name, pred in val_predictions.items():
        ensemble_pred += weights[name] * pred
    
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    print(f"\\n🚀 가중 앙상블 AUC: {ensemble_auc:.4f}")
    
    # 단순 평균과 비교
    simple_ensemble = np.mean(list(val_predictions.values()), axis=0)
    simple_auc = roc_auc_score(y_val, simple_ensemble)
    print(f"📊 단순 평균 앙상블 AUC: {simple_auc:.4f}")
    
    # 최고 성능 선택
    best_approach = 'weighted' if ensemble_auc > simple_auc else 'simple'
    best_auc = max(ensemble_auc, simple_auc)
    print(f"\\n🏆 선택된 앙상블 방식: {best_approach} (AUC: {best_auc:.4f})")
    
    # 최종 예측
    print("\\n🔮 === 최종 예측 중 ===")
    
    # 테스트 데이터 예측
    test_predictions = {}
    for name, model in models.items():
        test_predictions[name] = model.predict_proba(X_test_all)[:, 1]
        print(f"✅ {name} 예측 완료")
    
    # 최적 앙상블 적용
    if best_approach == 'weighted':
        final_probs = np.zeros(len(test_predictions['xgb']))
        for name, pred in test_predictions.items():
            final_probs += weights[name] * pred
        print("🎯 가중 앙상블 적용")
    else:
        final_probs = np.mean(list(test_predictions.values()), axis=0)
        print("📊 단순 평균 앙상블 적용")
    
    print(f"\\n📈 최종 예측 분포:")
    print(f"  최소값: {final_probs.min():.4f}")
    print(f"  최대값: {final_probs.max():.4f}")
    print(f"  평균값: {final_probs.mean():.4f}")
    print(f"  표준편차: {final_probs.std():.4f}")
    
    # 제출 파일 생성
    sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    sample_submission['generated'] = final_probs
    
    filename = './ultimate_ensemble_submission.csv'
    sample_submission.to_csv(filename, index=False)
    
    print(f"\\n🎉 최고 성능 제출 파일 생성 완료: {filename}")
    print(f"🚀 예상 성능: AUC {best_auc:.4f}")
    print(f"💪 이번엔 반드시 성공!")

if __name__ == "__main__":
    main()
