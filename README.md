# 📈 주식 예측 머신러닝 모델

이 프로젝트는 다양한 머신러닝 알고리즘을 사용하여 주식 가격을 예측하는 종합적인 도구입니다.

## 🚀 주요 기능

### 1. 기본 예측 모델 (`stock_predictor.py`)
- **Random Forest** 기반 주식 예측
- 기술적 지표 기반 특성 생성
- 다음 날 주가 예측
- 모델 성능 평가 및 시각화

### 2. 고급 예측 모델 (`advanced_stock_predictor.py`)
- **다중 알고리즘 지원**:
  - Random Forest
  - XGBoost
  - LightGBM
  - LSTM (Long Short-Term Memory)
- 앙상블 예측
- 모델 성능 비교

### 3. 웹 인터페이스 (`streamlit_app.py`)
- **Streamlit** 기반 대화형 웹 앱
- 실시간 데이터 시각화
- 기술적 분석 차트
- 모델 성능 비교

## 📦 설치 방법

### 1. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python -m venv stock_model

# 가상환경 활성화 (Windows)
Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source stock_model/bin/activate
```

### 2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 추가 패키지 설치 (선택사항)
```bash
# 고급 모델을 위한 추가 패키지
pip install xgboost lightgbm tensorflow
```

## 🎯 사용 방법

### 1. 기본 모델 실행
```bash
python stock_predictor.py
```

### 2. 고급 모델 실행
```bash
python advanced_stock_predictor.py
```

### 3. 웹 인터페이스 실행
```bash
streamlit run streamlit_app.py
```

## 📊 지원하는 주식 심볼

### 미국 주식
- `AAPL` - Apple Inc.
- `MSFT` - Microsoft Corporation
- `GOOGL` - Alphabet Inc.
- `AMZN` - Amazon.com Inc.
- `TSLA` - Tesla Inc.

### 한국 주식
- `005930.KS` - 삼성전자
- `000660.KS` - SK하이닉스
- `035420.KS` - NAVER
- `051910.KS` - LG화학
- `006400.KS` - 삼성SDI

## 🔧 기술적 지표

### 가격 기반 지표
- 이동평균 (MA 5, 10, 20, 50)
- 지수이동평균 (EMA 12, 26)
- 볼린저 밴드
- 파라볼릭 SAR

### 모멘텀 지표
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- 스토캐스틱 오실레이터
- Williams %R
- CCI (Commodity Channel Index)

### 거래량 지표
- 거래량 이동평균
- 거래량 가격 추세
- On-Balance Volume

### 변동성 지표
- ATR (Average True Range)
- 변동성 (20일 이동 표준편차)

## 📈 모델 성능 평가

### 평가 지표
- **MSE (Mean Squared Error)**: 예측 오차의 제곱 평균
- **MAE (Mean Absolute Error)**: 예측 오차의 절댓값 평균
- **R² Score**: 결정계수 (1에 가까울수록 좋음)

### 시각화
- 실제 vs 예측 주가 비교
- 예측 오차 분석
- 특성 중요도 분석
- 모델 성능 비교 차트

## 🌐 웹 인터페이스 기능

### 1. 데이터 개요
- 실시간 주가 정보
- 캔들스틱 차트
- 기본 통계 정보

### 2. 예측 모델
- 기본 모델 (Random Forest)
- 고급 모델 (다중 알고리즘)
- 모델 성능 비교
- 다음 날 예측

### 3. 차트 분석
- 기술적 분석 차트
- RSI, MACD, 볼린저 밴드
- 거래량 분석

### 4. 상세 정보
- 회사 정보
- 수익률 분석
- 변동성 분석

## ⚠️ 주의사항

1. **투자 조언 아님**: 이 도구는 교육 및 연구 목적으로만 사용하세요.
2. **과거 성과**: 과거 데이터 기반 예측이므로 미래 성과를 보장하지 않습니다.
3. **리스크 관리**: 실제 투자 시에는 적절한 리스크 관리가 필요합니다.
4. **데이터 품질**: Yahoo Finance 데이터의 정확성을 항상 확인하세요.

## 🔄 업데이트 및 개선

### 향후 계획
- [ ] 감정 분석 추가 (뉴스, 소셜 미디어)
- [ ] 더 많은 기술적 지표 추가
- [ ] 실시간 데이터 업데이트
- [ ] 포트폴리오 최적화 기능
- [ ] 백테스팅 기능

### 기여 방법
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 문의 및 지원

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**면책 조항**: 이 소프트웨어는 교육 목적으로만 제공됩니다. 실제 투자 결정에 사용하기 전에 전문가의 조언을 구하시기 바랍니다.
