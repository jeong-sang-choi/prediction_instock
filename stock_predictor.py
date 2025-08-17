import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import sys
import locale
print(sys.path)
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 한글 인코딩 설정
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'Korean_Korea.949')
    except:
        pass

class StockPredictor:
    def __init__(self, symbol, start_date=None, end_date=None):
        """
        주식 예측 모델 초기화
        
        Args:
            symbol (str): 주식 심볼 (예: 'AAPL', '005930.KS')
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
        """
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        
    def fetch_data(self):
        """Yahoo Finance에서 주식 데이터 가져오기"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(start=self.start_date, end=self.end_date)
            print(f"{self.symbol} 데이터를 성공적으로 가져왔습니다. (총 {len(self.data)}개 데이터)")
            return self.data
        except Exception as e:
            print(f"데이터 가져오기 실패: {e}")
            return None
    
    def create_features(self):
        """기술적 지표들을 사용하여 특성 생성"""
        if self.data is None:
            print("먼저 데이터를 가져와주세요.")
            return None
        
        df = self.data.copy()
        
        # 기본 가격 특성
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'] - df['Open']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # 이동평균
        df['MA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['MA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # 이동평균 비율
        df['MA_5_Ratio'] = df['Close'] / df['MA_5']
        df['MA_10_Ratio'] = df['Close'] / df['MA_10']
        df['MA_20_Ratio'] = df['Close'] / df['MA_20']
        
        # 볼린저 밴드
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'])
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # 스토캐스틱
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # 거래량 지표
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # ATR (Average True Range)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # 목표 변수 생성 (다음 날 종가)
        df['Target'] = df['Close'].shift(-1)
        
        # NaN 값 제거
        df = df.dropna()
        
        self.data = df
        print(f"특성 생성 완료. 총 {len(df.columns)}개 특성")
        return df
    
    def prepare_data(self, test_size=0.2):
        """모델 학습을 위한 데이터 준비"""
        if self.data is None:
            print("먼저 특성을 생성해주세요.")
            return None, None, None, None
        
        # 특성 선택 (Target 제외)
        feature_columns = [col for col in self.data.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = self.data[feature_columns]
        y = self.data['Target']
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, model_type='random_forest'):
        """모델 학습"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        if X_train is None:
            return None
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            print(f"지원하지 않는 모델 타입: {model_type}")
            return None
        
        # 모델 학습
        self.model.fit(X_train, y_train)
        
        # 예측
        y_pred = self.model.predict(X_test)
        
        # 성능 평가
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n모델 성능:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict_next_day(self):
        """다음 날 주가 예측"""
        if self.model is None:
            print("먼저 모델을 학습해주세요.")
            return None
        
        # 최신 데이터로 특성 생성
        latest_features = self.data.iloc[-1:][[col for col in self.data.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]]
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # 예측
        prediction = self.model.predict(latest_features_scaled)[0]
        current_price = self.data['Close'].iloc[-1]
        
        print(f"\n다음 날 예측:")
        print(f"현재 가격: ${current_price:.2f}")
        print(f"예측 가격: ${prediction:.2f}")
        print(f"예상 변동: ${prediction - current_price:.2f} ({((prediction - current_price) / current_price * 100):.2f}%)")
        
        return prediction
    
    def plot_predictions(self, results):
        """예측 결과 시각화"""
        if results is None:
            return
        
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        plt.figure(figsize=(15, 10))
        
        # 예측 vs 실제
        plt.subplot(2, 2, 1)
        plt.plot(y_test.index, y_test.values, label='실제', alpha=0.7)
        plt.plot(y_test.index, y_pred, label='예측', alpha=0.7)
        plt.title('실제 vs 예측 주가')
        plt.xlabel('날짜')
        plt.ylabel('주가')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 산점도
        plt.subplot(2, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('실제 주가')
        plt.ylabel('예측 주가')
        plt.title('실제 vs 예측 산점도')
        
        # 잔차
        plt.subplot(2, 2, 3)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측 주가')
        plt.ylabel('잔차')
        plt.title('잔차 분석')
        
        # 잔차 히스토그램
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('잔차')
        plt.ylabel('빈도')
        plt.title('잔차 분포')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance(self):
        """특성 중요도 분석"""
        if self.model is None:
            print("먼저 모델을 학습해주세요.")
            return
        
        feature_columns = [col for col in self.data.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        importance = self.model.feature_importances_
        
        # 특성 중요도 데이터프레임 생성
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_df.head(15), x='Importance', y='Feature')
        plt.title('특성 중요도 (상위 15개)')
        plt.xlabel('중요도')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df

def main():
    """메인 실행 함수"""
    print("=== 주식 예측 모델 ===")
    
    # 사용자 입력
    symbol = input("주식 심볼을 입력하세요 (예: AAPL, 005930.KS): ").strip()
    
    # 모델 생성
    predictor = StockPredictor(symbol)
    
    # 데이터 가져오기
    print(f"\n{symbol} 데이터를 가져오는 중...")
    predictor.fetch_data()
    
    # 특성 생성
    print("\n특성을 생성하는 중...")
    predictor.create_features()
    
    # 모델 학습
    print("\n모델을 학습하는 중...")
    results = predictor.train_model()
    
    if results:
        # 결과 시각화
        print("\n결과를 시각화하는 중...")
        predictor.plot_predictions(results)
        
        # 특성 중요도
        print("\n특성 중요도를 분석하는 중...")
        predictor.feature_importance()
        
        # 다음 날 예측
        print("\n다음 날 주가를 예측하는 중...")
        predictor.predict_next_day()

if __name__ == "__main__":
    main()
