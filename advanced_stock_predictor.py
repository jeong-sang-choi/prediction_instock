import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 고급 모델들을 위한 import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost가 설치되지 않았습니다. pip install xgboost로 설치하세요.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM이 설치되지 않았습니다. pip install lightgbm으로 설치하세요.")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow가 설치되지 않았습니다. pip install tensorflow로 설치하세요.")

class AdvancedStockPredictor:
    def __init__(self, symbol, start_date=None, end_date=None):
        """
        고급 주식 예측 모델 초기화
        
        Args:
            symbol (str): 주식 심볼 (예: 'AAPL', '005930.KS')
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
        """
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.models = {}
        self.scalers = {}
        
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
    
    def create_advanced_features(self):
        """고급 기술적 지표들을 사용하여 특성 생성"""
        if self.data is None:
            print("먼저 데이터를 가져와주세요.")
            return None
        
        df = self.data.copy()
        
        # 기본 가격 특성
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Change'] = df['Close'] - df['Open']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # 이동평균
        for window in [5, 10, 20, 50, 100]:
            df[f'MA_{window}'] = ta.trend.sma_indicator(df['Close'], window=window)
            df[f'MA_{window}_Ratio'] = df['Close'] / df[f'MA_{window}']
        
        # 지수이동평균
        for window in [12, 26]:
            df[f'EMA_{window}'] = ta.trend.ema_indicator(df['Close'], window=window)
        
        # 볼린저 밴드
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['BB_Squeeze'] = df['BB_Width'] / df['BB_Middle']
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'])
        df['RSI_MA'] = ta.trend.sma_indicator(df['RSI'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        df['MACD_Histogram_MA'] = ta.trend.sma_indicator(df['MACD_Histogram'], window=9)
        
        # 스토캐스틱
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        df['Stoch_K_D'] = df['Stoch_K'] - df['Stoch_D']
        
        # Williams %R
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # CCI (Commodity Channel Index)
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        
        # 거래량 지표
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Price_Trend'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
        df['On_Balance_Volume'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # ATR (Average True Range)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        # 파라볼릭 SAR
        df['Parabolic_SAR'] = ta.trend.psar_down(df['High'], df['Low'], df['Close'])
        
        # 모멘텀 지표
        df['ROC'] = ta.momentum.roc(df['Close'])
        df['TSI'] = ta.momentum.tsi(df['Close'])
        
        # 변동성 지표
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # 가격 패턴
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        
        # 시간 특성
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        # 목표 변수 생성 (다음 날 종가)
        df['Target'] = df['Close'].shift(-1)
        df['Target_Return'] = df['Target'] / df['Close'] - 1
        
        # NaN 값 제거
        df = df.dropna()
        
        self.data = df
        print(f"고급 특성 생성 완료. 총 {len(df.columns)}개 특성")
        return df
    
    def prepare_data_for_traditional_models(self, test_size=0.2):
        """전통적인 ML 모델을 위한 데이터 준비"""
        if self.data is None:
            print("먼저 특성을 생성해주세요.")
            return None, None, None, None
        
        # 특성 선택 (Target 제외)
        feature_columns = [col for col in self.data.columns 
                          if col not in ['Target', 'Target_Return', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = self.data[feature_columns]
        y = self.data['Target']
        
        # 시간 순서를 고려한 분할
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 특성 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['traditional'] = scaler
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns
    
    def prepare_data_for_lstm(self, lookback=60, test_size=0.2):
        """LSTM 모델을 위한 시계열 데이터 준비"""
        if self.data is None:
            print("먼저 특성을 생성해주세요.")
            return None, None, None, None
        
        # 특성 선택
        feature_columns = [col for col in self.data.columns 
                          if col not in ['Target', 'Target_Return', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # 데이터 정규화
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.data[feature_columns + ['Close']])
        
        # 시계열 데이터 생성
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, :-1])  # 특성들
            y.append(scaled_data[i, -1])  # 종가
        
        X, y = np.array(X), np.array(y)
        
        # 훈련/테스트 분할
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.scalers['lstm'] = scaler
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self):
        """Random Forest 모델 학습"""
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data_for_traditional_models()
        
        if X_train is None:
            return None
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 성능 평가
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nRandom Forest 성능:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        self.models['random_forest'] = model
        
        return {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_columns': feature_columns
        }
    
    def train_xgboost(self):
        """XGBoost 모델 학습"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost가 설치되지 않았습니다.")
            return None
        
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data_for_traditional_models()
        
        if X_train is None:
            return None
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 성능 평가
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nXGBoost 성능:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        self.models['xgboost'] = model
        
        return {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_columns': feature_columns
        }
    
    def train_lightgbm(self):
        """LightGBM 모델 학습"""
        if not LIGHTGBM_AVAILABLE:
            print("LightGBM이 설치되지 않았습니다.")
            return None
        
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data_for_traditional_models()
        
        if X_train is None:
            return None
        
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 성능 평가
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nLightGBM 성능:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        self.models['lightgbm'] = model
        
        return {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_columns': feature_columns
        }
    
    def train_lstm(self):
        """LSTM 모델 학습"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow가 설치되지 않았습니다.")
            return None
        
        X_train, X_test, y_train, y_test = self.prepare_data_for_lstm()
        
        if X_train is None:
            return None
        
        # LSTM 모델 구성
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # 모델 학습
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 성능 평가
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nLSTM 성능:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        self.models['lstm'] = model
        
        return {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred,
            'history': history
        }
    
    def ensemble_predict(self, models_to_use=['random_forest', 'xgboost', 'lightgbm']):
        """앙상블 예측"""
        predictions = []
        
        for model_name in models_to_use:
            if model_name in self.models:
                if model_name == 'lstm':
                    # LSTM은 다른 방식으로 예측
                    continue
                else:
                    # 전통적인 모델들
                    X_train, X_test, y_train, y_test, feature_columns = self.prepare_data_for_traditional_models()
                    if X_train is not None:
                        pred = self.models[model_name].predict(X_test)
                        predictions.append(pred)
        
        if predictions:
            # 평균 예측
            ensemble_pred = np.mean(predictions, axis=0)
            
            # 성능 평가
            mse = mean_squared_error(y_test, ensemble_pred)
            mae = mean_absolute_error(y_test, ensemble_pred)
            r2 = r2_score(y_test, ensemble_pred)
            
            print(f"\n앙상블 모델 성능:")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Mean Absolute Error: {mae:.2f}")
            print(f"R² Score: {r2:.4f}")
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'y_test': y_test,
                'y_pred': ensemble_pred
            }
        
        return None
    
    def compare_models(self):
        """모든 모델의 성능 비교"""
        results = {}
        
        # Random Forest
        rf_result = self.train_random_forest()
        if rf_result:
            results['Random Forest'] = rf_result
        
        # XGBoost
        xgb_result = self.train_xgboost()
        if xgb_result:
            results['XGBoost'] = xgb_result
        
        # LightGBM
        lgb_result = self.train_lightgbm()
        if lgb_result:
            results['LightGBM'] = lgb_result
        
        # LSTM
        lstm_result = self.train_lstm()
        if lstm_result:
            results['LSTM'] = lstm_result
        
        # 앙상블
        ensemble_result = self.ensemble_predict()
        if ensemble_result:
            results['Ensemble'] = ensemble_result
        
        # 결과 비교 시각화
        self.plot_model_comparison(results)
        
        return results
    
    def plot_model_comparison(self, results):
        """모델 성능 비교 시각화"""
        if not results:
            return
        
        # 성능 지표 추출
        models = list(results.keys())
        mse_scores = [results[model]['mse'] for model in models]
        mae_scores = [results[model]['mae'] for model in models]
        r2_scores = [results[model]['r2'] for model in models]
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # MSE 비교
        axes[0].bar(models, mse_scores, color='skyblue')
        axes[0].set_title('Mean Squared Error (낮을수록 좋음)')
        axes[0].set_ylabel('MSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE 비교
        axes[1].bar(models, mae_scores, color='lightgreen')
        axes[1].set_title('Mean Absolute Error (낮을수록 좋음)')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R² 비교
        axes[2].bar(models, r2_scores, color='salmon')
        axes[2].set_title('R² Score (높을수록 좋음)')
        axes[2].set_ylabel('R²')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 최고 성능 모델 출력
        best_model = max(results.keys(), key=lambda x: results[x]['r2'])
        print(f"\n최고 성능 모델: {best_model}")
        print(f"R² Score: {results[best_model]['r2']:.4f}")

def main():
    """메인 실행 함수"""
    print("=== 고급 주식 예측 모델 ===")
    
    # 사용자 입력
    symbol = input("주식 심볼을 입력하세요 (예: AAPL, 005930.KS): ").strip()
    
    # 모델 생성
    predictor = AdvancedStockPredictor(symbol)
    
    # 데이터 가져오기
    print(f"\n{symbol} 데이터를 가져오는 중...")
    predictor.fetch_data()
    
    # 특성 생성
    print("\n고급 특성을 생성하는 중...")
    predictor.create_advanced_features()
    
    # 모든 모델 학습 및 비교
    print("\n모든 모델을 학습하고 성능을 비교하는 중...")
    results = predictor.compare_models()

if __name__ == "__main__":
    main()
