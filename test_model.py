#!/usr/bin/env python3
"""
주식 예측 모델 테스트 스크립트
"""

import sys
import traceback
from datetime import datetime, timedelta

def test_basic_model():
    """기본 모델 테스트"""
    print("=== 기본 모델 테스트 ===")
    
    try:
        from stock_predictor import StockPredictor
        
        # 테스트용 주식 심볼 (애플)
        symbol = "AAPL"
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"테스트 주식: {symbol}")
        print(f"기간: {start_date} ~ {end_date}")
        
        # 모델 생성
        predictor = StockPredictor(symbol, start_date, end_date)
        
        # 데이터 가져오기
        print("\n1. 데이터 가져오기...")
        data = predictor.fetch_data()
        if data is None:
            print("❌ 데이터 가져오기 실패")
            return False
        print(f"✅ 데이터 가져오기 성공: {len(data)}개 데이터")
        
        # 특성 생성
        print("\n2. 특성 생성...")
        features = predictor.create_features()
        if features is None:
            print("❌ 특성 생성 실패")
            return False
        print(f"✅ 특성 생성 성공: {len(features.columns)}개 특성")
        
        # 모델 학습
        print("\n3. 모델 학습...")
        results = predictor.train_model()
        if results is None:
            print("❌ 모델 학습 실패")
            return False
        print("✅ 모델 학습 성공")
        print(f"   - MSE: {results['mse']:.2f}")
        print(f"   - MAE: {results['mae']:.2f}")
        print(f"   - R²: {results['r2']:.4f}")
        
        # 다음 날 예측
        print("\n4. 다음 날 예측...")
        prediction = predictor.predict_next_day()
        if prediction is None:
            print("❌ 예측 실패")
            return False
        print("✅ 예측 성공")
        
        print("\n🎉 기본 모델 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 기본 모델 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_advanced_model():
    """고급 모델 테스트"""
    print("\n=== 고급 모델 테스트 ===")
    
    try:
        from advanced_stock_predictor import AdvancedStockPredictor
        
        # 테스트용 주식 심볼 (마이크로소프트)
        symbol = "MSFT"
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"테스트 주식: {symbol}")
        print(f"기간: {start_date} ~ {end_date}")
        
        # 모델 생성
        predictor = AdvancedStockPredictor(symbol, start_date, end_date)
        
        # 데이터 가져오기
        print("\n1. 데이터 가져오기...")
        data = predictor.fetch_data()
        if data is None:
            print("❌ 데이터 가져오기 실패")
            return False
        print(f"✅ 데이터 가져오기 성공: {len(data)}개 데이터")
        
        # 특성 생성
        print("\n2. 고급 특성 생성...")
        features = predictor.create_advanced_features()
        if features is None:
            print("❌ 특성 생성 실패")
            return False
        print(f"✅ 특성 생성 성공: {len(features.columns)}개 특성")
        
        # Random Forest 모델 테스트
        print("\n3. Random Forest 모델 학습...")
        rf_result = predictor.train_random_forest()
        if rf_result is None:
            print("❌ Random Forest 학습 실패")
            return False
        print("✅ Random Forest 학습 성공")
        print(f"   - R²: {rf_result['r2']:.4f}")
        
        # XGBoost 모델 테스트 (가능한 경우)
        try:
            print("\n4. XGBoost 모델 학습...")
            xgb_result = predictor.train_xgboost()
            if xgb_result is None:
                print("⚠️ XGBoost 학습 실패 (패키지 미설치 또는 오류)")
            else:
                print("✅ XGBoost 학습 성공")
                print(f"   - R²: {xgb_result['r2']:.4f}")
        except Exception as e:
            print(f"⚠️ XGBoost 테스트 건너뜀: {e}")
        
        print("\n🎉 고급 모델 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 고급 모델 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_data_fetch():
    """데이터 가져오기 테스트"""
    print("\n=== 데이터 가져오기 테스트 ===")
    
    try:
        import yfinance as yf
        
        test_symbols = ["AAPL", "MSFT", "005930.KS", "000660.KS"]
        
        for symbol in test_symbols:
            print(f"\n테스트: {symbol}")
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1mo")
                
                if not data.empty:
                    print(f"✅ 성공: {len(data)}개 데이터")
                    print(f"   최신 가격: ${data['Close'].iloc[-1]:.2f}")
                else:
                    print(f"❌ 실패: 데이터 없음")
                    
            except Exception as e:
                print(f"❌ 실패: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터 가져오기 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 주식 예측 모델 테스트 시작")
    print("=" * 50)
    
    # 패키지 설치 확인
    print("📦 패키지 설치 확인...")
    
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'ta', 'scikit-learn',
        'matplotlib', 'seaborn', 'plotly', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (미설치)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 다음 패키지들을 설치해주세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return
    
    print("\n✅ 모든 필수 패키지가 설치되어 있습니다.")
    
    # 테스트 실행
    success_count = 0
    total_tests = 3
    
    # 데이터 가져오기 테스트
    if test_data_fetch():
        success_count += 1
    
    # 기본 모델 테스트
    if test_basic_model():
        success_count += 1
    
    # 고급 모델 테스트
    if test_advanced_model():
        success_count += 1
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    print(f"성공: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 모든 테스트가 성공했습니다!")
        print("\n다음 단계:")
        print("1. python stock_predictor.py - 기본 모델 실행")
        print("2. python advanced_stock_predictor.py - 고급 모델 실행")
        print("3. streamlit run streamlit_app.py - 웹 인터페이스 실행")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
        print("오류 메시지를 확인하고 필요한 패키지를 설치해주세요.")

if __name__ == "__main__":
    main()
