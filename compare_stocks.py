import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def analyze_stock(symbol, days=365):
    """주식 분석"""
    try:
        # 데이터 가져오기
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{days}d")
        
        if len(data) == 0:
            return None
            
        # 기본 통계
        current_price = data['Close'].iloc[-1]
        start_price = data['Close'].iloc[0]
        total_return = (current_price - start_price) / start_price * 100
        
        # 변동성
        daily_returns = data['Close'].pct_change()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # 연간 변동성
        
        # 최고/최저
        high_52w = data['High'].max()
        low_52w = data['Low'].min()
        high_ratio = (current_price - low_52w) / (high_52w - low_52w) * 100
        
        # 이동평균
        ma_20 = data['Close'].rolling(20).mean().iloc[-1]
        ma_50 = data['Close'].rolling(50).mean().iloc[-1]
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'total_return': total_return,
            'volatility': volatility,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'high_ratio': high_ratio,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'rsi': current_rsi,
            'data': data
        }
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def compare_stocks():
    """주식 비교 분석"""
    symbols = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    print("=== 주식 비교 분석 ===\n")
    
    results = []
    for symbol in symbols:
        print(f"{symbol} 분석 중...")
        result = analyze_stock(symbol)
        if result:
            results.append(result)
            print(f"  현재가: ${result['current_price']:.2f}")
            print(f"  수익률: {result['total_return']:.2f}%")
            print(f"  변동성: {result['volatility']:.2f}%")
            print(f"  RSI: {result['rsi']:.1f}")
            print()
    
    if not results:
        print("분석할 수 있는 데이터가 없습니다.")
        return
    
    # 결과를 데이터프레임으로 변환
    df_results = pd.DataFrame(results)
    
    # 투자 추천 분석
    print("=== 투자 추천 분석 ===\n")
    
    # 1. 수익률 기준
    best_return = df_results.loc[df_results['total_return'].idxmax()]
    print(f"최고 수익률: {best_return['symbol']} ({best_return['total_return']:.2f}%)")
    
    # 2. 변동성 기준 (낮을수록 안정적)
    lowest_vol = df_results.loc[df_results['volatility'].idxmin()]
    print(f"최저 변동성: {lowest_vol['symbol']} ({lowest_vol['volatility']:.2f}%)")
    
    # 3. RSI 기준 (과매수/과매도)
    oversold = df_results[df_results['rsi'] < 30]
    overbought = df_results[df_results['rsi'] > 70]
    
    if not oversold.empty:
        print(f"과매도 (RSI < 30): {', '.join(oversold['symbol'].tolist())}")
    if not overbought.empty:
        print(f"과매수 (RSI > 70): {', '.join(overbought['symbol'].tolist())}")
    
    # 4. 52주 고점 대비 위치
    low_high_ratio = df_results.loc[df_results['high_ratio'].idxmin()]
    print(f"52주 고점 대비 최저: {low_high_ratio['symbol']} ({low_high_ratio['high_ratio']:.1f}%)")
    
    # 5. 이동평균 기준
    above_ma20 = df_results[df_results['current_price'] > df_results['ma_20']]
    above_ma50 = df_results[df_results['current_price'] > df_results['ma_50']]
    
    print(f"20일 이동평균 위: {', '.join(above_ma20['symbol'].tolist())}")
    print(f"50일 이동평균 위: {', '.join(above_ma50['symbol'].tolist())}")
    
    # 종합 추천
    print("\n=== 종합 추천 ===")
    
    # 점수 계산
    scores = []
    for _, row in df_results.iterrows():
        score = 0
        
        # 수익률 점수 (높을수록 좋음)
        if row['total_return'] > 0:
            score += 2
        elif row['total_return'] > -10:
            score += 1
        
        # 변동성 점수 (낮을수록 좋음)
        if row['volatility'] < 30:
            score += 2
        elif row['volatility'] < 50:
            score += 1
        
        # RSI 점수 (30-70 사이가 좋음)
        if 30 <= row['rsi'] <= 70:
            score += 2
        elif 20 <= row['rsi'] <= 80:
            score += 1
        
        # 이동평균 점수
        if row['current_price'] > row['ma_20']:
            score += 1
        if row['current_price'] > row['ma_50']:
            score += 1
        
        scores.append(score)
    
    df_results['score'] = scores
    
    # 점수별 정렬
    df_results = df_results.sort_values('score', ascending=False)
    
    print("\n종합 점수 (높을수록 추천):")
    for _, row in df_results.iterrows():
        print(f"{row['symbol']}: {row['score']}점 (현재가: ${row['current_price']:.2f})")
    
    # 최고 점수 주식 추천
    best_stock = df_results.iloc[0]
    print(f"\n🎯 최고 추천: {best_stock['symbol']}")
    print(f"   현재가: ${best_stock['current_price']:.2f}")
    print(f"   수익률: {best_stock['total_return']:.2f}%")
    print(f"   변동성: {best_stock['volatility']:.2f}%")
    print(f"   RSI: {best_stock['rsi']:.1f}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 수익률 비교
    axes[0, 0].bar(df_results['symbol'], df_results['total_return'], color='skyblue')
    axes[0, 0].set_title('수익률 비교 (%)')
    axes[0, 0].set_ylabel('수익률 (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 변동성 비교
    axes[0, 1].bar(df_results['symbol'], df_results['volatility'], color='lightcoral')
    axes[0, 1].set_title('변동성 비교 (%)')
    axes[0, 1].set_ylabel('변동성 (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # RSI 비교
    axes[1, 0].bar(df_results['symbol'], df_results['rsi'], color='lightgreen')
    axes[1, 0].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='과매수')
    axes[1, 0].axhline(y=30, color='blue', linestyle='--', alpha=0.7, label='과매도')
    axes[1, 0].set_title('RSI 비교')
    axes[1, 0].set_ylabel('RSI')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend()
    
    # 종합 점수
    axes[1, 1].bar(df_results['symbol'], df_results['score'], color='gold')
    axes[1, 1].set_title('종합 점수')
    axes[1, 1].set_ylabel('점수')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return df_results

if __name__ == "__main__":
    compare_stocks()
