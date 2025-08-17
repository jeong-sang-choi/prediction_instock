import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 기본 모델 import
from stock_predictor import StockPredictor

# 고급 모델 import (선택적)
try:
    from advanced_stock_predictor import AdvancedStockPredictor
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

def main():
    st.set_page_config(
        page_title="주식 예측 모델",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 주식 예측 머신러닝 모델")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("설정")
    
    # 주식 심볼 입력
    symbol = st.sidebar.text_input(
        "주식 심볼",
        value="AAPL",
        help="예: AAPL (애플), 005930.KS (삼성전자), MSFT (마이크로소프트)"
    )
    
    # 날짜 범위 설정
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "시작 날짜",
            value=datetime.now() - timedelta(days=365*2),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "종료 날짜",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # 모델 선택
    model_type = st.sidebar.selectbox(
        "모델 타입",
        ["기본 모델 (Random Forest)", "고급 모델 (다중 알고리즘)"],
        help="고급 모델은 XGBoost, LightGBM, LSTM 등을 포함합니다."
    )
    
    # 분석 시작 버튼
    if st.sidebar.button("🚀 분석 시작", type="primary"):
        if symbol:
            with st.spinner("데이터를 분석하는 중..."):
                run_analysis(symbol, start_date, end_date, model_type)
        else:
            st.error("주식 심볼을 입력해주세요.")

def run_analysis(symbol, start_date, end_date, model_type):
    """주식 분석 실행"""
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["📊 데이터 개요", "🔮 예측 모델", "📈 차트 분석", "📋 상세 정보"])
    
    with tab1:
        show_data_overview(symbol, start_date, end_date)
    
    with tab2:
        if "고급 모델" in model_type and ADVANCED_AVAILABLE:
            show_advanced_prediction(symbol, start_date, end_date)
        else:
            show_basic_prediction(symbol, start_date, end_date)
    
    with tab3:
        show_chart_analysis(symbol, start_date, end_date)
    
    with tab4:
        show_detailed_info(symbol, start_date, end_date)

def show_data_overview(symbol, start_date, end_date):
    """데이터 개요 표시"""
    st.header("📊 데이터 개요")
    
    try:
        # 데이터 가져오기
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            st.error(f"{symbol}에 대한 데이터를 찾을 수 없습니다.")
            return
        
        # 기본 정보 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "현재 가격",
                f"${data['Close'].iloc[-1]:.2f}",
                f"{data['Close'].pct_change().iloc[-1]*100:.2f}%"
            )
        
        with col2:
            st.metric(
                "최고가",
                f"${data['High'].max():.2f}",
                f"{(data['High'].max() - data['Close'].iloc[-1])/data['Close'].iloc[-1]*100:.2f}%"
            )
        
        with col3:
            st.metric(
                "최저가",
                f"${data['Low'].min():.2f}",
                f"{(data['Low'].min() - data['Close'].iloc[-1])/data['Close'].iloc[-1]*100:.2f}%"
            )
        
        with col4:
            st.metric(
                "거래량",
                f"{data['Volume'].iloc[-1]:,.0f}",
                f"{data['Volume'].pct_change().iloc[-1]*100:.2f}%"
            )
        
        # 가격 차트
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="OHLC"
        ))
        
        fig.update_layout(
            title=f"{symbol} 주가 차트",
            xaxis_title="날짜",
            yaxis_title="가격 ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 통계 정보
        st.subheader("📈 통계 정보")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**수익률 통계**")
            returns = data['Close'].pct_change().dropna()
            stats_df = pd.DataFrame({
                '지표': ['평균 수익률', '수익률 표준편차', '최대 수익률', '최소 수익률', '샤프 비율'],
                '값': [
                    f"{returns.mean()*100:.2f}%",
                    f"{returns.std()*100:.2f}%",
                    f"{returns.max()*100:.2f}%",
                    f"{returns.min()*100:.2f}%",
                    f"{returns.mean()/returns.std():.2f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.write("**거래량 통계**")
            volume_stats = pd.DataFrame({
                '지표': ['평균 거래량', '최대 거래량', '최소 거래량'],
                '값': [
                    f"{data['Volume'].mean():,.0f}",
                    f"{data['Volume'].max():,.0f}",
                    f"{data['Volume'].min():,.0f}"
                ]
            })
            st.dataframe(volume_stats, use_container_width=True)
    
    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")

def show_basic_prediction(symbol, start_date, end_date):
    """기본 예측 모델 실행"""
    st.header("🔮 기본 예측 모델 (Random Forest)")
    
    try:
        # 모델 생성 및 학습
        predictor = StockPredictor(symbol, start_date, end_date)
        
        # 데이터 가져오기
        data = predictor.fetch_data()
        if data is None:
            st.error("데이터를 가져올 수 없습니다.")
            return
        
        # 특성 생성
        predictor.create_features()
        
        # 모델 학습
        results = predictor.train_model()
        
        if results:
            # 성능 지표 표시
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("MSE", f"{results['mse']:.2f}")
            
            with col2:
                st.metric("MAE", f"{results['mae']:.2f}")
            
            with col3:
                st.metric("R² Score", f"{results['r2']:.4f}")
            
            # 예측 결과 시각화
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=results['y_test'].index,
                y=results['y_test'].values,
                mode='lines',
                name='실제 가격',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=results['y_test'].index,
                y=results['y_pred'],
                mode='lines',
                name='예측 가격',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="실제 vs 예측 주가",
                xaxis_title="날짜",
                yaxis_title="가격 ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 다음 날 예측
            prediction = predictor.predict_next_day()
            if prediction:
                st.subheader("🎯 다음 날 예측")
                current_price = data['Close'].iloc[-1]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "현재 가격",
                        f"${current_price:.2f}"
                    )
                
                with col2:
                    st.metric(
                        "예측 가격",
                        f"${prediction:.2f}",
                        f"{(prediction - current_price):.2f}"
                    )
    
    except Exception as e:
        st.error(f"예측 모델 실행 중 오류가 발생했습니다: {e}")

def show_advanced_prediction(symbol, start_date, end_date):
    """고급 예측 모델 실행"""
    st.header("🔮 고급 예측 모델 (다중 알고리즘)")
    
    try:
        # 모델 생성
        predictor = AdvancedStockPredictor(symbol, start_date, end_date)
        
        # 데이터 가져오기
        data = predictor.fetch_data()
        if data is None:
            st.error("데이터를 가져올 수 없습니다.")
            return
        
        # 특성 생성
        predictor.create_advanced_features()
        
        # 모델 선택
        model_options = st.multiselect(
            "학습할 모델 선택",
            ["Random Forest", "XGBoost", "LightGBM", "LSTM"],
            default=["Random Forest", "XGBoost"]
        )
        
        if st.button("모델 학습 시작"):
            with st.spinner("모델을 학습하는 중..."):
                results = {}
                
                # 선택된 모델들 학습
                if "Random Forest" in model_options:
                    rf_result = predictor.train_random_forest()
                    if rf_result:
                        results["Random Forest"] = rf_result
                
                if "XGBoost" in model_options:
                    xgb_result = predictor.train_xgboost()
                    if xgb_result:
                        results["XGBoost"] = xgb_result
                
                if "LightGBM" in model_options:
                    lgb_result = predictor.train_lightgbm()
                    if lgb_result:
                        results["LightGBM"] = lgb_result
                
                if "LSTM" in model_options:
                    lstm_result = predictor.train_lstm()
                    if lstm_result:
                        results["LSTM"] = lstm_result
                
                # 결과 표시
                if results:
                    # 성능 비교
                    st.subheader("📊 모델 성능 비교")
                    
                    performance_data = []
                    for model_name, result in results.items():
                        performance_data.append({
                            '모델': model_name,
                            'MSE': result['mse'],
                            'MAE': result['mae'],
                            'R²': result['r2']
                        })
                    
                    perf_df = pd.DataFrame(performance_data)
                    st.dataframe(perf_df, use_container_width=True)
                    
                    # 성능 차트
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=('MSE (낮을수록 좋음)', 'MAE (낮을수록 좋음)', 'R² (높을수록 좋음)')
                    )
                    
                    fig.add_trace(
                        go.Bar(x=perf_df['모델'], y=perf_df['MSE'], name='MSE'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(x=perf_df['모델'], y=perf_df['MAE'], name='MAE'),
                        row=1, col=2
                    )
                    
                    fig.add_trace(
                        go.Bar(x=perf_df['모델'], y=perf_df['R²'], name='R²'),
                        row=1, col=3
                    )
                    
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 최고 성능 모델 찾기
                    best_model = max(results.keys(), key=lambda x: results[x]['r2'])
                    st.success(f"🏆 최고 성능 모델: {best_model} (R²: {results[best_model]['r2']:.4f})")
    
    except Exception as e:
        st.error(f"고급 예측 모델 실행 중 오류가 발생했습니다: {e}")

def show_chart_analysis(symbol, start_date, end_date):
    """차트 분석 표시"""
    st.header("📈 기술적 분석 차트")
    
    try:
        # 데이터 가져오기
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            st.error(f"{symbol}에 대한 데이터를 찾을 수 없습니다.")
            return
        
        # 기술적 지표 계산
        # RSI
        data['RSI'] = ta.momentum.rsi(data['Close'])
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()
        
        # 볼린저 밴드
        bb = ta.volatility.BollingerBands(data['Close'])
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        data['BB_Middle'] = bb.bollinger_mavg()
        
        # 이동평균
        data['MA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['MA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        
        # 차트 생성
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('주가 & 볼린저 밴드', 'RSI', 'MACD', '거래량'),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # 주가 & 볼린저 밴드
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="OHLC"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='orange')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_50'],
            mode='lines',
            name='MA 50',
            line=dict(color='red')
        ), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD_Signal'],
            mode='lines',
            name='MACD Signal',
            line=dict(color='red')
        ), row=3, col=1)
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['MACD_Histogram'],
            name='MACD Histogram'
        ), row=3, col=1)
        
        # 거래량
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume'
        ), row=4, col=1)
        
        fig.update_layout(
            title=f"{symbol} 기술적 분석",
            height=800,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 기술적 지표 해석
        st.subheader("📋 기술적 지표 해석")
        
        current_rsi = data['RSI'].iloc[-1]
        current_macd = data['MACD'].iloc[-1]
        current_macd_signal = data['MACD_Signal'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        current_bb_upper = data['BB_Upper'].iloc[-1]
        current_bb_lower = data['BB_Lower'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**RSI 분석**")
            if current_rsi > 70:
                st.warning("과매수 구간 (RSI > 70)")
            elif current_rsi < 30:
                st.success("과매도 구간 (RSI < 30)")
            else:
                st.info("중립 구간")
            st.write(f"현재 RSI: {current_rsi:.2f}")
        
        with col2:
            st.write("**MACD 분석**")
            if current_macd > current_macd_signal:
                st.success("상승 신호")
            else:
                st.warning("하락 신호")
            st.write(f"MACD: {current_macd:.4f}")
            st.write(f"Signal: {current_macd_signal:.4f}")
        
        with col3:
            st.write("**볼린저 밴드 분석**")
            if current_price > current_bb_upper:
                st.warning("상단 밴드 돌파 (과매수)")
            elif current_price < current_bb_lower:
                st.success("하단 밴드 돌파 (과매도)")
            else:
                st.info("밴드 내 정상 범위")
    
    except Exception as e:
        st.error(f"차트 분석 중 오류가 발생했습니다: {e}")

def show_detailed_info(symbol, start_date, end_date):
    """상세 정보 표시"""
    st.header("📋 상세 정보")
    
    try:
        # 데이터 가져오기
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            st.error(f"{symbol}에 대한 데이터를 찾을 수 없습니다.")
            return
        
        # 회사 정보
        st.subheader("🏢 회사 정보")
        
        try:
            info = ticker.info
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**회사명:** {info.get('longName', 'N/A')}")
                st.write(f"**섹터:** {info.get('sector', 'N/A')}")
                st.write(f"**산업:** {info.get('industry', 'N/A')}")
                st.write(f"**국가:** {info.get('country', 'N/A')}")
            
            with col2:
                st.write(f"**시가총액:** ${info.get('marketCap', 0):,.0f}")
                st.write(f"**52주 최고가:** ${info.get('fiftyTwoWeekHigh', 0):.2f}")
                st.write(f"**52주 최저가:** ${info.get('fiftyTwoWeekLow', 0):.2f}")
                st.write(f"**베타:** {info.get('beta', 0):.2f}")
        
        except:
            st.warning("회사 정보를 가져올 수 없습니다.")
        
        # 수익률 분석
        st.subheader("📊 수익률 분석")
        
        # 일별 수익률
        data['Returns'] = data['Close'].pct_change()
        data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
        
        # 월별 수익률
        monthly_returns = data['Close'].resample('M').last().pct_change()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**일별 수익률 통계**")
            returns_stats = data['Returns'].describe()
            st.dataframe(returns_stats, use_container_width=True)
        
        with col2:
            st.write("**월별 수익률**")
            monthly_df = pd.DataFrame({
                '월': monthly_returns.index.strftime('%Y-%m'),
                '수익률': monthly_returns.values * 100
            }).dropna()
            st.dataframe(monthly_df, use_container_width=True)
        
        # 변동성 분석
        st.subheader("📈 변동성 분석")
        
        # 20일 이동 표준편차
        data['Volatility_20'] = data['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volatility_20'],
            mode='lines',
            name='20일 변동성',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="연간 변동성 (20일 이동)",
            xaxis_title="날짜",
            yaxis_title="변동성 (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 최근 데이터 테이블
        st.subheader("📅 최근 거래 데이터")
        recent_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']]
        st.dataframe(recent_data, use_container_width=True)
    
    except Exception as e:
        st.error(f"상세 정보를 가져오는 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
