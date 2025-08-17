import yfinance as yf

# MSFT 현재가 확인
ticker = yf.Ticker('MSFT')
data = ticker.history(period='1d')

if len(data) > 0:
    current_price = data['Close'].iloc[-1]
    print(f"MSFT 현재가: ${current_price:.2f}")
else:
    print("데이터를 가져올 수 없습니다.")
