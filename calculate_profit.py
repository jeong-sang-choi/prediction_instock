import yfinance as yf

# MSFT 현재가 확인
ticker = yf.Ticker('MSFT')
data = ticker.history(period='1d')
current_price = data['Close'].iloc[-1]

print(f"MSFT 현재가: ${current_price:.2f}")

# 1% 상승 시 수익 계산
shares = 1
price_increase = current_price * 0.01
total_profit = shares * price_increase

print(f"\n=== 수익 계산 ===")
print(f"구매 주식 수: {shares}주")
print(f"1% 상승 시 주가: ${current_price + price_increase:.2f}")
print(f"주당 수익: ${price_increase:.2f}")
print(f"총 수익: ${total_profit:.2f}")

# 원화로 환산 (환율 1,350원 기준)
exchange_rate = 1350
profit_krw = total_profit * exchange_rate

print(f"\n=== 원화 환산 ===")
print(f"환율: 1달러 = {exchange_rate:,}원")
print(f"총 수익: {profit_krw:,.0f}원")

# 거래 수수료 고려 (보통 $10-20)
print(f"\n=== 수수료 고려 ===")
print(f"거래 수수료: $10-20 (약 13,500-27,000원)")
print(f"수수료 제외 실질 수익: ${total_profit - 15:.2f} (약 {profit_krw - 20250:,.0f}원)")

# 투자 금액 대비 수익률
investment_amount = current_price * exchange_rate
roi = (profit_krw / investment_amount) * 100

print(f"\n=== 투자 대비 수익률 ===")
print(f"투자 금액: {investment_amount:,.0f}원")
print(f"수익률: {roi:.2f}%")
