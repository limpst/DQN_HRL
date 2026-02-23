import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ==========================================
# 1. 초정밀 설정 (Hyper-Parameters)
# ==========================================
BASE_LIMIT = 200000  # 기본 리스크 한도 (USD)
TIMEOUT_BASE = 180  # 기본 대기 시간 (분)
S_i = 0.15  # Tier 1 익절 기준

SKEW_INTENSITY = 1.5
PASSIVE_MARGIN = 0.00
PLATFORM_FEE = 0.0

RATES_PATH = 'C:/Users/leeli/Downloads/finnode/data/환율(KST).xlsx'
TRADES_PATH = 'C:/Users/leeli/Downloads/finnode/data/거래데이터(KST).csv'

# [저장 경로 설정] 실행 시점별로 고유 폴더 생성
timestamp_dir = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f"./v6_usd_audit_{timestamp_dir}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. 유틸리티 및 지표 엔진
# ==========================================
def get_bank_status(ts: pd.Timestamp):
    hour = ts.hour
    if 2 <= hour < 9: return None, True
    if 9 <= hour < 16: return 0.30, False
    return 0.60, False


def clean_columns(df):
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
    return df


print(f"🚀 [V6 Hybrid] 엔진 가동 - Audit Mode 활성화 (저장소: {OUTPUT_DIR})")

# 데이터 로드
df_rates = clean_columns(pd.read_excel(RATES_PATH)).sort_values('시간(KST)')
df_trades = clean_columns(pd.read_csv(TRADES_PATH)).sort_values('체결시간')
df_rates['시간(KST)'] = pd.to_datetime(df_rates['시간(KST)'])
df_trades['체결시간'] = pd.to_datetime(df_trades['체결시간'])
df_trades = df_trades[df_trades['통화'] == 'USD'].copy()

# [Quant] 변동성(HARCH) 및 기대 듀레이션(ACD)
returns = df_rates['USD'].pct_change()
df_rates['vol'] = (0.5 * returns.rolling(30).std() + 0.3 * returns.rolling(120).std() + 0.2 * returns.rolling(
    480).std()).bfill().fillna(0.0001)

durations = df_trades['체결시간'].diff().dt.total_seconds().fillna(60).clip(lower=1)
omega, alpha, beta = 0.1, 0.15, 0.75
psi = np.zeros(len(durations));
psi[0] = durations.mean()
for k in range(1, len(durations)): psi[k] = omega + alpha * durations.iloc[k - 1] + beta * psi[k - 1]
df_trades['expected_dur'] = psi

df_trades = pd.merge_asof(df_trades.sort_values('체결시간'), df_rates[['시간(KST)', 'USD', 'vol']],
                          left_on='체결시간', right_on='시간(KST)', direction='backward')

trade_times = df_trades['체결시간'].unique()
trade_groups = {t: rows for t, rows in df_trades.groupby('체결시간')}

# ==========================================
# 3. 시뮬레이션 메인 루프 (Log & Audit 기능 탑재)
# ==========================================
results, states_history, pending_lots = [], [], []
inventory, netting_profit, trading_pnl = 0.0, 0.0, 0.0

# 실시간 로그 헤더
log_fmt = "{:<20} | {:^7} | {:>10} | {:>12} | {:>12} | {:>8} | {:>6}"
print("-" * 110)
print(log_fmt.format("Timestamp", "Status", "Inventory", "Netting PnL", "Trade PnL", "Limit", "Skew"))
print("-" * 110)

for i, t in enumerate(trade_times):
    curr_t = pd.Timestamp(t)
    group = trade_groups[t]
    c_rate, c_vol, c_dur = float(group.iloc[-1]['USD']), float(group.iloc[-1].get('vol', 0.0001)), float(
        group.iloc[-1].get('expected_dur', 60.0))
    bank_s, is_closed = get_bank_status(curr_t)
    bank_s_eff = 0.0 if is_closed else float(bank_s)

    # Dynamic Parameters
    vol_adj = np.clip(0.0001 / (c_vol + 1e-9), 0.7, 1.8)
    dur_adj = np.clip(300.0 / (c_dur + 1e-9), 0.8, 1.5)
    dyn_limit = BASE_LIMIT * vol_adj
    dyn_timeout = TIMEOUT_BASE * vol_adj * dur_adj
    curr_skew = -(inventory / dyn_limit) * SKEW_INTENSITY

    # Step B: Entry
    for _, row in group.iterrows():
        qty, action = float(row['수량']), row['주문유형']
        entry_p = (c_rate - PASSIVE_MARGIN + curr_skew) if action == '매수' else (c_rate + PASSIVE_MARGIN + curr_skew)

        # 넷팅 처리
        mm_delta = qty if action == '매수' else -qty
        if inventory != 0 and (inventory * mm_delta) < 0:
            matched = min(abs(inventory), abs(qty))
            netting_profit += matched * (bank_s_eff * 2)
            inventory += (matched if inventory < 0 else -matched)
            qty -= matched
            if qty <= 0: continue

        inventory += (qty if action == '매수' else -qty)
        pending_lots.append({'Side': action, 'Entry_Rate': entry_p, 'Entry_Time': curr_t, 'Qty': qty})

    # Step C: Liquidation
    if not is_closed and pending_lots:
        active = []
        for o in pending_lots:
            pnl_unit = (c_rate - bank_s_eff - o['Entry_Rate']) if o['Side'] == '매수' else (
                        o['Entry_Rate'] - (c_rate + bank_s_eff))
            duration = (curr_t - o['Entry_Time']).total_seconds() / 60

            method = ""
            if pnl_unit >= S_i:
                method = "Tier1_Alpha"
            elif duration >= dyn_timeout:
                method = "Tier2_Time"
            elif abs(inventory) > dyn_limit:
                method = "Tier3_Risk"

            if method:
                p_total = pnl_unit * o['Qty'] - PLATFORM_FEE
                trading_pnl += p_total
                o.update({'Exit_Time': curr_t, 'Exit_Rate': c_rate, 'PnL': p_total, 'Method': method,
                          'Status_At_Exit': 'CLOSED' if is_closed else 'OPEN'})
                results.append(o)
                inventory -= (o['Qty'] if o['Side'] == '매수' else -o['Qty'])
            else:
                active.append(o)
        pending_lots = active

    # [Audit Trail] 매 스텝의 모든 엔진 변수 기록
    states_history.append({
        'Timestamp': curr_t, 'Status': 'CLOSED' if is_closed else 'OPEN', 'Inventory': inventory,
        'Netting_PnL': netting_profit, 'Trade_PnL': trading_pnl, 'Limit': dyn_limit,
        'Skew': curr_skew, 'USD_Rate': c_rate, 'Vol': c_vol, 'Expected_Dur': c_dur
    })

    # 실시간 로그 출력 (1000스텝당 한 번)
    if i % 1000 == 0 or i == len(trade_times) - 1:
        print(log_fmt.format(str(curr_t)[:19], 'CLOSED' if is_closed else 'OPEN', f"{inventory:,.0f}",
                             f"{netting_profit:,.0f}", f"{trading_pnl:,.0f}", f"{dyn_limit:,.0f}", f"{curr_skew:.2f}"))

# ==========================================
# 4. 데이터 저장 및 결과 분석
# ==========================================
print("-" * 110)
print(f"📂 시뮬레이션 종료. CSV 데이터 저장 중...")

# 1. 체결 데이터 저장
df_trades_out = pd.DataFrame(results)
df_trades_out.to_csv(f"{OUTPUT_DIR}/trade_details_USD.csv", index=False, encoding='utf-8-sig')

# 2. 엔진 상태 히스토리 저장 (매 시점의 변수들)
df_states_out = pd.DataFrame(states_history)
df_states_out.to_csv(f"{OUTPUT_DIR}/engine_audit_trail_USD.csv", index=False, encoding='utf-8-sig')

# 3. 요약 리포트 생성 및 저장
final_pnl = netting_profit + trading_pnl
summary_dict = {
    'Metric': ['Total PnL', 'Netting PnL', 'Trading PnL', 'Tier 1 Success Rate', 'Max Inventory', 'Total Trades'],
    'Value': [
        f"{final_pnl:,.0f}", f"{netting_profit:,.0f}", f"{trading_pnl:,.0f}",
        f"{(df_trades_out['Method'] == 'Tier1_Alpha').mean() * 100:.2f}%" if not df_trades_out.empty else "0%",
        f"{df_states_out['Inventory'].abs().max():,.0f}", len(df_trades_out)
    ]
}
pd.DataFrame(summary_dict).to_csv(f"{OUTPUT_DIR}/final_summary.csv", index=False, encoding='utf-8-sig')

print(f"✅ 저장 완료: {OUTPUT_DIR}")
print(f"💰 최종 수익: {final_pnl:,.0f} KRW")

# 시각화 (Equity Curve)
if not df_trades_out.empty:
    df_trades_out = df_trades_out.sort_values('Exit_Time')
    df_trades_out['CumPnL'] = df_trades_out['PnL'].cumsum() + (netting_profit / len(df_trades_out)) * np.arange(
        len(df_trades_out))
    plt.figure(figsize=(12, 6))
    plt.plot(df_trades_out['Exit_Time'], df_trades_out['CumPnL'], color='navy', label='V6 Hybrid')
    plt.title(f"V6 Hybrid Equity Curve (Final: {final_pnl:,.0f} KRW)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/equity_curve.png")
    plt.show()