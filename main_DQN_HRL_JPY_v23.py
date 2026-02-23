import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ==========================================
# 1. JPY 최적화 설정 (Hyper-Parameters for JPY)
# ==========================================
BASE_LIMIT = 10000000  # JPY 리스크 한도
TIMEOUT_BASE = 180  # 기본 대기 시간 (분)
S_i = 0.0015  # 익절 기준 (1엔당 0.0015 KRW)

SKEW_INTENSITY = 1.5
PASSIVE_MARGIN = 0.000  # 시장 진입 마진
PLATFORM_FEE = 0.0

RATES_PATH = 'C:/Users/leeli/Downloads/finnode/data/환율(KST).xlsx'
TRADES_PATH = 'C:/Users/leeli/Downloads/finnode/data/거래데이터(KST).csv'

# [저장 경로 설정]
timestamp_dir = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f"./v6_jpy_audit_{timestamp_dir}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. 유틸리티 및 지표 엔진
# ==========================================
def get_bank_status(ts: pd.Timestamp):
    hour = ts.hour
    if 2 <= hour < 9: return None, True  # 은행 폐장
    if 9 <= hour < 16: return 0.0030, False  # 주간 스프레드
    return 0.0060, False  # 야간 스프레드


def clean_columns(df):
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
    return df


# MM 액션 정규화 함수
def normalize_mm_action(raw_action):
    """원천 데이터를 MM 표준 액션(BUY/SELL)으로 변환"""
    mapping = {
        '매수': 'BUY', 'BUY': 'BUY', 'BID': 'BUY', '1': 'BUY',
        '매도': 'SELL', 'SELL': 'SELL', 'ASK': 'SELL', '2': 'SELL'
    }
    return mapping.get(str(raw_action).upper(), 'UNKNOWN')


print(f"🚀 [V6 JPY-Hybrid] 엔진 가동 - MM 기준 최적화 (저장소: {OUTPUT_DIR})")

# 데이터 로드 및 정규화
df_rates = clean_columns(pd.read_excel(RATES_PATH)).sort_values('시간(KST)')
df_trades = clean_columns(pd.read_csv(TRADES_PATH)).sort_values('체결시간')

df_rates['시간(KST)'] = pd.to_datetime(df_rates['시간(KST)'])
df_trades['체결시간'] = pd.to_datetime(df_trades['체결시간'])

df_trades = df_trades[df_trades['통화'] == 'JPY'].copy()
df_rates['JPY_norm'] = df_rates['JPY'] / 100.0
df_trades['가격_norm'] = df_trades['가격'] / 100.0

# [Quant] 변동성 및 기대 듀레이션(ACD) 계산
returns = df_rates['JPY_norm'].pct_change()
df_rates['vol'] = (0.5 * returns.rolling(30).std() + 0.3 * returns.rolling(120).std() +
                   0.2 * returns.rolling(480).std()).bfill().fillna(0.0001)

durations = df_trades['체결시간'].diff().dt.total_seconds().fillna(60).clip(lower=1)
omega, alpha, beta = 0.1, 0.15, 0.75
psi = np.zeros(len(durations))
psi[0] = durations.mean()
for k in range(1, len(durations)):
    psi[k] = omega + alpha * durations.iloc[k - 1] + beta * psi[k - 1]
df_trades['expected_dur'] = psi

df_trades = pd.merge_asof(df_trades.sort_values('체결시간'),
                          df_rates[['시간(KST)', 'JPY_norm', 'vol']],
                          left_on='체결시간', right_on='시간(KST)', direction='backward')

trade_times = df_trades['체결시간'].unique()
trade_groups = {t: rows for t, rows in df_trades.groupby('체결시간')}

# ==========================================
# 3. 시뮬레이션 메인 루프 (MM Logic 적용)
# ==========================================
results, states_history, pending_lots = [], [], []
inventory, netting_profit, trading_pnl = 0.0, 0.0, 0.0

log_fmt = "{:<20} | {:^7} | {:>12} | {:>12} | {:>12} | {:>10} | {:>6}"
print("-" * 120)
print(log_fmt.format("Timestamp", "Status", "Inventory(Y)", "Netting PnL", "Trade PnL", "Limit", "Skew"))
print("-" * 120)

for i, t in enumerate(trade_times):
    curr_t = pd.Timestamp(t)
    group = trade_groups[t]

    c_rate = float(group.iloc[-1]['JPY_norm'])
    c_vol = float(group.iloc[-1].get('vol', 0.0001))
    c_dur = float(group.iloc[-1].get('expected_dur', 60.0))

    bank_s, is_closed = get_bank_status(curr_t)
    bank_s_eff = 0.0 if is_closed else float(bank_s)

    # Dynamic Risk Parameters
    vol_adj = np.clip(0.0001 / (c_vol + 1e-9), 0.7, 1.8)
    dur_adj = np.clip(300.0 / (c_dur + 1e-9), 0.8, 1.5)
    dyn_limit = BASE_LIMIT * vol_adj
    dyn_timeout = TIMEOUT_BASE * vol_adj * dur_adj
    curr_skew = -(inventory / dyn_limit) * SKEW_INTENSITY

    # Step B: Entry (MM Action 기준)
    for _, row in group.iterrows():
        # --- 수정 및 MM 표준화 적용 부분 ---
        qty = float(row['수량'])
        action = normalize_mm_action(row['주문유형'])

        if action == 'UNKNOWN': continue

        # 진입 가격 결정 (MM Skew 반영)
        entry_p = (c_rate - PASSIVE_MARGIN + curr_skew) if action == 'BUY' else (c_rate + PASSIVE_MARGIN + curr_skew)

        # 넷팅 처리 (Inventory 상쇄 시 수익 확정)
        mm_delta = qty if action == 'BUY' else -qty
        if inventory != 0 and (inventory * mm_delta) < 0:
            matched = min(abs(inventory), abs(qty))
            netting_profit += matched * (bank_s_eff * 2)
            inventory += (matched if inventory < 0 else -matched)
            qty -= matched
            if qty <= 0: continue

        inventory += (qty if action == 'BUY' else -qty)
        pending_lots.append({'Side': action, 'Entry_Rate': entry_p, 'Entry_Time': curr_t, 'Qty': qty})

    # Step C: Liquidation (Exit)
    if not is_closed and pending_lots:
        active = []
        for o in pending_lots:
            pnl_unit = (c_rate - bank_s_eff - o['Entry_Rate']) if o['Side'] == 'BUY' else (
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
                o.update({'Exit_Time': curr_t, 'Exit_Rate': c_rate, 'PnL': p_total, 'Method': method})
                results.append(o)
                inventory -= (o['Qty'] if o['Side'] == 'BUY' else -o['Qty'])
            else:
                active.append(o)
        pending_lots = active

    # Audit Trail
    states_history.append({
        'Timestamp': curr_t, 'Status': 'CLOSED' if is_closed else 'OPEN', 'Inventory': inventory,
        'Netting_PnL': netting_profit, 'Trade_PnL': trading_pnl, 'Limit': dyn_limit,
        'Skew': curr_skew, 'JPY_Rate_1Yen': c_rate
    })

    if i % 1000 == 0 or i == len(trade_times) - 1:
        print(log_fmt.format(str(curr_t)[:19], 'CLOSED' if is_closed else 'OPEN', f"{inventory:,.0f}",
                             f"{netting_profit:,.0f}", f"{trading_pnl:,.0f}", f"{dyn_limit:,.0f}", f"{curr_skew:.4f}"))

# ==========================================
# 4. 결과 저장 및 분석
# ==========================================
df_trades_out = pd.DataFrame(results)
df_states_out = pd.DataFrame(states_history)

df_trades_out.to_csv(f"{OUTPUT_DIR}/jpy_trade_details.csv", index=False, encoding='utf-8-sig')
df_states_out.to_csv(f"{OUTPUT_DIR}/jpy_engine_audit.csv", index=False, encoding='utf-8-sig')

final_pnl = netting_profit + trading_pnl
print(f"\n✅ JPY 시뮬레이션 완료 | 최종 수익: {final_pnl:,.0f} KRW")

# Equity Curve 시각화
if not df_trades_out.empty:
    df_trades_out = df_trades_out.sort_values('Exit_Time')
    df_trades_out['CumPnL'] = df_trades_out['PnL'].cumsum()
    plt.figure(figsize=(12, 6))
    plt.plot(df_trades_out['Exit_Time'], df_trades_out['CumPnL'], color='firebrick')
    plt.title(f"JPY MM Equity Curve (Final: {final_pnl:,.0f} KRW)")
    plt.grid(True, alpha=0.3)
    plt.show()