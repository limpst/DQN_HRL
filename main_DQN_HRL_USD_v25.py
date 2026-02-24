import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ==========================================
# 1. 초정밀 설정 (Hyper-Parameters)
# ==========================================
BASE_LIMIT = 200000
TIMEOUT_BASE = 180
S_i = 0.15  # Tier 1 익절 기준 (이 마진을 확보하는 것이 핵심)

SKEW_INTENSITY = 1.5
PASSIVE_MARGIN = 0.00
PLATFORM_FEE = 0.0

# 데이터 경로 (사용자 환경에 맞게 유지)
RATES_PATH = 'C:/Users/leeli/Downloads/finnode/data/환율(KST).xlsx'
TRADES_PATH = 'C:/Users/leeli/Downloads/finnode/data/거래데이터(KST).csv'

timestamp_dir = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f"./v6_1_netting_fix_{timestamp_dir}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_bank_status(ts: pd.Timestamp):
    hour = ts.hour
    if 2 <= hour < 9: return None, True
    if 9 <= hour < 16: return 0.30, False
    return 0.60, False


def clean_columns(df):
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
    return df


# 데이터 로드 및 전처리
df_rates = clean_columns(pd.read_excel(RATES_PATH)).sort_values('시간(KST)')
df_trades = clean_columns(pd.read_csv(TRADES_PATH)).sort_values('체결시간')
df_rates['시간(KST)'] = pd.to_datetime(df_rates['시간(KST)'])
df_trades['체결시간'] = pd.to_datetime(df_trades['체결시간'])
df_trades = df_trades[df_trades['통화'] == 'USD'].copy()

# 지표 계산
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
# 2. 시뮬레이션 메인 루프 (진입 시 넷팅 제거 버전)
# ==========================================
results, states_history, pending_lots = [], [], []
inventory, netting_profit, trading_pnl = 0.0, 0.0, 0.0

log_fmt = "{:<20} | {:^7} | {:>10} | {:>12} | {:>12} | {:>8} | {:>6}"
print(f"🚀 [V6.1 Fix] 넷팅의 역설 방어 모드 가동")
print("-" * 110)
print(log_fmt.format("Timestamp", "Status", "Inventory", "Netting PnL", "Trade PnL", "Limit", "Skew"))
print("-" * 110)

for i, t in enumerate(trade_times):
    curr_t = pd.Timestamp(t)
    group = trade_groups[t]
    c_rate = float(group.iloc[-1]['USD'])
    c_vol = float(group.iloc[-1].get('vol', 0.0001))
    c_dur = float(group.iloc[-1].get('expected_dur', 60.0))
    bank_s, is_closed = get_bank_status(curr_t)
    bank_s_eff = 0.0 if is_closed else float(bank_s)

    # Dynamic Parameters
    vol_adj = np.clip(0.0001 / (c_vol + 1e-9), 0.7, 1.8)
    dyn_limit = BASE_LIMIT * vol_adj
    dyn_timeout = TIMEOUT_BASE * vol_adj * np.clip(300.0 / (c_dur + 1e-9), 0.8, 1.5)
    curr_skew = -(inventory / dyn_limit) * SKEW_INTENSITY

    # [Step B] Entry: 넷팅 없이 모든 주문을 개별 Lot으로 생성
    for _, row in group.iterrows():
        qty, action = float(row['수량']), row['주문유형']
        entry_p = (c_rate - PASSIVE_MARGIN + curr_skew) if action == '매수' else (c_rate + PASSIVE_MARGIN + curr_skew)

        inventory += (qty if action == '매수' else -qty)
        pending_lots.append({
            'Side': action,
            'Entry_Rate': entry_p,
            'Entry_Time': curr_t,
            'Qty': qty,
            'Status': 'ACTIVE'
        })

    # [Step C] Liquidation: 청산 단계에서만 조건부 넷팅 및 청산 실행
    if not is_closed and pending_lots:
        active = []
        # 리스크 한도 초과 여부 확인
        over_limit = abs(inventory) > dyn_limit

        # 방향별로 로트 분리 (넷팅 청산용)
        buys = [o for o in pending_lots if o['Side'] == '매수']
        sells = [o for o in pending_lots if o['Side'] == '매도']

        for o in pending_lots:
            pnl_unit = (c_rate - bank_s_eff - o['Entry_Rate']) if o['Side'] == '매수' else (
                        o['Entry_Rate'] - (c_rate + bank_s_eff))
            duration = (curr_t - o['Entry_Time']).total_seconds() / 60

            method = ""
            # 1. Tier 1: 목표 마진 도달 (수익의 핵심)
            if pnl_unit >= S_i:
                method = "Tier1_Alpha"
            # 2. Tier 2: 시간 경과
            elif duration >= dyn_timeout:
                method = "Tier2_Time"
            # 3. Tier 3: 리스크 한도 초과 시 반대 방향 로트와 넷팅 청산
            elif over_limit:
                # 현재 로트와 반대 방향의 로트가 있다면 넷팅 처리
                if (o['Side'] == '매수' and inventory > 0 and sells) or (o['Side'] == '매도' and inventory < 0 and buys):
                    method = "Tier3_Netting_Exit"
                    # 넷팅으로 절감한 비용(은행 스프레드 2배)을 별도 기록
                    netting_profit += o['Qty'] * (bank_s_eff * 2)
                else:
                    method = "Tier3_Risk_Market"

            if method:
                p_total = pnl_unit * o['Qty'] - PLATFORM_FEE
                trading_pnl += p_total
                o.update({'Exit_Time': curr_t, 'Exit_Rate': c_rate, 'PnL': p_total, 'Method': method,
                          'Status_At_Exit': 'OPEN'})
                results.append(o)
                inventory -= (o['Qty'] if o['Side'] == '매수' else -o['Qty'])
            else:
                active.append(o)
        pending_lots = active

    # Audit Trail 기록
    states_history.append({
        'Timestamp': curr_t, 'Status': 'CLOSED' if is_closed else 'OPEN', 'Inventory': inventory,
        'Netting_PnL': netting_profit, 'Trade_PnL': trading_pnl, 'Limit': dyn_limit,
        'Skew': curr_skew, 'USD_Rate': c_rate
    })

    if i % 1000 == 0 or i == len(trade_times) - 1:
        print(log_fmt.format(str(curr_t)[:19], 'CLOSED' if is_closed else 'OPEN', f"{inventory:,.0f}",
                             f"{netting_profit:,.0f}", f"{trading_pnl:,.0f}", f"{dyn_limit:,.0f}", f"{curr_skew:.2f}"))

# ==========================================
# 3. 결과 분석 및 저장
# ==========================================
df_trades_out = pd.DataFrame(results)
df_states_out = pd.DataFrame(states_history)

# 순수 트레이딩 수익 중심의 요약
summary_dict = {
    'Metric': ['Trading PnL (Real)', 'Netting PnL (Cost Saving)', 'Total Combined', 'Tier 1 Success Rate',
               'Max Inventory', 'Total Trades'],
    'Value': [
        f"{trading_pnl:,.0f}", f"{netting_profit:,.0f}", f"{(trading_pnl + netting_profit):,.0f}",
        f"{(df_trades_out['Method'] == 'Tier1_Alpha').mean() * 100:.2f}%" if not df_trades_out.empty else "0%",
        f"{df_states_out['Inventory'].abs().max():,.0f}", len(df_trades_out)
    ]
}
pd.DataFrame(summary_dict).to_csv(f"{OUTPUT_DIR}/final_summary_v6_1.csv", index=False, encoding='utf-8-sig')

print("-" * 110)
print(f"✅ 분석 완료. 진입 넷팅을 제거하여 Tier 1 기회를 최대화했습니다.")
print(f"💰 순수 트레이딩 수익: {trading_pnl:,.0f} KRW")
print(f"🛡️ 넷팅 절감 비용: {netting_profit:,.0f} KRW")