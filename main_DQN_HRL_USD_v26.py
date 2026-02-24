import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ==========================================
# 1. 하이퍼 파라미터 최적화 (Trading PnL 극대화)
# ==========================================
BASE_LIMIT = 300000  # 리스크 한도 확대
TIMEOUT_BASE = 360  # 대기 시간 확대 (6시간) - 시장 반등 기회 확보
S_I_MIN = 0.05  # 최소 익절 마진 (이것이 우리의 주 수익원)

SKEW_INTENSITY = 2.5  # 스큐 강화 (인벤토리 쏠림 시 평단가 방어)
PASSIVE_MARGIN = 0.01  # 진입 시 미세한 유리한 가격 설정

RATES_PATH = 'C:/Users/leeli/Downloads/finnode/data/환율(KST).xlsx'
TRADES_PATH = 'C:/Users/leeli/Downloads/finnode/data/거래데이터(KST).csv'

timestamp_dir = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f"./v6_2_2_final_fix_{timestamp_dir}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_bank_status(ts: pd.Timestamp):
    hour = ts.hour
    if 2 <= hour < 9: return None, True
    if 9 <= hour < 16: return 0.30, False
    return 0.60, False


def clean_columns(df):
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
    return df


# 데이터 로드
df_rates = clean_columns(pd.read_excel(RATES_PATH)).sort_values('시간(KST)')
df_trades = clean_columns(pd.read_csv(TRADES_PATH)).sort_values('체결시간')
df_rates['시간(KST)'] = pd.to_datetime(df_rates['시간(KST)'])
df_trades['체결시간'] = pd.to_datetime(df_trades['체결시간'])
df_trades = df_trades[df_trades['통화'] == 'USD'].copy()

returns = df_rates['USD'].pct_change()
df_rates['vol'] = returns.rolling(120).std().bfill().fillna(0.0001)
df_rates_indexed = df_rates.set_index('시간(KST)')

# ==========================================
# 2. 시뮬레이션 엔진 (요구사항 반영 버전)
# ==========================================
results, states_history = [], []
pending_buys, pending_sells = [], []
inventory, trading_pnl, cost_saving = 0.0, 0.0, 0.0

trade_times = df_trades['체결시간'].unique()
trade_groups = {t: rows for t, rows in df_trades.groupby('체결시간')}

print(f"🚀 [V6.2.2] 엔진 가동 - 넷팅의 역설 해결 및 수익 최적화")
log_fmt = "{:<20} | {:>10} | {:>12} | {:>10} | {:>8}"
print("-" * 75)
print(log_fmt.format("Timestamp", "Inventory", "Trading PnL", "Limit", "Skew"))
print("-" * 75)

for i, t in enumerate(trade_times):
    curr_t = pd.Timestamp(t)
    group = trade_groups[t]

    try:
        idx = df_rates_indexed.index.get_indexer([curr_t], method='backward')[0]
        c_rate = float(df_rates_indexed.iloc[idx]['USD'])
        c_vol = float(df_rates_indexed.iloc[idx]['vol'])
    except:
        continue

    bank_s, is_closed = get_bank_status(curr_t)
    bank_s_eff = 0.0 if is_closed else float(bank_s)

    dyn_limit = BASE_LIMIT * np.clip(0.0001 / (c_vol + 1e-9), 0.5, 2.0)
    curr_skew = -(inventory / dyn_limit) * SKEW_INTENSITY
    dyn_s_i = max(S_I_MIN, S_I_MIN * (1 + c_vol * 5000))

    # [Core] 고객 주문 처리: 새로운 흐름(Flow)이 들어오면 '청산' 도구로 먼저 활용
    for _, row in group.iterrows():
        qty, action = float(row['수량']), row['주문유형']

        if action == '매수':
            # 1. 청산 단계의 넷팅 (우리가 숏 포지션일 때 고객 매수가 들어오면 상계 청산)
            while qty > 0 and pending_sells:
                target = pending_sells[0]
                matched = min(qty, target['Qty'])
                # 유동성 매칭 수익 (은행 수수료 없이 직접 청산)
                pnl = (target['Entry_Rate'] - (c_rate + curr_skew)) * matched
                trading_pnl += pnl
                cost_saving += matched * (bank_s_eff * 2)  # 은행을 안 써서 아낀 비용
                results.append({'Side': '매도', 'Entry_Rate': target['Entry_Rate'], 'Exit_Time': curr_t,
                                'Exit_Rate': c_rate, 'PnL': pnl, 'Method': 'Flow_Matching', 'Qty': matched})
                inventory += matched;
                qty -= matched;
                target['Qty'] -= matched
                if target['Qty'] <= 0: pending_sells.pop(0)

            # 2. 청산하고 남은 수량은 신규 포지션으로 진입 (넷팅 없이)
            if qty > 0:
                entry_p = c_rate - PASSIVE_MARGIN + curr_skew
                pending_buys.append({'Side': '매수', 'Entry_Rate': entry_p, 'Entry_Time': curr_t, 'Qty': qty})
                inventory += qty

        else:  # 매도 주문
            while qty > 0 and pending_buys:
                target = pending_buys[0]
                matched = min(qty, target['Qty'])
                pnl = ((c_rate + curr_skew) - target['Entry_Rate']) * matched
                trading_pnl += pnl
                cost_saving += matched * (bank_s_eff * 2)
                results.append({'Side': '매수', 'Entry_Rate': target['Entry_Rate'], 'Exit_Time': curr_t,
                                'Exit_Rate': c_rate, 'PnL': pnl, 'Method': 'Flow_Matching', 'Qty': matched})
                inventory -= matched;
                qty -= matched;
                target['Qty'] -= matched
                if target['Qty'] <= 0: pending_buys.pop(0)

            if qty > 0:
                entry_p = c_rate + PASSIVE_MARGIN + curr_skew
                pending_sells.append({'Side': '매도', 'Entry_Rate': entry_p, 'Entry_Time': curr_t, 'Qty': qty})
                inventory -= qty

    # [Step D] 은행 청산 로직 (익절/손절/타임아웃)
    if not is_closed:
        for p_list, side in [(pending_buys, '매수'), (pending_sells, '매도')]:
            active = []
            for o in p_list:
                if side == '매수':
                    pnl_unit = (c_rate - bank_s_eff) - o['Entry_Rate']
                else:
                    pnl_unit = o['Entry_Rate'] - (c_rate + bank_s_eff)

                duration = (curr_t - o['Entry_Time']).total_seconds() / 60
                method = None
                if pnl_unit >= dyn_s_i:
                    method = "Bank_Alpha"
                elif duration >= TIMEOUT_BASE:
                    method = "Bank_Time"
                elif abs(inventory) > dyn_limit:
                    method = "Bank_Risk"

                if method:
                    p_total = pnl_unit * o['Qty']
                    trading_pnl += p_total
                    results.append({**o, 'Exit_Time': curr_t, 'Exit_Rate': c_rate, 'PnL': p_total, 'Method': method})
                    inventory -= (o['Qty'] if side == '매수' else -o['Qty'])
                else:
                    active.append(o)
            if side == '매수':
                pending_buys = active
            else:
                pending_sells = active

    if i % 1000 == 0 or i == len(trade_times) - 1:
        states_history.append({'Timestamp': curr_t, 'Inventory': inventory, 'TradingPnL': trading_pnl})
        print(log_fmt.format(str(curr_t)[:19], f"{inventory:,.0f}", f"{trading_pnl:,.0f}", f"{dyn_limit:,.0f}",
                             f"{curr_skew:.2f}"))

# ==========================================
# 3. 결과 분석 (KeyError 방지)
# ==========================================
df_res = pd.DataFrame(results) if results else pd.DataFrame(columns=['Method', 'PnL'])

summary = {
    'Real Trading PnL(본질수익)': trading_pnl,
    'Netting Cost Saving(비용절감)': cost_saving,
    'Total Combined': trading_pnl + cost_saving,
    'Alpha Trades': len(df_res[df_res['Method'] == 'Bank_Alpha']) if not df_res.empty else 0
}

print("\n" + "=" * 50)
for k, v in summary.items(): print(f"{k}: {v:,.0f} KRW")
print("=" * 50)

# 시각화: Trading PnL 기준 우상향 확인
if states_history:
    plt.figure(figsize=(12, 6))
    plt.plot([s['Timestamp'] for s in states_history], [s['TradingPnL'] for s in states_history], color='blue')
    plt.title(f"V6.2.2 Equity Curve (Trading PnL Only)")
    plt.grid(True, alpha=0.3);
    plt.savefig(f"{OUTPUT_DIR}/trading_pnl_curve.png")