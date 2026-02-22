import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from datetime import datetime
import os
import sys


# ==========================================
# 1. DQN 신경망 및 에이전트
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x): return self.fc(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.memory = deque(maxlen=5000)
        self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay = 0.95, 1.0, 0.01, 0.998
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_dim, action_dim).to(self.device)
        self.target_model = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.update_target()

        self.action_map = {
            0: (0.25, 60, 100000), 1: (0.25, 240, 150000), 2: (0.25, 600, 250000),
            3: (0.15, 60, 150000), 4: (0.15, 240, 200000), 5: (0.15, 600, 300000),
            6: (0.08, 60, 200000), 7: (0.08, 240, 300000), 8: (0.08, 600, 500000)
        }

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        if np.random.rand() <= self.epsilon: return random.randrange(self.action_dim)
        st = torch.FloatTensor(state).to(self.device)
        return torch.argmax(self.model(st)).item()

    def train(self, batch_size=64):
        if len(self.memory) < batch_size: return
        batch = random.sample(self.memory, batch_size)
        s, a, r, ns, d = zip(*batch)
        s, a, r, ns, d = torch.FloatTensor(np.array(s)).to(self.device), torch.LongTensor(a).to(self.device), \
            torch.FloatTensor(r).to(self.device), torch.FloatTensor(np.array(ns)).to(self.device), \
            torch.FloatTensor(d).to(self.device)
        curr_q = self.model(s).gather(1, a.unsqueeze(1))
        next_q = self.target_model(ns).max(1)[0].detach()
        target_q = r + (1 - d) * self.gamma * next_q
        loss = nn.MSELoss()(curr_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay


# ==========================================
# 2. 데이터 로드 및 전처리
# ==========================================
def get_state(vol, dur, inv, limit):
    return np.array([vol * 1000, dur / 600, inv / limit], dtype=np.float32)


def clean_columns(df):
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
    return df


rates_path = 'v:/PythonProject1/sideProject1/data/환율(KST).xlsx'
trades_path = 'v:/PythonProject1/sideProject1/data/거래데이터(KST).csv'

df_rates = clean_columns(pd.read_excel(rates_path)).sort_values('시간(KST)')
df_trades = clean_columns(pd.read_csv(trades_path)).sort_values('체결시간')
df_rates['시간(KST)'] = pd.to_datetime(df_rates['시간(KST)'])
df_trades['체결시간'] = pd.to_datetime(df_trades['체결시간'])
df_trades = df_trades[df_trades['통화'] == 'USD'].copy()

returns = df_rates['USD'].pct_change()
df_rates['vol'] = returns.rolling(120).std().bfill().fillna(0.0001)
durations = df_trades['체결시간'].diff().dt.total_seconds().fillna(60).clip(lower=1)
psi = np.zeros(len(durations));
psi[0] = durations.mean()
for k in range(1, len(durations)): psi[k] = 0.1 + 0.15 * durations.iloc[k - 1] + 0.75 * psi[k - 1]
df_trades['expected_dur'] = psi
df_trades = pd.merge_asof(df_trades.sort_values('체결시간'), df_rates[['시간(KST)', 'USD', 'vol']],
                          left_on='체결시간', right_on='시간(KST)', direction='backward')

trade_times = df_trades['체결시간'].sort_values().unique()
trade_groups = {t: rows for t, rows in df_trades.groupby('체결시간')}

# ==========================================
# 3. 시뮬레이션 메인 루프 (로그 강화 버전)
# ==========================================
agent = DQNAgent(state_dim=3, action_dim=9)
results, pending_lots = [], []
inventory, netting_profit, current_trade_pnl = 0.0, 0.0, 0.0
manager_interval = 100
current_action_idx = 4
total_steps = len(trade_times)

print(f"🚀 DQN-HRL 시뮬레이션 시작 (Total: {total_steps} events)")
print("-" * 100)

for i, t in enumerate(trade_times):
    current_time = pd.Timestamp(t)
    group = trade_groups[t]
    c_rate = float(group.iloc[-1]['USD'])
    c_vol = float(group.iloc[-1].get('vol', 0.0001))
    c_dur = float(group.iloc[-1].get('expected_dur', 60))

    # --- [Manager] DQN Decision ---
    S_i, T_w_base, BASE_LIMIT = agent.action_map[current_action_idx]
    state = get_state(c_vol, c_dur, inventory, BASE_LIMIT)

    if i % manager_interval == 0 and i > 0:
        # 보상 함수: $R = \Delta PnL - (Inventory \times Cost)$
        reward = (current_trade_pnl + netting_profit) - (abs(inventory) * 0.00005)
        agent.memory.append((prev_state, current_action_idx, reward, state, False))
        agent.train()
        if i % 1000 == 0: agent.update_target()
        current_action_idx = agent.select_action(state)
        prev_state = state

    elif i == 0:
        prev_state = state

    # --- 실시간 로그 출력 ---
    if i % 500 == 0 or i == total_steps - 1:
        total_realized = current_trade_pnl + netting_profit
        progress = (i / total_steps) * 100
        print(f"[{progress:5.1f}%] {current_time} | Action: {current_action_idx} | "
              f"Inv: {inventory:10,.0f} | Realized PnL: {total_realized:12,.0f}원")

    # --- [Worker] Trading Logic ---
    bank_s, is_closed = (0.3, False) if 9 <= current_time.hour < 16 else (0.6, False)
    if 2 <= current_time.hour < 9: is_closed = True

    for _, row in group.iterrows():
        qty = float(row['수량'])
        mm_side = '매도' if row['주문유형'] == '매수' else '매수'
        cu_qty = -qty if mm_side == '매도' else qty

        # 넷팅 수익 발생 여부 체크
        if inventory != 0 and (inventory * cu_qty) < 0:
            matched = min(abs(inventory), abs(cu_qty))
            netting_profit += matched * (0.0 if is_closed else bank_s) * 2

        inventory += cu_qty
        lot = row.to_dict()
        lot.update({
            'MM_Side': mm_side, 'Entry_Rate': float(row.get('체결단가', row.get('가격'))),
            'Entry_Time': current_time, 'S_i': S_i, 'T_w': T_w_base, 'Limit': BASE_LIMIT
        })
        pending_lots.append(lot)

    if is_closed or not pending_lots: continue

    active = []
    for o in pending_lots:
        expected = (o['Entry_Rate'] - (c_rate + bank_s)) if o['MM_Side'] == '매도' else (
                    (c_rate - bank_s) - o['Entry_Rate'])
        time_diff = (current_time - o['Entry_Time']).total_seconds() / 60

        method = ""
        if expected >= o['S_i']:
            method = "Tier1_TP"
        elif time_diff >= o['T_w']:
            method = "Tier2_TimeOut"
        elif abs(inventory) > o['Limit']:
            method = "Tier3_RiskLimit"

        if method:
            pnl = expected * float(o['수량'])
            current_trade_pnl += pnl
            o.update({
                'Exit_Time': current_time,
                'Exit_Rate': (c_rate + bank_s if o['MM_Side'] == '매도' else c_rate - bank_s),
                'PnL': pnl, 'Exit_Method': method
            })
            results.append(o)
            inventory += (float(o['수량']) if o['MM_Side'] == '매도' else -float(o['수량']))
        else:
            active.append(o)
    pending_lots = active

# ==========================================
# 4. 결과 요약 및 시각화
# ==========================================
df_res = pd.DataFrame(results)
final_total = current_trade_pnl + netting_profit
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


out_trades = f"dqn_hrl_simulation_trades_{timestamp}.csv"

if not df_res.empty:
    df_res.to_csv(out_trades, index=False, encoding='utf-8-sig')

print(f"\n✅ 파일 저장 완료: {out_trades}")

print("\n" + "=" * 60)
print(f"✅ 시뮬레이션 종료 | 최종 실현 손익: {final_total:,.0f} 원")
print(f"💰 넷팅(스프레드) 수익: {netting_profit:,.0f} 원 | 거래 손익: {current_trade_pnl:,.0f} 원")
if not df_res.empty:
    print(f"📊 Tier1(익절) 성공률: {(df_res['Exit_Method'] == 'Tier1_TP').mean() * 100:.1f}%")
print("=" * 60)

# Cumulative PnL Graph
if not df_res.empty:
    df_res['CumPnL'] = df_res['PnL'].cumsum() + (netting_profit / len(df_res)) * np.arange(len(df_res))
    plt.figure(figsize=(12, 5))
    plt.plot(df_res['Exit_Time'], df_res['CumPnL'], label='DQN-HRL MM Performance', color='#1f77b4')
    plt.fill_between(df_res['Exit_Time'], df_res['CumPnL'], alpha=0.1, color='#1f77b4')
    plt.title(f"DQN-HRL Market Making Cumulative PnL\nFinal: {final_total:,.0f} KRW", fontsize=12)
    plt.grid(True, alpha=0.3);
    plt.legend()
    plt.tight_layout()
    plt.show()
