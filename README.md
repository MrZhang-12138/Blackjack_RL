# Blackjack RL

Q-learning 智能体在 Gymnasium Blackjack-v1 环境上的实现，胜率约 43%（接近理论最优）。

## 安装

```bash
pip install gymnasium numpy matplotlib tqdm
```

## 使用

```bash
# 互动游戏（AI 顾问模式）
python play.py

# 仅训练 + 评估
python Blackjack.py
```

### 游戏操作

| 按键 | 动作 |
|------|------|
| `h` | hit（要牌） |
| `s` | stick（停牌） |
| `q` | 退出 |

每局 AI 会给出建议，用户可以自行决定是否采纳。游戏结束后显示胜率统计。

## 算法

基于 Q-learning 的表格型强化学习：

- 状态: `(玩家点数, 庄家明牌, 是否有可用Ace)`
- 动作: 要牌 / 停牌
- 探索策略: 指数衰减 ε-greedy
- 学习率: 指数衰减（0.05 → 0.001）
