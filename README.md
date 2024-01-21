# reinforcement-learning-for-trading
Code for my thesis: `Reinforcement learning for financial trading`

## TODO
- [ ] Create eval script for evaluations - calculate KPIs and write findings to DOC.


### Ideas
- Periods when to trade: BUll marker long only, bear market short only. SMA - 150 * 24 vs 50 * 24. Short above long - BULL, Short below - BEAR, Uncertain close / few days after cross.
- Reward: smaller reward for early stage of trade AND full reward for switching position. for example R^(8-x) where R is reward
- Consider adding minimal profit you need make
- Teach model to replicate model - label dataset and add bonus or negative feedback if model make correct decision.
