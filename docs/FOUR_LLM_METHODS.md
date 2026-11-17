# DGLight、LLMLight、Traffic-R1、CoLLMLight 比较

2026-11-15：核对论文相关正文、结果表及已保存源码。比较方法机制，不跨协议直接排名。

| 工作 | 核心问题 | LLM 的训练信号 | 信用分配粒度 | 主要边界 |
|---|---|---|---|---|
| LLMLight / LightGPT | 把教师与交通 critic 的知识迁入小 LLM | GPT-4 高质量轨迹、critic 过滤，IFT + critic-guided policy refinement | 回复/候选轨迹级 | critic 主要指导筛选与排序，不是在线 PPO 的 value head；解释合理不证明推理忠实 |
| DGLight | 用交通 Q 分数直接驱动 GRPO | 先训练、冻结 CoLight DQN，再对最终相位评分；保存 prompt+全部动作分数，collect-then-train | 一个回复一个分数 | 依赖 learned critic；未实现推理 span 归因；初始化继承 LightGPT |
| Traffic-R1 | 专家规则启蒙后，在仿真交互中优化 LLM 控制 | 离线专家动作准确/格式奖励；在线交通反馈与 STPO | 在线阶段将轨迹回报平均分配给交通决策 | 公开训练资产不完整；尚无细粒度时空或 span 归因 |
| CoLLMLight | 多路口协同与实时推理成本折中 | 教师轨迹/SFT，再 PPO；公开代码以 Best_Signal 一致性及长度/效用调节奖励 | SR/RD 两个模块/回复级 | Best_Signal 是仿真动作比较得到的标签；不等于每个推理 span 的反事实价值 |

## DGLight 的实际流程

1. 在 Hangzhou1 训练 CoLight DQN critic，使用结构化路口状态与邻居图信息，交通奖励为 -0.25×队列长度。
2. 从已经经过 LLMLight 流程训练的 LightGPT（Llama3-8B）开始；不是从通用模型零起步做这一步。
3. 冻结 critic，输出各相位分数。LLM 生成多个“理由+相位”候选，采样阶段执行得分最高候选。
4. 保存 prompt 和完整相位分数向量。GRPO 阶段重新生成回复，解析最终相位并查询对应分数，组内标准化更新 LoRA。
5. 主方法两轮 episode，3600s episode、30s action interval。critic 是交通 Q 网络；GRPO 本身不需要额外语言 value critic。

论文把奖励写为 Q(s,a)，但源码 `choose_action_with_value` 对 Q/0.05 做 softmax，再 min-max 到近似 [0,1]。
`CoLightQRewarder.get_q_values` 返回该变换后的向量。按原始 Bellman Q 解释奖励绝对值会产生误解。
同一个 prompt 下，同一个相位的不同语言回复得到相同交通分数。

另有 JS-GRPO 比较变体：保存 CityFlow 快照，对多个路口联合候选动作做短期分支 rollout，
再将联合回报投影成单路口动作分数，保存后训练。测试 h=3、6。它不是主方法，也不能直接等同于逐 span 因果干预。

## 结果应该如何解读

DGLight Table 5 的 LightGPT→DGLight：

| 场景 | LightGPT ATT | DGLight ATT | 判断 |
|---|---:|---:|---|
| Jinan1 | 277.38 | 275.92 | 小幅改善 |
| Jinan2 | 270.14 | 269.68 | 小幅改善 |
| Jinan3 | 262.67 | 262.16 | 小幅改善 |
| Hangzhou1 | 316.12 | 316.59 | 略变差 |
| Hangzhou2 | 333.95 | 333.14 | 小幅改善 |

大幅 base Llama→DGLight 改善不能全部归给新增 GRPO；LightGPT 初始化已经贡献主要性能。
JS-GRPO-h3/h6 在此表没有优于 DGLight，也没有证明简单加长 rollout 一定更好。
该稿自称 dissertation，arXiv v1 于 2026-04-28 发布；不按顶会已审稿 SOTA 对待。
这些是不同论文不同协议的结果，不把 CoLLMLight/Traffic-R1 表格与此表直接混合排名。

## 与本项目的关系

`minimal_cityflow_grpo.py` 借鉴 DGLight 的 collect-then-train 和 prompt-score-vector 接口，
实际使用固定规则后续 rollout 得到连续队列分数，没有训练 CoLight critic，故不是 DGLight 复现。
其状态集固定、后续规则非 LLM，也不是 Traffic-R1 的在线轨迹优化。

对用户的 span 信用分配工作：DGLight 是直接的 action-score GRPO 对照；其 §7.2.1 已把 token-level reward/masking 写为未来方向。
因此仅“给推理/动作 token 不同权重”不足以形成清楚的新意。应明确干预方式、span 对后续决策的影响、
如何处理仅改解释但动作不变的情况，以及局部优势与全局 GRPO 优势的组合及预算匹配消融。
CoLLMLight 更适合作为第二篇快慢控制工作的直接参照；Traffic-R1 是在线轨迹信用分配的重要参照。

## 来源与代码

- DGLight 全文：https://arxiv.org/html/2604.25259v1
- DGLight 官方代码：https://github.com/yyccbb/FYP_LLMTSC
- 核心源码：`vendor/DGLight/src/rl/critic/colight_q_rewarder.py`、`rollout/collector.py`、`training/grpo_runner.py`、`rollout/joint_scored_collector.py`
- Q 值归一化：`vendor/DGLight/src/modelling/agent/colight_agent.py:360`
- LLMLight：https://arxiv.org/html/2312.16044
- Traffic-R1：https://aclanthology.org/2026.acl-long.995.pdf
- CoLLMLight：https://openreview.net/forum?id=KeJqoEVOeY
- 既有详细笔记：`RL_READING.md`、`RESEARCH_EXPANSION.md`
