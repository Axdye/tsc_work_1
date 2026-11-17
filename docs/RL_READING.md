# 首轮全文与代码对照阅读（2026-11-02）

## 范围与证据等级

重点：LLMLight/LightGPT、CoLLMLight、Traffic-R1、OracleTSC；HeraldLight 作为可变时长与训练范式对照，
LLM-Assisted Light 作为后续原生 TransSimHub 功能线。
GENEX-TSC/VLALight 的源码已下载，但不声称首轮精读了它们。

本地保存五篇主要论文 PDF 和提取正文，另保存 CoLLMLight arXiv 初版及补充最近邻 ProcessLight。
CoLLMLight 的 ICLR 最终版通过 OpenReview 官方全文读取器核对方法、实验与附录；
终端下载遇到 403，未把 arXiv 初版冒充最终版。
论文可访问不代表训练资产齐全，源码审计不等于成功运行，更不等于复现表格。

## 一、共同问题：两种“RL”必须分开

传统交通 RL：状态是队列/相位/邻居等，动作是相位或时长，奖励来自交通动态，
通过 Bellman 更新或 actor-critic 优化跨交通时刻的累计收益。

LLM 强化微调：生成一段文本，末尾解析出交通动作；训练可使用教师动作一致性、
仿真 rollout 得到的动作标签、实际环境奖励、格式奖励等。文本 token 的信用分配
和多路口、多周期的信用分配不是同一问题。

输出一次动作不等于单步优化；有多段 CoT 也不等于解决了多周期控制。

## 二、LLMLight / LightGPT

全文：https://arxiv.org/abs/2312.16044
代码：vendor/LLMTSCS

方法链路：GPT-4 与场景交互 -> 训练好的交通 action-value critic 过滤动作 -> LoRA SFT -> critic 排序对齐。
critic 架构来自 Advanced-CoLight，用 Bellman 目标训练；保留 a=argmax_a Q(o,a) 的 GPT 轨迹。
CGPR 对同状态的 k 条模型回复，按末尾动作的 Q 值排序，优化带 boundary constraint 的 ranking loss。
这不是直接拿 ATT 对 LLM 做在线 PPO。文字解释与动作的质量并没有被分别验证。

代码入口：

- `run_colight.py` / `run_presslight.py`：传统 RL 的训练入口。
- `utils/llm_aft_trainer.py:117`：initialize_critic；加载 model_weights 中的交通 critic。
- 同文件 `choose_action_with_value` 调用与 q_value 索引：给候选动作评分。
- `finetune/run_imitation_finetune.py`：SFT。
- `finetune/run_policy_refinement_data_collection.py`：采样 CGPR 数据。
- `finetune/run_policy_refinement.py`：排序对齐入口。

状态主要是结构化队列、接近车辆、相位映射的文本；常见动作是四相位选择。
论文的绿灯/黄灯/全红描述与当前 wrapper 的实际切换要分别记录：
当前 config 默认 30s 决策、5s phase-0 过渡包含在区间内；重复同相位没有过渡。

实验读法：收益需分解为教师数据、critic 过滤、SFT、CGPR。
论文表 3 的部分模型中 SFT 是主要收益来源，CGPR 的增益相对小；不能全部归给推理。
训练/测试同域、跨域、扩网结果需要分别比较。全套复现还需要 critic、训练语料、LLM checkpoint。

## 三、CoLLMLight：本课题最近邻

最终版：https://openreview.net/forum?id=KeJqoEVOeY
原始代码：vendor/CoLLMLight（commit 见 sources.lock.json）。

### 论文机制

SR：利用本地/上下游、空间关系和历史交互，选择性进行关键车道、时空分析、反思等步骤，生成建议。
RD：当前状态 + 最近缓存建议 -> 动作，并自行输出建议效用 U（代码字段 used_atr）。
训练先挑选最短且效果最佳的 SR-RD 教师轨迹做 SFT，再 PPO 精调。

关键奖励（论文 §3.2、代码 ppo_ft.py）：

R_RD = +1 if a == a* else -1

R_SR = R_RD * [ beta*(1-L/Lmax) + (1-beta)*U ]

a* 来自仿真枚举候选相位后的较长期队列评价；U 是 RD 自报建议是否有用。
因此“交通仿真生成监督 + label-reward PPO”比“直接在线优化 ATT”更准确地描述当前公开训练脚本。
U 是使用/有用的自报告，不能等同于建议相对无建议实际带来的边际交通收益。

### 代码实际做了什么

1. `framework/FTSample.py:253` 的采样循环保存快照，枚举目标路口四个相位，
   后续步骤用默认策略，依据队列等指标选 Best_Signal；`run_fts.py` 的 reward_period 默认 5。
   它已经有多步反事实 rollout，不能把“加入 rollout”作为新贡献。
2. `FTSample.py:846` 同时保存 eng.snapshot() 和 wrapper deepcopy；恢复控制状态同样必要。
3. `ppo_ft.py:183` generate_offline_data 先生成 ATR、再 RA，读取固定样本的 Best_Signal 算 ±1。
4. `ppo_ft.py:429,444` 的 reward 函数中，长度 L 是回复 `##` 模式的计数并截断至 8，
   不是直接测量输出 token 数或真实推理耗时。
5. `framework/CoLLMlight.py:408` 当前评测函数串行调用 advance_traffic_reasoning 和 reactive_action。
   `models/CoLLMLightAgent.py:448,587` 对历史作 t-1 截取，模拟上一时刻建议。
   不能把此代码的串行 wall-clock 执行说成已实现真实后台并发缓存服务；也不能因此否定其模拟异步设计。
6. PPO 数据是每个状态 ATR/RA 两条记录，32 个状态变成 64 个 query；
   batch_size 的 TRL 兼容性、验证函数误传 RA response、best checkpoint 选择时序是具体运行风险。
   未静默更改上游；REPRODUCTION.md 记录详情。

### 实验与真正未解决的部分

最终版主实验强调 Syn-Train 训练后跨场景 zero-shot，不能把表格和原版同城训练结果直接混排。
最终版 §4.6 **已有** Stale SR（50% 概率额外延迟一步）与 Communication Failure（50% 丢一个邻居）实验。
之前“它没考虑建议过时”的说法过强，现已撤回。
但该扰动测试与学习“哪条建议此刻还有实际增益、应该采用/刷新/丢弃”是不同问题。
这一差异是候选 idea 的出发点，不是对整个领域的首创证明。
附录说明未显式建模通信 latency；真正 deployment-time 事件队列/多种延迟需另建协议。

## 四、Traffic-R1

全文（正式 ACL 2026）：https://aclanthology.org/2026.acl-long.995/
权重：https://huggingface.co/Season998/Traffic-R1

阶段 A：3000 个专家确认的最终动作，基于 Qwen2.5-3B 做 GRPO；奖励含动作准确与格式。
没有标注推理过程，因此训练奖励验证的是最终动作，而非每步推理的真实性。
阶段 B：CityFlow 4x4、约 8000veh/h 的模拟交通中在线学习。
轨迹奖励结合队列与等待，STPO 将同一个轨迹回报平均分给每个交通决策：r_t=R_traj/T。
这减少了长轨迹文本拼接，但没有精细区分“哪个路口、哪个动作造成了收益/损失”。
通信采用路口位置奇偶分组、半步交替消息传递；不应理解为任意网络延迟下的调度机制。

论文动作仍是预设相位，基准绿灯 15s 加 3s yellow/2s red；不能把任务定义里的 duration 泛称
当成实际实验已经联合优化时长。
研究中 OOD 事件、传统 benchmark、现场流程是不同证据。现场包含配时计划生成与人工审核，
不等于验证无人监督秒级闭环。解释通顺也不等于证明内部推理忠实。

复现边界：Public 0.1 模型卡声明是较早 checkpoint，未公开全部离线样本。
没有完整官方在线训练代码可拉；登记为 checkpoint evaluation，不能冒充训练复现。

## 五、OracleTSC

全文：https://arxiv.org/abs/2605.08516 （本地 PDF 显示 TMLR 04/2026）。
未找到可确认的官方代码；下载 LibSignal 不等于下载了此方法。

模型针对同一交通观测采样 G 条回复，提取相位，第一条的相位执行到环境。
环境奖励采用队列减少量；R_hurdle = R_env - HR。
根据 G 条回复中各相位出现的次数做 temperature softmax，得到 p_j；
对实际选择相位 j_c，在 R_env>HR 时增加 p_jc 奖励，否则为 0。
因此它实际鼓励的是“交通表现过阈值时，所选动作在多回复中的一致性”，不是无条件降低文本熵。
用 PPO、value head 和 GAE 更新；“token 轨迹”与“交通时刻轨迹”仍应分清。

关键证据：CityFlow1x1（8 相位）与 Cologne1（4 相位）两个单路口；10s 决策，切换加 5s yellow。
文中 75% 的旅行时间下降主要是相对于未训练 LLaMA3-8B，不能说相对强 RL 下降 75%。
正文 §4.4：单路口上专用 RL 有约 100–120s 的旅行时间，LLM 146.9s；它自己也承认强 RL 仍有优势。
跨场景转移须看绝对值：Cologne 训练 -> CityFlow 为 517.5s，对比 base 594.9s，离同域训练 146.9s 仍远。
HR 随模型/场景变化，需要调参；报告 best checkpoint 的选择与独立测试隔离还需核对。
常数 reward shift 在不同时间/终止/优势实现下的效果不能简单视为通用最优性理论。

对课题的启发：不要用语言不确定性替代动作后果不确定性；
多个动作可能交通效果接近，分歧不一定意味着差决策。应比较 value gap/后果分布和相位投票熵。

## 六、HeraldLight（重要的动作空间对照）

全文：https://arxiv.org/abs/2511.00136；代码 vendor/HeraldLight。
先学习 queue-to-release-time 映射，再用距离/速度估计接近车辆是否赶得上当前服务窗口。
Herald 的预测、阈值与不平衡规则被写入提示；LLM 输出 phase+duration，时长可到 1s 分辨率。
ChatGPT 教师生成数据 -> Llama3.1-8B LoRA imitation -> 更强 LLM 纠错打分 -> ranking loss。

**它的 LLM-Critic 是语言评价器，不是 actor-critic RL 里的 V/Q critic。**
`fintune/RankTrainer/RankTrainer.py:167` 是 response log-likelihood 的排序损失。
不要把 HeraldLight 当成第三个在线 PPO 方法。

`models/Llama.py:207` 的 prompt 有明确规则；`heraldmaker_v2` 负责构造预测。
`models/herald.py` 记录/更新 queue2duration，部分实现采用离散表；论文的 piecewise mapping 与当前文件需分别解释。

论文表 II：SFT 303.581 -> 加 dual correction 297.852（杭州2 ATT），
主改进不能都算作双 LLM。表 I 杭州2 AWT 为120.03，对照 DynamicLight53.99；
优化 ATT 可能牺牲等待，应报告不同指标。DynamicLight5s vs Herald1s 的控制分辨率是混杂因素。
预测-only、规则-only、相同动作空间下 RL/LLM 的配对实验更能判断 LLM 贡献。

## 七、LLM-Assisted Light：下一条环境线

全文：https://arxiv.org/abs/2403.08337；代码 vendor/LLM-Assisted-Light。
原论文将交通感知、辅助控制器建议、可行动作与决策解释包装成工具，面向应急车辆/路障/传感器故障。
原论文以 UniTSA 等 RL 作辅助决策；当前重构版公开代码以 MP 提供参考建议，不能混称原论文原样复现。

当前 `la_light/llm/tools.py` 在一次决策开始时取得一致 snapshot 并缓存工具返回；
MP expert_advice 已预先放入 snapshot，get_phase_recommendation 不一定增加信息。
`run_llm_tsc.py` 先完成 ReAct 工具调用，再 env.step，未让交通在 API 等待期间持续前进。
其时延主要是 wall-clock 日志，不能直接解释为已经作用于交通的物理延迟。
事件由 YAML 触发，三个基线共享环境；非法输出尝试重格式化后可回退传统动作。

最有价值的后续研究是“调用工具/采用建议的实际增量价值”，
而不是重复把已在 snapshot 中的 MP 建议再调用一次、再声称获得额外信息。

## 八、ProcessLight：不能忽略的过程监督最近邻

全文：https://arxiv.org/abs/2609.22746；代码：https://github.com/wenzhaoabc/processlight。
本轮补充读取方法 §4、实验 §5、局限及关键训练代码；不声称像前六篇一样覆盖所有附录。

方法链路：同一仿真快照枚举相位、固定 horizon 评价 -> 教师生成与最优相位一致的分步推理 ->
SFT warm-up -> STeP-PO。六个证据步骤分别检查相位映射、局部需求、下游、邻居协调、流量趋势和公平性，
最后输出相位。它研究的是语言推理步骤的信用分配，实验仍为四相位、15s 绿灯 + 3s 黄灯 + 2s 全红。

三个核心监督：

1. Step quality：生成可核验 claim，用当前交通观测验证；错误 claim 受罚，正确 claim 按教师关键结论覆盖率评分。
2. Step importance：删除一个推理步骤，测量最终相位分布相对完整推理的 JSD，归一化为重要性。
   论文明确这些分数在训练前按 prompt 预计算，后续 rollout 复用，并非每条新生成回复都重新做反事实。
3. Semantic step tree：相同前缀下聚合步骤，分别计算结果优势与步骤质量优势，再映射到对应 token。
   最终 outcome 仍含最优相位准确度与格式完成度。

代码入口：`src/rltsc/databuild/samples_simu.py`、`samples_claims.py`、`samples_importance.py`；
`src/rltsc/trainer/rewards.py`、`advantage.py::compute_processlight_tree_advantage`；
`src/rltsc/verify/` 是 claim verifier，`patches/verl-processlight.patch` 是训练框架适配。
实际 `samples_importance.py::process_sample` 对完整/删步 prompt 分别查询 A/B/C/D 分布并计算 JSD。

实验按 Synthetic-1 训练、济南/杭州五个流量测试；表 4 对照 outcome-only 与 step-wise 监督，
表 3 逐步加入 quality、importance、tree。基线主要是跨域协议，不能把结果直接解释为胜过所有同域专用 RL。
作者局限也承认 verifier 依赖结构化观测、预设步骤和规则，删步预计算有额外成本，复杂路口适用性未解决。

它排除了“只加过程奖励/分步优势”这种宽泛 idea。
但改变相位分布不等于改善交通；JSD 大也可能只是改变了两个几乎等效的动作。
这支持把研究问题收窄到建议/工具在实际执行时的交通增量价值，而不是宣称首次细粒度信用分配。

## 阅读结论

首轮建议以 CoLLMLight 为最近邻、以传统 RL 为快策略，研究建议的增量价值监督与执行时效。
Traffic-R1 提供轨迹级信用分配的对照，OracleTSC 提供多回复一致性与环境效用区别的对照，
HeraldLight 强制提醒动作/时长公平性，LA-Light 提供后续工具与事件功能迁移环境，
ProcessLight 要求我们把语言步骤信用与交通时间信用明确区分。
所有创新判断都需要进一步最近邻检索与实验；不能把代码问题本身包装为方法贡献。
