# RL-TSC 扩展调研：论文正文与源码价值

调研日期：2026-11-02。近一年窗口：2025-11-02 至 2026-11-02，并补读与项目直接相关的较早工作。
这是一轮有范围的文献调研，不是全领域穷尽搜索，也不是统一榜单。
本轮只下载文献/源码、静态阅读及记录来源，没有安装依赖、运行仿真或训练。

## 一、范围与阅读证据

以下 11 篇均进入正文阅读：问题定义、方法、训练目标、实验设置、主结果与局限；
按相关性核对关键附录和源码，不表示每个附录、每个源码文件都逐行审计。
本地 PDF、官方 HTML、版本不同的稿件分别登记，不以摘要或 README 代替论文正文。

| 工作 | 本轮实际阅读版本 | 全文入口 | 源码入口 |
|---|---|---|---|
| DynamicLight，Liang Zhang 等 | arXiv v2，2024-05-02；初稿 2022，不虚标正式会议 | https://arxiv.org/html/2211.01025v2 | https://github.com/LiangZhang1996/DynamicLight |
| X-Light，Haoyuan Jiang 等 | IJCAI 2024；阅读 arXiv v2 扩展全文 | https://arxiv.org/html/2404.12090v2 | https://github.com/jiang-haoyuan/X-Light |
| Unicorn，Yifeng Zhang 等 | T-ITS 2026；阅读 2026-03-25 arXiv v2 | https://arxiv.org/html/2503.11488v2 | https://github.com/marmotlab/Unicorn |
| RobustLight，Mingyuan Li 等 | ICML 2025，PMLR 正文及附录，23 页 | https://proceedings.mlr.press/v267/li25cs.html | https://github.com/MJLee00/RobustLight |
| DataLight，Liang Zhang 等 | 实际精读的是 2024-05-02 arXiv v2，14 页 | https://arxiv.org/html/2303.10828v2 | https://github.com/LiangZhang1996/DataLight |
| CFLight，Mingyuan Li 等 | KDD 2026；阅读 2025-12 arXiv v2 | https://arxiv.org/html/2512.09368v2 | https://github.com/MJLee00/CFLight |
| iLLM-TSC，Aoyu Pang 等 | 实际阅读 2024 arXiv v1，18 页；作者仓库另列 TVT 2026 版本 | https://arxiv.org/pdf/2407.06025 | https://github.com/Traffic-Alpha/iLLM-TSC |
| VLMLight，Maonan Wang 等 | NeurIPS 2025 正式版，32 页 | https://proceedings.neurips.cc/paper_files/paper/2025/file/3849b5861dcaeaf4758eef0979a98cc6-Paper-Conference.pdf | https://github.com/Traffic-Alpha/VLMLight |
| PromptGAT，Longchao Da 等 | AAAI 2024 正式版，9 页 | https://ojs.aaai.org/index.php/AAAI/article/download/27758/27557 | https://github.com/DaRL-LibSignal/PromptGAT |
| GenEx-TSC，Hao Huang 等 | CAAI Transactions on Intelligence Technology，2026-03-05 正式全文 | https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.70113 | https://github.com/Run-Aground/GENEX-TSC |
| DGLight，Chenbo Yu | 2026-04-28 arXiv v1；论文自称 dissertation，不当作顶会论文 | https://arxiv.org/html/2604.25259v1 | https://github.com/yyccbb/FYP_LLMTSC |

版本注意：DataLight 当前 README 指向 ECML PKDD 2026 工作，BibTeX 的 proceedings 年为 2027；
2026 Zenodo 补充材料确实存在，但本轮完整阅读的是它明确链接的历史 v2。
没有把历史稿表格当作 2026 正式版实验。RobustLight/CFLight 论文指向 SmartCity 聚合仓库，
本地整理的是作者 MJLee00 名下独立项目；二者没有核验为同一 commit。

## 二、传统 RL：最有价值的六篇

### 1. DynamicLight：可变时长，也有真正异步的环境推进

问题：固定间隔选相位无法在执行前明确本次绿灯持续多久。
方法：先选相位，再用所选相位特征预测时长；DQN 分阶段训练相位头与时长头。
时长集合为 10、15、20、25、30、35、40 秒；还比较固定顺序、MP、最大队列及 Lite 变体。
实验含济南、杭州、纽约和合成异构路口；时长消融支持时长模块有贡献，纽约2并未胜过 Advanced-CoLight。
作者明确承认邻居信息与协同不足。不是通过训练数据证明工业部署。

源码最有价值的是 `utils/cityflow_env.py::step`：每个路口维护剩余动作队列，
环境推进到最早需要新动作的时刻，返回 `list_need_action`。这是可借用的事件驱动控制骨架。
`models/dynamiclight.py::choose_action` 及 `prepare_Xs_Y` 对应双头和分阶段更新；
TD 更新仍用固定 GAMMA，因此扩展时要明确奖励累计和交通时间折扣口径。
不能仅复制网络而保留固定间隔 wrapper。

判断：你的 CityFlow 相位/时长方向优先级最高；环境推进代码比网络层数更值得读。

### 2. X-Light：无需 LLM 的轨迹上下文与跨城市协同

方法：Lower Transformer 聚合本路口和邻居的当前观测、上一步动作/奖励；
Upper Transformer 处理历史上下文，配合动态预测辅助损失与 PPO，跨场景混合训练。
GPI 统一异构路口；主实验在 SUMO，固定 15 秒、预设相位选择。
实验含留出整个场景的 zero-shot；因此“传统 RL 无法跨城市”不成立。
它依赖结构映射、历史奖励和邻居信息，不是任意新路口都能直接接入，也没有联合优化绿灯时长。

源码：`onpolicy/algorithms/r_mappo/algorithm/sumo_nn.py::TransformerEncoderPolicy`；
`r_mappo.py`、`runner/shared/sumo_runner.py`、`envs/sumo_files_marl/`。
输入切片中有固定数量邻居和特征维度；移植 CityFlow 需要保留语义映射、mask 与时序对齐。

判断：适合研究快策略本身的泛化，作为“历史/邻居建模是否需要大模型”的重要对照。

### 3. Unicorn：异构路口表示与协同 critic

方法：以 movement 为单位统一输入，将相位表示为 movement 的集合；UTR 做相位条件特征提取，
ISR 使用变分表示与对比学习建模路口特性，协同训练引入邻居动作依赖。
SUMO 八个网络，既有同域和多场景联合训练，也有额外的留出 Grid5×5 零样本实验。
主动作仍是固定时长下选择合法相位。多场景训练结果不能全部称作未见路网泛化。

源码：`models/Unicorn.py` 的 ActorNet、CriticNet；`env/tls.py` 的相位/路口表示；
`runner_unicorn.py` 处理轨迹，`driver_unicorn.py` 更新策略。
movement-phase 映射、有效相位 mask、邻居动作向量比“使用 Transformer”更可复用。
仓库有 maps 和训练/评测结构，MIT；但依赖 Ray/SUMO，自带包装层不是 CityFlow 接口。

判断：异构路口和强快策略方向优先读；做 CityFlow 项目时先借表示与训练协议，不应假定直接兼容。

### 4. RobustLight：先恢复坏观测，再交给控制器

方法：把状态恢复本身建模成 DSI agent，历史状态与上次动作为条件，扩散 actor 输出恢复状态，
critic 用交通奖励优化；测试时分别使用 denoise 和 repaint 处理噪声与缺失。
CityFlow 济南/杭州/纽约实验，与 MP、CoLight 等组合，报告交通 ATT 而不只重建误差。
大幅改善来自受扰动测试条件，不能当作正常交通 ATT 的同等提升。
正文讨论了预先检测噪声/缺失的假设，且承认大路网扩散推理较慢。

源码：`inferences/rl_dynamic_state_filling.py::Diffusion_Predictor`、
`inferences/diffusion.py`、`utils/make_mask_noise.py` 和 `models/advanced_maxpressure_agent.py`。
有独立的状态恢复、噪声注入和下游控制接口；checkpoint 路径、设备索引和旧依赖需核对。

判断：若研究传感器故障，必须先看这种非语言解法，才能判断 LLM 提供的增益是否不可替代。

### 5. DataLight：离线数据的质量和覆盖，比在线探索更关键

方法：用分段道路的车辆数与速度饱和/不饱和特征，空间编码后学习 Q；训练目标含 TD、ER、CQL。
v2 比较随机、FRAP、Advanced-CoLight 和固定周期产生的数据，并研究少数据与跨域情况。
它使用仿真生成离线数据，没有因此获得真实城市的日志覆盖保证；固定周期数据也容易缺少动作支持。

源码：`models/datalight.py`、`attentionlightBC.py`、`attentionlightBCQ.py`、
`attentionlightCQL.py`、`lightDT.py`；`memory/readme.md` 有数据来源及质量/时长说明。
离线数据另放 Google Drive/Zenodo，本轮未下载；仓库 GPL-3.0、TensorFlow 旧栈。

静态核对发现：当前 `train_network` 中 `berror` 为 [B,1]，
`reduce_mean(berror, axis=1, keepdims=True)` 仍为该值，随后平方差使 ER 项退化为零。
此外 CQL 对 chosen Q 先作全 batch 求和，需要核对与期望公式的尺度一致性。
这些是当前源码表达式的静态结论，不是通过实验认定论文结果无效；上游未修改。

判断：数据协议和离线比较代码很有价值；当前训练损失不能未经检查直接当作可信实现。

### 6. CFLight：反事实已经用于安全控制

方法：事故前状态与替代动作输入 BiCoGAN 近似的 SCM，预测后果并构造反事实轨迹；
再以 reward、loss 或 Q 分支注入安全学习。SUMO 合成/Cologne 单路口，涉及许可左转/保护相位，
同时报告碰撞、等待和吞吐。安全收益可能牺牲效率，主张不应只看碰撞降幅。
其人工风险设定和 SCM 的独立噪声/单调性假设，限制了向普通 CityFlow 队列任务推广。

源码：`CFLight_r.py`、`CFLight_loss.py`、`CFLight_q.py`、`gan_cf.py`、`buffer.py`。
在碰撞样本附近生成替代轨迹并加入 CF buffer 的实现值得读。
README 提到的 `CFLight.py` 在归档中不存在；部分消融文件拼写也不同，依赖版本较旧，未见根代码许可证。

判断：反事实研究的重要最近邻；不能再把“回看替代动作后果”本身称为新贡献。
它不是开箱即用的 CityFlow baseline，更不是实地交通安全保证。

## 三、LLM 与 RL：五种不同接法

### 7. iLLM-TSC：先 RL，再由 LLM 审查/改动作

预训练 PPO 先提出动作，LLM 读取交通描述、缺失标记与特殊车辆信息，保留或修改动作，格式失败则回退。
它不对 LLM 做 PPO/GRPO。所读 v1 是 SUMO 单路口、固定 5 秒槽；退化观测与应急车辆是主要场景。
应急收益并不意味着所有普通车辆指标最好；正常条件下 SARL-TSC 的普通交通效果更好。

源码：`rl_llm_tsc.py`、`TSCAssistant/tsc_assistant.py::agent_run`、
`TSCEnvironment/`、`sb3_ppo.py`。PPO checkpoint 与归一化统计已放 models，Apache-2.0。
LLM 输入还从 infos 读取额外字段，比较时应对齐信息权限；当前调用是同步后再推进环境。

判断：LA-Light/快慢组合的直接前作，接口清楚；不能仅以“LLM 修正 RL”作为新 idea。

### 8. VLMLight：视觉感知、模式选择、快慢双分支

VLM 从多视角图像生成描述；LLM 选择 routine RL 或分阶段推理，由相位映射、规划和规则检查完成控制。
RL 分支采用 PPO。正式版在 Songdo、油麻地和 Massy 路口仿真，重点是应急车辆优先。
不是大规模多路口异步协同的完整证据，也没有展示 LLM 参数的在线强化训练。

源码：`vlm_tsc_en/vlmlight_decision.py`、`utils/tsc_agent/llm_agents.py`、
`rl_tsc/` 和 `sim_envs/`。当前入口在回到 RL 分支前仍调用视觉理解、场景分析与模式选择；
因此“采用快动作”不意味着“省掉感知和语言调用”。场景判断 prompt 重点限制为明确应急车辆。
env.step 出现在推理之后；推理耗时没有自动构成交通持续演化下的延迟实验。

正文有一处值得留意：称正常交通 ATT 增幅均 <1%，但表1油麻地 38.10→39.80、Massy 57.84→60.84，
约为 4.5% 和 5.2%。应据表述为普通效率与应急优先的权衡，不能重复泛称全部 <1%。
同名提醒：此为 Traffic-Alpha/VLMLight，不是 usail-hkust/VLALight。

判断：与你的快慢、事件、TransSimHub 功能线高度重合；事件路由和视觉/相位映射代码优先级高。

### 9. PromptGAT：LLM 帮助动力学适配，不直接替代 RL 控制器

根据天气、道路和车流询问 LLM，提取加减速度、启动延迟等知识，辅助前向模型；
前向模型与逆模型联合形成 grounded action transformation，再训练 DQN。
“GAT”在这里是 Grounded Action Transformation，不是 Graph Attention Network。
论文将 CityFlow 设为模拟域、SUMO 设为目标域，明确属于 sim-to-sim；没有真实上路迁移验证。

源码：`trainer/sim2real_trainer.py`、`sim2real.py`、`common/stat_utils.py`、
`prompt_data/ChatGPT/`、`world/world_cityflow.py` 与 `world/world_sumo.py`。
有前向/逆向模型、提示与缓存知识的路径，适合借鉴跨模拟器适配；旧 API、固定路径和预收集数据依赖需核对。

判断：提醒我们 LLM 可以放在训练与建模阶段；不必在每个信号决策时调用大模型。

### 10. GenEx-TSC：周期级连续配时与 critic 加权语言训练

TD3 用车辆→车道→相位→路口的层级表示，输出下一周期各相位的绿灯系数；
乘 30 秒、取整并施加最小 5 秒约束，各相位顺序执行。
LLM 先从教师轨迹蒸馏，再用 TD3 critic 的 Q 分数 softmax 加权 MLE。
这是 critic 辅助的监督对齐，不能说 LLM 本身做了在线 PPO。
实验覆盖十种合成路口类型与新流量；测试流量变化不等于留出整个拓扑。
对同域专用 DRL 并非普遍胜出，停止车辆总量也不能与别的论文 ATT 混成榜单。

源码：`agent.py`、`env.py::step`、`Score_Train_LLM.py::ContrastiveTrainer.compute_loss`、`PN_data_chat.py`。
核心损失确实为 `softmax(scores)` 加权 response NLL；文件名 `aft_rank_loss.py` 不代表必然是 ranking 优化。
权重目录 README 仍涉及外部获取；资产完整性需与正文公开声明分别判断。

判断：你关心“更加灵活配时”时很值得读；周期级接口比单纯换一个语言模型更有启发。

### 11. DGLight：CityFlow + PyTorch CoLight critic + GRPO

固定 critic 对每个候选回复的最终相位评分，收集 prompt 与全相位分数向量，再单独做 GRPO；
采样阶段执行候选中得分最高的动作。附录另有快照分支联合 rollout 评分变体 JS-GRPO。
主实验是四相位、固定间隔，初始化继承 LightGPT；改善幅度较小，不能把 base Llama→DGLight 的巨大提升全算给 GRPO。
该文是公开学位工作/预印本，证据地位与同行评审顶会工作不同。

源码分层较好：`src/rl/critic/colight_q_rewarder.py`、`rollout/collector.py`、
`training/grpo_runner.py`、`pipeline/qguided_grpo_pipeline.py`；
`rollout/joint_scored_collector.py` 值得读快照、联合动作、horizon 与分数投影。

两项实际边界：

1. `src/modelling/agent/colight_agent.py::choose_action_with_value` 实际先对 Q/0.05 做 softmax，
   再 min-max 到近似 [0,1]；rewarder 返回的不是原始 Bellman Q。论文的 raw-Q 写法与源码评分尺度需要区分。
   非法动作默认 0，因此不能按“所有 Q 为负、非法动作更好”作错误诊断。
2. 配置保留本机绝对模型路径与外部 critic checkpoint；没有根 README/许可证。
   模块适合借鉴，不等于论文全部资产已齐或默认配置可直接运行。

判断：在新增项目中，最适合优先学习 CityFlow 上 LLM 强化训练接口的代码之一。
但“用 Q critic 替代稀疏回报”“增加 GRPO”本身已被覆盖，不能作为独立新意。

## 四、代码阅读优先级与环境边界

| 你的目的 | 优先项目/模块 | 本轮判断 |
|---|---|---|
| CityFlow 联合相位/时长 | DynamicLight 的 step 与双头 | 高：环境时钟最值得复用 |
| CityFlow LLM RL 接口 | DGLight 的 rewarder/collector/GRPO，已有 ProcessLight | 高：模块明确；需核对评分和资产 |
| 故障与缺失观测 | RobustLight 的 noise、denoise、repaint | 高：LLM 方法应面对的强非语言解法 |
| 跨城市快策略 | X-Light 的轨迹编码、Unicorn 的 movement 表示与 mask | 高：原生 SUMO，移植成本实在存在 |
| 离线日志研究 | DataLight 的数据协议及 BC/BCQ/CQL/DT 对照 | 高：训练损失有静态疑点 |
| TransSimHub 事件和快慢分支 | iLLM-TSC、VLMLight，已有 LA-Light | 高：接口与语义清楚；目前是同步推理闭环 |
| 周期级配时 | GenEx-TSC 的 TD3、执行器、Q 加权语言损失 | 高：实验口径与相位选择不同 |
| 反事实安全 | CFLight 的 SCM 与 CF replay | 机制阅读高，直接纳入项目中等偏低 |
| LLM 辅助环境适配 | PromptGAT 的前向/逆模型 | 机制阅读高，旧工程整体复用中等 |

GPL、MIT、Apache 或未声明许可证在源码清单中记录；不把“可下载”视为任意复制授权。
没有执行任何项目，以上为静态可读性和模块价值判断，不是运行通过或复现实验成功的评级。

## 五、检索到但未纳入这轮精读结论

- DuaLight（AAMAS 2024）：https://arxiv.org/abs/2312.14532 ，https://github.com/lujiaming-12138/DuaLight 。与 X-Light/Unicorn 构成场景共享/特有知识路线；本轮未逐项对照全文与代码。
- DTLight（arXiv 2023）：https://arxiv.org/html/2312.07795v1 ，https://github.com/XingshuaiHuang/DTLight 。读取了方法和数据公开说明；未完成代码对照，不与上面11篇同等计数。DTRL 有16种离线数据设定，部分大路网数据未随仓库上传。
- CrossLight（KDD 2024）：https://zhoujingbo.github.io/paper/2024CrossLightKDD.pdf 。离线到在线跨城市路线；本轮未确认官方可用实现，不给出代码复用推荐。
- FutureLight（VLDB 2026 仓库声明）：https://github.com/Allxz0523/FutureLight 。发现实际源码目录，未完成正式全文核对，不因“新”就推荐为已审计实现。
- ReasonLight（2026-05 预印本）：https://arxiv.org/abs/2605.29425 。与事件下语义修正直接相关；本轮先读其前序 iLLM-TSC/VLMLight，不对新稿结果作全文级判断。
- TSC-LLM（TR-C 2026-10）：https://doi.org/10.1016/j.trc.2026.105837 。检索到 PPO/MCP 路线，但未获得全文与确认官方代码，不把摘要描述当成实现事实。
- TARL-TSC（2026-10）：https://doi.org/10.1016/j.engappai.2026.116083 。检索到缓存事件文本嵌入的轻量 RL 路线，页面称代码待发布；未完成全文审读，不列为可复用代码。

## 六、对当前选题的影响

1. 不应把跨路网泛化当作 LLM 独有优势：X-Light/Unicorn 已有传统 RL 证据。
2. 不应把快慢双分支、LLM 改 RL、Q-guided GRPO 或反事实 rollout 本身当作新贡献。
3. 当前更值得收窄的问题是：建议/新增观测在实际执行时相对快策略的增量价值，及其时效、成本和归因。
4. 在传感器故障问题上，须区分缺失检测、状态恢复和语义事件；RobustLight 与 LLM 可能解决不同部分。
5. 若研究更灵活控制，DynamicLight 的异步动作到期执行器与 GenEx-TSC 的周期配时值得优先借鉴；
   仍要明确相同动作/时间/信息/训练预算下的比较。现阶段只提出阅读结论，没有新实验结果。
