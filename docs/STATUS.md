# 当前状态

2026-11-17：新增独立在线反事实 GRPO 实验套件 `scripts/tsc_experiment.py`、
`tsc_experiments/`，包含传统/学习对照、消融、统一评测、成本日志、图表与拥堵 GIF。
运行文档见 `CREDIT_EXPERIMENTS.md`，研究路线见 `CREDIT_RESEARCH_STORY.md`。
已完成 CPU 检查和济南2传统基线验证，未执行 GPU 训练；具体证据及跳过项见 `CREDIT_VALIDATION.md`。
默认 Qwen3.5-0.8B、单卡单任务、8 卡并行方法/seed；不使用旧原型的固定分数数据训练。

2026-11-15：按用户新指令，已执行本地 CPU baseline，不包含 LLM 训练。
新增 `cpu_upstream_baselines.py`，直接使用 LLMLight 公开环境、两种控制器和最终指标语句。
五个真实车流 × 两个控制器 × 一个 seed 已完成，原始日志独立复算通过。
依赖 NumPy/Pandas 仅安装在仓库 `.build/cpu-deps`。没有改 vendor。
结果：`outputs/cpu-upstream-suite-20261115/`；setup 审计：`CPU_SETUP_AUDIT.md`。
济南1 MaxPressure ATT 与表格相差约 1 秒，杭州2 FixedTime 有小差异，未做修正追表；其余结果接近或匹配。
四方法比较与 DGLight 代码边界见 `FOUR_LLM_METHODS.md`。

以下为历史状态，CPU 执行权限已被 11 月 15 日的新授权更新；LLM 训练仍未执行。

2026-11-03，用户提供 8 x A100 80GB 资源，并明确要求：**只允许写好代码，用户自行调试**。
新增最小 CityFlow 固定状态分数采集、LoRA＋GRPO 训练、闭环评估入口与单卡/8卡配置。
模型已按用户确认改为 `Qwen/Qwen3.5-0.8B`：纯文本加载、关闭 thinking，
LoRA rank 8、all-linear；目标接口更新为 Transformers 5.2.0 / TRL 0.29.0。
该固定状态原型已从当前分支清理。现有在线实验见 [工作二运行手册](CREDIT_EXPERIMENTS.md)。
没有连接服务器，没有在后台安装、仿真或训练。

此前 2026-11-02 的范围为只阅读论文、整理代码和总结；以下为保留的阅读记录。

## 已整理

- 本地独立 tsc Git 仓库已创建；未发布远端仓库。
- 首轮 10 个方法/环境源码已获取：LLMTSCS、CoLLMLight、HeraldLight、LLM-Assisted-Light、
  CityFlow、LibSignal、GENEX-TSC、VLALight、TransSimHub、ProcessLight。
- 扩展检索后新增 DynamicLight、X-Light、Unicorn、DataLight、RobustLight、CFLight、
  iLLM-TSC、VLMLight、PromptGAT、DGLight，当前共 20 个源码目录；没有执行这些新增项目。
- 重点正文与代码对照见 RL_READING.md；来源和校验记录见 sources.lock.json、papers/sources.json。
- 缺官方实现的部分明确标为缺失，没有用自行实现冒充官方代码。
- 已完成最近邻检索，补充 ProcessLight 与 guarded supervision，收紧候选 idea。

## 阅读范围

- LLMLight：正文方法/实验、关键附录与训练代码。
- CoLLMLight：ICLR 最终版方法、实验及相关附录；核心采样/推理/奖励代码。
- HeraldLight：正文与预测、提示、时长、ranking loss 代码。
- Traffic-R1：正式 ACL 全文及模型公开边界；没有完整训练代码可审阅。
- OracleTSC：全文及算法附录；没有确认官方方法实现。
- LA-Light：正文、工具与事件实现；区分原论文与当前重构版。
- ProcessLight：补充最近邻，只读方法与代码，不执行。
- 扩展轮另重点阅读 11 篇正文并对照核心源码（含此前仅归档的 GenEx-TSC）；
  实际稿件版本、代码模块、候选与未完成全文核验部分见 RESEARCH_EXPANSION.md。

## 原请求阶段的历史产物

用户缩小范围之前，已经生成统一复现入口、完成本地 CityFlow portability build，
并跑过直接 Engine 的 CPU 诊断。这些保留在 scripts/、.build/、outputs/，
不视作论文指标复现，也不作为当前方法创新的实验证据。
TransSimHub 依赖安装先后遇到软件源 TLS/超时及镜像 403，已停止，不再处理安装问题。

当前没有任何论文的完整 LLM 训练或表格复现完成；本次交付是阅读与代码整理。
