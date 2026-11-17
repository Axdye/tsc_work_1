# 第一篇工作：代码覆盖与运行边界

更新：2026-11-18。

**核心方法已有实现，可以进入开发机 GPU 冒烟和 pilot；尚不能称为已跑通训练或取得论文效果。**
本次审查按 [METHOD.md](METHOD.md) 的单模型、多路口、多步、配对反事实 GRPO 方案进行。

## 代码与研究设定对照

| 设定 | 当前实现 | 验证情况 |
|---|---|---|
| 一个共享策略控制多个路口 | `policy.py`；Qwen3.5-0.8B＋LoRA，每路口一条 response | 接口存在，真实模型尚未加载测试 |
| 在线多步环境交互 | `runner.py`、`environment.py`；每轮重新访问状态与采集 | 真实 CityFlow 的脚本策略检查通过 |
| 配对反事实比较 | `credit.py`；相同快照、相同其他路口当前动作，改变目标动作，续跑 H 步 | 预算、恢复和已知干扰测试通过 |
| 交通 reward | H 步折扣负排队成本；同相位回复平均；非法回复加惩罚 | CPU 数值与采集检查通过 |
| GRPO 更新 | `trainer.py`；保留采样 token，token ratio clipping，response 级优势，基座 KL | CUDA 前反向与保存读取待验 |
| baseline 和消融 | FixedTime、MaxPressure、同基座；trajectory/RTG/joint/unpaired，H/M/局部奖励/规则续跑 | 采集逻辑检查通过；学习结果待跑 |
| 统一评测 | ATT/AWT/AQL、三个训练 seed 与三个环境 seed、final checkpoint | 传统基线和指标复算已验 |
| 机制与成本分析 | 重复反事实估计诊断、组信号统计、仿真秒/token/调用数 | CPU 链路已验；机制优势待证 |
| 论文图表 | 主表、消融、学习曲线、训练信号图、拥堵 GIF | 文件生成链路已验 |

主入口是 `scripts/tsc_experiment.py`。旧 `scripts/minimal_cityflow_grpo.py` 的固定状态训练流程不是这篇工作的在线训练入口；新入口只复用其中的模型加载函数。

## 本次修正

1. **未配对消融的无效生成**：此前每个 context 先生成一批背景，随后每个候选又生成背景，第一批没有使用。采集器和独立诊断均删除该批生成；未配对方法仍需每候选独立背景，不能把其 token 开销宣称与配对方法相同。
2. **配置静默失效**：trajectory/RTG 的局部奖励、规则续跑和多背景，以及 joint 的局部奖励/多背景，现在会提前报错。拒绝 NaN、Inf、负格式惩罚和非整数预算等参数。
3. **学习信号可见性**：新增合法回复交通分数标准差、交通对照组比例、平均相位数和非法比例。`credit_status` 区分零优势、仅格式、微弱交通对照和有交通对照；有分差不代表归因准确，仍需独立诊断与闭环效果验证。
4. **离线运行**：新增 `local_files_only=true`，传递到 tokenizer、config、模型和 adapter 加载。CityFlow 构建提前检查本地头文件，明确关闭 Git 子模块下载。

未改变交通奖励公式、组标准化或 GRPO 目标。新增加的统计用于排错，不参与 reward。

## 服务器第一轮应验证什么

先跑 smoke，然后用保存的 adapter 闭环评测，命令见 [CREDIT_EXPERIMENTS.md](CREDIT_EXPERIMENTS.md)。

- 模型语言权重完整加载，生成长度与显存可接受，合法相位比例正常。
- 同组存在不同动作和交通分差；不要把单纯惩罚非法回复产生的非零优势当成交通学习。
- loss、梯度有限；有学习信号时 LoRA 参数发生变化；adapter 保存后能够重新加载。
- 五种训练方式的小 pilot 都能完成，再扩大主表和消融。不要首次直接执行 525 个作业。

依赖版本是目标 API 约束，尚未形成经服务器验证的 GPU lockfile。不支持完整的中途训练恢复；本地没有 torch，本轮跳过 torch 数值测试，具体执行记录见 [CREDIT_VALIDATION.md](CREDIT_VALIDATION.md)。

## 工作边界

- 这是动作/决策级信用分配；每条 response 的 token 共享优势。没有实现 span 级归因或 LLM judge。
- 配对只固定其他路口的**当前动作**；分支后续 LLM 根据各自状态重新采样，没有固定全部未来随机性。
- 不等于完整 Traffic-R1、SocialLight 或 DGLight 复现。`joint` 为 JS-GRPO 风格受控基线；trajectory/RTG 为本项目对照。
- 快慢策略、通信建议缓存和工具调用属于下一篇工作。当前没有这些模块。
- 没有任何新方法的实测收益、SOTA 或显著性结论。已有论文主表和消融预期文档不能当作本项目训练结果。
