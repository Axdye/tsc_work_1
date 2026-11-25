# 工作二实验运行手册

[定稿方法与实验协议](METHOD.md)给出公式、主表和消融表。本仓库的统一入口是 `scripts/tsc_experiment.py`，用于训练、评测、组建实验套件和汇总结果。

## 环境

在联网准备机获取上游源码、数据、构建依赖、目标平台的 Python wheels 和完整 Qwen3.5-0.8B 权重。离线开发机可直接接收这些文件。模型、环境和实验输出均在仓库之外保存。

```bash
python scripts/bootstrap.py --only CityFlow LLMTSCS CoLLMLight --build-deps
python -m pip install -r requirements-experiments.txt
python scripts/build_cityflow.py
export TSC_MODEL=/data/models/Qwen3.5-0.8B
```

离线安装时使用目标 Python ABI、Linux 架构和 CUDA 版本对应的 wheelhouse，安装命令为 `python -m pip install --no-index --find-links /path/to/wheelhouse -r requirements-experiments.txt`。先准备与驱动兼容的 PyTorch，再编译 CityFlow。编译产物必须与训练环境使用同一个 Python。可设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。Qwen3.5 的文本模型加载、token 保留和 LoRA 更新共用同一入口。

## 中文策略观测

策略读取当前相位、相位持续时间和切相清空成本。四个候选相位逐方向列出进口排队、近中远三段行驶车辆、平均和最长连续停车时间，以及实际接收车道的排队、容量加权和最高占用率。另提供一跳邻居的当前相位及总排队，路网边界显式标出。三段位置由当前车辆距离估算，接收容量按 7.5 米有效车间距近似。回复采用一句简短中文理由和唯一的 `<signal>NTST</signal>` 相位标签。LLM 方法共用此模板，CoLight 从同一观测构造张量。经验 COMA 聚类仍使用相位级的 16 维特征。[Traffic-R1 附录 A.1](https://aclanthology.org/2026.acl-long.995.pdf) · [LLMLight](https://arxiv.org/abs/2312.16044v5) · [CoLLMLight](https://openreview.net/forum?id=KeJqoEVOeY)

## 冒烟与预实验

CityFlow 和 NumPy 可用后，先用脚本策略检查 CPU 采集、分支与预算。该模式没有模型前向和梯度。

```bash
python scripts/tsc_experiment.py train --scripted \
  --config configs/credit_smoke.json --set branch_workers=2 \
  --output outputs/work2-cpu-smoke
```

首次用 GPU 时先沿用 smoke 配置，并用 `--set model="$TSC_MODEL" --set local_files_only=true` 指定权重。然后用正式参数做单种子 8 窗口预实验。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tsc_experiment.py train \
  --config configs/credit_main.json --set model="$TSC_MODEL" \
  --set local_files_only=true --set rounds=8 --set max_responses=6144 \
  --output outputs/work2-pilot
```

在 `training.jsonl` 核对合法回复率、有效交通优势比例、快照复跑次数、留一与均匀基线的信用均值、簇锐度和回退率、RMS 裁剪率，以及生成、更新和分支耗时。失配时采集器会报错。`rollouts.jsonl` 保存实际回复 token、相位、候选分支回报和训练优势。`clusters.json` 保存冻结的标准化器和质心。分析生成长度与可执行相位时应同时查看非法输出率。

## 正式实验与等墙钟对照

```bash
python scripts/tsc_experiment.py suite --model "$TSC_MODEL" \
  --output outputs/work2-main
python scripts/tsc_experiment.py suite --model "$TSC_MODEL" \
  --output outputs/work2-main --execute --gpus 0 1 2 3 4 5 6 7
```

第一条命令只生成 `plan.json` 与实验配置。正式套件固定使用训练种子 0，共 7 个等回复预算的 LLM 作业、1 个 CPU CoLight 风格 DQN 作业和 1 个等墙钟 Trajectory-GRPO 作业。每个留出场景仍用环境种子 0、1 评测。每个 LLM 作业独占一张 GPU。正式配置为 `K=4, T=12, H=4, gamma=0.95`，40 个完整窗口共 30,720 条训练回复。训练和分支截止于 2700 秒，末 900 秒供合成留出验证。测试在济南 1、济南 3 和杭州 1 上运行完整 3600 秒。

CoLight 基线使用与 LLM 相同的结构化相位、车道和方向对齐邻居观测，以共享图注意力 Double DQN 在同一训练时段学习。它按与 LLM 主轨迹相同的 1,920 次环境控制步训练，模型跨路网共享且无需改变输出维数。该实现是协议对齐的 CoLight 风格对照。固定相序与 MaxPressure 使用仓库已有的上游适配器。

完整方法为等墙钟基线提供时间上限。时间包含校准、模拟、收集和参数更新，单独记录含验证的端到端时间。同一时钟协议下，等墙钟作业使用在上限内完成的最后一个更新。若某个完整窗口超时，回退该窗口的模型参数并记录超出的实际时间。比较时尽量使用同型 GPU 和相同 CPU 配额。跨卡共享 CPU 资源不足时，等墙钟实验可用 `--gpus 0` 串行运行。

默认多进程分支在各自 CityFlow 引擎中用原生内存快照复用实际联合分支。上游原生 Archive 含有引擎对象引用，工作进程用真实联合动作同步至采集快照，再在本引擎内恢复分支。工作进程缓存已经同步的前缀，日志分别报告标准分支仿真量与同步、复跑开销。每个切片的实际首步奖励会被交叉校验。使用 `branch_workers=1` 时直接在采集引擎内评分，省去跨进程同步。

## 汇总与可选诊断

```bash
python scripts/tsc_experiment.py report --root outputs/work2-main \
  --output outputs/work2-report
python scripts/tsc_experiment.py diagnose --config configs/credit_main.json \
  --set model="$TSC_MODEL" --set local_files_only=true \
  --training-run outputs/work2-main/train/full-s0 --output outputs/work2-diagnostic
```

汇总包含逐场景 ATT、AWT、AQL、剩余车辆、配对 ATT 与相对 AQL 差值、训练成本和验证 ATT 曲线。可选诊断从验证时段抽取 12 个可评分的非空状态，比较 MaxPressure 与冻结 LLM 贪心续跑的信用符号和相位选择。`audit` 可复算闭环指标，`animate` 可按路网方向生成多策略的拥堵 GIF。详细参数由 `python scripts/tsc_experiment.py --help` 提供。

## 实现口径

下游占用以受控接收车道车辆数除以车道存储量近似，存储量按 7.5 米有效车间距计算，边界接收填零。留一经验基线按定稿排除样本自身动作。批内样本来自关联的交通轨迹，严格无偏性仍取决于条件独立等额外假设。实验日志支持检验信用漂移与规则续跑误差。
