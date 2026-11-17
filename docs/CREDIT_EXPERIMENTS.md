# 配对反事实 GRPO：实验代码与运行手册

更新：2026-11-18。研究脉络、任务定义及下一部分快慢思考见 [CREDIT_RESEARCH_STORY.md](CREDIT_RESEARCH_STORY.md)，当前代码覆盖与待验事项见 [FIRST_WORK_READINESS.md](FIRST_WORK_READINESS.md)。

## 0. 交付范围与验证边界

本入口新增 **在线交替采集—分支评分—LoRA 更新**，与旧 `minimal_cityflow_grpo.py` 固定数据原型分开。

包括：

- 共享 Qwen3.5-0.8B 的多路口、多步控制，LoRA、显式 GRPO clipped loss、初始基座 KL。
- paired / unpaired / joint / trajectory / RTG 五种训练信号，以及 H、续跑策略、局部奖励、背景数消融。
- LLMLight 官方 FixedTime、MaxPressure；未交通训练的相同基座 LLM；同协议闭环评测。
- 采集/分支/更新日志、成本统计、checkpoint、独立指标复算。
- 3 个训练 seed × 3 个环境 seed 的实验清单与多 GPU 作业调度。
- 表格、PNG/PDF 指标图、学习曲线、信用估计质量诊断、并排拥堵 GIF。

**不是** Traffic-R1、SocialLight、DGLight 的逐项官方复现。`joint` 是 JS-GRPO 风格受控实现：使用统一模型、奖励、prompt、更新器和训练目标路口数量，便于归因比较。

CPU 检查使用 `ScriptedPolicy` 验证采集和记录链路，不产生任何 LLM 训练结果。CUDA 模型加载、梯度与收益仍须在你的服务器验证。具体已运行检查见 [CREDIT_VALIDATION.md](CREDIT_VALIDATION.md)。

## 1. 代码结构

```text
scripts/tsc_experiment.py        train/evaluate/suite/report/animate/audit/diagnose
tsc_experiments/
  common.py                    配置、组标准化、RTG、文件与 hash
  environment.py               官方环境适配、完整快照、统一观测与统计
  policy.py                    LLM/LoRA 加载、真实 token 保留、相位解析
  credit.py                    五种采集与信用分配方式
  trainer.py                   显式 token 级 clipped GRPO + KL，无 value head
  runner.py                    在线训练、验证、checkpoint、闭环测试
  diagnostics.py               高预算动作差参考与重复低预算估计
  suite.py                     实验清单与单 GPU 作业调度
  visualization.py             CSV/MD/PNG/PDF/GIF、指标独立审计
configs/credit_smoke.json       GPU 冒烟：2 轮、G=4、H=3、1 目标路口
configs/credit_main.json        主实验：仿真预算 200000 秒，最多 1000 轮
configs/credit_pilot_suite.json 5 种训练方式、小型等轮数 pilot
configs/credit_suite.json       主比较 + 消融，3×3 seeds
tests/test_credit_experiments.py 无 GPU 算法/预算检查；可选 torch loss 检查
tests/test_credit_cityflow.py    可选真实引擎快照和指标检查
```

不修改 `vendor/` 的交通动力学。Qwen3.5 文本模型加载复用 `scripts/minimal_cityflow_grpo.py:load_policy`，它显式提取 text_config 并映射语言权重，遇到缺失/不匹配的语言权重立即失败。

## 2. 服务器环境

建议单独 Python 3.11/3.12 环境。首先按服务器驱动安装合适的 CUDA PyTorch，再装实验依赖。仓库中的 `vendor/` 不进入 Git，复制仓库时必须带上相应数据与源码；可以用既有 bootstrap 资产流程准备，详见旧 `REPRODUCTION.md` 和 `sources.lock.json`。

从开发机新克隆时，先准备本实验需要的三个上游源码/数据目录和构建依赖：

```bash
git clone git@github.com:Axdye/tsc_work_1.git
cd tsc_work_1
python3 scripts/bootstrap.py --only CityFlow LLMTSCS CoLLMLight --build-deps
```

bootstrap 从官方分支获取源码，会更新本地 `sources.lock.json`；分支内容可能变化，保留下载后的 hash 记录。原论文 PDF、模型权重、历史实验输出、本地虚拟环境和编译产物均不上传 GitHub。

```bash
cd /path/to/tsc
python -m venv .venv-credit
source .venv-credit/bin/activate
python -m pip install --upgrade pip
# 先按服务器 CUDA/驱动安装合适的 torch wheel。
python -m pip install -r requirements-experiments.txt
```

CityFlow 需要针对**这个 Python ABI** 编译，不能把本机 `.cpython-39-darwin.so` 复制到 Linux：

```bash
python scripts/build_cityflow.py
export PYTHONPATH="$PWD/.build/python:$PYTHONPATH"
python -c 'import cityflow; print(cityflow.__file__)'
```

构建脚本依赖 CMake、C++ 编译器，以及已准备好的 `vendor/CityFlow`、`vendor/pybind11-v2.10.4`、`vendor/rapidjson`。此脚本是已记录的 portability build，不是宣称论文作者完全同版工具链。

默认模型 `Qwen/Qwen3.5-0.8B`，可用本地模型目录：

```bash
export TSC_MODEL=/path/to/Qwen3.5-0.8B
```

首次模型加载可能由 Hugging Face 下载；本地离线运行可设置 `HF_HUB_OFFLINE=1`。保留模型快照和 manifest 中的 base_commit。依赖是目标约束而非已运行的 GPU lockfile，成功后建议另存 `pip freeze`。

### 开发机无法联网时

上面的 bootstrap、pip 下载和模型下载在**联网准备机**完成。向开发机手动上传仓库、完整模型目录、`vendor/CityFlow`、`vendor/LLMTSCS`、`vendor/CoLLMLight`、`vendor/pybind11-v2.10.4`、`vendor/rapidjson`，以及目标平台的 wheelhouse。保留 `sources.lock.json`。

wheelhouse 必须匹配开发机的 Linux 架构、Python 版本和 CUDA PyTorch；不要用 macOS 默认下载的 wheel 直接给 Linux 安装。具体 Torch wheel 仍须按开发机驱动选择；当前没有这些平台信息，仓库不假定一个通用 CUDA 安装包。

```bash
# 目标平台匹配的联网准备机，先确定适合服务器的 torch 来源/版本。
python -m pip download --only-binary=:all: \
  -r requirements-experiments.txt -d wheelhouse

# 离线开发机，同一个目标 Python 环境。
python -m pip install --no-index --find-links=wheelhouse -r requirements-experiments.txt
python -m pip check
python scripts/build_cityflow.py
export PYTHONPATH="$PWD/.build/python:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TSC_MODEL=/data/models/Qwen3.5-0.8B
```

构建仍需要本地 CMake、C++ 编译器和 Python 开发头文件。构建脚本会检查源码/依赖并关闭 Git 子模块下载，缺文件时直接报错。模型需要包括 config、tokenizer/chat template 和完整权重，不能只有权重文件。

单个 train/evaluate/diagnose 命令加 `--set local_files_only=true --set model="$TSC_MODEL"`，可显式限制模型与 adapter 只从本地文件/缓存加载。运行 suite 时，将所用 base config 的 `local_files_only` 设为 `true`，并通过 `--model "$TSC_MODEL"` 指定本地路径。配置默认保留 `false`，兼容联网准备机；离线环境变量仍会生效。

### 8 张 A100 如何使用

**每个训练任务单卡**，一个 0.8B LoRA 任务不必用 8 卡 DDP。套件在 8 张卡上并行不同方法/seed；单卡足够容纳预期模型规模，但峰值显存仍取决于长度、批量和 Qwen3.5 内核。

当前不实现 DDP/FSDP，不要使用 `torchrun --nproc_per_node=8` 启动训练。`micro_batch_size × gradient_accumulation_steps` 定义每次更新累计的 response 数；为了降低全词表 logits 的峰值，当前逐 response 反向传播，不进行该 micro batch 内的向量化。

## 3. 不用 GPU 的检查

### 3.1 纯逻辑测试

```bash
TSC_TEST_CITYFLOW=1 python -m unittest discover -s tests -p 'test_credit_*.py' -v
```

没有 torch 时只跳过 loss 数值测试，其余无需 GPU；没有 matplotlib 时跳过绘图统计测试。CityFlow 测试要求 numpy/pandas/cityflow，验证 Python 计时/记录与引擎同步恢复，且适配器一步与上游 `env.step` 的观测/记录一致。

### 3.2 脚本策略采集冒烟

```bash
python scripts/tsc_experiment.py train --scripted \
  --config configs/credit_smoke.json --output outputs/plumbing-paired
```

这只是随机合法相位策略，无梯度、无模型加载、无 adapter。输出不可放进 LLM 主结果表。可分别用 `--set method=unpaired`、`joint`、`trajectory`、`rtg` 检查其他路径。

### 3.3 统一传统基线

```bash
python scripts/tsc_experiment.py evaluate --policy fixedtime --dataset jinan2 \
  --seed 0 --config configs/credit_main.json --output outputs/cpu-fixedtime-jinan2
python scripts/tsc_experiment.py evaluate --policy maxpressure --dataset jinan2 \
  --seed 0 --config configs/credit_main.json --output outputs/cpu-maxpressure-jinan2
python scripts/tsc_experiment.py audit --run outputs/cpu-maxpressure-jinan2
```

所有输出目录必须为空，避免覆盖。旧 CPU 套件仍可使用，新适配器用于本次学习方法的共同入口。

## 4. GPU 先冒烟，再做 pilot

### 4.1 单个 paired 冒烟

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tsc_experiment.py train \
  --config configs/credit_smoke.json --set model="$TSC_MODEL" \
  --output outputs/gpu-smoke-paired
```

检查 `training.jsonl`：loss/gradient_norm 有限；非零优势存在时 `max_lora_delta` 应有变化；观察截断、非法回复和零优势组比例。所有组优势为零且 KL 为零时，不应期待参数更新；先检查状态、生成合法性和动作多样性，不能伪造非零奖励。

新增 `traffic_signal_group_fraction`（合法回复交通分数存在差异的组比例）、`mean_unique_phases`、`invalid_response_fraction` 和 `credit_status`。`format_only` 表示这一窗口非零优势仅来自非法输出惩罚，没有可辨别的合法交通分差；`zero_advantage` 表示所有组优势均为零。`traffic_signal` 只说明存在分差，不说明反事实估计准确或控制效果改善。`groups.jsonl` 保存组级 `valid_traffic_reward_std` 供排查。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tsc_experiment.py evaluate \
  --config configs/credit_main.json --set model="$TSC_MODEL" \
  --dataset jinan2 --seed 0 --policy llm \
  --adapter outputs/gpu-smoke-paired/checkpoints/final/adapter \
  --method-label paired --train-seed 0 --output outputs/gpu-smoke-eval
```

两轮不会证明交通收益，主要检查模型权重映射、生成、旧/参考概率、梯度和 adapter 保存读取。

### 4.2 小套件

```bash
# 默认只生成清单，不执行。
python scripts/tsc_experiment.py suite --spec configs/credit_pilot_suite.json \
  --model "$TSC_MODEL" --output outputs/credit-pilot

# 阅读 plan.json、configs/ 后执行同一清单。
python scripts/tsc_experiment.py suite --spec configs/credit_pilot_suite.json \
  --model "$TSC_MODEL" --output outputs/credit-pilot \
  --gpus 0 1 2 3 4 5 6 7 --execute
```

Pilot 按相同轮数组织，只用于排错与早期趋势，不代表成本匹配。它的训练轮数较少，不能拿来下最终结论。

## 5. 正式主实验与消融

```bash
python scripts/tsc_experiment.py suite --spec configs/credit_suite.json \
  --model "$TSC_MODEL" --output outputs/credit-main \
  --gpus 0 1 2 3 4 5 6 7 --execute
```

默认完整矩阵是 **30 个训练作业 + 495 个评测作业 = 525 个作业**：10 个训练变体 × 3 个训练 seed，每个测 5 数据集 × 3 环境 seed，另有 3 种基线 × 5 数据集 × 3 seed。不是建议第一次直接运行全部。先删除 spec 中不需要的变体/场景，保存为自己的新 spec。

| 名称 | 内容 | 要回答什么 |
|---|---|---|
| `llm-base` | 相同初始基座，无交通 RL | RL 是否有效 |
| `fixedtime` / `maxpressure` | 官方传统控制器，同环境 | 是否超过强简单基线 |
| `trajectory` | 从同一窗口起点采样 G 条多步轨迹，交通总分均匀给窗口各回复 | 粗粒度交通信号对照；**不是 Traffic-R1 STPO 复现** |
| `rtg` | 相同轨迹采集，使用逐时刻 return-to-go，跨轨迹分组 | 排除只超过过弱均匀分配的情况 |
| `joint` | G 个联合动作分支，H 步续跑，按目标路口相位平均投影 | JS-GRPO 风格直接竞争者 |
| `paired` | 目标路口 G 回复，共享其他当前路口动作 | 主融合方法 |
| `unpaired` | 目标候选各自重采其他当前路口动作 | 最重要配对消融 |
| `paired-h1` / `paired-h6` | H=1 / 6，主方法 H=3 | 延迟效应与成本 |
| `paired-rule` | 分支未来使用官方 MaxPressure | 当前 LLM continuation 的作用 |
| `paired-local` | 仅使用目标路口排队成本 | 局部目标与网络协作 |
| `paired-m2` | M=2 个背景，主方法 M=1 | 更多背景评估是否划算 |

Trajectory/RTG 使用 8 步窗口，其他主变体 2 步采集窗口。这个选择是明确的实验设定，不应把其差异全归为优势公式；必要时增加相同窗口长度的敏感性实验。所有同类配对消融保持其他参数不变。

### 预算

- `max_sim_seconds=200000` 记录**实际运行的所有 Engine 物理秒**，包括 warmup、训练访问轨迹和所有反事实分支，恢复不会抵消开销。
- 开始一个新窗口前预估完整开销，不足时停止；预算可能剩余不足一个窗口，记录实际值。
- `rounds=1000` 是第二个上限；最终以先触及者停止，查看 summary 的 stop_reason。
- 相同仿真预算不等于相同 token、更新数和访问状态数；输出全部成本并同时画学习曲线。
- M、H、目标路口数改变会改变每窗口开销。可另建等窗口 spec：`max_sim_seconds=0`、固定 rounds，作为补充分析。
- 验证仿真不计入训练预算，单独保存日志和 wall time。方法训练窗口数不同会导致验证次数不同，最终报告需说明。

### checkpoint 与运行状态

默认统一使用 **final** 做主比较，避免不同验证频率造成 best 选择机会不均。`checkpoint=best` 仅在有验证且明确采用相同选择协议时使用。

验证只用同一 Synthetic8000 流量的另一个 Engine 随机实现，不是新的需求分布；测试济南/杭州不参与选 checkpoint。数据内容完全相同的合成 seed 验证不能称为严格分布外验证。

`checkpoints/` 保存 adapter、optimizer.pt 和 training_state.json。**暂不支持中途恢复训练**：CityFlow 进度、生成 RNG、采集窗口和 optimizer 必须共同恢复，当前没有假装提供这个能力。中断训练应换输出目录重跑。suite 可以跳过已完成作业；失败/中断的输出须人工保留或移动，并清理对应 status 条目后再启动，不能直接覆盖。

## 6. 信用估计质量诊断

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tsc_experiment.py diagnose \
  --config configs/credit_main.json --set model="$TSC_MODEL" \
  --states 4 --repeats 8 --reference-contexts 32 \
  --output outputs/credit-main/diagnostics/base-seed0
```

诊断枚举四个物理相位，比较 paired/unpaired 在相同状态上的低预算相位差估计，并以**独立的高预算分支均值**作参考。输出差值 MSE、重复估计方差和非平局相位对的符号一致率，保存全部估计。

它不训练策略，默认使用基座；传 `--adapter /path/to/checkpoints/final/adapter` 可诊断训练后策略。参考依然有限样本、有截断、有噪声，不是“真实因果贡献”。默认开销不小，先把 states/repeats/reference-contexts 改小检查；`--scripted` 只验证文件链路。

## 7. 汇总、画图、动图

```bash
python scripts/tsc_experiment.py report --root outputs/credit-main \
  --output outputs/credit-main-report
```

输出 `runs.csv`、`summary.csv/json`、`RESULTS.md`、每数据集三指标 PNG/PDF、预算学习曲线、训练成本 JSON，存在诊断结果时输出信用质量图。

新日志还会输出 `training-signal.png/pdf`，展示交通对照组比例、相位多样性、非法和截断比例；旧日志缺字段时不会伪造该统计。

报告自动排除 validation 和 scripted，拒绝同数据集下混合路网/流量/时长/换道协议，拒绝同方法、训练 seed、评测 seed 的重复记录。不要把一个方法多个 checkpoint 混进同一汇总目录。

误差条为 sample SD，不是假装由 3 个 seed 得到精确的置信区间。主表先平均同一训练实例下的多个评测 seed，再汇总训练 seed。

并排动图示例：

```bash
python scripts/tsc_experiment.py animate \
  --runs outputs/cpu-fixedtime-jinan2 outputs/cpu-maxpressure-jinan2 \
  --labels FixedTime MaxPressure --stride 2 --fps 8 \
  --output outputs/figures/jinan2-congestion.gif
```

对比训练模型时把其中一个目录替换为它的评测目录。要求相同路网、流量、环境 seed、时间轴和控制时长；共享固定颜色范围，颜色表示**每路口进口排队车辆总数**。绿线显示记录边界的实际相位允许方向，清空阶段不伪装为绿灯。它不是每秒车辆位置视频，动画时间由采样步长和 fps 压缩。

输出 GIF 和最后一帧 PNG。默认使用所有帧，较大路网/多个面板可能占较多内存，可增大 `--stride`；多张图之间需要一致颜色范围时传 `--queue-max`。

## 8. 原始记录与审计

训练目录：

```text
config.json / manifest.json / model.json
episodes/episode-*/engine files + provenance.json
rollouts.jsonl       prompt、真实生成 token、文字、相位、分数、优势、上下文回报
groups.jsonl         零优势、相位多样性、非法率、reward std
training.jsonl       每轮 loss、KL、clip、梯度、LoRA 变化、预算、token、validation
validation/          独立闭环评测，不混入训练访问轨迹
checkpoints/         adapter + optimizer archive，非完整恢复快照
summary.json         停止原因与最终/最佳 adapter 路径
```

评测目录：`decisions.jsonl`（每步队列、等待、真实边界相位、动作和文字）、`arrivals.json`、`metrics.json`、manifest、数据副本。`audit` 仅依赖这些原始记录重新计算 ATT/AWT/AQL，不使用训练分数。

训练阶段上游 `signal_inter_*.txt` 会包含分支运行产生的信号记录，不应当作一条部署轨迹；拥堵动图只读取独立闭环评测的 `decisions.jsonl`。

## 9. 常见排错

| 现象 | 检查 |
|---|---|
| 找不到 CityFlow/ABI 错误 | 重新针对当前 Python 编译；检查 PYTHONPATH，不加载 macOS 历史扩展 |
| Missing/mismatched language keys | 模型实际版本、Transformers 与 text_config/key mapping；不要跳过检查训练随机层 |
| Qwen3.5 内核警告/较慢 | 官方支持无可选 DeltaNet 内核的回退；先跑通，之后再匹配 CUDA/torch 安装加速依赖 |
| 全部零优势 | 空路网、回复全部相同相位、H 太短、流量未进入；先查看 groups 和原始相位分数 |
| 非法回复多 | 查看实际 token/text、chat template、max_new_tokens；提高长度前先确认是否发生截断 |
| OOM | 先减 generation_batch_size/max_new_tokens；反向传播当前逐 response，保持配置长度不过大 |
| 有优势但 LoRA delta=0 | 查梯度、PEFT 可训练参数和学习率；不要直接认为训练已经有效 |
| 比别人论文数字不同 | 检查 phase/interval/clearance/laneChange/需求/指标定义；不从不同论文直接搬分数 |
| report 找不到结果 | 先执行 evaluate；训练日志不是测试结果；scripted/validation 不进主表 |
| suite 部分作业失败 | 检查 logs 与 status；依赖失败的评测自动 blocked，不输出伪结果 |

## 10. 本实现没有暗含的能力

没有额外 PPO critic；没有 LLM judge；没有 span 内归因；没有慢协调模型；没有可变绿灯时长；没有工业信号机接入；没有自动保证超过 MaxPressure。第一部分完成后再按研究文档扩展。

模型 API 参考：[Qwen3.5 Transformers 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、[PEFT LoRA 文档](https://huggingface.co/docs/peft/package_reference/lora)。
