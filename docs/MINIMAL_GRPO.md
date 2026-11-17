# 最小 CityFlow + LoRA + GRPO 实验

2026-11-03：按用户要求，仅编写代码和运行说明，由用户自行部署调试。
**本入口尚未运行、未完成依赖兼容验证；没有启动数据生成、训练或闭环评估。**

## 范围与代码来源

训练组织参考 DGLight 的 `src/rl/training/grpo_runner.py` 与 collect-then-train 接口，
直接调用 Hugging Face TRL `GRPOTrainer`。新入口在本项目 `scripts/minimal_cityflow_grpo.py`，
不改上游 vendor，不加载 CoLight critic，不使用 CoLLMLight 的 PPO。
直接 Engine 的 phase/pressure 辅助函数来自本项目已有 `scripts/cityflow_diagnostic.py`。

当前只验证最基本链路：

1. 在真实 CityFlow 状态上，从同一快照尝试四个相位。
2. 首步替换一个目标路口，其余路口按固定队列压力规则行动，之后全路网都按该规则闭环运行。
3. 固定三个 30 秒控制步，按每秒采样的受控进口车道排队总数积分，再除以 90 秒，得到连续分数。
4. 采集固定状态集；GRPO 对每个 prompt 重新生成回复，按解析相位查询分数，训练 LoRA。
5. 独立运行基座模型和 LoRA 模型的闭环评估。

这是**固定状态分布上的 GRPO 微调原型**。它不是每次生成都重新查询仿真的在线 RL，
后续仿真策略也不是当前 LLM，因此分数不等于当前 LLM 的完整长期 Q 值。
全局交通回报、slot judge、反事实 span 优势的对照实验在此基础上另加。

## 默认配置

| 项目 | 默认值 |
|---|---|
| 模型 | `Qwen/Qwen3.5-0.8B`，可用 `--model` 换成本地目录；仅加载文本模型 |
| LoRA | rank 8、alpha 16、`all-linear`，dropout 0；不适配输出 head |
| 算法 | TRL GRPO；无价值 critic、无 judge、无过程奖励 |
| 回复 | 一句简短理由 + 唯一 `<signal>ETWT</signal>`，相位四选一 |
| Thinking | 训练采样与评估均显式 `enable_thinking=False`；简短理由仍属于输出 |
| 精度 | BF16，非量化，开启 gradient checkpointing |
| 采样 | 每组 4 条回复，最长回复 192 tokens，最长 prompt 2048 tokens |
| 单卡 | micro-batch 1、累积 4、8 次优化更新 |
| 8 卡 | 每卡 micro-batch 1、累积 4、100 次优化更新 |
| 生成引擎 | 模型直接生成，暂不启用独立 vLLM 服务 |
| 信号切换 | 固定 30 秒控制间隔；切换时先执行 5 秒 phase 0，包含在 30 秒内 |

仅支持随仓库提供的直角网格、四相位 benchmark 路网。入口检查 phase 1..4 与
ETWT/NTST/ELWL/NLSL 的几何映射，不匹配时直接报错。
Phase 0 沿用路网中的清空相位，右转权限沿用路网。
这与论文中“30 秒绿灯再加黄灯”的设置不同，不能直接混比表格。

## 服务器环境（以下命令由用户手动执行）

建议独立 Linux Python 3.10 环境。保留服务器上可用的 CUDA/PyTorch 组合；
`requirements-grpo.txt` 固定 Transformers 5.2.0、TRL 0.29.0、PEFT 0.18.1，
是按公开源码接口选定的目标组合，尚不是实测 lockfile。旧的 Transformers 4.x 约束已移除。
安装前按服务器驱动选择 PyTorch CUDA wheel，不要直接复制本地 Mac 的编译扩展。

官方 Qwen3.5 权重包含视觉模块。入口使用 `Qwen3_5ForCausalLM` 与 `text_config`，
将检查点的 `model.language_model.*` 映射至 `model.*`，不实例化视觉编码器。
语言权重缺失或形状不匹配会报错，避免无意中从随机权重开始训练。
此加载分支也支持保存为纯文本配置的 Qwen3.5 检查点。

Qwen3.5 同时包含 Gated DeltaNet 和 full attention；`all-linear` 适配两类模块及 MLP，
避免只给 q/k/v/o 加 LoRA 而漏掉线性注意力层。设置在两个配置文件的
`lora_target_modules` 中。纯文本模型中没有视觉线性层。
当前不要求安装 `flash-linear-attention`、`causal-conv1d` 等额外 CUDA 扩展；
未安装时 DeltaNet 使用 PyTorch 回退实现，速度和显存需自行验证。
TRL 0.29.0 已移除 `max_prompt_length` 参数，本项目保留配置中的 2048 token 上限，
训练前检查完整聊天 prompt，超过则报错，不截断交通状态。

```bash
cd /path/to/tsc
python -m pip install -r requirements-grpo.txt
```

CityFlow 需要 g++、CMake 和 Python 头文件。当前 vendor 的 CityFlow Python binding
是此前 portability build 用的 pybind11 2.10.4，交通动力学源码未改。
如果已安装兼容的 CityFlow，直接使用。如果使用现有构建脚本：

```bash
python scripts/build_cityflow.py
```

此脚本依赖 `vendor/pybind11-v2.10.4/`、`vendor/rapidjson/` 和编译工具，输出到 `.build/python/`。
只同步本项目 Git 跟踪文件不会带走被忽略的 `vendor/`；部署时必须另行同步这些源码和数据。
构建完成后，在数据生成和评估命令加入 `--engine-module-dir .build/python`。

## 1. 生成小规模交通数据

默认数据在 `vendor/CoLLMLight/data/Synthetic/4_4/`，路网和车流可以分别覆盖。
输出目录已有 `train.jsonl` 时拒绝覆盖。

```bash
python scripts/minimal_cityflow_grpo.py collect \
  --engine-module-dir .build/python \
  --samples 32 \
  --warmup-steps 6 \
  --horizon-steps 3 \
  --seed 0 \
  --output outputs/grpo-data-smoke
```

输出：`train.jsonl` 和 `manifest.json`。每条记录包含聊天 prompt 和四个连续 phase_scores；
**分数只作为奖励函数 metadata，不进入 actor 的 prompt**。

采样只执行 1 个 Engine 线程。首次状态的四相位分支重复计算一次，检查结果能否恢复一致；
失败会报错，不静默继续训练。该检查不证明所有随机反事实可识别。
每次首步只替换一个目标路口；该四相位比较不是多路口联合最优搜索。

采集过程采用压力规则加目标路口随机探索，不执行得分最高的分支来强行挑状态。
零分差状态保留并统计；若所有状态的四相位得分完全相同，训练入口拒绝运行。

## 2. 单卡 LoRA + GRPO 冒烟

将模型目录替换为服务器上的真实目录；未提供本地路径时可使用配置中的 HF ID。
训练命令不需要 CityFlow，只读取上一步的分数表。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/minimal_cityflow_grpo.py train \
  --config configs/grpo_smoke.json \
  --model /path/to/Qwen3.5-0.8B \
  --data outputs/grpo-data-smoke/train.jsonl \
  --output outputs/grpo-single-smoke
```

期望检查：有效输出率、奖励组标准差、有限的 loss/grad norm、非零 LoRA 参数变化。
配置仅 8 次更新，不期待得出交通性能改进结论。
所有采样回复如果均为同一相位，组优势可能为零，即使分数表内有差异也无法学习；
查看 TRL 的 `frac_reward_zero_std`、回复长度与奖励日志，再调采样和提示。

非法回复会得到“该状态最差合法分数 - 1”，不会用可能优于负排队奖励的零分。
最终 adapter 保存在 `outputs/grpo-single-smoke/adapter/`。
`run_manifest.json` 记录目标配置、包版本、数据哈希和 LoRA 最大参数变化；
参数没有发生有限非零更新时直接报错。
本入口不自动恢复旧训练；输出目录必须为空。

## 3. 8 x A100 LoRA

先确认单卡链路，再使用 8 进程 DDP。这里每卡保存一个完整小模型副本，
没有 tensor parallel / ZeRO / FSDP，也不训练全量模型参数。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone --nproc_per_node=8 \
  scripts/minimal_cityflow_grpo.py train \
  --config configs/grpo_8gpu.json \
  --model /path/to/Qwen3.5-0.8B \
  --data outputs/grpo-data-smoke/train.jsonl \
  --output outputs/grpo-eight
```

当前配置的有效 generation batch 为 `1 * 8 * 4 = 32`，能被组大小 4 整除。
至少需要 8 个不同 prompt。两份配置均使用同一个 0.8B 模型，
8 卡通过数据并行增加采样吞吐，不改变模型规格。

## 4. 独立闭环评估

使用相同配置、相同车流和 seed 对比更新前后。下面只用 600 秒检查执行链路；
训练入口不会自动运行评估。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/minimal_cityflow_grpo.py evaluate \
  --engine-module-dir .build/python \
  --policy llm \
  --model /path/to/Qwen3.5-0.8B \
  --seed 100 --run-seconds 600 \
  --output outputs/eval-before

CUDA_VISIBLE_DEVICES=0 python scripts/minimal_cityflow_grpo.py evaluate \
  --engine-module-dir .build/python \
  --policy llm \
  --model /path/to/Qwen3.5-0.8B \
  --adapter outputs/grpo-single-smoke/adapter \
  --seed 100 --run-seconds 600 \
  --output outputs/eval-after

python scripts/minimal_cityflow_grpo.py evaluate \
  --engine-module-dir .build/python \
  --policy pressure \
  --seed 100 --run-seconds 600 \
  --output outputs/eval-pressure
```

`pressure` 是本项目诊断用的队列压力规则，不是已审计的官方 MaxPressure 实现。
`fixedtime` 也可用于检查基础控制链路。
LLM 每个路口只生成一条回复，贪心执行；不使用仿真 oracle 挑答案。
非法回复保持原相位，同时计入 `invalid_response_rate`。

输出 `decisions.jsonl`、`metrics.json`，包括平均受控队列、时域末未完成车辆数、原生 Engine ATT。
原生 ATT 的未完成车辆口径需要保留，不能当作所有车辆完成后的旅行时间。
仅更换 seed 不等于独立测试需求；正式结果应更换留出的车流文件/路网，并做多 seed 对比。

## 后续方法接入

先保持相同的状态集、actor、采样预算和连续交通回报，建立 outcome-only GRPO 对照。
之后再扩展 response 为 demand/downstream/decision，并增加 judge 与自定义 token advantage。
当前文件没有伪装成已实现 slot 信用分配。

## 参考

- DGLight: https://github.com/yyccbb/FYP_LLMTSC
- Qwen3.5-0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Transformers Qwen3.5: https://huggingface.co/docs/transformers/model_doc/qwen3_5
- 文本模型与加载映射接口（5.2.0）: https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- TRL 0.29.0 GRPO: https://huggingface.co/docs/trl/v0.29.0/grpo_trainer
- CityFlow: https://github.com/cityflow-project/CityFlow
