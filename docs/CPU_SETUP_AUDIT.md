# CPU baseline 与论文 setup 对齐（2026-11-15）

## 仓库原来是什么设置

这不是单一论文的完整复现仓库，而是多个上游源码加本项目适配器。

| 入口 | 来源与用途 | 能否直接作为论文复现 |
|---|---|---|
| `scripts/cityflow_diagnostic.py` / `reproduce.py cityflow` | 本项目自写诊断；LLMTSCS 数据；30s 固定决策、5s phase 0 包含其中；关闭换道；可截流并排空 | 不能：压力公式、车道集合、采样与 ATT 口径都不同 |
| `scripts/minimal_cityflow_grpo.py` | 本项目最小 GRPO 原型；训练组织参考 DGLight；默认数据为 CoLLMLight Synthetic 4×4；奖励由固定规则续跑的快照分支生成 | 不能称为 DGLight 或 Traffic-R1 复现；无在线 LLM 轨迹更新、无 DQN critic |
| `vendor/LLMTSCS` | LLMLight 官方公开环境、传统/RL 控制器、统计代码 | 本次 CPU 对齐的主要代码依据 |
| `vendor/CoLLMLight` | CoLLMLight 官方环境与控制/采样/PPO 代码 | 保留独立协议；不默认等同 Traffic-R1 |
| `scripts/cpu_upstream_baselines.py` | 本次新增的轻量官方代码适配器 | 直接使用 LLMLight 环境、FixedtimeAgent、MaxPressureAgent 和原指标语句；差异详见下文 |

## 论文协议对照

| 工作 | 环境、动作、时域 | 训练/测试关系 | 主指标 |
|---|---|---|---|
| LLMLight | CityFlow；4 相位；常规场景 3600s；正文写 30s 绿灯 + 3s 黄灯 + 2s 全红 | 有目标数据训练及跨场景泛化，不能全部视作同一种 zero-shot | ATT/AWT/AQL |
| DGLight | CityFlow；相同五个济南/杭州场景；正文写 30+3+2；代码/配置使用 30s action interval | CoLight critic 在 Hangzhou1 训练；LLM 继承 LightGPT，之后进行 critic-guided GRPO | ATT/AWT/AQL |
| Traffic-R1 | CityFlow；4 相位；3600s；正文写 15s 绿灯 + 3s 黄灯 + 2s 红灯 | 专家动作阶段 + 合成 4×4 在线探索；测试真实场景零样本 | 常规任务 ATT/AWT；另有事件指标 |
| CoLLMLight 最终版 | CityFlow；4 相位；3600s；正文写 30+3+2 | Syn-Train：4×4、8000 辆；真实场景 zero-shot。论文 Jinan 是 4365 辆，对应 LLMLight Jinan2，而不是 Jinan1 | ATT/AWT/AQL；另有推理延迟和 token 数 |

重要：LLMLight/CoLLMLight 正文的 30+3+2 与公开代码并非字面一致。
LLMTSCS 的 `MIN_ACTION_TIME=30`、`YELLOW_TIME=5`、`ALL_RED_TIME=0`，
切换才运行 phase 0，且 5s 包含于 30s 决策步，保持相位不插入 phase 0。
FixedTime 依据上游 phase duration 计数选择下一相位，而不是本项目诊断脚本的每 30s 强制轮换。
因此本次协议名为 **llmlight-public-code-3600s-v1**，对齐公开代码；不声称字面实现正文的 35s 周期。

## 本次固定评测实现

- 不改 vendor 的环境、agent、压力公式和指标语句。
- 从 `my_utils.py` 用 AST 仅加载环境需要的 `load_json`、`calculate_road_length`；从 `config.py` 读取原配置字面量。
  用独立包命名空间加载上游模块，避免无关的 TensorFlow、torch、wandb 全量导入；没有伪造这些依赖。
- 从 `OneLine.train` 直接执行 `vehicle_travel_times` 到 `results` 的原指标语句。
- 交通引擎沿用已构建 CityFlow；`laneChange=True`、1s 物理步、120 个 30s 决策步、3600s 固定终点、不排空、不截取车流。
- 只关闭 replay 文件输出，保留原始道路、流量文件和全部动作/指标记录。
- 先设置 NumPy seed，再保留上游 `randint(0,100)` 的 Engine seed 生成方式；两种 seed 都记录。NumPy seed=0 对应 Engine seed=44。
- Python 3.9.6，NumPy 1.26.4，Pandas 2.2.3，依赖安装在 `.build/cpu-deps`，不修改系统环境。
- 现有 CityFlow portability build 使用 pybind11 2.10.4，动力学代码未改；源代码和扩展文件 SHA256 保存在 manifest。

指标的操作定义：

1. **ATT**：把每辆车在各受控路口进口车道的进入/离开时间差相加，再对记录到的车辆求平均；
   未离开记录用 3600s 补齐，过滤名称含 shadow 的车辆。不是完整行程排空后的 Engine ATT。
2. **AWT**：每个决策步结束时，对上游 `waiting_vehicle_list` 中 `time` 求均值，无等待记录时取 0；
   再平均 120 个采样值。该列表由上游每秒速度 <0.1、drivable 变化等逻辑维护。
   不是“每辆完成车辆一生的累计等待时间”的平均。
3. **AQL**：每个决策步结束时，汇总全部路口全部进口车道的排队车辆数，再平均 120 个采样值；
   不除路口数。论文中出现的“per intersection”措辞不应覆盖实际代码的网络总量。

保留 `remaining_vehicles_including_waiting` 和原生 `engine_ATT_at_horizon` 作为诊断字段，不能替换主表 ATT。

## 运行

```bash
cd /path/to/tsc
UV_CACHE_DIR=.build/uv-cache uv pip install --python /usr/bin/python3 \
  --target .build/cpu-deps numpy==1.26.4 pandas==2.2.3
/usr/bin/python3 scripts/cpu_upstream_baselines.py \
  --seeds 0 --output outputs/cpu-upstream-suite-20261115
```

`/usr/bin/python3` 是本机 ABI 匹配的 Python；服务器应使用自身已安装 CityFlow 的 Python。
可用 `--datasets jinan2 hangzhou1` 选择 CoLLMLight 的两个较小真实测试场景。
默认运行 LLMLight 的 Jinan1/2/3、Hangzhou1/2 和两个控制器；无 GPU、无训练、无外部模型调用。
每次输出目录须为空，失败记录保留，不覆盖。

## 来源

- LLMLight §4.1–4.2、Table 2：https://arxiv.org/html/2312.16044
- DGLight §5.1、Table 2–6：https://arxiv.org/html/2604.25259v1
- Traffic-R1 §5.1：https://aclanthology.org/2026.acl-long.995.pdf
- CoLLMLight 最终版 §4.1、附录 A.3：https://openreview.net/pdf/07b5ef2f45a88fefa5f4737f9eabfa0efb0a0ea0.pdf
- 上游代码：`vendor/LLMTSCS/utils/{config,cityflow_env,oneline}.py`、`models/{fixedtime_agent,maxpressure_agent}.py`

## 结果状态

完整结果见 `outputs/cpu-upstream-suite-20261115/`。单 seed 是协议核对，不代表随机性或跨场景稳健性已验证。

## 实测与 LLMLight Table 2 对照

2026-11-15，五个车流、两个控制器、NumPy seed=0 / Engine seed=44。每项显示“论文 / 实测”，均保留两位小数。

| 数据集 | 方法 | ATT 秒 | AWT 秒 | AQL 辆 |
|---|---|---:|---:|---:|
| jinan1 | fixedtime | 481.79 / 481.79 | 70.99 / 70.99 | 491.03 / 491.03 |
| jinan1 | maxpressure | 281.58 / 282.58 | 44.53 / 44.53 | 170.71 / 170.71 |
| jinan2 | fixedtime | 441.19 / 441.19 | 66.72 / 66.72 | 294.14 / 294.14 |
| jinan2 | maxpressure | 273.20 / 273.20 | 38.25 / 38.25 | 106.58 / 106.58 |
| jinan3 | fixedtime | 450.11 / 450.11 | 69.19 / 69.19 | 394.34 / 394.34 |
| jinan3 | maxpressure | 265.75 / 265.75 | 40.20 / 40.20 | 133.90 / 133.90 |
| hangzhou1 | fixedtime | 616.02 / 616.02 | 73.99 / 73.99 | 301.33 / 301.32 |
| hangzhou1 | maxpressure | 325.33 / 325.33 | 49.60 / 49.60 | 68.99 / 68.99 |
| hangzhou2 | fixedtime | 486.72 / 486.77 | 72.80 / 72.80 | 425.15 / 425.23 |
| hangzhou2 | maxpressure | 347.74 / 347.74 | 70.58 / 70.58 | 215.53 / 215.53 |

全部 10 次运行已从保存的 arrivals.json 和 decisions.jsonl 独立复算，ATT/AWT/AQL 均与输出一致（绝对容差 1e-9）。
济南1 MaxPressure ATT 实测 282.58、表中 281.58，相差约 1.00s；杭州2 FixedTime ATT/AQL 分别相差约 0.05s/0.08 辆。没有人为修正或调参追表；论文历史版本/统计细节/引擎差异未完全归因。
这表明公开代码协议与历史表格高度接近，但不能声称全部逐项精确复现。其他论文的 baseline 数字不可直接替换本表。
济南1 的预检查与完整套件即使 Engine seed 相同，ATT 也出现约 1e-3s 以内差异；固定随机 seed 不等于已证明引擎逐位确定性。
完整 CSV、原始日志、源码/数据哈希、逐项差值见 outputs/cpu-upstream-suite-20261115/。
