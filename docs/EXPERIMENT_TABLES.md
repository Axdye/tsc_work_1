# 第一部分毕业设计：最终实验表格与填写规范

日期：2026-11-17。对应 `configs/credit_suite.json`。所有学习方法统一 Qwen3.5-0.8B、LoRA rank=8。

**这是最终报告的表格模板，不是已经取得的学习结果。** “待跑”表示对应完整实验没有执行；目前已有的单 seed CPU 检查结果单独列在文末，不填充为正式多 seed 结果。

## 统计与公平性

- 训练：Synthetic 4×4、8000 辆需求场景；测试：Jinan1/2/3、Hangzhou1/2。
- 评测：4 相位；30s 控制间隔包含切换时 5s 清空；3600s；laneChange=True；固定终点、不排空。
- 学习方法：3 个训练 seed × 3 个评测环境 seed。先对每个训练实例的评测 seed 求均值，再对 3 个训练 seed 报 mean±sample SD；不是 9 个独立训练实例。
- 传统控制器与未经交通训练的基座：报告 3 个评测环境 seed 的 mean±sample SD。其 SD 与学习方法 SD 对应的随机性来源不同，在表注说明。
- 训练总仿真预算上限 200000 物理秒，包含 warmup、分支和已部署训练轨迹；完整窗口前预算不足即停止，填实际开销。最终使用 final checkpoint。
- 同预算不等于同 token、更新次数或访问状态数；成本表和学习曲线同时给出。H/M 改变时尤其不能隐藏新增分支开销。
- ATT/AWT 单位秒；AQL 是全网进口排队车辆总数的时间均值。三项均越低越好。使用已有统一实现，不能换成原生 Engine ATT。
- 不从其他论文复制不同基座、不同初始化或不同信号时长的分数。`joint`、`trajectory` 为本项目控制对照，不称为官方 JS-GRPO/Traffic-R1 逐项复现。

## 表 1：主实验 ATT（↓）

回答：同一基座与可比较的交通评测协议下，配对反事实训练能否改善旅行时间？

| 方法 | Jinan1 | Jinan2 | Jinan3 | Hangzhou1 | Hangzhou2 |
|---|---:|---:|---:|---:|---:|
| FixedTime | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| MaxPressure | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Qwen3.5-0.8B，无交通 RL | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Trajectory-GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| RTG-GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Joint-GRPO，JS 风格 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Paired-CF-GRPO，本方法 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

每个单元格填写 `mean ± SD`。方法命名固定：Trajectory 使用窗口总交通收益均匀分配；RTG 使用窗口内 return-to-go；Joint 使用联合候选分支回报按相位投影；Paired 在目标路口候选间共享其他路口当前动作。

## 表 2：主实验 AWT（↓）

回答：旅行时间的改善是否伴随等待改善，还是转移了等待负担？

| 方法 | Jinan1 | Jinan2 | Jinan3 | Hangzhou1 | Hangzhou2 |
|---|---:|---:|---:|---:|---:|
| FixedTime | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| MaxPressure | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Qwen3.5-0.8B，无交通 RL | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Trajectory-GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| RTG-GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Joint-GRPO，JS 风格 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Paired-CF-GRPO，本方法 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

## 表 3：主实验 AQL（↓）

回答：是否降低网络排队，而不是只优化某个终点统计量？

| 方法 | Jinan1 | Jinan2 | Jinan3 | Hangzhou1 | Hangzhou2 |
|---|---:|---:|---:|---:|---:|
| FixedTime | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| MaxPressure | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Qwen3.5-0.8B，无交通 RL | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Trajectory-GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| RTG-GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Joint-GRPO，JS 风格 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Paired-CF-GRPO，本方法 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

ATT/AWT/AQL 可以合并成一张宽表，每数据集下分三列；这里拆开方便填写和阅读。正文若篇幅有限，可放 ATT 主表和精简消融，AWT/AQL 完整表放附录，不能只展示有利指标。

## 表 4：机制消融

主配置：paired、H=3、G=4、M=1、每状态 2 个目标路口、LLM 续跑、全局排队奖励。

| 变体 | 唯一主要变化 | Jinan2 ATT | Jinan2 AWT | Jinan2 AQL | Hangzhou1 ATT | Hangzhou1 AWT | Hangzhou1 AQL |
|---|---|---:|---:|---:|---:|---:|---:|
| 完整方法 `paired` | 基准配置 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| `unpaired` | 每个目标候选独立采样其他路口当前动作 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| `paired-rule` | 分支未来由 MaxPressure 续跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| `paired-local` | 只评价目标路口队列 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

`paired` vs `unpaired` 是最关键消融。不能把取消配对与取消多步、换奖励、换模型同时实施，再声称收益来自配对。

默认完整套件会对五个场景都测。这里用济南/杭州各一个场景做紧凑展示，完整结果保留；如需控制实验量，应在运行前固定场景选择，不按结果挑最好看的场景。

## 表 5：时间范围与背景数敏感性

| 变体 | H（包含第一步） | M | 分支物理时域，通常值 | Jinan2 ATT | Jinan2 AWT | Jinan2 AQL | Hangzhou1 ATT | Hangzhou1 AWT | Hangzhou1 AQL |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `paired-h1` | 1 | 1 | 30s | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| `paired` | 3 | 1 | 90s | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| `paired-h6` | 6 | 1 | 180s | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| `paired-m2` | 3 | 2 | 每个背景 90s | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

接近 3600s 终点时会截断到剩余完整控制步，不能越过评测时域。固定总预算下更大 H/M 会减少可访问状态/窗口数，因此必须与成本表一起解释。等窗口数结果可作补充，不替换固定预算主比较。

## 表 6：信用估计质量

运行 `diagnose`，在同一冻结 checkpoint、同一批交通状态上重复估计。高预算参考独立采样，不复用低预算样本。

| 估计器 | 相位对收益差 MSE ↓ | 重复估计差值方差 ↓ | 与高预算参考的符号一致率 ↑ | 每次背景数 M | 参考背景数 |
|---|---:|---:|---:|---:|---:|
| Unpaired | 待跑 | 待跑 | 待跑 | 1 | 32 |
| Paired | 待跑 | 待跑 | 待跑 | 1 | 32 |

表注：参考是有限样本均值，不是真实因果 oracle。排除参考几乎平局的相位对再统计符号一致率，阈值为代码中 1e-6。每组枚举四相位、共有六个相位对，不能把所有相位对当独立交通状态。至少同时报告独立状态数、重复次数、checkpoint、seed 和总诊断预算。

建议分两次报告：同一未交通训练基座上的估计器比较；同一选定训练 checkpoint 上的估计器比较。不要用“paired 训练后的策略”与“unpaired 训练后的策略”的估计方差直接证明配对机制，因为策略分布本身不同。

## 表 7：训练成本与部署表现

| 方法 | 实际训练仿真秒数 | 训练窗口数 | Optimizer 更新数 | 训练生成 token 数 | 训练 wall time（含验证） | 五场景宏平均 ATT | 测试非法回复率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trajectory-GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| RTG-GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Joint-GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Unpaired-CF-GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| Paired-CF-GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

若做统一部署开销对照，可补一张表列全部 LLM 方法的评测生成 token/决策、总评测 wall time、非法率和剩余车辆数。当前代码记录的是整段评测 wall time，包含仿真与记录，**不是纯 LLM GPU 推理耗时或单次推理 p95**；这些尚未单独计时，不能从现有 wall time 字段改名得到。

五场景宏平均 ATT = 五个场景 ATT 等权平均，只作辅助汇总，不是把所有车辆混起来的加权 ATT。收益相对某基线时，先逐场景算 `(baseline-ours)/baseline ×100%` 再平均，并明确基线名称，不能用原始 ATT 差直接跨场景排名。

### 配置推导的每窗口开销，非实测结果

| 变体 | 常规窗口全部仿真工作（物理秒） |
|---|---:|
| `trajectory` / `rtg`（8 步窗口） | 960 |
| `joint`（2 步采集、H=3） | 780 |
| `paired` / `unpaired` / `paired-rule` / `paired-local` | 1500 |
| `paired-h1` | 540 |
| `paired-h6` / `paired-m2` | 2940 |

以上包括分支与部署轨迹，单独 warmup 为每 episode 180s。接近 episode 终点或窗口缩短时实际值改变，正式成本必须填日志中的实际值。

## 表 8：已完成的 CPU 链路检查，单独展示

数据 Jinan2、输入 seed=0 / Engine seed=44、3600s；n=1，不报告 SD，不表示学习方法主结果已经完成。

| 方法 | ATT (s) ↓ | AWT (s) ↓ | AQL（全网辆）↓ | 状态 |
|---|---:|---:|---:|---|
| FixedTime | 441.19 | 66.72 | 294.14 | 已运行，原始记录审计通过 |
| MaxPressure | 273.20 | 38.25 | 106.58 | 已运行，原始记录审计通过 |

来源：`outputs/credit-validation-report/summary.json`。完整精度与边界见 `CREDIT_VALIDATION.md`。

## 配套图与运行入口

| 图 | 内容 | 当前支持情况 |
|---|---|---|
| 主指标图 | 各场景 ATT/AWT/AQL、mean±SD | `report` 已支持 |
| 学习曲线 | 验证 ATT/AWT/AQL 对实际训练仿真秒数 | `report` 已支持；每个 seed 单独画线 |
| 训练诊断图 | KL、loss、零优势组比例 | `report` 已支持 |
| 排队/等待时间曲线 | 独立闭环测试的网络队列与等待随时间变化 | `report` 已支持 |
| 信用质量图 | 差值 MSE、方差和符号一致率 | `diagnose` + `report` 已支持 |
| 路口拥堵动画 | 相同交通输入、相同色标的并排路口队列与相位 | `animate` 已支持 |

```bash
# 正式实验矩阵；开发机离线时先手动准备代码、vendor、wheelhouse 和本地模型目录。
python scripts/tsc_experiment.py suite --spec configs/credit_suite.json \
  --model /data/models/Qwen3.5-0.8B --output outputs/credit-main \
  --gpus 0 1 2 3 4 5 6 7 --execute

# 运行后汇总：当前自动导出为长表，并产生上述图；本文宽表按长表填写。
python scripts/tsc_experiment.py report --root outputs/credit-main \
  --output outputs/credit-main-report
```

结果文件：`summary.csv/json`、`runs.csv`、`RESULTS.md`、`training-costs.json`，以及诊断的 `credit_quality.json`。本文模板未声称所有宽表均已自动排版。

最小完成集：表 1–5、表 7，加学习曲线和拥堵案例。若要解释“为什么是信用分配在起作用”，表 6 优先级很高。表 8 仅作为实现核验，不替代正式实验。
