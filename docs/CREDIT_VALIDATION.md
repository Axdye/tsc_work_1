# 本轮代码验证记录

> 历史记录。当前工作二代码与复现入口见 [运行手册](CREDIT_EXPERIMENTS.md)。

日期：2026-11-18。本文件只记录实际执行的验证，不把待运行命令当成功结果。

## 已执行

- 新实验模块与 CLI 的 Python 语法检查、ruff 未使用/未定义变量检查。
- `test_credit_experiments.py`：五种采集方式的预算、主轨迹前进与分支恢复、终点截断、严格相位解析、同相位文字共享评分、非法格式惩罚、已知背景干扰的配对比较。
- `test_credit_cityflow.py`：真实 CityFlow 快照恢复引擎与 Python 状态；等待计时、相位、车辆到离记录一致；适配器一步和官方 `env.step` 一致；短程指标独立复算。
- `--scripted` 真实引擎 paired 冒烟：1 个窗口，180s warmup 后部署到 210s；实际仿真工作 570s，其中分支/部署窗口 390s；没有模型或 optimizer 更新。
- 其他四种采集方式也分别完成真实引擎单窗口 scripted 检查：unpaired/joint 共 570s 仿真工作，trajectory/RTG 共 300s；每个已部署轨迹都只前进至 210s，分支没有污染部署时间。
- `diagnose --scripted` 小规模文件链路检查。随机控制器结果不用于支持配对方法有效。
- 主套件计划生成：525 个作业，**未启动 GPU 作业**。
- 汇总统计测试：先平均环境 seed、再统计训练 seed；拒绝重复 checkpoint 和混合时长协议。
- 济南2两个传统基线各 3600s 闭环仿真及 ATT/AWT/AQL 原始记录审计。
- 指标 PNG/PDF、并排拥堵 GIF/末帧 PNG 导出与人工图像检查。

2026-11-18 修正后新增检查：配对/未配对采集器与诊断的实际生成数量；合法交通分差与格式惩罚的区分；无效配置和非有限参数提前失败；显式离线配置解析。

`test_credit_*.py` 合并检查：**14 项，13 项通过，1 项因缺 torch 跳过**。真实引擎检查使用 `TSC_TEST_CITYFLOW=1`，绘图检查使用本地 `.build/plot-deps`。跳过项不计为通过。

修正后的 CityFlow 构建脚本已在本机完成增量构建，显式 `GIT_SUBMODULE=OFF`，没有下载依赖。这验证的是本机 macOS/Python 3.9 构建，不代表目标 Linux/CUDA 环境已验。

修正后再次完成五种方式各一个真实引擎 scripted 窗口：paired/unpaired/joint 共 570s 仿真工作，trajectory/RTG 共 300s，主轨迹均至 210s。训练生成调用按回复数统计分别为 164/212/208/64/64；这些是脚本策略调用计数，不是实际 LLM token 成本。套件计划仍为 525 个作业，未执行。记录位于 `outputs/credit-review-20261118-afj_rsd8/`，不进 Git。

新增训练信号图的导出由临时合成 fixture 检查，fixture 不是训练数据或论文结果。

## 新入口 CPU baseline 数值

数据：Jinan2；输入 seed=0，沿用上游 NumPy draw 得到 Engine seed=44。四相位、30s 间隔包含切换时 5s 清空、laneChange=True、3600s 不排空。

| 方法 | ATT (s) | AWT (s) | AQL（全网辆） |
|---|---:|---:|---:|
| FixedTime | 441.1901489117984 | 66.71992629537749 | 294.14166666666665 |
| MaxPressure | 273.1993127147766 | 38.248717948276514 | 106.575 |

两次 `audit` 的三项误差均为 0；显示到两位小数与既有 LLMLight 公共代码核对相符。它们是环境/统计链路验证，不是新训练方法的效果。

本地结果（不进 Git 历史）：

```text
outputs/credit-validation/fixedtime-jinan2-s0/
outputs/credit-validation/maxpressure-jinan2-s0/
outputs/credit-validation-report/summary.csv
outputs/credit-validation-report/jinan2-metrics.png
outputs/credit-validation-report/jinan2-metrics.pdf
outputs/credit-validation-report/jinan2-congestion.gif
outputs/credit-validation-report/jinan2-congestion.png
```

## 尚未执行

- Qwen3.5-0.8B checkpoint 加载和 LoRA adapter 保存读取。
- CUDA forward/backward、实际梯度与 optimizer 更新、初始基座 KL 的端到端模型测试。
- 所有学习方法的 GPU 训练、多 seed 性能、消融效果或 SOTA 比较。
- CPU torch loss 数值单测：当前检查解释器没有 torch，标记 skipped；服务器装依赖后会自动执行。
- 快慢模型、建议缓存、多工具调度、工业真实闭环。

不支持中途训练恢复。GPU 依赖文件是目标 API 约束，不是已验证服务器 lockfile；服务器先运行 smoke 再扩大实验。
