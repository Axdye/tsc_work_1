# TSC research workspace

本地独立 Git 仓库；2026-11-02 建立。

## 开发机快速开始

```bash
git clone git@github.com:Axdye/tsc_work_1.git
cd tsc_work_1
python3 scripts/bootstrap.py --only CityFlow LLMTSCS CoLLMLight --build-deps
```

上游源码/数据放在不入 Git 的 `vendor/`。接着按下方运行手册创建 Python 环境、安装 CUDA PyTorch、
编译 CityFlow，并先跑 `credit_smoke.json`；模型权重和本机实验输出不在仓库中。

**2026-11-17：新增在线反事实 GRPO 实验套件（用户自行运行 GPU 训练）。**

- [训练、baseline、消融、评测、绘图与拥堵 GIF 运行手册](docs/CREDIT_EXPERIMENTS.md)
- [TSC 定义、相关工作脉络、当前工作与下一篇快慢思考定位](docs/CREDIT_RESEARCH_STORY.md)
- [实际验证范围与结果](docs/CREDIT_VALIDATION.md)
- [第一篇工作代码覆盖、2026-11-18 修正与待验事项](docs/FIRST_WORK_READINESS.md)
- [方法说明](docs/METHOD.md)

新入口 `python scripts/tsc_experiment.py --help`。默认 Qwen3.5-0.8B＋LoRA，
单卡单作业、8 卡并行方法与 seed；与下方历史固定状态 GRPO 原型分开。

**2026-11-15 更新：用户已授权 CPU baseline 执行。** 已新增并运行 LLMLight 官方代码轻量适配器，
五个完整一小时的济南/杭州车流、FixedTime 和 MaxPressure，统一 ATT/AWT/AQL；
详见 [setup 来源、协议和实测对照](docs/CPU_SETUP_AUDIT.md)。LLM 训练仍未执行。
[DGLight / LLMLight / Traffic-R1 / CoLLMLight 比较](docs/FOUR_LLM_METHODS.md)。

**当前范围（2026-11-03）：阅读与研究设计，并编写最小 LoRA＋GRPO 实验代码；用户自行部署调试。**
以下为 11 月 3 日的历史范围；11 月 15 日已按新授权执行本地 CPU 仿真及其依赖安装，未连接服务器或执行 LLM 训练。
此前按原请求生成的复现脚本和运行产物保留为历史，不表示已跑通新的训练入口。
重点是 CityFlow 中的 RL/LLM 与快慢控制，另保留 LLM-Assisted Light 原生 TransSimHub 源码。

当前阅读入口：

- [最小 CityFlow＋LoRA＋GRPO 代码与运行说明](docs/MINIMAL_GRPO.md)（新增，未运行验证）
- [扩展调研：11 篇 RL/LLM 正文与源码价值](docs/RESEARCH_EXPANSION.md)
- [总体总结](docs/SUMMARY.md)
- [论文与代码讲解](docs/RL_READING.md)
- [代码导航与阅读顺序](docs/CODE_INDEX.md)
- [候选 idea 与最近邻边界](docs/IDEAS.md)
- [当前状态](docs/STATUS.md)

## 阅读顺序

- [全文与代码对照](docs/RL_READING.md)：重点机制、奖励、信用分配、实现差异及阅读边界。
- [候选 idea 和最小实验](docs/IDEAS.md)：已有工作覆盖了什么、还需要验证什么。
- [当前状态](docs/STATUS.md)：本轮范围、阅读边界及此前产物。

`vendor/` 保存官方源码（部分通过 Git，部分因连接失败改用 GitHub 官方源码归档）；
`sources.lock.json` 保存 Git commit 或归档 SHA256 和源码文件哈希。
归档 hash 锁定下载的字节，但分支 URL 本身可变，不能伪称已经验证成对应 commit。
上游源码不复制进本仓库的 Git 历史；原许可证保留在 vendor 内。
论文原始 PDF/TXT 放在 `papers/`，来源与版本记录在 `papers/sources.json`。

**没有官方训练代码的 Traffic-R1/OracleTSC 不会被包装成已完成训练复现。**

## 历史产物

原请求阶段生成的复现脚本、tests/ 及 [复现协议](docs/REPRODUCTION.md) 仅保留归档，当前停止执行。
此前直接 Engine 的 CPU 诊断不是论文指标复现，不用于支持本轮研究判断。
