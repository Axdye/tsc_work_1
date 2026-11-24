# TSC research workspace

## 工作二 · 多路口时空反事实 GRPO

工作二使用单个共享 Qwen3.5-0.8B + LoRA，结合真实轨迹优势与固定前瞻时域的反事实信用。统一入口为 `scripts/tsc_experiment.py`。方法、正式实验和开发机运行方式分别见[论文定稿](docs/METHOD.md)与[运行手册](docs/CREDIT_EXPERIMENTS.md)。

```bash
git clone git@github.com:Axdye/tsc_work_1.git
cd tsc_work_1
python scripts/bootstrap.py --only CityFlow LLMTSCS CoLLMLight --build-deps
python scripts/tsc_experiment.py --help
```

上游源码和交通数据存放于 `vendor/`，论文原文放在 `papers/`。归档来源及校验记录见 `sources.lock.json` 和 `papers/sources.json`。离线安装、模型路径与 GPU 作业调度见运行手册。

## 调研与复现资料

- [主流 LLM 方法横向对比](docs/FOUR_LLM_METHODS.md)
- [论文与代码阅读笔记](docs/RL_READING.md)和[扩展调研](docs/RESEARCH_EXPANSION.md)
- [已有规则基线的实测协议](docs/CPU_SETUP_AUDIT.md)
- [快慢协同、区域规划与交通工具的研究方向](docs/CREDIT_RESEARCH_STORY.md)
- [上游方法的原生入口与复现记录](docs/REPRODUCTION.md)
