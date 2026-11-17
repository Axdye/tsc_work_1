# 源码阅读导航

当前只读整理；下列入口不是“全部已运行”的清单。

| 目录 | 用途 | 首先读什么 | 需要分清的概念 |
|---|---|---|---|
| vendor/LLMTSCS | LLMLight、LightGPT 和传统 RL baseline | utils/config.py -> utils/cityflow_env.py -> models/colight_agent.py -> utils/llm_aft_trainer.py -> finetune/ | Q critic、SFT、ranking refinement 与在线 PPO 不同 |
| vendor/CoLLMLight | 协同推理、快决策、反事实数据与 PPO | models/CoLLMLightAgent.py -> framework/CoLLMlight.py -> framework/FTSample.py -> ppo_ft.py | 建议来源时刻、实际执行时刻、自报效用与交通增益 |
| vendor/HeraldLight | 预测辅助的相位/时长与双 LLM 纠错 | models/herald.py -> models/Llama.py -> utils/cityflow_env.py -> fintune/RankTrainer/RankTrainer.py | 语言 critic 不等于价值 critic；动作空间更细不等于推理更强 |
| vendor/LLM-Assisted-Light | 原生 TransSimHub 工具调用和特殊事件 | la_light/runtime.py -> la_light/env/ -> la_light/llm/tools.py -> la_light/llm/agent.py -> run_llm_tsc.py | 工具可能读取同一个 snapshot；当前参考策略为 MP |
| vendor/CityFlow | C++ 仿真器 | src/engine/engine.cpp -> src/engine/archive.cpp -> src/flow/flow.cpp | native ATT 的未完成车辆口径、快照的 RNG 与相位状态 |
| vendor/TransSimHub | LA-Light 的 SUMO 环境 | tshub/tshub_env/tshub_env.py -> base_sumo_env.py | API 等待的 wall time 不自动变成交通物理时间 |
| vendor/ProcessLight | 分步过程监督与 GRPO 扩展 | src/rltsc/databuild/ -> verify/ -> trainer/rewards.py -> trainer/advantage.py | 删步骤改变动作分布，不等于步骤改善了交通后果 |
| vendor/LibSignal | 通用交通 RL 环境 | README.md -> world/ -> agent/ | 它不是 OracleTSC 的官方实现 |
| vendor/GENEX-TSC | 周期级控制与 critic 引导语言训练 | config.py -> env.py -> agent.py -> Score_Train_LLM.py | TD3 连续配时；LLM 阶段是 Q 加权 MLE |
| vendor/VLALight | 后续视觉动作资料 | README.md -> VLALight/ | 不与另一团队同名论文混淆，本轮只归档 |

## 2026-11-02 扩展调研新增源码

对应论文、实验解释和静态疑点见 [扩展调研](RESEARCH_EXPANSION.md)。所有项目均未执行。

| 目录 | 最有价值的入口 | 环境与边界 |
|---|---|---|
| vendor/DynamicLight | utils/cityflow_env.py:751、models/dynamiclight.py:95 | CityFlow；按路口动作到期推进时间，联合相位/时长 |
| vendor/X-Light | onpolicy/algorithms/r_mappo/algorithm/sumo_nn.py:344、runner/shared/sumo_runner.py | SUMO；历史 o/a/r 的两层 Transformer，不是 Advanced-XLight |
| vendor/Unicorn | models/Unicorn.py、env/tls.py、runner_unicorn.py | SUMO；movement-phase 统一表示、mask、协同 critic |
| vendor/DataLight | memory/readme.md、models/datalight.py:130、models/attentionlightCQL.py | CityFlow 离线 RL；数据外置，ER/CQL 计算需核对 |
| vendor/RobustLight | inferences/rl_dynamic_state_filling.py、inferences/diffusion.py、utils/make_mask_noise.py | CityFlow；噪声/缺失恢复，作者独立仓库与聚合仓库版本未等同 |
| vendor/CFLight | CFLight_r.py、gan_cf.py、buffer.py | SUMO；反事实安全轨迹；README 主入口缺失 |
| vendor/iLLM-TSC | rl_llm_tsc.py、TSCAssistant/tsc_assistant.py:97、sb3_ppo.py | TransSimHub/SUMO；PPO 建议后 LLM 审查，不训练 LLM |
| vendor/VLMLight | vlm_tsc_en/vlmlight_decision.py、utils/tsc_agent/llm_agents.py、rl_tsc/ | TransSimHub；视觉/模式选择也有成本，与 VLALight 分开 |
| vendor/PromptGAT | trainer/sim2real_trainer.py、common/stat_utils.py、prompt_data/ChatGPT/ | CityFlow→SUMO；Grounded Action Transformation |
| vendor/DGLight | src/rl/critic/colight_q_rewarder.py、rollout/collector.py、training/grpo_runner.py | CityFlow；PyTorch critic/GRPO；实际用归一化 Q 分数 |

## CoLLMLight 五个最值得逐行检查的位置

1. `framework/FTSample.py::test`：从同一快照枚举动作，产生 Best_Signal。
2. `framework/FTSample.py::save_env/load_env`：Engine 与 Python wrapper 联合还原。
3. `ppo_ft.py::calculate_reactive_action_reward`：动作与标签相同才 +1。
4. `ppo_ft.py::calculate_advance_reasoning_reward`：used_atr 自报告、## 数量与动作奖励相乘。
5. `framework/CoLLMlight.py::Asynchronous_Reasoning_and_Acting` +
   `models/CoLLMLightAgent.py::advance_traffic_reasoning_prompt`：串行执行与 t-1 历史输入分别承担什么角色。

## 不能提供“官方代码讲解”的边界

- Traffic-R1：只有公开较早模型、模型卡、论文和 LLMLight 兼容评测建议，没有完整训练实现。
- OracleTSC：未找到可确认官方方法代码；只按全文与算法伪代码分析，不声称已审阅其实现。
- EvoSignal：当前官方页面说明接受后发布源码，故未用空介绍仓库充当代码。

来源 URL、归档校验和与已核对的 Git commit 见根目录 sources.lock.json。
GitHub 源码归档下载不含 .git 元数据；没有把 archive hash 虚称为 commit。
