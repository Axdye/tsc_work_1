# 统一复现协议

## 仓库与环境

已收集：LLMTSCS、CoLLMLight、HeraldLight、LLM-Assisted-Light、CityFlow、LibSignal、
GENEX-TSC、VLALight、TransSimHub。后两种视觉/周期方法只作为后续资料，首轮未声称精读或复现。
OracleTSC 无可确认官方实现；LibSignal 只能提供它所用的测试环境。Traffic-R1 仅公开较早权重和接入建议。

环境建议分开：

1. CPU 诊断：Python 3.9.6、CMake 4.1.2、CityFlow C++ 源码、pybind11 2.10.4。
   本机通过 CMake policy 兼容开关与较新 Python binding 编译；交通动力学源码未改。
   `python3 scripts/bootstrap.py --build-deps` 获取头文件，再执行 `build_cityflow.py`。
   正式 Linux 复现应优先使用各上游原生依赖，另记录此处 portability build 差异。
2. LLMLight/传统 RL：Linux，按 LLMTSCS README 的 TensorFlow 与 numpy 等版本准备独立环境。
3. CoLLMLight：Linux/CUDA，torch/transformers/TRL/vLLM 版本按上游；需模型权重和教师生成资产。
4. HeraldLight：独立 CUDA 环境，vLLM 版本与上一项不同；需要配置多个硬编码路径。
5. LA-Light：独立 Python>=3.10 环境，安装 SUMO 与 TransSimHub。
   将 `vendor/TransSimHub` 安装为 editable，并装上游需要的 openai/httpx/pyyaml/loguru。
   本阶段不需要 3D 画面时先不安装 `[all]` 的全部渲染依赖。

当前环境没有 CUDA GPU，不能在此声称完成 8B LLM PPO/SFT 训练。

## 官方入口派发

统一工具不会偷偷改上游算法、默认训练轮数或时间口径。

```bash
# RL baseline，原版训练入口（默认可能运行 100 轮）
python3 scripts/reproduce.py native --method presslight --python /env/cityflow/bin/python --dataset jinan
python3 scripts/reproduce.py native --method colight --python /env/cityflow/bin/python --dataset jinan
python3 scripts/reproduce.py native --method advanced-colight --python /env/cityflow/bin/python --dataset hangzhou

# CoLLMLight checkpoints：先自行启动文档指定的 OpenAI-compatible 推理服务
python3 scripts/reproduce.py native --method collmlight --python /env/collm/bin/python --dataset jinan -- --model_path /models/collm --port 23333
# 反事实采样，固定状态枚举相位并向后 rollout；需完整依赖
python3 scripts/reproduce.py native --method collmlight-sample --python /env/collm/bin/python
# 必须先配置 vendor/CoLLMLight/config/ppo_config.yaml 和数据/模型路径
python3 scripts/reproduce.py native --method collmlight-ppo --python /env/collm/bin/python

# LA-Light 保留自己的环境与事件配置
python3 scripts/reproduce.py native --method la-maxpressure --python /env/tshub/bin/python --scenario 4way --event special_vehicles_accident.yaml
python3 scripts/reproduce.py native --method la-light --python /env/tshub/bin/python --scenario 4way --event special_vehicles_accident.yaml
```

CPU 脚本的 `--seeds` 真正写入 Engine 配置；官方入口的 seed 不统一，launcher 明确保留原默认值，
**不能把 shell 的 PYTHONHASHSEED 当作覆盖学习器/仿真器 seed**。上游多进程和 seed 还需单独适配。
原版入口自身可能在子进程失败后仍返回 0，因此 launcher 的 process_exited 也不等于复现成功。

## 原版已发现的问题（未静默修改）

- CoLLMLight `ppo_ft.py`：32 个状态被转换为 64 条 ATR/RA 训练记录，但 PPO batch_size 配置为 32。
  与 TRL 0.9.2 的 batch 检查兼容性须在正式训练前验证；测试已直接调用该纯函数验证 2N 记录。
- 同文件 `evaluate_model` 给 ATR reward 传入 RA response；可能影响验证长度/效用评分。
- 同文件在更新前计算 validation score，却复制更新后的 checkpoint 为 best；选择时序需要修正并记录。
- `FTSample.py` 使用 eng.snapshot()+完整 wrapper deepcopy；只复制 Engine 不足以还原全部控制状态。
- CoLLMLight 当前评测控制流串行调用 ATR/RA；ATR 输入截取 t-1 历史，等价时序应与真正 wall-clock 异步分开验证。
- HeraldLight `run_dynamiclight.py` 有 `defaut=0` 拼写错误，注册为 blocked。
- HeraldLight 的 `run_llama.py` 默认开启 Hangzhou；增加 `-jinan` 并不会关闭 Hangzhou。
  它还内置训练/模型路径，不能假装通过统一 dataset 参数完成所有配置。
- CoLLMLight 的 PPO 使用 `Best_Signal` 标签奖励；不要将它误报为直接在线优化网络 ATT 的 PPO。

## 指标与时间

CPU 诊断默认 1s 物理步、30s 决策；切换时 5s phase-0 clearance 占用此区间，保持相位不付切换损失。
phase 0 的可通行连接由数据决定（常含右转）；不能把它称为所有方向全红。
这是独立 MP 变体：进口队列减可达出口车道平均队列，去掉重复 laneLink；不是原版 MaxPressure 精确克隆。
因此它的数字只能检验环境、指标和滞后风险，不能填进论文的原版复现栏。

交通需求在 H 秒截止，然后最多 drain D 秒；未排空时 all_vehicle_att=null。
CityFlow `get_average_travel_time()` 在未结束时包含仍在系统中车辆已经经历的时间，见 engine.cpp:682。
脚本分别记录 H 时剩余数、H 时 API ATT、排空后全车辆 ATT、网络排队车辆时间均值。
它们和 LLMLight/CoLLMLight 的 wrapper ATT/AWT/AQL 不能混用。

正式实验需要同流量、同相位、同过渡、同控制机会、同物理时间、独立验证/测试与匹配仿真预算。
多 seed 若数值完全相同，只能说明该场景/协议下结果重复，不能宣称完成了随机性鲁棒验证。
