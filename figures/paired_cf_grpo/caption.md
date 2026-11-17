# Figure caption

## English

**Overview of Paired-CF-GRPO for multi-intersection traffic signal control.** A single parameter-shared Qwen3.5-0.8B policy maps junction observations to textual responses and signal phases. For a selected target junction, the training-only evaluator restores the same simulator-and-Python snapshot and compares candidate actions while reusing the current actions of all other junctions. Each branch then continues for H control steps under the frozen behavior policy, which reacts to that branch's evolving state. Discounted returns from interval-averaged network queue costs are pooled for responses selecting the same phase and normalized within the candidate group. A clipped GRPO objective with initial-base-policy KL regularization updates the LoRA parameters; one advantage is shared by all tokens of a response. Separately, the visited trajectory advances using sampled policy actions, rather than a rollout argmax. Counterfactual evaluation is used during training only.

The three displayed rows represent candidates 1, 2, and G; the default is G=4 and candidate phases may repeat. The schematic focuses on one selected junction i and one current-action background (M=1); the evaluator is reused for the selected targets. Default H=3 includes the intervened first step and two later control steps. Solid arrows denote data/traffic flow; gray dashed arrows denote snapshot/parameter-control dependencies. The 4×4 network is a topology schematic, not measured congestion or an experimental result.

## 中文

**Paired-CF-GRPO 多路口信号控制方法概览。** 一个参数共享的 Qwen3.5-0.8B 根据各路口观测生成文字回复与信号相位。针对选定目标路口，训练评估器恢复相同的引擎与 Python 环境快照，复用其他路口的当前动作，只替换目标路口候选。各分支随后由冻结旧策略根据自身新状态进行 H 步续跑。控制间隔内平均网络队列成本形成折扣收益；选择相同相位的回复共享平均评分，组内标准化后用于带初始基座 KL 正则的 clipped GRPO 更新。完整回复的生成 token 共享一个优势。实际访问轨迹单独执行策略采样动作，不执行分支最优动作，反事实评估仅用于训练。

图中展示候选 1、2、G，默认 G=4，允许重复相位；只展开一个目标路口与一个当前动作背景（M=1）。默认 H=3 包括被替换的第一步与后续两步。灰色虚线表示快照或参数控制依赖，实线表示数据/交通流。4×4 网格只表示路网拓扑，没有表达任何拥堵实测结果。

## Exports and editability

- `overview.drawio`: authoritative, fully editable module/edge/text source.
- `overview.svg`: vector figure with editable text, derived from draw.io XML by `build_figure.py`.
- `overview.pdf`: vector export with embedded TrueType font.
- `overview.png`: 4000px-wide preview/export; `paper-scale.png` is the reduced-width inspection view.
- `build_figure.py`: generator plus limited XML-to-vector renderer, using Matplotlib.
- No draw.io desktop CLI was available; exports use the documented XML renderer. Native draw.io editor layout has not been separately screenshot-tested, so minor text-wrap differences may occur when editing. Geometry and all textual content are preserved in XML.
- The diagram uses the default H=3/M=1 layout; it does not expand the optional multiple-background loop or full optimizer internals.

For manuscript layout, use the PDF or SVG at full two-column width. Keep the frame/label proportions when resizing; mathematical indices and the legend are secondary annotations.
