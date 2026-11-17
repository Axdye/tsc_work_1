# Diagram brief

## Goal
Editable paper overview of the current Paired-CF-GRPO implementation, for a full-width figure in a machine-learning/traffic-control paper. English figure labels; Chinese explanatory caption supplied separately. No experimental results are drawn.

## Sources and traceability
| Source | Role | Visible evidence |
|---|---|---|
| docs/METHOD.md | Scientific content and terminology | Four stages; same snapshot/current background; old-policy continuation; same-phase pooling; group-relative credit; LoRA update |
| tsc_experiments/credit.py | Operation/data dependencies | Candidate group and rollout, not four guaranteed distinct phases; representative rows 1,2,G |
| tsc_experiments/trainer.py | Update semantics | One response advantage shared by all reply tokens, clipped loss plus KL |
| configs/credit_main.json | Defaults | Qwen3.5-0.8B, G=4, H=3, M=1 |
| academic-figures-drawer style contract | Style/layout | Landscape, semantic palette, editable primitives, contribution accent |

## Must communicate
- A single parameter-shared LLM is invoked for multiple junctions.
- ξt includes engine and Python traffic state.
- Only the target junction's current action changes within a paired comparison.
- Future actions react separately to each branch; old policy is frozen during collection.
- Return pooling and group advantages train LoRA; no learned critic is shown.
- The actual visitation path executes sampled policy actions, never rollout argmax.

## Scientific assumptions / exclusions
- The 4×4 grid is the synthetic-training road topology as a schematic, not a measured congestion map.
- Rows 1,2,G summarize G sampled responses; they are not three different networks or model copies.
- The timeline shows H=3 for the default setting; no new fixed-future-action assumption is introduced.
- Topology edges are roads, not algorithmic data-flow arrows.
- No claims about SOTA, numerical traffic improvements, reasoning-span credit, or fast/slow roles.
- Multiple backgrounds M are summarized by the default badge rather than an additional nested loop.

## Edges
| Relation | Route / direction | Semantics |
|---|---|---|
| CityFlow → observation → shared policy | Left local chain | Current actor input |
| CityFlow → paired context | Dashed top bypass | Save/restore engine+Python snapshot |
| policy → candidate rows | Three fan-out ports | Sample G token responses |
| replies → first joint actions → rollout states → Z | Main left-to-right rows | Parse, execute intervention, allow future policy reaction |
| {Z} → pool → advantage → GRPO | Right merge and downward chain | Traffic scoring and standard update |
| GRPO → policy | Dashed bottom bypass | Parameter update after a complete collection window |
| policy → advance environment → CityFlow | Separate green left loop | Sampled actual trajectory; no branch selection |

## Export choice
No draw.io desktop CLI is installed. XML is the authoritative geometry/text source; a small primitive renderer reads the XML to derive SVG/PDF/PNG. All nodes/edges remain editable in draw.io; exports are not screenshots of an editor.
