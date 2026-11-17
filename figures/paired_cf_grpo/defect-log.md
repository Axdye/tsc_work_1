# Defect log

## Pass 0 — Plan review
- Status: 
- Open risks: 

## Screenshot review cycles

### Cycle 1
- Screenshot (canvas-only): 
- P0/P1 inventory: 
- Fixes and verification: 

### Cycle 2
- Screenshot (canvas-only): 
- P0/P1 inventory: 
- Fixes and verification: 

### Cycle 3
- Screenshot (canvas-only): 
- P0/P1 inventory: 
- Fixes and verification: 

## Red-team audit
- Text: 
- Arrows: 
- Boxes/overlap: 
- Spacing/layout: 
- Color/typography: 
- Icons/assets: 
- Semantics/regressions: 

## Self-score
| Dimension | Score /10 | Evidence |
|---|---:|---|
| Text readability | | |
| Arrow accuracy | | |
| Color coherence | | |
| Layout consistency | | |
| Style/spec match | | |
| **Total /50** | | |

## Remaining gaps
- 

## Preflight before first preview
- Initial automated pass: 4 FAILs (stage_a/stage_b/cityflow_title/candidate_head text estimate). Enlarged two stage heading bounds and shortened two labels; also shortened observation and frozen-policy annotation.
- Recheck preflight-1b.json: zero FAIL, 19 WARN.
- WARN review: three spacing findings combine unrelated panels with the road icon; per-family spacing remains regular. Timeline t cells are actual physical control states, not decorative tokens. Target i is the evaluated junction. Five legend dots encode the five documented semantic categories. Empty dashed cityflow/evaluator frames contain their explicit titles and children. Two road-icon density warnings count topology lines, not algorithmic arrows; fan-out and return-merge hotspots are legitimate dependency junctions and will be checked visually. No warning is silently removed or downgraded; these are accepted semantic exceptions pending screenshot review.
- No raster preview was made before zero-FAIL preflight.

## Screenshot review — cycle 1 (review-1.png)
Full canvas inspected in all nine zones. Found glyph failures in actual export, not just an estimator warning.
| Zone | Genuine findings / action | Severity |
|---|---|---|
| 1 Text | y1/a1, y2/a2, Z1/Z2, A_k/q_k and the vertical ellipsis render as missing-glyph boxes. Replace unsupported Unicode subscripts with explicit ASCII indices and a supported ellipsis. | P0 |
| 2 Arrows | Candidate return merge rails are traceable; snapshot and update bypasses do not cross boxes. Short return strokes are nearly hidden by adjacent Z labels; inspect again after glyph fix. | P2 |
| 3 Boxes | Coral joint-action cards are readable, but the evaluator frame leaves only 10px below the last candidate card. Increase internal bottom breathing room in a later geometry pass. | P2 |
| 4 Spacing | Narrow 12px gaps before the Z labels feel cramped; row connector gaps otherwise consistent. | P2 |
| 5 Color | Coral identifies first-step intervention; standard future states recede. No undesignated color. | Verified |
| 6 Typography | Legend markers and labels have visibly excessive separation in the first entry; align the legend family in next pass. Policy block is over-bold across all three lines. | P2 |
| 7 Composition | Z is not defined anywhere; add a return definition. Frozen-policy continuation should be explicitly named at its column. | P1 |
| 8 Icons | 4×4 target-junction icon is clear; target i is not a heatmap/data result. | Verified |
| 9 Coherence | Overall method is recognizable, but missing glyphs prevent paper use. | P0 |

Fix plan: ASCII mathematical indices throughout XML, supported ellipsis, explicit frozen-policy rollout column, define Z as discounted queue return, move update note below its feedback edge. Only real defects are counted; no synthetic findings are invented to meet a checklist quota.

## Screenshot review — cycle 2 (review-2.png)
Glyph boxes are gone. All nine zones rescanned; core semantic paths are intact.
| Zone | Finding | Severity / decision |
|---|---|---|
| 1 | ASCII indices are readable but visually code-like; replace with editable rich-text subscripts using ordinary glyphs, not unsupported Unicode characters. | P2, improve |
| 2 | Short strokes before Z look like minus signs rather than return-flow arrows. Make these short links arrowed and increase their gap. | P1, fix |
| 3 | Last candidate card still has only 10px to its container; the return explanation sits outside the proposed evaluator. Extend frame to include its explanation. | P2, improve |
| 4 | Legend swatches are followed by unequal visual whitespace because labels are centered in wide cells. Left-align all legend labels. | P2, improve |
| 5 | Palette is consistent; proposed first-step actions and future steps remain distinguishable. | Verified |
| 6 | Policy card uses bold on every line, flattening hierarchy. Keep first line bold and model/adapter descriptions regular. | P2, improve |
| 7 | Training-only scope is implicit in the dashed group; make it explicit in the stage heading. | P1, fix |
| 8 | Three representative candidate rows plus ellipsis correctly imply G, not three model copies. Four-step coverage is not claimed. | Verified |
| 9 | The diagram is coherent but can improve mathematical typography and endpoint precision. | P2, improve |

Cycle-1 fix verification: all missing-glyph labels fixed; frozen-policy continuation now explicit; Z return now defined; update annotation moved below feedback line. No P0 remains. Real findings are logged rather than manufacturing 15 defects.

## Screenshot review — cycle 3 (review-3.png and paper-scale.png)
Full-size and 1000px-wide exports inspected across all nine zones.
| Zone | Finding / verification | Severity |
|---|---|---|
| 1 | Rich subscripts now render; no missing glyphs. The small mathematical indices are necessarily secondary at reduced width. | P2 accepted |
| 2 | Return links now have visible arrowheads. Red group border and gray feedback path are only 10px apart, making two dashed lines look crowded at reduced width. | P2 fix |
| 3 | Evaluator now encloses the return definition; no bottom-card clipping. | Verified |
| 4 | Legend label gaps now aligned. Need more vertical room below the evaluator for feedback explanation and legend. | P2 fix |
| 5 | Five fill categories remain consistent. Gray dashed meaning is explained in accompanying caption/spec, not a separate inline line-style legend. | P2 accepted |
| 6 | Actor typography now has clear first-line emphasis; mathematical notation uses ordinary characters in editable sub/sup markup. | Verified |
| 7 | Training-only scope now explicit. The schematic shows one target junction; caption must state sequential reuse for selected targets. | Caption requirement |
| 8 | Target junction i and representative candidate/time icons remain semantically grounded. | Verified |
| 9 | Current figure is suitable as an overview; publication caption must distinguish control-step reward and final evaluation metrics. | Caption requirement |

Cycle-2 fix verification: code-like indices replaced by rich sub/sup text, pre-Z strokes replaced with arrows, legend aligned, actor hierarchy corrected, training-only label added. No P0/P1 remains. Final polish: lower the parameter-feedback route and legend to separate dashed baselines.

## Cycle-3 fix verification — review-4.png
The feedback bypass is now at y=830, 30px below the red evaluator boundary. The update explanation and legend occupy separate rows. The final full-canvas export was re-inspected; no text clipping, glyph loss, box collision or required-edge omission is visible. Final preflight adds one unrelated-family spacing warning (joint cards vs legend marker in the same heuristic column); accepted because these are different layout families, not a repeated component stack.

## Red-team audit — final full canvas and reduced-width export
Role switched from author to reviewer. Genuine residual limitations and verification outcomes are listed; no artificial defects are invented to inflate the finding count.
| # | Zone / check | Final observation | Outcome |
|---|---|---|---|
| 1 | Text / glyphs | Subscripts now use ordinary glyphs in HTML runs, no tofu boxes | Pass |
| 2 | Text / scale | Mathematical superscripts and legend are smaller than main labels at reduced width | P2 accepted; use full two-column PDF/SVG |
| 3 | Text / reward | Z is explicitly defined as a discounted network-queue return | Pass |
| 4 | Arrows / candidate fan | One actor feeds representative rows; no model-copy interpretation required | Pass |
| 5 | Arrows / returns | Three non-crossing merge rails feed same-phase pooling | Pass; mildly long routes are P2 |
| 6 | Arrows / actual state | Sampled policy action feeds environment advancement; never branches' argmax | Pass |
| 7 | Arrows / gradient | Dashed parameter update returns to the policy after collection | Pass |
| 8 | Boxes | No accidental overlap; empty-frame warnings refer to titled semantic containers | Accepted checker exception |
| 9 | Spacing | Parser mixes road-grid rows, computation cards and legend into a single row/column | Accepted checker exception; families visually regular |
| 10 | Palette | Each of five fills has an explicit legend meaning; no performance heatmap is fabricated | Pass |
| 11 | Typography | Actor headline and stage hierarchy preserved; Arial font embedded in PDF | Pass |
| 12 | Composition | State input is vertically staggered relative to actor; left region is less dense than evaluator | P2 accepted to preserve independent visitation loop |
| 13 | Icons | Grid denotes topology; target i is an intervention target, not measured congestion | Pass |
| 14 | Semantics / future | Only current background reused; frozen-policy rollout allows branch-specific future state/action responses | Pass; detailed wording in caption |
| 15 | Semantics / G | Three schematic rows are 1,2,G; G=4 does not guarantee four distinct phases | Pass; caption explicit |
| 16 | Semantics / M | Multiple-background loop not expanded; default M=1 declared | Deliberate approximation |
| 17 | Semantics / KL | Initial-base reference and token-mean details not separately expanded | Deliberate overview abstraction, caption explicit |
| 18 | Export provenance | XML renderer used instead of native draw.io CLI | Disclosed limitation; native editor wrap differences not independently tested |

## Self-score
| Dimension | Score /10 | Evidence and deduction |
|---|---:|---|
| Text readability | 8 | All full-size labels readable; tiny indices and legend require full-width placement |
| Arrow accuracy | 9 | Correct direction and separate loops; merge rail routes remain relatively long |
| Color coherence | 9 | Restrained semantic fills; dense coral header dominates slightly |
| Layout consistency | 8 | Regular branches; left state/actor regions intentionally have unequal density |
| Style/spec match | 9 | Native editable primitives, consistent strokes/type; renderer is not native editor |
| Total | 43/50 | No P0/P1 unresolved |

## Validation status and remaining approximations
- Structural validator: pass, zero errors and zero warnings.
- Visual preflight: zero FAIL, 20 reviewed WARN; not claimed warning-free. Warnings cover semantic grid/time/legend markers, titled empty containers, unrelated-family spacing, and genuine topology/fan/merge density.
- Four canvas-only renders reviewed, including a reduced-width raster; cycle-1/2/3 fixes have a final verification render.
- No experimental measurements or SOTA claims appear in the figure.
- All scientific structure/labels are editable; native draw.io editor wrapping was not separately tested. SVG/PDF/PNG are exports derived directly from XML geometry and content.
