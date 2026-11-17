# Layout grid
- A: traffic context x=40..320, y=200..750; topology and observed input above, actual visitation below.
- B: shared policy x=370..640, y=350..510. Candidate fan-out rail at x=680; update rail at x=650.
- C: training-only paired evaluator x=700..1450, y=200..800. Candidate, first action, three-step continuation, then return. Three representative rows use y=350/500/650.
- D: return pooling x=1540..1960,y=320..440; advantage y=490..595; GRPO y=630..755.
- Return merge rails x=1465/1490/1515, kept outside computation boxes.
- Snapshot bypass y=180; parameter-update bypass y=830; legend y=894..942.
- Forbidden crossing zones: stage headings, context header text, cards, formula, legend.
