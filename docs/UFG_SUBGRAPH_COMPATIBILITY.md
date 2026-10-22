# UFGTime fixed-roster compatibility evidence

This is preparation for the required recent pure-GNN comparison, not a performance result. The Stage57ChatTime source pilot continues unchanged from its immutable71371b1snapshot. No extra optimizer update, target label or GPU prediction was used here.

The official UFGTime effective forward and its pinned source hashes are unchanged. A new input-only wrapper presents the same maximum9-node physical history roster, target at0, node-wise origin MinMax minus0.5, and normalized zero fill for missing/padded slots. Fewer than9real nodes use explicit masked-zero virtual trailing roles. No future or label argument exists in the input packer. Outputs are inverse transformed to raw units without clipping.

Use official default hidden width128, not a reduced width selected to weaken the baseline. Its original frequency/node-sized parameters require distinct12→12and48→24models; the9slots denote query-roster roles, not persistent physical-node IDs. This is an explicit transfer adaptation, not strict reproduction of the original per-dataset paper experiment. UFG retains its own learned frequency graph and does not receive the external road adjacency. Padding and history encoding differences must be reported in any future comparison.

Seven isolated CPU checks passed in2.11seconds with CUDA disabled/OMP2/MKL2: exact official forward/raw inverse for2/3/9real nodes and both shapes, finite active-parameter gradients, auxiliary-history gradients, missing-input gradient exclusion, constant and negative histories, padding/hidden garbage invariance, checkpoint reload equality, causal history-only access and task-shape rejection. Old UFG normalization/complex warnings remain disclosed. No UFG performance score or ranking is established.

Only after the real ChatTime source evidence justifies target work should a full source-only UFG training/selection protocol and matched16/64support-query few-shot budgets be frozen. Existing old-dataset UFG results cannot stand in for the new fixed-six-target comparison. Honest results take priority over any requested ordering.
