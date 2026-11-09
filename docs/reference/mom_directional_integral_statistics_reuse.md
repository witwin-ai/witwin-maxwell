# Guarded automatic integral statistics

A directional RWG DAG projection bound to AutomaticDirectionalDAGPlan reuses the plan's existing CUDA physical-tree statistics after owner validation. Statistics array identity, version and gradient mode are part of the planner state and saved native backward chain. The default unguarded traversal remains unchanged. There is no second owner statistics array; borrowed planner storage still contributes to whole-resource accounting.

See docs/assessments/sie_directional_integral_statistics_reuse.md for terminal precision, mutation and all-near magnetic boundary evidence. No measured whole-resource performance or production qualification is claimed.
