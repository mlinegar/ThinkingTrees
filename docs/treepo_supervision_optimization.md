# TreePO: Tree-Based Supervision Optimization

Canonical entrypoint for the supervision-first optimization design.

The detailed formalization map lives in [treepo_preference_optimization.md](./treepo_preference_optimization.md), which now documents the supervision-first surface:

- `SupervisionDataset` as the primary stored training object
- scalar response judgments and comparative judgments as first-class records
- binary optimizer projections as derived views for DPO/reward-model style objectives

Use this file as the stable documentation entrypoint for the V2 cutover.

Teacher-first staged swap plan:
- [two_stage_teacher_first_swap_plan.md](./two_stage_teacher_first_swap_plan.md)
- [tree_neural_teacher_first_handoff_2026-03-23.md](./tree_neural_teacher_first_handoff_2026-03-23.md)

CPU tutorial ladder:
- [supervision_tutorials/README.md](./supervision_tutorials/README.md)
- [supervision_tutorials/17_decision_guide.md](./supervision_tutorials/17_decision_guide.md)
