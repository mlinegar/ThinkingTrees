/-!
# OPT Module: Oracle Preference Training

## Overview

This module formalizes the theory of **oracle-preserving summarization** for preference learning.
The key insight is that **local testable conditions** (L1, L2, L3) on a summarizer imply
**global training equivalence** for DPO, GRPO, and other preference learning methods.

## Main Results

| Theorem | File | Statement |
|---------|------|-----------|
| `multi_round_proper` | ExpectationTheory | L1+L2+L3 ⟹ E[dist(f*(Z), f*(x))] = 0 |
| `dpo_equivalence` | PreferenceLearning | Zero distortion ⟹ DPO loss equivalent |
| `grpo_equivalence` | PreferenceLearning | Zero distortion ⟹ GRPO-PL loss equivalent |
| `grpo_rl_equivalence` | PreferenceLearning | Zero distortion ⟹ GRPO-RL loss equivalent |
| `dpo_gap_bounded` | PreferenceBounds | Quantitative gap: |L(X) - L(Z)| ≤ L × Δ |
| `grpo_pl_gap_bounded` | PreferenceBounds | GRPO-PL quantitative gap |
| `grpo_rl_gap_bounded` | PreferenceBounds | GRPO-RL quantitative gap |
| `unified_preference_gap_bounded` | PreferenceBounds | Unified framework for all gap bounds |

## File Structure

```
OPT/
├── CoreDefinitions.lean      # Basic types: BinTree, Summarizer, Policy
├── OracleMeasurable.lean     # Oracle-measurable predicates (lightweight)
├── PreferenceNoise.lean      # Abstract preference noise models
├── SamplingModel.lean        # Generative model for preference datasets
├── TreeProperties.lean       # Tree operations and counting lemmas
├── LocalLaws.lean            # L1, L2, L3 local consistency conditions
├── PreservationTheorems.lean # Tree reduction preserves oracle
├── ExpectationTheory.lean    # Multi-round preservation (main CLT-style result)
├── GlobalAssumptions.lean    # Global A1, A2, A3 and derivations
├── PreferenceLearning.lean   # DPO, GRPO loss definitions and equivalence
├── PreferenceBounds.lean     # Quantitative gap bounds (Lipschitz)
├── AuditBounds.lean          # Violation probability bounds
├── Audit.lean                # Empirical audit framework
├── MeasureTheoreticAudit.lean # Hoeffding inequality connection
├── TrainingPipeline.lean     # Multi-stage gap composition
├── CounterexampleExistence.lean # L3 is substantive (counterexample)
├── ScoreTransport.lean       # Score transport theory
└── MainTheorems.lean         # Curated exports with documentation
```

## Axioms Used

This module uses **1 axiom** (see `FormalProofs/Axioms.lean` for full documentation):

- `ExpectedGroupLossLipschitz` - Expected loss over groups is Lipschitz

This axiom is justified by the **Random Utility Model** (McFadden 1974).
Under continuous noise distributions, ranking ties have measure zero, so expected losses
are Lipschitz even though pointwise ranking functions are discontinuous.

The axiom is instantiated for specific loss functions:
- `ExpectedGRPOLossLipschitz` - GRPO-PL (Plackett-Luce ranking loss)
- `ExpectedGRPORLLossLipschitz` - GRPO-RL (PPO-style clipped surrogate)

## Key Concepts

### Local Laws (L1, L2, L3)

- **L1 (Sufficiency)**: Summarizing leaves preserves oracle: E[D(g(b), b)] = 0
- **L2 (Merge)**: Merge preserves oracle: E[D(g(u·v), g(g(u)·g(v)))] = 0
- **L3 (Idempotence)**: Re-summarizing is inert: E[D(g(Z), Z)] = 0 for Z ∈ range(g)

### Oracle-Measurability

A function is **oracle-measurable** if it depends on documents only through the oracle f*:
```
dist(f*(x), f*(x')) = 0 ⟹ h(x) = h(x')
```

### Training Equivalence

When local laws hold and losses are oracle-measurable:
```
L_DPO(π; μ_X) = L_DPO(π; μ_Z)
```
where μ_Z is the distribution over summaries.

## Entry Point

For a curated view of the main theorems, see `MainTheorems.lean`.
-/
