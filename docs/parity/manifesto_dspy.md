# Manifesto DSPy

## Reference

The reference is the manifesto expert/teacher label carried by the qsentence
grids. DSPy/chat models are compared through the same grid summary schema:
external expert Pearson/MAE and internal teacher Pearson/MAE.

## Current Evidence

Usable completed artifact:
`outputs/manifesto_qsentence_diffusiongemma_full_leaf1/grid_summary.json`

The first two rows are the transport/scorer check:

| iteration | n eval | external expert Pearson | external expert MAE | internal Pearson |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 48 | 0.9986 | 0.00245 | 0.999999 |
| 1 | 48 | 0.9986 | 0.00245 | 0.999999 |

The third row is a trained merge stage and is not a parity row:

| iteration | n eval | external expert Pearson | external expert MAE |
| ---: | ---: | ---: | ---: |
| 2 | 47 | 0.1159 | 0.2662 |

## Example

See `examples/parity/manifesto_dspy.md` for the inspection command and the
matching ladder command shape.

## Reading

The completed DiffusionGemma DSPy rows show that the chat/DSPy scorer path can
preserve the expert label when used directly. They do not establish that every
trained merge stage is good; the trained merge stage in this artifact is a
negative check.

