import FormalProofs.ML.NeuralOperatorArchitecture

/-!
# FormalProofs/ML/TransformerAsNeuralOperator.lean

Kovachki Proposition 6, in the architecture-level form used by C-TreePO.

We formalize the transformer claim narrowly: a pre-normalized single-head
attention block is represented by a discretized neural-operator kernel layer,
and finite stacks of such blocks are equation-(6)-style neural operators.
Tokenizer behavior, checkpoint details, KV-cache engineering, and runtime
systems are deliberately outside this theorem surface.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace ML

/-! ## Attention as a discretized kernel layer -/

/-- A discretized kernel layer acts on a function represented over grid/index
points. This is the discrete surface of the continuum kernel layer in
Kovachki Proposition 6. -/
structure DiscretizedKernelLayer (Grid Value : Type*) where
  map : (Grid -> Value) -> Grid -> Value

/-- A single-head attention block represented by its induced kernel layer.
The softmax, query/key/value, residual, and output maps are abstracted into
`kernelLayer`; concrete choices instantiate this structure. -/
structure SingleHeadAttention (Grid Value : Type*) where
  kernelLayer : DiscretizedKernelLayer Grid Value

/-- The usual single-head attention map. -/
def singleHeadAttention {Grid Value : Type*}
    (attn : SingleHeadAttention Grid Value) : (Grid -> Value) -> Grid -> Value :=
  attn.kernelLayer.map

/-- The same attention block, read explicitly as a discretized kernel layer. -/
def singleHeadAttentionKernelLayer {Grid Value : Type*}
    (attn : SingleHeadAttention Grid Value) : DiscretizedKernelLayer Grid Value :=
  attn.kernelLayer

/-- Kovachki Proposition 6 surface: single-head attention is exactly the
corresponding discretized neural-operator kernel layer. -/
theorem singleHeadAttention_eq_discretizedKernelLayer
    {Grid Value : Type*} (attn : SingleHeadAttention Grid Value) :
    singleHeadAttention attn = (singleHeadAttentionKernelLayer attn).map := by
  rfl

/-! ## Transformer blocks as neural-operator layers -/

/-- A transformer block consists of a single-head attention layer plus an
arbitrary local/feed-forward adapter. Layer norm and residual conventions are
included in this adapter when they are part of the formal model. -/
structure TransformerBlock (Grid Value : Type*) where
  attention : SingleHeadAttention Grid Value
  feedForward : (Grid -> Value) -> Grid -> Value

/-- The function-space map implemented by one transformer block. -/
def transformerBlockMap {Grid Value : Type*}
    (block : TransformerBlock Grid Value) : (Grid -> Value) -> Grid -> Value :=
  fun x => block.feedForward (singleHeadAttention block.attention x)

/-- Read a transformer block as a homogeneous neural-operator layer. -/
def transformerBlockLayer {Grid Value : Type*}
    (block : TransformerBlock Grid Value) :
    NeuralOperatorLayer (Grid -> Value) where
  map := transformerBlockMap block

/-- A transformer block is a neural-operator layer in the equation-(6) hidden
space. -/
theorem transformerBlock_is_neuralOperatorLayer
    {Grid Value : Type*} (block : TransformerBlock Grid Value) :
    (transformerBlockLayer block).map = transformerBlockMap block := by
  rfl

/-! ## Finite transformer encoders as equation-(6) neural operators -/

/-- A finite transformer encoder stack. -/
def transformerEncoder {Grid Value : Type*}
    (blocks : List (TransformerBlock Grid Value)) :
    (Grid -> Value) -> Grid -> Value :=
  fun x => blocks.foldl (fun h block => transformerBlockMap block h) x

/-- The equation-(6) neural-operator architecture corresponding to a finite
transformer encoder stack. The lift and projection are identities; the hidden
layers are the transformer blocks. -/
def transformerEncoderAsEquation6NeuralOperator {Grid Value : Type*}
    (blocks : List (TransformerBlock Grid Value)) :
    Equation6NeuralOperator (Grid -> Value) (Grid -> Value) (Grid -> Value) where
  lift := id
  layers := blocks.map transformerBlockLayer
  project := id

private theorem foldl_transformer_layers_eq {Grid Value : Type*}
    (blocks : List (TransformerBlock Grid Value)) (x : Grid -> Value) :
    (blocks.map transformerBlockLayer).foldl
        (fun h layer => layer.map h) x =
      blocks.foldl (fun h block => transformerBlockMap block h) x := by
  induction blocks generalizing x with
  | nil =>
      rfl
  | cons block blocks ih =>
      simp [transformerBlockLayer, ih]

/-- Finite transformer encoder stacks are equation-(6)-style neural
operators. -/
theorem transformerEncoder_is_equation6NeuralOperator
    {Grid Value : Type*} (blocks : List (TransformerBlock Grid Value)) :
    (transformerEncoderAsEquation6NeuralOperator blocks).realize =
      transformerEncoder blocks := by
  funext x
  simp [Equation6NeuralOperator.realize, Equation6NeuralOperator.hidden,
    NeuralOperators.ArchCore.Equation6NeuralOperator.realize,
    NeuralOperators.ArchCore.Equation6NeuralOperator.hidden,
    transformerEncoderAsEquation6NeuralOperator, transformerEncoder,
    foldl_transformer_layers_eq]

/-- The realized finite transformer encoder belongs to the equation-(6)
architecture class. -/
theorem transformerEncoder_mem_equation6Class
    {Grid Value : Type*} (blocks : List (TransformerBlock Grid Value)) :
    transformerEncoder blocks ∈
      Equation6NeuralOperatorClass (Grid -> Value) (Grid -> Value) (Grid -> Value) := by
  refine ⟨transformerEncoderAsEquation6NeuralOperator blocks, ?_⟩
  exact transformerEncoder_is_equation6NeuralOperator blocks

end ML
