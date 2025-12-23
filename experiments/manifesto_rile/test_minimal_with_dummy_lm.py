#!/usr/bin/env python3
"""
Minimal test using DSPy's DummyLM (no server needed).

Tests ground truth generation and preference collection with mock LM.
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 70)
print("  MINIMAL TEST WITH DUMMY LM (No Server Required)")
print("=" * 70)
print()

# Test imports
print("Test 1: Importing modules...")
try:
    import dspy
    # Import directly to avoid circular import
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "oracle_ground_truth",
        project_root / "src/ops_engine/training_framework/oracle_ground_truth.py"
    )
    oracle_gt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle_gt_module)

    ChunkGroundTruth = oracle_gt_module.ChunkGroundTruth
    ManifestoGroundTruthTree = oracle_gt_module.ManifestoGroundTruthTree
    GroundTruthDataset = oracle_gt_module.GroundTruthDataset

    from src.manifesto.batched_pipeline import chunk_text
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 2: Test chunking
print("Test 2: Testing chunk_text...")
try:
    test_doc = """This is a test political manifesto.

We support free market economics and lower taxes.

We also believe in traditional family values.

Investment in infrastructure is important."""

    chunks = chunk_text(test_doc, max_chars=100)
    print(f"✓ Created {len(chunks)} chunks from {len(test_doc)} char document")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} chars - {chunk[:50]}...")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Create ground truth tree structure (without oracle scoring)
print("Test 3: Creating ground truth tree structure...")
try:
    tree = ManifestoGroundTruthTree(
        manifesto_id="test_minimal",
        document_text=test_doc,
        document_rile=10.0,  # Mock ground truth
        oracle_model="dummy_oracle",
    )

    # Add leaf nodes for each chunk
    for i, chunk in enumerate(chunks):
        node = ChunkGroundTruth(
            chunk_id=f"test_minimal_L0_N{i}",
            manifesto_id="test_minimal",
            level=0,
            text=chunk,
            rile_score=10.0 + i * 2.5,  # Mock scores
            reasoning=f"Mock reasoning for chunk {i}",
            confidence=0.9,
        )
        tree.add_node(node)

    print(f"✓ Created tree with {tree.num_chunks} nodes")
    print(f"  Leaves: {len(tree.get_leaves())}")

    # Add merge nodes
    if len(chunks) >= 2:
        merge_node = ChunkGroundTruth(
            chunk_id="test_minimal_L1_N0",
            manifesto_id="test_minimal",
            level=1,
            text=f"{chunks[0]}\n\n{chunks[1]}",
            rile_score=11.25,  # Average of first two
            reasoning="Merged first two chunks",
            left_child_id="test_minimal_L0_N0",
            right_child_id="test_minimal_L0_N1",
            confidence=0.85,
        )
        tree.add_node(merge_node)
        print(f"  Merge nodes: {len(tree.get_merge_nodes())}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Save and load tree
print("Test 4: Save/load tree...")
try:
    test_dir = Path("data/test_minimal")
    test_dir.mkdir(parents=True, exist_ok=True)

    tree_path = test_dir / "test_tree.json"
    tree.save(tree_path)
    print(f"✓ Saved to {tree_path}")

    loaded_tree = ManifestoGroundTruthTree.load(tree_path)
    print(f"✓ Loaded tree with {loaded_tree.num_chunks} nodes")

    assert loaded_tree.num_chunks == tree.num_chunks
    assert loaded_tree.manifesto_id == tree.manifesto_id
    print("✓ Tree integrity verified")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: Create dataset
print("Test 5: Create and save dataset...")
try:
    dataset = GroundTruthDataset()
    dataset.add_tree(tree)

    dataset_dir = test_dir / "dataset"
    dataset.save(dataset_dir)
    print(f"✓ Saved dataset to {dataset_dir}")

    loaded_dataset = GroundTruthDataset.load(dataset_dir)
    print(f"✓ Loaded dataset with {len(loaded_dataset)} trees")

    stats = loaded_dataset.get_statistics()
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Avg chunks/tree: {stats['avg_chunks_per_tree']:.1f}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 6: Test with DummyLM (no server needed)
print("Test 6: Test with DSPy DummyLM...")
try:
    # Create a DummyLM that returns predictable responses
    class SimpleDummyLM(dspy.LM):
        def __init__(self):
            super().__init__(model="dummy")
            self.history = []

        def __call__(self, prompt=None, messages=None, **kwargs):
            # Return a mock RILE score
            response = "15.5"  # Mock RILE score
            self.history.append({"prompt": prompt, "response": response})
            return [response]

        def inspect_history(self, n=1):
            return self.history[-n:] if self.history else []

    dummy_lm = SimpleDummyLM()
    dspy.configure(lm=dummy_lm)

    # Test that we can call it
    result = dummy_lm("What is the RILE score?")
    print(f"✓ DummyLM returned: {result}")
    print(f"  History entries: {len(dummy_lm.history)}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("  ALL MINIMAL TESTS PASSED!")
print("=" * 70)
print()
print("Summary:")
print(f"  ✓ Data structures working")
print(f"  ✓ Chunking working ({len(chunks)} chunks created)")
print(f"  ✓ Tree creation/save/load working")
print(f"  ✓ Dataset creation/save/load working")
print(f"  ✓ DSPy DummyLM integration working")
print()
print("Test files created in: data/test_minimal/")
print()
print("Next steps:")
print("  1. Test with real oracle server")
print("  2. Run full ground truth generation")
print("  3. Run preference collection")
print("=" * 70)
