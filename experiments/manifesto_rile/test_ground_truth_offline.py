#!/usr/bin/env python3
"""
Offline test for ground truth data structures.

Tests the data structures and serialization without requiring servers.
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 70)
print("  OFFLINE GROUND TRUTH TEST")
print("=" * 70)
print()

# Test 1: Import modules
print("Test 1: Importing modules...")
try:
    from src.ops_engine.training_framework.oracle_ground_truth import (
        ChunkGroundTruth,
        ManifestoGroundTruthTree,
        GroundTruthDataset,
    )
    from src.ops_engine.training_framework.preference import PreferencePair
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print()

# Test 2: Create a chunk ground truth
print("Test 2: Creating ChunkGroundTruth...")
try:
    chunk = ChunkGroundTruth(
        chunk_id="test_doc_L0_N0",
        manifesto_id="test_doc",
        level=0,
        text="This is a test political text about economic policy.",
        rile_score=15.5,
        reasoning="Contains right-wing economic indicators",
        left_indicators="",
        right_indicators="free market, deregulation",
        confidence=0.9,
    )
    print(f"✓ Created chunk: {chunk.chunk_id}")
    print(f"  RILE score: {chunk.rile_score}")
    print(f"  Level: {chunk.level}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print()

# Test 3: Create a tree
print("Test 3: Creating ManifestoGroundTruthTree...")
try:
    tree = ManifestoGroundTruthTree(
        manifesto_id="test_doc",
        document_text="Full document text here...",
        document_rile=20.0,
        oracle_model="test_oracle",
    )

    # Add some nodes
    for i in range(4):
        node = ChunkGroundTruth(
            chunk_id=f"test_doc_L0_N{i}",
            manifesto_id="test_doc",
            level=0,
            text=f"Chunk {i} text",
            rile_score=10.0 + i * 5,
            confidence=0.8,
        )
        tree.add_node(node)

    # Add a merge node
    merge_node = ChunkGroundTruth(
        chunk_id="test_doc_L1_N0",
        manifesto_id="test_doc",
        level=1,
        text="Merged chunks 0 and 1",
        rile_score=12.5,
        left_child_id="test_doc_L0_N0",
        right_child_id="test_doc_L0_N1",
        confidence=0.85,
    )
    tree.add_node(merge_node)

    print(f"✓ Created tree: {tree.manifesto_id}")
    print(f"  Nodes: {tree.num_chunks}")
    print(f"  Levels: {tree.num_levels}")
    print(f"  Leaves: {len(tree.get_leaves())}")
    print(f"  Merges: {len(tree.get_merge_nodes())}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Save and load tree
print("Test 4: Save and load tree...")
try:
    test_dir = Path("data/test_ground_truth")
    test_dir.mkdir(parents=True, exist_ok=True)

    tree_path = test_dir / "test_tree.json"
    tree.save(tree_path)
    print(f"✓ Saved tree to {tree_path}")

    loaded_tree = ManifestoGroundTruthTree.load(tree_path)
    print(f"✓ Loaded tree: {loaded_tree.manifesto_id}")
    print(f"  Nodes match: {loaded_tree.num_chunks == tree.num_chunks}")
    print(f"  Levels match: {loaded_tree.num_levels == tree.num_levels}")

    assert loaded_tree.num_chunks == tree.num_chunks
    assert loaded_tree.manifesto_id == tree.manifesto_id
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: Create a dataset
print("Test 5: Creating GroundTruthDataset...")
try:
    dataset = GroundTruthDataset()
    dataset.add_tree(tree)

    # Create another tree
    tree2 = ManifestoGroundTruthTree(
        manifesto_id="test_doc_2",
        document_text="Another document",
        document_rile=-10.0,
        oracle_model="test_oracle",
    )
    for i in range(3):
        node = ChunkGroundTruth(
            chunk_id=f"test_doc_2_L0_N{i}",
            manifesto_id="test_doc_2",
            level=0,
            text=f"Chunk {i}",
            rile_score=-5.0 - i * 2,
            confidence=0.9,
        )
        tree2.add_node(node)
    dataset.add_tree(tree2)

    print(f"✓ Created dataset with {len(dataset)} trees")

    # Save dataset
    dataset_dir = test_dir / "dataset"
    dataset.save(dataset_dir)
    print(f"✓ Saved dataset to {dataset_dir}")

    # Load dataset
    loaded_dataset = GroundTruthDataset.load(dataset_dir)
    print(f"✓ Loaded dataset with {len(loaded_dataset)} trees")

    stats = loaded_dataset.get_statistics()
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total leaves: {stats['total_leaves']}")
    print(f"  Total merges: {stats['total_merge_nodes']}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 6: Tree statistics
print("Test 6: Tree statistics...")
try:
    stats = tree.get_statistics()
    print(f"✓ Tree statistics:")
    print(f"  Num chunks: {stats['num_chunks']}")
    print(f"  Num levels: {stats['num_levels']}")
    print(f"  Num leaves: {stats['num_leaves']}")
    print(f"  Num merge nodes: {stats['num_merge_nodes']}")
    print(f"  RILE mean: {stats['rile_mean']:.2f}")
    print(f"  RILE range: [{stats['rile_min']:.2f}, {stats['rile_max']:.2f}]")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 7: PreferencePair compatibility
print("Test 7: PreferencePair compatibility with ground truth...")
try:
    # Get a chunk from the tree
    chunk = tree.get_node("test_doc_L0_N0")

    # Create a preference pair linked to this chunk
    pair = PreferencePair(
        pair_id="test_pair_001",
        source_example_id=chunk.chunk_id,
        original_text=chunk.text,
        rubric="Preserve RILE score",
        ground_truth_score=chunk.rile_score,
        law_type="sufficiency",
        summary_a="Summary A",
        summary_b="Summary B",
        preferred="A",
        reasoning="Summary A better preserves RILE",
        confidence=0.85,
        score_estimate_a=12.0,
        score_estimate_b=8.0,
        oracle_error_a=abs(12.0 - chunk.rile_score),
        oracle_error_b=abs(8.0 - chunk.rile_score),
        judge_model="test_judge",
    )

    print(f"✓ Created preference pair: {pair.pair_id}")
    print(f"  Linked to chunk: {pair.source_example_id}")
    print(f"  Ground truth: {pair.ground_truth_score}")
    print(f"  Oracle error A: {pair.oracle_error_a:.2f}")
    print(f"  Oracle error B: {pair.oracle_error_b:.2f}")
    print(f"  Preferred: {pair.preferred}")

    # Verify preference matches oracle errors
    if pair.oracle_error_a < pair.oracle_error_b:
        expected_preferred = "A"
    else:
        expected_preferred = "B"

    print(f"  Expected preferred (by oracle error): {expected_preferred}")
    print(f"  Actual preferred: {pair.preferred}")
    print(f"  Match: {pair.preferred == expected_preferred}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("  ALL TESTS PASSED!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Start oracle server: ./scripts/start_oracle_server.sh")
print("  2. Test with real data: python experiments/manifesto_rile/test_minimal_generation.py")
print("=" * 70)
