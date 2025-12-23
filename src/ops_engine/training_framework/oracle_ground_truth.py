"""
Oracle Ground Truth Trees for OPS Preference Learning.

This module provides data structures for storing oracle-scored hierarchical
trees of manifesto chunks. Supports all three OPS laws:
- Sufficiency: Each chunk has oracle score
- Idempotence: Re-summarize and compare to original oracle score
- Merge: Parent nodes have oracle scores for merged content

Design:
- Reuses patterns from preference.py
- Extensible to multi-dimensional scoring
- Compatible with existing batched pipeline tree structure
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ChunkGroundTruth:
    """
    Ground truth oracle scores for a single chunk/node in the tree.

    Supports hierarchical structure where internal nodes represent
    merged content from child nodes.
    """
    # Identification
    chunk_id: str
    manifesto_id: str
    level: int  # 0 = leaf chunk, 1+ = internal merge node

    # Content
    text: str

    # Oracle scores (primary)
    rile_score: float

    # Multi-dimensional scores (future extension)
    dimension_scores: Optional[Dict[str, float]] = None
    # Example: {"economic": -15.0, "social": 20.0, "environmental": 30.0}

    # Oracle reasoning/metadata
    reasoning: str = ""
    left_indicators: str = ""  # LEFT political signals found
    right_indicators: str = ""  # RIGHT political signals found
    confidence: float = 1.0

    # Tree structure (for merge nodes)
    left_child_id: Optional[str] = None
    right_child_id: Optional[str] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "manifesto_id": self.manifesto_id,
            "level": self.level,
            "text": self.text,
            "rile_score": self.rile_score,
            "dimension_scores": self.dimension_scores,
            "reasoning": self.reasoning,
            "left_indicators": self.left_indicators,
            "right_indicators": self.right_indicators,
            "confidence": self.confidence,
            "left_child_id": self.left_child_id,
            "right_child_id": self.right_child_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkGroundTruth':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ManifestoGroundTruthTree:
    """
    Complete ground truth tree for a manifesto document.

    Stores oracle scores for all nodes (leaves and internal merges)
    to enable testing all three OPS laws with known ground truth.
    """
    # Document identification
    manifesto_id: str
    document_text: str
    document_rile: float  # Original manifesto RILE score

    # Tree structure
    nodes: Dict[str, ChunkGroundTruth] = field(default_factory=dict)
    levels: List[List[str]] = field(default_factory=list)
    # levels[i] = list of chunk_ids at level i

    # Statistics
    num_chunks: int = 0
    num_levels: int = 0

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    oracle_model: str = ""

    def add_node(self, node: ChunkGroundTruth):
        """Add a node to the tree."""
        self.nodes[node.chunk_id] = node

        # Ensure levels list is long enough
        while len(self.levels) <= node.level:
            self.levels.append([])

        # Add to appropriate level
        if node.chunk_id not in self.levels[node.level]:
            self.levels[node.level].append(node.chunk_id)

        # Update statistics
        self.num_chunks = len(self.nodes)
        self.num_levels = len(self.levels)

    def get_node(self, chunk_id: str) -> Optional[ChunkGroundTruth]:
        """Get a node by ID."""
        return self.nodes.get(chunk_id)

    def get_level(self, level: int) -> List[ChunkGroundTruth]:
        """Get all nodes at a specific level."""
        if level >= len(self.levels):
            return []
        return [self.nodes[chunk_id] for chunk_id in self.levels[level]]

    def get_leaves(self) -> List[ChunkGroundTruth]:
        """Get all leaf nodes (level 0)."""
        return self.get_level(0)

    def get_merge_nodes(self) -> List[ChunkGroundTruth]:
        """Get all internal merge nodes (level > 0)."""
        merge_nodes = []
        for level in range(1, self.num_levels):
            merge_nodes.extend(self.get_level(level))
        return merge_nodes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": "1.0",
            "manifesto_id": self.manifesto_id,
            "document_text": self.document_text,
            "document_rile": self.document_rile,
            "nodes": {chunk_id: node.to_dict() for chunk_id, node in self.nodes.items()},
            "levels": self.levels,
            "num_chunks": self.num_chunks,
            "num_levels": self.num_levels,
            "created_at": self.created_at,
            "oracle_model": self.oracle_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManifestoGroundTruthTree':
        """Create from dictionary."""
        tree = cls(
            manifesto_id=data["manifesto_id"],
            document_text=data["document_text"],
            document_rile=data["document_rile"],
            levels=data["levels"],
            num_chunks=data["num_chunks"],
            num_levels=data["num_levels"],
            created_at=data["created_at"],
            oracle_model=data.get("oracle_model", ""),
        )

        # Reconstruct nodes
        for chunk_id, node_data in data["nodes"].items():
            tree.nodes[chunk_id] = ChunkGroundTruth.from_dict(node_data)

        return tree

    def save(self, path: Path):
        """Save tree to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved ground truth tree with {self.num_chunks} nodes to {path}")

    @classmethod
    def load(cls, path: Path) -> 'ManifestoGroundTruthTree':
        """Load tree from JSON file."""
        with open(path) as f:
            data = json.load(f)

        tree = cls.from_dict(data)
        logger.info(f"Loaded ground truth tree with {tree.num_chunks} nodes from {path}")
        return tree

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics about the tree."""
        if not self.nodes:
            return {"num_chunks": 0, "num_levels": 0}

        rile_scores = [node.rile_score for node in self.nodes.values()]

        return {
            "num_chunks": self.num_chunks,
            "num_levels": self.num_levels,
            "num_leaves": len(self.get_leaves()),
            "num_merge_nodes": len(self.get_merge_nodes()),
            "rile_mean": sum(rile_scores) / len(rile_scores),
            "rile_min": min(rile_scores),
            "rile_max": max(rile_scores),
            "document_rile": self.document_rile,
            "oracle_model": self.oracle_model,
        }


class GroundTruthDataset:
    """
    Collection of ground truth trees for multiple manifestos.

    Enables batch operations and statistics across documents.
    """

    def __init__(self, trees: Optional[List[ManifestoGroundTruthTree]] = None):
        self.trees: Dict[str, ManifestoGroundTruthTree] = {}
        if trees:
            for tree in trees:
                self.trees[tree.manifesto_id] = tree

    def add_tree(self, tree: ManifestoGroundTruthTree):
        """Add a tree to the dataset."""
        self.trees[tree.manifesto_id] = tree

    def get_tree(self, manifesto_id: str) -> Optional[ManifestoGroundTruthTree]:
        """Get a tree by manifesto ID."""
        return self.trees.get(manifesto_id)

    def __len__(self) -> int:
        return len(self.trees)

    def save(self, directory: Path):
        """Save all trees to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for manifesto_id, tree in self.trees.items():
            filename = f"{manifesto_id}_ground_truth.json"
            tree.save(directory / filename)

        # Save index
        index = {
            "num_trees": len(self.trees),
            "manifesto_ids": list(self.trees.keys()),
            "created_at": datetime.now().isoformat(),
        }
        with open(directory / "index.json", 'w') as f:
            json.dump(index, f, indent=2)

        logger.info(f"Saved {len(self.trees)} ground truth trees to {directory}")

    @classmethod
    def load(cls, directory: Path) -> 'GroundTruthDataset':
        """Load all trees from a directory."""
        directory = Path(directory)

        # Load index
        with open(directory / "index.json") as f:
            index = json.load(f)

        # Load all trees
        dataset = cls()
        for manifesto_id in index["manifesto_ids"]:
            filename = f"{manifesto_id}_ground_truth.json"
            tree = ManifestoGroundTruthTree.load(directory / filename)
            dataset.add_tree(tree)

        logger.info(f"Loaded {len(dataset)} ground truth trees from {directory}")
        return dataset

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics across all trees."""
        if not self.trees:
            return {"num_trees": 0}

        total_chunks = sum(tree.num_chunks for tree in self.trees.values())
        total_leaves = sum(len(tree.get_leaves()) for tree in self.trees.values())
        total_merges = sum(len(tree.get_merge_nodes()) for tree in self.trees.values())

        return {
            "num_trees": len(self.trees),
            "total_chunks": total_chunks,
            "total_leaves": total_leaves,
            "total_merge_nodes": total_merges,
            "avg_chunks_per_tree": total_chunks / len(self.trees),
            "avg_levels": sum(tree.num_levels for tree in self.trees.values()) / len(self.trees),
        }
