"""
Tests for core data models (Node, Tree).
"""

import pytest
from src.core.data_models import (
    Node, Tree, AuditStatus, AuditResult,
    leaf, node
)


class TestNode:
    """Tests for Node class."""

    def test_create_default_node(self):
        """Default node creation produces valid node."""
        n = Node()
        assert n.id is not None
        assert n.level == 0
        assert n.is_leaf
        assert n.audit_result.status == AuditStatus.PENDING

    def test_create_leaf_node(self):
        """Leaf nodes have raw_text_span and level=0."""
        n = leaf("Test content", node_id="test_1")
        assert n.id == "test_1"
        assert n.level == 0
        assert n.raw_text_span == "Test content"
        assert n.ops_span == "Test content"
        assert n.summary == "Test content"
        assert n.is_leaf

    def test_create_internal_node(self):
        """Internal nodes have children and level > 0."""
        left_node = leaf("Left content")
        right_node = leaf("Right content")
        parent = node(left_node, right_node, "Parent summary")

        assert parent.level == 1
        assert not parent.is_leaf
        assert parent.left_child is left_node
        assert parent.right_child is right_node
        assert parent.ops_span == "PART 1:\nLeft content\n\nPART 2:\nRight content"
        assert parent.summary == "Parent summary"

    def test_is_leaf_property(self):
        """is_leaf returns True iff no children."""
        leaf_node = Node(summary="leaf")
        assert leaf_node.is_leaf

        # Add children
        leaf_node.left_child = Node(summary="child1")
        leaf_node.right_child = Node(summary="child2")
        assert not leaf_node.is_leaf

    def test_is_root_property(self):
        """is_root returns True iff no parent."""
        root = Node(summary="root")
        assert root.is_root

        root.parent = Node(summary="grandparent")
        assert not root.is_root

    def test_node_id_unique(self):
        """Each node gets a unique ID by default."""
        nodes = [Node() for _ in range(100)]
        ids = [n.id for n in nodes]
        assert len(set(ids)) == len(ids)

    def test_children_property(self):
        """Children property returns correct list."""
        n = Node()
        assert n.children == []

        left_node = Node()
        n.left_child = left_node
        n.right_child = Node()
        assert len(n.children) == 2
        assert n.children[0] is left_node

    def test_has_both_children(self):
        """has_both_children is True only when both set."""
        n = Node()
        assert not n.has_both_children

        n.left_child = Node()
        assert not n.has_both_children

        n.right_child = Node()
        assert n.has_both_children

    def test_parent_reference_set_by_factory(self):
        """node() sets parent references."""
        left_node = leaf("left")
        right_node = leaf("right")
        parent = node(left_node, right_node, "parent")

        assert left_node.parent is parent
        assert right_node.parent is parent

    def test_audit_methods(self):
        """Audit pass/fail methods work correctly."""
        n = Node()

        n.set_audit_passed(score=0.05, reasoning="Minor issues")
        assert n.audit_passed
        assert n.discrepancy_score == 0.05

        n.set_audit_failed(score=0.8, reasoning="Major loss")
        assert not n.audit_passed
        assert n.discrepancy_score == 0.8

    def test_validate_valid_leaf(self):
        """Valid leaf passes validation."""
        leaf_node = leaf("content")
        violations = leaf_node.validate()
        assert violations == []

    def test_validate_invalid_leaf_level(self):
        """Leaf with non-zero level fails validation."""
        leaf_node = Node(level=1, raw_text_span="content")
        violations = leaf_node.validate()
        assert any("level" in v.lower() for v in violations)

    def test_validate_invalid_internal_level(self):
        """Internal node with level 0 fails validation."""
        left_node = leaf("left")
        right_node = leaf("right")
        internal = Node(level=0, left_child=left_node, right_child=right_node)
        violations = internal.validate()
        assert any("level 0" in v for v in violations)

    def test_validate_single_child(self):
        """Node with only one child fails validation."""
        n = Node(level=1, left_child=Node())
        violations = n.validate()
        assert any("one child" in v for v in violations)

    def test_repr(self):
        """Node repr is readable."""
        n = leaf("Some content here")
        repr_str = repr(n)
        assert "Node" in repr_str
        assert "Leaf" in repr_str


class TestTree:
    """Tests for Tree class."""

    def test_create_single_node_tree(self, single_node_tree):
        """Tree with one leaf (root = leaf)."""
        tree = single_node_tree
        assert tree.root is not None
        assert tree.root.is_leaf
        assert tree.height == 0
        assert tree.node_count == 1
        assert tree.leaf_count == 1

    def test_create_binary_tree(self, simple_tree):
        """Standard binary tree structure."""
        tree = simple_tree
        assert tree.root is not None
        assert not tree.root.is_leaf
        assert tree.node_count == 7  # 4 leaves + 2 internal + 1 root
        assert tree.leaf_count == 4

    def test_height_calculation(self, simple_tree):
        """Tree height is max depth."""
        assert simple_tree.height == 2  # root -> internal -> leaf

    def test_final_summary(self, simple_tree):
        """final_summary returns root summary."""
        assert simple_tree.final_summary == "Root summary of all leaves"

    def test_leaves_property(self, simple_tree):
        """leaves property returns leaf nodes in order."""
        leaves = simple_tree.leaves
        assert len(leaves) == 4
        assert all(leaf.is_leaf for leaf in leaves)

    def test_internal_nodes_property(self, simple_tree):
        """internal_nodes returns non-leaf nodes."""
        internals = simple_tree.internal_nodes
        assert len(internals) == 3  # 2 at level 1, 1 root
        assert all(not n.is_leaf for n in internals)

    def test_traverse_preorder(self, simple_tree):
        """Preorder traversal visits root first."""
        nodes = list(simple_tree.traverse_preorder())
        assert len(nodes) == 7
        assert nodes[0] is simple_tree.root

    def test_traverse_postorder(self, simple_tree):
        """Postorder traversal visits root last."""
        nodes = list(simple_tree.traverse_postorder())
        assert len(nodes) == 7
        assert nodes[-1] is simple_tree.root

    def test_traverse_inorder(self, simple_tree):
        """Inorder traversal visits left, root, right."""
        nodes = list(simple_tree.traverse_inorder())
        assert len(nodes) == 7

    def test_traverse_level_order(self, simple_tree):
        """Level order traversal (BFS)."""
        nodes = list(simple_tree.traverse_level_order())
        assert len(nodes) == 7
        assert nodes[0] is simple_tree.root
        # Level 1 nodes come before level 0 (leaves)
        assert not nodes[1].is_leaf

    def test_find_node_by_id(self, simple_tree):
        """Locate node by ID."""
        # Find root
        found = simple_tree.find_node("root")
        assert found is simple_tree.root

        # Find leaf
        found = simple_tree.find_node("leaf_0")
        assert found is not None
        assert found.is_leaf

    def test_tree_roundtrip_preserves_node_metadata(self, simple_tree):
        """Tree.to_dict/from_dict should preserve Node.metadata."""
        for idx, leaf_node in enumerate(simple_tree.leaves):
            leaf_node.metadata["leaf_idx"] = idx
        simple_tree.root.metadata["root_tag"] = {"a": 1, "b": [2, 3]}

        restored = Tree.from_dict(simple_tree.to_dict())
        assert restored.root is not None
        assert restored.root.metadata.get("root_tag") == {"a": 1, "b": [2, 3]}
        assert restored.root.ops_span == simple_tree.root.ops_span
        assert [n.metadata.get("leaf_idx") for n in restored.leaves] == [0, 1, 2, 3]

        # Not found
        found = simple_tree.find_node("nonexistent")
        assert found is None

    def test_get_path_to_root(self, simple_tree):
        """Path from leaf to root."""
        leaf = simple_tree.find_node("leaf_0")
        path = simple_tree.get_path_to_root(leaf)

        assert len(path) == 3  # leaf -> internal -> root
        assert path[0] is leaf
        assert path[-1] is simple_tree.root

    def test_audit_failure_rate(self, simple_tree):
        """Calculate proportion of failed audits."""
        # Initially no failures
        assert simple_tree.audit_failure_rate == 0.0

        # Mark some as failed
        simple_tree.root.set_audit_failed(0.5)
        rate = simple_tree.audit_failure_rate
        assert rate == pytest.approx(1/7)  # 1 of 7 nodes failed

    def test_get_failed_audits(self, simple_tree):
        """Retrieve all failed nodes."""
        assert len(simple_tree.get_failed_audits()) == 0

        simple_tree.root.set_audit_failed(0.5)
        failed = simple_tree.get_failed_audits()
        assert len(failed) == 1
        assert failed[0] is simple_tree.root

    def test_validate_valid_tree(self, simple_tree):
        """Valid tree passes validation."""
        violations = simple_tree.validate()
        assert violations == []

    def test_apply_to_all(self, simple_tree):
        """Apply function to all nodes."""
        levels = simple_tree.apply_to_all(lambda n: n.level)
        assert len(levels) == 7
        assert 0 in levels
        assert 1 in levels
        assert 2 in levels

    def test_repr(self, simple_tree):
        """Tree repr is readable."""
        repr_str = repr(simple_tree)
        assert "Tree" in repr_str
        assert "height" in repr_str
