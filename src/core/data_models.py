"""
Core data models for OPS (Oracle-Preserving Summarization) trees.

This module defines the fundamental data structures:
- OPSNode: Individual nodes in the summarization tree
- OPSTree: Container for the complete tree structure
"""

from dataclasses import dataclass, field
from typing import Optional, List, Iterator, Callable, Any
from enum import Enum
import uuid


class AuditStatus(Enum):
    """Status of node audit verification."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AuditResult:
    """Result of an audit check on a node."""
    status: AuditStatus
    discrepancy_score: float = 0.0
    reasoning: Optional[str] = None
    trace: Optional[dict] = None


@dataclass
class OPSNode:
    """
    A node in the OPS (Oracle-Preserving Summarization) tree.

    Leaves contain raw text spans from the original document.
    Internal nodes contain summaries of their children.

    Attributes:
        id: Unique identifier for this node
        level: Depth in tree (0 = leaf)
        raw_text_span: Original text (only for leaves)
        summary: The summary text at this node
        left_child: Left child node (None for leaves)
        right_child: Right child node (None for leaves)
        parent: Parent node (None for root)
        audit_result: Result of verification audit
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: int = 0

    # Content
    raw_text_span: Optional[str] = None
    summary: str = ""

    # Structure
    left_child: Optional['OPSNode'] = None
    right_child: Optional['OPSNode'] = None
    parent: Optional['OPSNode'] = None

    # Audit state
    audit_result: AuditResult = field(
        default_factory=lambda: AuditResult(status=AuditStatus.PENDING)
    )

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return self.left_child is None and self.right_child is None

    @property
    def is_root(self) -> bool:
        """Check if this node is the root (no parent)."""
        return self.parent is None

    @property
    def has_both_children(self) -> bool:
        """Check if node has both left and right children."""
        return self.left_child is not None and self.right_child is not None

    @property
    def children(self) -> List['OPSNode']:
        """Get list of children (0, 1, or 2 nodes)."""
        result = []
        if self.left_child is not None:
            result.append(self.left_child)
        if self.right_child is not None:
            result.append(self.right_child)
        return result

    @property
    def audit_passed(self) -> bool:
        """Check if audit passed."""
        return self.audit_result.status == AuditStatus.PASSED

    @property
    def discrepancy_score(self) -> float:
        """Get the discrepancy score from audit."""
        return self.audit_result.discrepancy_score

    def set_audit_passed(self, score: float = 0.0, reasoning: str = "") -> None:
        """Mark this node as having passed audit."""
        self.audit_result = AuditResult(
            status=AuditStatus.PASSED,
            discrepancy_score=score,
            reasoning=reasoning
        )

    def set_audit_failed(self, score: float, reasoning: str = "") -> None:
        """Mark this node as having failed audit."""
        self.audit_result = AuditResult(
            status=AuditStatus.FAILED,
            discrepancy_score=score,
            reasoning=reasoning
        )

    def validate(self) -> List[str]:
        """
        Check node invariants and return list of violations.

        Returns:
            List of violation descriptions (empty if valid)
        """
        violations = []

        # Leaf invariants
        if self.is_leaf:
            if self.level != 0:
                violations.append(f"Leaf node has non-zero level: {self.level}")
        else:
            # Internal node invariants
            if self.level == 0:
                violations.append("Internal node has level 0")
            if self.raw_text_span is not None:
                violations.append("Internal node has raw_text_span set")

        # Binary tree constraint: both children or neither
        has_left = self.left_child is not None
        has_right = self.right_child is not None
        if has_left != has_right:
            violations.append("Node has exactly one child (must have 0 or 2)")

        # Parent-child consistency
        for child in self.children:
            if child.parent is not self:
                violations.append(f"Child {child.id} doesn't reference this node as parent")

        return violations

    def __repr__(self) -> str:
        node_type = "Leaf" if self.is_leaf else "Internal"
        summary_preview = self.summary[:30] + "..." if len(self.summary) > 30 else self.summary
        return f"OPSNode({node_type}, id={self.id}, level={self.level}, summary='{summary_preview}')"


@dataclass
class OPSTree:
    """
    Container for an OPS (Oracle-Preserving Summarization) tree.

    The tree is built bottom-up from document chunks (leaves) through
    recursive summarization to a single root node.

    Attributes:
        root: The root node containing the final summary
        rubric: Information preservation criteria for summarization
        metadata: Additional information about source document
    """
    root: OPSNode
    rubric: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        """Get the height of the tree (max depth from root to leaf)."""
        if self.root is None:
            return 0
        return self._calculate_height(self.root)

    def _calculate_height(self, node: OPSNode) -> int:
        """Recursively calculate height from a node."""
        if node.is_leaf:
            return 0
        left_height = self._calculate_height(node.left_child) if node.left_child else 0
        right_height = self._calculate_height(node.right_child) if node.right_child else 0
        return 1 + max(left_height, right_height)

    @property
    def node_count(self) -> int:
        """Get total number of nodes in the tree."""
        return len(list(self.traverse_preorder()))

    @property
    def leaf_count(self) -> int:
        """Get number of leaf nodes."""
        return len(self.leaves)

    @property
    def leaves(self) -> List[OPSNode]:
        """Get all leaf nodes in left-to-right order."""
        return [node for node in self.traverse_inorder() if node.is_leaf]

    @property
    def internal_nodes(self) -> List[OPSNode]:
        """Get all internal (non-leaf) nodes."""
        return [node for node in self.traverse_preorder() if not node.is_leaf]

    @property
    def final_summary(self) -> str:
        """Get the root summary (final output)."""
        return self.root.summary if self.root else ""

    @property
    def audit_failure_rate(self) -> float:
        """Calculate proportion of failed audits."""
        all_nodes = list(self.traverse_preorder())
        if not all_nodes:
            return 0.0
        failed = sum(1 for n in all_nodes if n.audit_result.status == AuditStatus.FAILED)
        return failed / len(all_nodes)

    def traverse_preorder(self) -> Iterator[OPSNode]:
        """Traverse tree in preorder (root, left, right)."""
        if self.root is None:
            return
        yield from self._preorder(self.root)

    def _preorder(self, node: OPSNode) -> Iterator[OPSNode]:
        """Helper for preorder traversal."""
        yield node
        if node.left_child:
            yield from self._preorder(node.left_child)
        if node.right_child:
            yield from self._preorder(node.right_child)

    def traverse_postorder(self) -> Iterator[OPSNode]:
        """Traverse tree in postorder (left, right, root)."""
        if self.root is None:
            return
        yield from self._postorder(self.root)

    def _postorder(self, node: OPSNode) -> Iterator[OPSNode]:
        """Helper for postorder traversal."""
        if node.left_child:
            yield from self._postorder(node.left_child)
        if node.right_child:
            yield from self._postorder(node.right_child)
        yield node

    def traverse_inorder(self) -> Iterator[OPSNode]:
        """Traverse tree in inorder (left, root, right)."""
        if self.root is None:
            return
        yield from self._inorder(self.root)

    def _inorder(self, node: OPSNode) -> Iterator[OPSNode]:
        """Helper for inorder traversal."""
        if node.left_child:
            yield from self._inorder(node.left_child)
        yield node
        if node.right_child:
            yield from self._inorder(node.right_child)

    def traverse_level_order(self) -> Iterator[OPSNode]:
        """Traverse tree in level order (BFS)."""
        if self.root is None:
            return
        from collections import deque
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            yield node
            if node.left_child:
                queue.append(node.left_child)
            if node.right_child:
                queue.append(node.right_child)

    def find_node(self, node_id: str) -> Optional[OPSNode]:
        """Find a node by its ID."""
        for node in self.traverse_preorder():
            if node.id == node_id:
                return node
        return None

    def get_path_to_root(self, node: OPSNode) -> List[OPSNode]:
        """Get path from a node to the root."""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def get_failed_audits(self) -> List[OPSNode]:
        """Get all nodes that failed audit."""
        return [
            node for node in self.traverse_preorder()
            if node.audit_result.status == AuditStatus.FAILED
        ]

    def validate(self) -> List[str]:
        """
        Validate entire tree structure.

        Returns:
            List of violation descriptions (empty if valid)
        """
        violations = []

        if self.root is None:
            violations.append("Tree has no root")
            return violations

        # Check each node
        for node in self.traverse_preorder():
            node_violations = node.validate()
            for v in node_violations:
                violations.append(f"Node {node.id}: {v}")

        # Check that root has no parent
        if self.root.parent is not None:
            violations.append("Root node has a parent")

        # Check level consistency (children must have lower levels than parent)
        # Note: We allow children at any lower level (not just level-1) to
        # accommodate odd nodes that get promoted during tree construction
        for node in self.traverse_preorder():
            if not node.is_leaf:
                for child in node.children:
                    if child.level >= node.level:
                        violations.append(
                            f"Level inconsistency: parent {node.id} (level {node.level}) "
                            f"has child {child.id} at same or higher level ({child.level})"
                        )

        return violations

    def apply_to_all(self, func: Callable[[OPSNode], Any]) -> List[Any]:
        """Apply a function to all nodes and return results."""
        return [func(node) for node in self.traverse_preorder()]

    def __repr__(self) -> str:
        return (
            f"OPSTree(height={self.height}, nodes={self.node_count}, "
            f"leaves={self.leaf_count}, rubric='{self.rubric[:30]}...')"
        )


def create_leaf_node(text: str, node_id: Optional[str] = None) -> OPSNode:
    """
    Factory function to create a leaf node.

    Args:
        text: The raw text for this leaf
        node_id: Optional custom ID

    Returns:
        A properly configured leaf node
    """
    node = OPSNode(
        level=0,
        raw_text_span=text,
        summary=text  # Initial summary is the raw text
    )
    if node_id:
        node.id = node_id
    return node


def create_internal_node(
    left: OPSNode,
    right: OPSNode,
    summary: str,
    node_id: Optional[str] = None
) -> OPSNode:
    """
    Factory function to create an internal node.

    Args:
        left: Left child node
        right: Right child node
        summary: Summary text for this node
        node_id: Optional custom ID

    Returns:
        A properly configured internal node with parent refs set
    """
    level = max(left.level, right.level) + 1
    node = OPSNode(
        level=level,
        summary=summary,
        left_child=left,
        right_child=right
    )
    if node_id:
        node.id = node_id

    # Set parent references
    left.parent = node
    right.parent = node

    return node


def build_tree_from_leaves(
    leaves: List[OPSNode],
    summarize_fn: Callable[[str, str], str],
    rubric: str = ""
) -> OPSTree:
    """
    Build a complete OPS tree from leaf nodes.

    This implements bottom-up tree construction by recursively
    pairing nodes and summarizing until a single root remains.

    Args:
        leaves: List of leaf nodes (chunks)
        summarize_fn: Function that takes (content, rubric) and returns summary
        rubric: Information preservation criteria

    Returns:
        Complete OPSTree with root

    Raises:
        ValueError: If leaves list is empty
    """
    if not leaves:
        raise ValueError("Cannot build tree from empty list of leaves")

    if len(leaves) == 1:
        # Single leaf is also the root
        return OPSTree(root=leaves[0], rubric=rubric)

    # Build tree bottom-up
    current_level = list(leaves)

    while len(current_level) > 1:
        next_level = []

        # Pair nodes
        for i in range(0, len(current_level), 2):
            left = current_level[i]

            if i + 1 < len(current_level):
                # Pair exists - merge
                right = current_level[i + 1]
                combined_content = f"{left.summary}\n\n{right.summary}"
                summary = summarize_fn(combined_content, rubric)
                parent = create_internal_node(left, right, summary)
                next_level.append(parent)
            else:
                # Odd node - promote to next level
                # Create a "pass-through" parent
                # This maintains binary tree structure
                # Alternative: just promote the node directly
                next_level.append(left)

        current_level = next_level

    return OPSTree(root=current_level[0], rubric=rubric)
