"""
Tests for DSPy signatures module.

Note: These tests use mocked DSPy to avoid requiring actual LLM calls.
Integration tests with real LLMs should be separate.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSignatureDefinitions:
    """Tests for signature field definitions."""

    def test_recursive_summary_signature_exists(self):
        """RecursiveSummary signature is defined correctly."""
        from src.core.signatures import RecursiveSummary

        # Check that the class exists and has the right fields
        assert hasattr(RecursiveSummary, '__annotations__') or hasattr(RecursiveSummary, 'fields')

    def test_oracle_judge_signature_exists(self):
        """OracleJudge signature is defined correctly."""
        from src.core.signatures import OracleJudge

        assert hasattr(OracleJudge, '__annotations__') or hasattr(OracleJudge, 'fields')

    def test_sufficiency_check_signature_exists(self):
        """SufficiencyCheck signature is defined correctly."""
        from src.core.signatures import SufficiencyCheck

        assert hasattr(SufficiencyCheck, '__annotations__') or hasattr(SufficiencyCheck, 'fields')

    def test_merge_consistency_check_signature_exists(self):
        """MergeConsistencyCheck signature is defined correctly."""
        from src.core.signatures import MergeConsistencyCheck

        assert hasattr(MergeConsistencyCheck, '__annotations__') or hasattr(MergeConsistencyCheck, 'fields')


class TestSignatureImports:
    """Test that all expected items can be imported."""

    def test_import_signatures(self):
        """All signatures can be imported."""
        from src.core.signatures import (
            RecursiveSummary,
            OracleJudge,
            SufficiencyCheck,
            MergeConsistencyCheck,
        )
        assert RecursiveSummary is not None
        assert OracleJudge is not None
        assert SufficiencyCheck is not None
        assert MergeConsistencyCheck is not None

    def test_import_modules(self):
        """All module classes can be imported."""
        from src.core.signatures import (
            Summarizer,
            Judge,
            SufficiencyChecker,
            MergeChecker,
        )
        assert Summarizer is not None
        assert Judge is not None
        assert SufficiencyChecker is not None
        assert MergeChecker is not None


class TestSummarizerModule:
    """Tests for Summarizer DSPy module."""

    @pytest.fixture
    def mock_dspy(self):
        """Create mock DSPy environment."""
        with patch('src.core.signatures.dspy') as mock:
            # Mock ChainOfThought to return a mock module
            mock_cot = MagicMock()
            mock.ChainOfThought.return_value = mock_cot
            yield mock, mock_cot

    def test_summarizer_initialization(self, mock_dspy):
        """Summarizer initializes correctly."""
        mock, mock_cot = mock_dspy
        from src.core.signatures import Summarizer

        # This creates the module
        summarizer = Summarizer()
        mock.ChainOfThought.assert_called()

    def test_summarizer_forward_call(self, mock_dspy):
        """Summarizer forward method calls ChainOfThought."""
        mock, mock_cot = mock_dspy

        # Setup mock return
        mock_result = MagicMock()
        mock_result.summary = "This is a summary"
        mock_cot.return_value = mock_result

        from src.core.signatures import Summarizer
        summarizer = Summarizer()
        summarizer.summarize = mock_cot

        result = summarizer.forward("Some content", "Keep facts")

        assert result == "This is a summary"
        mock_cot.assert_called_with(rubric="Keep facts", content="Some content")


class TestJudgeModule:
    """Tests for Judge DSPy module."""

    @pytest.fixture
    def mock_dspy(self):
        """Create mock DSPy environment."""
        with patch('src.core.signatures.dspy') as mock:
            mock_cot = MagicMock()
            mock.ChainOfThought.return_value = mock_cot
            yield mock, mock_cot

    def test_judge_initialization(self, mock_dspy):
        """Judge initializes correctly."""
        mock, mock_cot = mock_dspy
        from src.core.signatures import Judge

        judge = Judge()
        mock.ChainOfThought.assert_called()

    def test_judge_forward_returns_dict(self, mock_dspy):
        """Judge forward returns dictionary with expected keys."""
        mock, mock_cot = mock_dspy

        mock_result = MagicMock()
        mock_result.is_congruent = True
        mock_result.discrepancy_score = 0.1
        mock_result.reasoning = "Minor differences"
        mock_cot.return_value = mock_result

        from src.core.signatures import Judge
        judge = Judge()
        judge.judge = mock_cot

        result = judge.forward("input A", "input B", "rubric")

        assert isinstance(result, dict)
        assert 'is_congruent' in result
        assert 'discrepancy_score' in result
        assert 'reasoning' in result
        assert result['is_congruent'] is True
        assert result['discrepancy_score'] == 0.1


class TestSufficiencyCheckerModule:
    """Tests for SufficiencyChecker DSPy module."""

    @pytest.fixture
    def mock_dspy(self):
        """Create mock DSPy environment."""
        with patch('src.core.signatures.dspy') as mock:
            mock_cot = MagicMock()
            mock.ChainOfThought.return_value = mock_cot
            yield mock, mock_cot

    def test_sufficiency_checker_forward(self, mock_dspy):
        """SufficiencyChecker returns expected dict."""
        mock, mock_cot = mock_dspy

        mock_result = MagicMock()
        mock_result.is_sufficient = True
        mock_result.missing_info = ""
        mock_result.confidence = 0.95
        mock_cot.return_value = mock_result

        from src.core.signatures import SufficiencyChecker
        checker = SufficiencyChecker()
        checker.check = mock_cot

        result = checker.forward("original text", "summary text", "preserve names")

        assert isinstance(result, dict)
        assert result['is_sufficient'] is True
        assert result['confidence'] == 0.95


class TestMergeCheckerModule:
    """Tests for MergeChecker DSPy module."""

    @pytest.fixture
    def mock_dspy(self):
        """Create mock DSPy environment."""
        with patch('src.core.signatures.dspy') as mock:
            mock_cot = MagicMock()
            mock.ChainOfThought.return_value = mock_cot
            yield mock, mock_cot

    def test_merge_checker_forward(self, mock_dspy):
        """MergeChecker returns expected dict."""
        mock, mock_cot = mock_dspy

        mock_result = MagicMock()
        mock_result.is_consistent = True
        mock_result.lost_content = ""
        mock_result.discrepancy_score = 0.05
        mock_cot.return_value = mock_result

        from src.core.signatures import MergeChecker
        checker = MergeChecker()
        checker.check = mock_cot

        result = checker.forward("child A\n\nchild B", "merged summary", "rubric")

        assert isinstance(result, dict)
        assert result['is_consistent'] is True
        assert result['discrepancy_score'] == 0.05


class TestSignatureDocstrings:
    """Tests for signature documentation."""

    def test_recursive_summary_has_docstring(self):
        """RecursiveSummary has descriptive docstring."""
        from src.core.signatures import RecursiveSummary
        assert RecursiveSummary.__doc__ is not None
        assert len(RecursiveSummary.__doc__) > 10

    def test_oracle_judge_has_docstring(self):
        """OracleJudge has descriptive docstring."""
        from src.core.signatures import OracleJudge
        assert OracleJudge.__doc__ is not None
        assert "oracle" in OracleJudge.__doc__.lower() or "compare" in OracleJudge.__doc__.lower()


class TestSignatureFieldDescriptions:
    """Tests for field descriptions in signatures."""

    def test_recursive_summary_fields_described(self):
        """RecursiveSummary fields have descriptions."""
        from src.core.signatures import RecursiveSummary
        # DSPy signatures store field info in model_fields
        # Check that the signature representation contains expected fields
        sig_str = str(RecursiveSummary)
        assert 'rubric' in sig_str
        assert 'content' in sig_str
        assert 'summary' in sig_str

    def test_oracle_judge_fields_described(self):
        """OracleJudge fields have descriptions."""
        from src.core.signatures import OracleJudge
        sig_str = str(OracleJudge)
        assert 'rubric' in sig_str
        assert 'input_a' in sig_str
        assert 'input_b' in sig_str
        assert 'is_congruent' in sig_str
        assert 'discrepancy_score' in sig_str
        assert 'reasoning' in sig_str


class TestPrefixCacheStability:
    """Tests that rubric precedes variable content in all key signatures.

    This ensures that vLLM/SGLang prefix caching can share the rubric
    portion of KV-cache across requests with different content.
    """

    def test_recursive_summary_rubric_before_content(self):
        """RecursiveSummary: rubric input field is declared before content."""
        from src.core.signatures import RecursiveSummary
        sig_str = str(RecursiveSummary)
        rubric_pos = sig_str.index('rubric')
        content_pos = sig_str.index('content')
        assert rubric_pos < content_pos, (
            "rubric must be declared before content in RecursiveSummary "
            "to maximise prefix cache reuse"
        )

    def test_oracle_judge_rubric_before_inputs(self):
        """OracleJudge: rubric input field is declared before input_a/input_b."""
        from src.core.signatures import OracleJudge
        sig_str = str(OracleJudge)
        rubric_pos = sig_str.index('rubric')
        input_a_pos = sig_str.index('input_a')
        input_b_pos = sig_str.index('input_b')
        assert rubric_pos < input_a_pos, (
            "rubric must be declared before input_a"
        )
        assert rubric_pos < input_b_pos, (
            "rubric must be declared before input_b"
        )

    def test_default_summarize_prompt_rubric_in_system(self):
        """Rubric is in the system message, not the user message."""
        from src.core.prompting import default_summarize_prompt
        msgs = default_summarize_prompt("some text", "preserve names and dates")
        system_msg = msgs[0]["content"]
        user_msg = msgs[1]["content"]
        assert "Preservation rubric" in system_msg
        assert "preserve names and dates" in system_msg
        assert "Preservation rubric" not in user_msg

    def test_default_merge_prompt_rubric_in_system(self):
        """Rubric is in the system message for merge prompts too."""
        from src.core.prompting import default_merge_prompt
        msgs = default_merge_prompt("left", "right", "preserve names and dates")
        system_msg = msgs[0]["content"]
        user_msg = msgs[1]["content"]
        assert "Preservation rubric" in system_msg
        assert "preserve names and dates" in system_msg
        assert "Preservation rubric" not in user_msg

    def test_summarize_system_msg_identical_for_same_rubric(self):
        """System message is byte-identical across calls with different content."""
        from src.core.prompting import default_summarize_prompt
        msgs_a = default_summarize_prompt("text A about politics", "preserve policy positions")
        msgs_b = default_summarize_prompt("text B about economics", "preserve policy positions")
        assert msgs_a[0]["content"] == msgs_b[0]["content"], (
            "System message must be identical for same rubric to enable "
            "KV-cache prefix sharing"
        )
