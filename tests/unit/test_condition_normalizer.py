"""Unit tests for condition normalizer."""

import pytest
from src.utils.condition_normalizer import (
    normalize_condition,
    get_condition_variants,
    get_search_terms,
    conditions_match,
    get_therapeutic_category,
)


class TestNormalizeCondition:
    """Tests for normalize_condition function."""

    def test_exact_match(self):
        """Test exact condition matching."""
        assert normalize_condition("diabetes") == "diabetes"
        assert normalize_condition("breast cancer") == "breast cancer"
        assert normalize_condition("heart failure") == "heart failure"

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert normalize_condition("DIABETES") == "diabetes"
        assert normalize_condition("Diabetes") == "diabetes"
        assert normalize_condition("BREAST CANCER") == "breast cancer"

    def test_variant_matching(self):
        """Test variant condition matching."""
        # Diabetes variants
        assert normalize_condition("Type 2 Diabetes Mellitus") == "diabetes"
        assert normalize_condition("type 2 diabetes") == "type 2 diabetes"
        assert normalize_condition("T2DM") == "type 2 diabetes"
        assert normalize_condition("NIDDM") == "type 2 diabetes"

        # Cancer variants
        assert normalize_condition("breast neoplasm") == "breast cancer"
        assert normalize_condition("breast carcinoma") == "breast cancer"
        assert normalize_condition("NSCLC") == "lung cancer"
        assert normalize_condition("non-small cell lung cancer") == "lung cancer"

    def test_abbreviation_matching(self):
        """Test abbreviation matching."""
        assert normalize_condition("RA") == "rheumatoid arthritis"
        assert normalize_condition("MS") == "multiple sclerosis"
        assert normalize_condition("COPD") == "copd"
        assert normalize_condition("HIV") == "hiv"
        assert normalize_condition("CHF") == "heart failure"

    def test_empty_input(self):
        """Test empty input handling."""
        assert normalize_condition("") == ""
        assert normalize_condition(None) == ""

    def test_unknown_condition(self):
        """Test unknown condition returns lowercase."""
        result = normalize_condition("xyzabc completely unknown")
        assert result == "xyzabc completely unknown"


class TestGetConditionVariants:
    """Tests for get_condition_variants function."""

    def test_diabetes_variants(self):
        """Test diabetes has expected variants."""
        variants = get_condition_variants("diabetes")
        assert "diabetes" in variants
        assert "diabetes mellitus" in variants

    def test_cancer_variants(self):
        """Test cancer has expected variants."""
        variants = get_condition_variants("breast cancer")
        assert "breast cancer" in variants
        assert "breast neoplasm" in variants
        assert "breast carcinoma" in variants

    def test_unknown_condition(self):
        """Test unknown condition returns itself."""
        variants = get_condition_variants("unknown condition xyz")
        assert variants == ["unknown condition xyz"]

    def test_normalized_input(self):
        """Test that normalized input works."""
        # Input as variant, should normalize and get all variants
        variants = get_condition_variants("T2DM")
        assert "type 2 diabetes" in variants
        assert "t2dm" in variants


class TestGetSearchTerms:
    """Tests for get_search_terms function."""

    def test_returns_set(self):
        """Test that search terms returns a set."""
        terms = get_search_terms("diabetes")
        assert isinstance(terms, set)

    def test_includes_variants(self):
        """Test that search terms include variants."""
        terms = get_search_terms("diabetes")
        assert "diabetes" in terms
        assert "diabetes mellitus" in terms or "mellitus" in terms

    def test_includes_word_parts(self):
        """Test that search terms include word parts."""
        terms = get_search_terms("type 2 diabetes")
        # Should include significant words
        assert "diabetes" in terms


class TestConditionsMatch:
    """Tests for conditions_match function."""

    def test_exact_match(self):
        """Test exact condition matching."""
        assert conditions_match("diabetes", "diabetes")
        assert conditions_match("breast cancer", "breast cancer")

    def test_variant_match(self):
        """Test variant matching."""
        assert conditions_match("diabetes", "diabetes mellitus")
        assert conditions_match("Type 2 Diabetes", "T2DM")
        assert conditions_match("breast cancer", "breast neoplasm")
        assert conditions_match("NSCLC", "non-small cell lung cancer")

    def test_non_match(self):
        """Test non-matching conditions."""
        assert not conditions_match("diabetes", "breast cancer")
        assert not conditions_match("heart failure", "lung cancer")


class TestGetTherapeuticCategory:
    """Tests for get_therapeutic_category function."""

    def test_oncology(self):
        """Test oncology category."""
        assert get_therapeutic_category("breast cancer") == "oncology"
        assert get_therapeutic_category("lung cancer") == "oncology"
        assert get_therapeutic_category("lymphoma") == "oncology"
        assert get_therapeutic_category("leukemia") == "oncology"

    def test_cardiovascular(self):
        """Test cardiovascular category."""
        assert get_therapeutic_category("heart failure") == "cardiovascular"
        assert get_therapeutic_category("hypertension") == "cardiovascular"
        assert get_therapeutic_category("atrial fibrillation") == "cardiovascular"

    def test_metabolic(self):
        """Test metabolic category."""
        assert get_therapeutic_category("diabetes") == "metabolic"
        assert get_therapeutic_category("obesity") == "metabolic"

    def test_neurology(self):
        """Test neurology category."""
        assert get_therapeutic_category("alzheimer") == "neurology"
        assert get_therapeutic_category("parkinson") == "neurology"
        assert get_therapeutic_category("multiple sclerosis") == "neurology"

    def test_unknown_category(self):
        """Test unknown condition returns None."""
        assert get_therapeutic_category("unknown xyz") is None
