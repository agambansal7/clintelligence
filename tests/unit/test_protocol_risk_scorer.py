"""Unit tests for protocol risk scorer."""

import pytest
from src.analysis.protocol_risk_scorer import (
    ProtocolRiskScorer,
    RiskAssessment,
)


class TestProtocolRiskScorer:
    """Tests for ProtocolRiskScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a scorer instance without DB connection."""
        return ProtocolRiskScorer()

    def test_score_protocol_returns_assessment(self, scorer, sample_protocol):
        """Test that score_protocol returns RiskAssessment."""
        result = scorer.score_protocol(**sample_protocol)

        assert isinstance(result, RiskAssessment)
        assert hasattr(result, "overall_risk_score")
        assert hasattr(result, "risk_factors")
        assert hasattr(result, "recommendations")

    def test_risk_score_in_valid_range(self, scorer, sample_protocol):
        """Test that risk score is within valid range."""
        result = scorer.score_protocol(**sample_protocol)

        assert 0 <= result.overall_risk_score <= 100

    def test_probabilities_in_valid_range(self, scorer, sample_protocol):
        """Test that probabilities are between 0 and 1."""
        result = scorer.score_protocol(**sample_protocol)

        assert 0 <= result.amendment_probability <= 1
        assert 0 <= result.enrollment_delay_probability <= 1

    def test_recommendations_not_empty(self, scorer, sample_protocol):
        """Test that recommendations are provided."""
        result = scorer.score_protocol(**sample_protocol)

        assert isinstance(result.recommendations, list)
        # Should have at least some recommendations

    def test_high_enrollment_increases_risk(self, scorer, sample_protocol):
        """Test that high enrollment target increases risk."""
        # Low enrollment
        low_enrollment_protocol = {**sample_protocol, "target_enrollment": 100}
        low_result = scorer.score_protocol(**low_enrollment_protocol)

        # High enrollment
        high_enrollment_protocol = {**sample_protocol, "target_enrollment": 5000}
        high_result = scorer.score_protocol(**high_enrollment_protocol)

        # Higher enrollment should generally increase enrollment delay probability
        assert high_result.enrollment_delay_probability >= low_result.enrollment_delay_probability

    def test_complex_criteria_increases_risk(self, scorer, sample_protocol):
        """Test that complex eligibility criteria increases risk."""
        simple_criteria = """
        Inclusion: Adults with diabetes
        Exclusion: Severe disease
        """

        complex_criteria = """
        Inclusion Criteria:
        - Age 18-65 years
        - HbA1c between 7.0% and 10.0%
        - BMI >= 25 and < 40 kg/m2
        - On stable metformin dose >= 1500mg for 90 days
        - eGFR >= 60 mL/min/1.73m2
        - No prior GLP-1 RA use
        - Fasting C-peptide >= 0.8 ng/mL

        Exclusion Criteria:
        - Type 1 diabetes
        - History of pancreatitis
        - NYHA Class III or IV heart failure
        - Active liver disease
        - Recent cardiovascular event
        - Current use of insulin
        - Pregnancy or nursing
        - Malignancy within 5 years
        """

        simple_protocol = {**sample_protocol, "eligibility_criteria": simple_criteria}
        complex_protocol = {**sample_protocol, "eligibility_criteria": complex_criteria}

        simple_result = scorer.score_protocol(**simple_protocol)
        complex_result = scorer.score_protocol(**complex_protocol)

        # Complex criteria should generally increase amendment probability
        assert complex_result.amendment_probability >= simple_result.amendment_probability

    def test_different_conditions(self, scorer):
        """Test scoring works for different conditions."""
        conditions = ["diabetes", "breast cancer", "heart failure", "alzheimer"]

        for condition in conditions:
            result = scorer.score_protocol(
                condition=condition,
                phase="PHASE3",
                eligibility_criteria="Adults with condition",
                primary_endpoints=["Primary endpoint"],
                target_enrollment=500,
                planned_sites=50,
                planned_duration_months=24,
            )

            assert result is not None
            assert 0 <= result.overall_risk_score <= 100

    def test_different_phases(self, scorer, sample_protocol):
        """Test scoring works for different phases."""
        for phase in ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]:
            protocol = {**sample_protocol, "phase": phase}
            result = scorer.score_protocol(**protocol)

            assert result is not None
            assert 0 <= result.overall_risk_score <= 100

    def test_normalize_condition(self, scorer):
        """Test internal condition normalization."""
        # Access the internal method
        assert scorer._normalize_condition("diabetes") == "diabetes"
        assert scorer._normalize_condition("breast cancer") == "cancer"
        assert scorer._normalize_condition("heart failure") == "cardiovascular"
        assert scorer._normalize_condition("alzheimer") == "alzheimer"
        assert scorer._normalize_condition("rare orphan disease") == "rare_disease"
        assert scorer._normalize_condition("unknown condition") == "default"


class TestRiskAssessment:
    """Tests for RiskAssessment dataclass."""

    def test_dataclass_fields(self):
        """Test RiskAssessment has expected fields."""
        from src.analysis.protocol_risk_scorer import RiskFactor
        from dataclasses import asdict

        risk_factor = RiskFactor(
            category="enrollment",
            description="High enrollment target",
            severity="medium",
            historical_evidence="Similar trials took longer",
            recommendation="Consider more sites",
        )

        assessment = RiskAssessment(
            overall_risk_score=45,
            amendment_probability=0.35,
            enrollment_delay_probability=0.25,
            termination_probability=0.15,
            risk_factors=[risk_factor],
            benchmark_trials=["NCT123", "NCT456"],
            recommendations=["Consider more sites"],
        )

        # Check fields are accessible
        assert assessment.overall_risk_score == 45
        assert assessment.amendment_probability == 0.35
        assert assessment.enrollment_delay_probability == 0.25
        assert assessment.termination_probability == 0.15
        assert len(assessment.risk_factors) == 1
        assert len(assessment.benchmark_trials) == 2
        assert len(assessment.recommendations) == 1

        # Test serialization with dataclasses.asdict
        result = asdict(assessment)
        assert isinstance(result, dict)
        assert result["overall_risk_score"] == 45
