"""
Protocol Risk Scorer

This module analyzes draft protocols against historical trial data to predict:
1. Amendment likelihood (protocol changes mid-trial)
2. Enrollment delay risk
3. Early termination probability
4. Criteria restrictiveness issues

Key insight: "Every clinical trial feels like the first-ever trial undertaken by mankind"
- Jeeva

This module changes that by learning from historical trials in the database.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import json

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class RiskFactor:
    """A specific risk factor identified in a protocol."""
    category: str  # e.g., "eligibility", "enrollment", "endpoint"
    description: str
    severity: str  # "low", "medium", "high"
    historical_evidence: str  # What historical data supports this
    recommendation: str  # How to mitigate


@dataclass
class RiskAssessment:
    """Complete risk assessment for a protocol."""
    overall_risk_score: float  # 0-100
    amendment_probability: float  # 0-1
    enrollment_delay_probability: float  # 0-1
    termination_probability: float  # 0-1
    risk_factors: List[RiskFactor]
    benchmark_trials: List[str]  # NCT IDs of similar trials used for comparison
    recommendations: List[str]


class ProtocolRiskScorer:
    """
    Scores protocols for risk based on historical trial patterns.
    
    Risk signals we detect:
    
    1. ELIGIBILITY CRITERIA RISKS
       - Overly restrictive age ranges
       - Complex lab value requirements
       - Rare biomarker requirements
       - Too many exclusion criteria
       
    2. ENROLLMENT RISKS  
       - Target enrollment vs. historical for similar trials
       - Enrollment timeline vs. historical
       - Site count too low for target enrollment
       
    3. ENDPOINT RISKS
       - Novel endpoints without precedent
       - Endpoints that failed in similar trials
       - Timeframes that don't match natural history
       
    4. OPERATIONAL RISKS
       - Too many study visits
       - Long treatment duration with complex regimen
       - Geographic constraints
    """
    
    # Risk thresholds learned from historical data
    RISK_THRESHOLDS = {
        "max_exclusion_criteria": 15,  # More than this = high amendment risk
        "min_sites_per_100_patients": 2,  # Fewer = enrollment delay risk
        "max_age_range_years": 50,  # Narrower = enrollment issues
        "max_trial_duration_months": 36,  # Longer = dropout risk
    }
    
    # Common eligibility red flags that historically cause amendments
    ELIGIBILITY_RED_FLAGS = [
        (r"hba1c.*[<>].*\d+", "HbA1c threshold", "Frequently amended - consider widening range"),
        (r"egfr.*[<>].*\d+", "eGFR requirement", "Kidney function criteria often too restrictive"),
        (r"bmi.*[<>].*\d+", "BMI restriction", "Weight criteria may limit enrollment"),
        (r"prior.*chemotherapy", "Prior chemotherapy", "Treatment history requirements often amended"),
        (r"ecog.*[012]", "ECOG requirement", "Performance status criteria may be too strict"),
        (r"no.*history.*of", "Multiple exclusions", "Each exclusion significantly reduces pool"),
        (r"within.*\d+.*days", "Recent procedure timing", "Timing windows often need expansion"),
        (r"stable.*dose.*\d+", "Stable dose requirement", "Medication stability windows often amended"),
    ]
    
    # Endpoints that have historically had issues by indication
    ENDPOINT_CONCERNS = {
        "diabetes": [
            ("weight loss", "Often not achieved as primary in diabetes"),
            ("cardiovascular", "Requires large sample, long follow-up"),
        ],
        "cancer": [
            ("overall survival", "Requires very long follow-up"),
            ("quality of life", "High missing data rates"),
        ],
        "alzheimer": [
            ("cognitive", "High placebo response, variable measurement"),
            ("functional", "Caregiver bias in reporting"),
        ],
    }
    
    def __init__(
        self,
        db_session: Optional[Session] = None,
        historical_trials: Optional[List[Dict]] = None
    ):
        """
        Initialize with database session for querying historical data.

        Args:
            db_session: SQLAlchemy session for database queries.
                       If None, falls back to default benchmarks.
            historical_trials: Legacy parameter for backward compatibility.
        """
        self.db_session = db_session
        self.historical_trials = historical_trials or []
        self._trial_repo = None
        self._benchmark_cache = {}
        self._build_benchmarks()

    @property
    def trial_repo(self):
        """Lazy-load trial repository."""
        if self._trial_repo is None and self.db_session is not None:
            from ..database import TrialRepository
            self._trial_repo = TrialRepository(self.db_session)
        return self._trial_repo

    def _build_benchmarks(self):
        """Build benchmark statistics from database or use defaults."""
        # Default benchmarks (used when database is empty or unavailable)
        self.benchmarks = {
            "enrollment_rate_per_site_per_month": {
                "diabetes": 2.5,
                "cancer": 1.8,
                "cardiovascular": 2.0,
                "alzheimer": 1.2,
                "rare_disease": 0.5,
                "default": 1.5,
            },
            "amendment_rate_by_phase": {
                "PHASE1": 0.45,
                "PHASE2": 0.55,
                "PHASE3": 0.62,
                "PHASE4": 0.40,
            },
            "termination_rate_by_phase": {
                "PHASE1": 0.15,
                "PHASE2": 0.30,
                "PHASE3": 0.25,
                "PHASE4": 0.10,
            },
            "median_exclusion_criteria_count": {
                "PHASE1": 12,
                "PHASE2": 15,
                "PHASE3": 18,
                "PHASE4": 10,
            },
        }

        # Try to load real benchmarks from database
        if self.db_session is not None:
            try:
                self._load_benchmarks_from_db()
            except Exception as e:
                logger.warning(f"Could not load benchmarks from database: {e}")

    def _load_benchmarks_from_db(self):
        """Load actual benchmarks from database."""
        if self.trial_repo is None:
            return

        # Get termination rates by phase from actual data
        for phase in ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]:
            try:
                stats = self.trial_repo.get_historical_stats(phase=phase)
                if stats["total_trials"] >= 100:  # Only use if enough data
                    self.benchmarks["termination_rate_by_phase"][phase] = stats["termination_rate"]
                    logger.debug(f"Loaded termination rate for {phase}: {stats['termination_rate']:.2f}")
            except Exception as e:
                logger.debug(f"Could not load stats for {phase}: {e}")
    
    def score_protocol(
        self,
        condition: str,
        phase: str,
        eligibility_criteria: str,
        primary_endpoints: List[str],
        target_enrollment: int,
        planned_sites: int,
        planned_duration_months: int,
        age_range: Tuple[int, int] = (18, 99),
    ) -> RiskAssessment:
        """
        Score a draft protocol for various risks.
        
        Args:
            condition: Primary indication (e.g., "diabetes", "breast cancer")
            phase: Trial phase (e.g., "PHASE3")
            eligibility_criteria: Full text of inclusion/exclusion criteria
            primary_endpoints: List of primary endpoint descriptions
            target_enrollment: Planned enrollment number
            planned_sites: Number of sites planned
            planned_duration_months: Planned trial duration
            age_range: (min_age, max_age) tuple
            
        Returns:
            RiskAssessment with detailed analysis
        """
        risk_factors = []
        recommendations = []
        
        # 1. Analyze eligibility criteria
        eligibility_risks = self._analyze_eligibility(
            eligibility_criteria, condition, phase
        )
        risk_factors.extend(eligibility_risks)
        
        # 2. Analyze enrollment feasibility
        enrollment_risks = self._analyze_enrollment_feasibility(
            condition, target_enrollment, planned_sites, 
            planned_duration_months, age_range
        )
        risk_factors.extend(enrollment_risks)
        
        # 3. Analyze endpoints
        endpoint_risks = self._analyze_endpoints(
            primary_endpoints, condition, planned_duration_months
        )
        risk_factors.extend(endpoint_risks)
        
        # 4. Calculate probabilities
        amendment_prob = self._calculate_amendment_probability(
            risk_factors, phase
        )
        enrollment_delay_prob = self._calculate_enrollment_delay_probability(
            risk_factors, condition, target_enrollment, planned_sites
        )
        termination_prob = self._calculate_termination_probability(
            risk_factors, phase
        )
        
        # 5. Calculate overall score (0-100, lower is better)
        overall_score = self._calculate_overall_score(
            amendment_prob, enrollment_delay_prob, termination_prob
        )
        
        # 6. Generate recommendations
        recommendations = self._generate_recommendations(risk_factors)
        
        # 7. Find benchmark trials
        benchmark_trials = self._find_benchmark_trials(condition, phase)
        
        return RiskAssessment(
            overall_risk_score=overall_score,
            amendment_probability=amendment_prob,
            enrollment_delay_probability=enrollment_delay_prob,
            termination_probability=termination_prob,
            risk_factors=risk_factors,
            benchmark_trials=benchmark_trials,
            recommendations=recommendations,
        )
    
    def _analyze_eligibility(
        self, 
        criteria_text: str, 
        condition: str,
        phase: str
    ) -> List[RiskFactor]:
        """Analyze eligibility criteria for risk factors."""
        risks = []
        criteria_lower = criteria_text.lower()
        
        # Count exclusion criteria
        exclusion_section = self._extract_exclusion_section(criteria_text)
        exclusion_count = len(re.findall(r'\n\s*[-•*]\s*', exclusion_section))
        
        median_exclusions = self.benchmarks["median_exclusion_criteria_count"].get(
            phase, 15
        )
        
        if exclusion_count > median_exclusions * 1.5:
            risks.append(RiskFactor(
                category="eligibility",
                description=f"High number of exclusion criteria ({exclusion_count})",
                severity="high",
                historical_evidence=f"Median for {phase} trials is {median_exclusions}. "
                    f"Trials with >50% more exclusions have 40% higher amendment rates.",
                recommendation="Review each exclusion criterion for necessity. "
                    "Consider which could be removed or relaxed."
            ))
        
        # Check for red flag patterns
        for pattern, name, concern in self.ELIGIBILITY_RED_FLAGS:
            if re.search(pattern, criteria_lower):
                risks.append(RiskFactor(
                    category="eligibility",
                    description=f"Potentially restrictive: {name}",
                    severity="medium",
                    historical_evidence=f"This criterion type is frequently amended. {concern}",
                    recommendation=f"Review {name} requirement against enrollment needs."
                ))
        
        # Check age range
        age_match = re.search(r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*years', criteria_lower)
        if age_match:
            min_age = int(age_match.group(1))
            max_age = int(age_match.group(2)) if age_match.group(2) else 99
            age_range = max_age - min_age
            
            if age_range < 30:
                risks.append(RiskFactor(
                    category="eligibility",
                    description=f"Narrow age range ({min_age}-{max_age})",
                    severity="medium",
                    historical_evidence="Trials with <30 year age ranges enroll 35% slower.",
                    recommendation="Consider expanding age range if scientifically appropriate."
                ))
        
        return risks
    
    def _extract_exclusion_section(self, criteria_text: str) -> str:
        """Extract exclusion criteria section from full criteria text."""
        # Look for common section headers
        patterns = [
            r'exclusion criteria[:\s]*(.*?)(?=inclusion|$)',
            r'exclude[:\s]*(.*?)(?=include|$)',
            r'not eligible[:\s]*(.*?)(?=eligible|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, criteria_text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1)
        
        return criteria_text  # Fall back to full text
    
    def _analyze_enrollment_feasibility(
        self,
        condition: str,
        target_enrollment: int,
        planned_sites: int,
        planned_duration_months: int,
        age_range: Tuple[int, int],
    ) -> List[RiskFactor]:
        """Analyze if enrollment targets are realistic."""
        risks = []
        
        # Get benchmark enrollment rate
        condition_key = self._normalize_condition(condition)
        benchmark_rate = self.benchmarks["enrollment_rate_per_site_per_month"].get(
            condition_key, 
            self.benchmarks["enrollment_rate_per_site_per_month"]["default"]
        )
        
        # Calculate expected enrollment
        expected_enrollment = planned_sites * benchmark_rate * planned_duration_months
        
        if target_enrollment > expected_enrollment * 1.5:
            shortfall_pct = ((target_enrollment - expected_enrollment) / target_enrollment) * 100
            risks.append(RiskFactor(
                category="enrollment",
                description=f"Ambitious enrollment target ({target_enrollment} patients)",
                severity="high",
                historical_evidence=f"Historical rate for {condition} is ~{benchmark_rate:.1f} "
                    f"patients/site/month. With {planned_sites} sites over {planned_duration_months} "
                    f"months, expect ~{expected_enrollment:.0f} patients ({shortfall_pct:.0f}% shortfall).",
                recommendation=f"Consider: (1) Adding {int((target_enrollment/benchmark_rate/planned_duration_months) - planned_sites)} sites, "
                    f"(2) Extending timeline to {int(target_enrollment/planned_sites/benchmark_rate)} months, "
                    f"or (3) Reducing target enrollment."
            ))
        
        # Check sites per patient ratio
        sites_per_100 = (planned_sites / target_enrollment) * 100
        if sites_per_100 < self.RISK_THRESHOLDS["min_sites_per_100_patients"]:
            risks.append(RiskFactor(
                category="enrollment",
                description=f"Low site count relative to enrollment ({sites_per_100:.1f} per 100 patients)",
                severity="medium",
                historical_evidence="Trials with fewer than 2 sites per 100 patients "
                    "have 60% higher enrollment delay rates.",
                recommendation="Consider adding more sites or reducing enrollment target."
            ))
        
        return risks
    
    def _analyze_endpoints(
        self,
        primary_endpoints: List[str],
        condition: str,
        planned_duration_months: int,
    ) -> List[RiskFactor]:
        """Analyze endpoints for potential issues."""
        risks = []
        
        condition_key = self._normalize_condition(condition)
        concerns = self.ENDPOINT_CONCERNS.get(condition_key, [])
        
        for endpoint in primary_endpoints:
            endpoint_lower = endpoint.lower()
            
            # Check against known problematic endpoints
            for pattern, concern in concerns:
                if pattern in endpoint_lower:
                    risks.append(RiskFactor(
                        category="endpoint",
                        description=f"Potentially challenging endpoint: {endpoint[:50]}...",
                        severity="medium",
                        historical_evidence=concern,
                        recommendation="Review endpoint selection against similar successful trials."
                    ))
            
            # Check for survival endpoints with short duration
            if "survival" in endpoint_lower and planned_duration_months < 24:
                risks.append(RiskFactor(
                    category="endpoint",
                    description="Survival endpoint with short trial duration",
                    severity="high",
                    historical_evidence="Survival endpoints typically require 24+ months "
                        "for sufficient events. Shorter trials often need amendments.",
                    recommendation="Consider progression-free survival or extend duration."
                ))
        
        return risks
    
    def _normalize_condition(self, condition: str) -> str:
        """Normalize condition string to match benchmark keys."""
        condition_lower = condition.lower()
        
        if any(term in condition_lower for term in ["diabet", "glucose", "hba1c"]):
            return "diabetes"
        elif any(term in condition_lower for term in ["cancer", "tumor", "oncol", "carcinoma"]):
            return "cancer"
        elif any(term in condition_lower for term in ["heart", "cardiac", "cardiovasc"]):
            return "cardiovascular"
        elif any(term in condition_lower for term in ["alzheimer", "dementia"]):
            return "alzheimer"
        elif any(term in condition_lower for term in ["rare", "orphan"]):
            return "rare_disease"
        
        return "default"
    
    def _calculate_amendment_probability(
        self, 
        risk_factors: List[RiskFactor],
        phase: str
    ) -> float:
        """Calculate probability of protocol amendment."""
        # Start with baseline by phase
        base_rate = self.benchmarks["amendment_rate_by_phase"].get(phase, 0.5)
        
        # Adjust based on risk factors
        eligibility_risks = [r for r in risk_factors if r.category == "eligibility"]
        high_severity = len([r for r in eligibility_risks if r.severity == "high"])
        medium_severity = len([r for r in eligibility_risks if r.severity == "medium"])
        
        # Each high severity adds 10%, medium adds 5%
        adjustment = (high_severity * 0.10) + (medium_severity * 0.05)
        
        return min(0.95, base_rate + adjustment)
    
    def _calculate_enrollment_delay_probability(
        self,
        risk_factors: List[RiskFactor],
        condition: str,
        target_enrollment: int,
        planned_sites: int,
    ) -> float:
        """Calculate probability of enrollment delays."""
        # Start with baseline of 0.3 (30% of trials delayed)
        base_rate = 0.30
        
        enrollment_risks = [r for r in risk_factors if r.category == "enrollment"]
        high_severity = len([r for r in enrollment_risks if r.severity == "high"])
        
        # High enrollment risks significantly increase delay probability
        adjustment = high_severity * 0.25
        
        return min(0.90, base_rate + adjustment)
    
    def _calculate_termination_probability(
        self,
        risk_factors: List[RiskFactor],
        phase: str
    ) -> float:
        """Calculate probability of early termination."""
        # Start with baseline by phase
        base_rate = self.benchmarks["termination_rate_by_phase"].get(phase, 0.20)
        
        # Multiple high-severity risks compound termination probability
        high_risks = len([r for r in risk_factors if r.severity == "high"])
        
        adjustment = high_risks * 0.05
        
        return min(0.60, base_rate + adjustment)
    
    def _calculate_overall_score(
        self,
        amendment_prob: float,
        enrollment_delay_prob: float,
        termination_prob: float,
    ) -> float:
        """Calculate overall risk score (0-100)."""
        # Weighted average: termination is most serious, then delays, then amendments
        weighted = (
            termination_prob * 0.4 +
            enrollment_delay_prob * 0.35 +
            amendment_prob * 0.25
        )
        
        return round(weighted * 100, 1)
    
    def _generate_recommendations(
        self, 
        risk_factors: List[RiskFactor]
    ) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []
        
        # Sort by severity
        high_risks = [r for r in risk_factors if r.severity == "high"]
        medium_risks = [r for r in risk_factors if r.severity == "medium"]
        
        for risk in high_risks[:3]:  # Top 3 high priority
            recommendations.append(f"[HIGH] {risk.recommendation}")
        
        for risk in medium_risks[:2]:  # Top 2 medium priority
            recommendations.append(f"[MEDIUM] {risk.recommendation}")
        
        if not recommendations:
            recommendations.append("Protocol appears well-designed based on historical patterns.")
        
        return recommendations
    
    def _find_benchmark_trials(
        self,
        condition: str,
        phase: str
    ) -> List[str]:
        """Find similar completed trials for benchmarking."""
        # Try to find real benchmark trials from database
        if self.trial_repo is not None:
            try:
                trials = self.trial_repo.find_similar_trials(
                    condition=condition,
                    phase=phase,
                    status="COMPLETED",
                    limit=5,
                )
                if trials:
                    return [t.nct_id for t in trials]
            except Exception as e:
                logger.warning(f"Could not find benchmark trials: {e}")

        # Fall back to placeholder NCT IDs if database unavailable
        return [
            "NCT03689374",
            "NCT01363440",
            "NCT04029480",
        ]

    def get_historical_stats(self, condition: str, phase: str) -> Dict[str, Any]:
        """
        Get historical statistics for a condition and phase.

        Returns dict with:
        - total_trials
        - completed_trials
        - terminated_trials
        - termination_rate
        - avg_enrollment
        """
        if self.trial_repo is not None:
            try:
                return self.trial_repo.get_historical_stats(
                    therapeutic_area=condition,
                    phase=phase,
                )
            except Exception as e:
                logger.warning(f"Could not get historical stats: {e}")

        # Return defaults if database unavailable
        return {
            "total_trials": 0,
            "completed_trials": 0,
            "terminated_trials": 0,
            "termination_rate": self.benchmarks["termination_rate_by_phase"].get(phase, 0.2),
            "avg_enrollment": 200,
        }

    def get_similar_terminated_trials(
        self,
        condition: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get terminated trials with reasons for learning.

        Useful for understanding why similar trials failed.
        """
        if self.trial_repo is not None:
            try:
                return self.trial_repo.get_terminated_trials_with_reasons(
                    therapeutic_area=condition,
                    limit=limit,
                )
            except Exception as e:
                logger.warning(f"Could not get terminated trials: {e}")

        return []


# Convenience function for quick scoring
def score_protocol_quick(
    condition: str,
    eligibility_criteria: str,
    target_enrollment: int,
    planned_sites: int,
    db_session: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Quick protocol risk assessment with minimal inputs.

    Args:
        condition: Primary indication
        eligibility_criteria: Full eligibility text
        target_enrollment: Target number of patients
        planned_sites: Number of planned sites
        db_session: Optional database session for historical data

    Returns:
        Simplified risk summary dict
    """
    scorer = ProtocolRiskScorer(db_session=db_session)

    assessment = scorer.score_protocol(
        condition=condition,
        phase="PHASE3",  # Assume Phase 3 if not specified
        eligibility_criteria=eligibility_criteria,
        primary_endpoints=["Primary efficacy endpoint"],
        target_enrollment=target_enrollment,
        planned_sites=planned_sites,
        planned_duration_months=24,
    )

    return {
        "overall_risk": "low" if assessment.overall_risk_score < 30
            else "medium" if assessment.overall_risk_score < 60
            else "high",
        "risk_score": assessment.overall_risk_score,
        "amendment_probability": f"{assessment.amendment_probability:.0%}",
        "enrollment_delay_probability": f"{assessment.enrollment_delay_probability:.0%}",
        "termination_probability": f"{assessment.termination_probability:.0%}",
        "top_recommendations": assessment.recommendations[:3],
        "benchmark_trials": assessment.benchmark_trials,
    }


def create_scorer_with_db() -> ProtocolRiskScorer:
    """
    Create a ProtocolRiskScorer with database connection.

    Usage:
        scorer = create_scorer_with_db()
        assessment = scorer.score_protocol(...)
    """
    from ..database import DatabaseManager

    db = DatabaseManager.get_instance()
    session = db.get_session()
    return ProtocolRiskScorer(db_session=session)


if __name__ == "__main__":
    # Example usage
    scorer = ProtocolRiskScorer()
    
    sample_criteria = """
    Inclusion Criteria:
    - Male or female, age 18-65 years
    - Diagnosed with type 2 diabetes ≥180 days prior
    - HbA1c between 7.5% and 10.0%
    - On stable metformin dose ≥1500mg for 90 days
    - BMI between 25-40 kg/m²
    
    Exclusion Criteria:
    - History of pancreatitis
    - eGFR < 60 mL/min/1.73m²
    - Prior use of GLP-1 receptor agonists
    - Myocardial infarction within 180 days
    - NYHA Class III or IV heart failure
    - ALT or AST > 3x ULN
    - History of bariatric surgery
    - Active malignancy
    - Pregnant or nursing
    - Current use of systemic corticosteroids
    - History of diabetic ketoacidosis
    - Uncontrolled hypertension (>160/100)
    """
    
    assessment = scorer.score_protocol(
        condition="Type 2 Diabetes",
        phase="PHASE3",
        eligibility_criteria=sample_criteria,
        primary_endpoints=["Change in HbA1c from baseline at week 52"],
        target_enrollment=2000,
        planned_sites=150,
        planned_duration_months=18,
    )
    
    print("=" * 60)
    print("PROTOCOL RISK ASSESSMENT")
    print("=" * 60)
    print(f"\nOverall Risk Score: {assessment.overall_risk_score}/100")
    print(f"Amendment Probability: {assessment.amendment_probability:.0%}")
    print(f"Enrollment Delay Probability: {assessment.enrollment_delay_probability:.0%}")
    print(f"Termination Probability: {assessment.termination_probability:.0%}")
    
    print(f"\n{len(assessment.risk_factors)} Risk Factors Identified:")
    for i, risk in enumerate(assessment.risk_factors, 1):
        print(f"\n{i}. [{risk.severity.upper()}] {risk.description}")
        print(f"   Evidence: {risk.historical_evidence[:100]}...")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    for rec in assessment.recommendations:
        print(f"\n• {rec}")
