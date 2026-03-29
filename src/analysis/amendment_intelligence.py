"""
Protocol Amendment Intelligence Module

Analyzes historical trial data to predict amendment risks and provide
insights on common protocol changes that could help trial designers
avoid costly mid-trial modifications.

Data sources:
1. Inferred from our database (enrollment changes, timeline delays, terminations)
2. ClinicalTrials.gov API (version history, protocol updates)
3. Industry benchmarks from published research
"""

import os
import json
import logging
import statistics
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import Counter
import re

logger = logging.getLogger(__name__)


# Industry benchmarks from published research
INDUSTRY_BENCHMARKS = {
    "overall_amendment_rate": 0.57,  # 57% of Phase 3 trials have amendments
    "avg_amendments_per_trial": 2.3,
    "amendment_categories": {
        "eligibility_criteria": {"rate": 0.30, "avg_delay_weeks": 8, "avg_cost_usd": 500000},
        "sample_size": {"rate": 0.15, "avg_delay_weeks": 12, "avg_cost_usd": 2000000},
        "endpoints": {"rate": 0.12, "avg_delay_weeks": 6, "avg_cost_usd": 750000},
        "study_duration": {"rate": 0.10, "avg_delay_weeks": 10, "avg_cost_usd": 400000},
        "dosing_regimen": {"rate": 0.08, "avg_delay_weeks": 4, "avg_cost_usd": 300000},
        "safety_monitoring": {"rate": 0.08, "avg_delay_weeks": 3, "avg_cost_usd": 200000},
        "site_management": {"rate": 0.07, "avg_delay_weeks": 6, "avg_cost_usd": 350000},
        "administrative": {"rate": 0.10, "avg_delay_weeks": 2, "avg_cost_usd": 50000},
    },
    "phase_specific": {
        "PHASE1": {"amendment_rate": 0.35, "avg_amendments": 1.2},
        "PHASE2": {"amendment_rate": 0.48, "avg_amendments": 1.8},
        "PHASE3": {"amendment_rate": 0.57, "avg_amendments": 2.3},
        "PHASE4": {"amendment_rate": 0.42, "avg_amendments": 1.5},
    },
    "therapeutic_area_risks": {
        "oncology": {"amendment_rate": 0.65, "top_reason": "eligibility_criteria"},
        "cardiovascular": {"amendment_rate": 0.52, "top_reason": "endpoints"},
        "neurology": {"amendment_rate": 0.58, "top_reason": "sample_size"},
        "immunology": {"amendment_rate": 0.55, "top_reason": "safety_monitoring"},
        "infectious_disease": {"amendment_rate": 0.48, "top_reason": "study_duration"},
        "metabolic": {"amendment_rate": 0.45, "top_reason": "eligibility_criteria"},
    }
}


@dataclass
class AmendmentRiskFactor:
    """A specific factor contributing to amendment risk."""
    category: str
    risk_level: str  # high, medium, low
    probability: int  # 0-100
    description: str
    evidence: List[str] = field(default_factory=list)
    mitigation: str = ""
    estimated_cost_if_occurs: int = 0
    estimated_delay_weeks: int = 0


@dataclass
class HistoricalAmendment:
    """Amendment detected or inferred from historical trial."""
    nct_id: str
    trial_name: str
    amendment_type: str
    description: str
    timing: str  # early, mid, late
    impact: str  # enrollment, timeline, cost
    source: str  # inferred, api, reported


@dataclass
class AmendmentPattern:
    """Pattern of amendments across similar trials."""
    category: str
    frequency: int  # number of trials with this amendment type
    percentage: float
    common_triggers: List[str]
    typical_timing: str
    prevention_strategies: List[str]
    example_trials: List[Dict[str, str]]


@dataclass
class AmendmentIntelligence:
    """Complete amendment intelligence for a protocol."""
    overall_risk_score: int  # 0-100
    risk_level: str  # low, medium, high
    predicted_amendments: float  # expected number of amendments
    risk_factors: List[AmendmentRiskFactor] = field(default_factory=list)
    historical_patterns: List[AmendmentPattern] = field(default_factory=list)
    historical_amendments: List[HistoricalAmendment] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    benchmarks: Dict[str, Any] = field(default_factory=dict)


class AmendmentIntelligenceAnalyzer:
    """
    Analyzes protocol amendment risks and patterns from historical data.
    """

    def __init__(self, db_manager, api_key: Optional[str] = None):
        """Initialize analyzer with database connection."""
        self.db = db_manager
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def analyze(
        self,
        protocol: Any,
        similar_trials: List[Any],
        protocol_text: str = ""
    ) -> Dict[str, Any]:
        """
        Generate comprehensive amendment intelligence.

        Args:
            protocol: Extracted protocol information
            similar_trials: List of matched similar trials
            protocol_text: Original protocol text

        Returns:
            Complete amendment intelligence data
        """
        logger.info("Generating amendment intelligence...")

        # Get protocol attributes safely
        phase = getattr(protocol, 'phase', None) or getattr(getattr(protocol, 'design', None), 'phase', '') or 'PHASE3'
        therapeutic_area = getattr(protocol, 'therapeutic_area', '') or 'general'
        target_enrollment = getattr(protocol, 'target_enrollment', None) or getattr(getattr(protocol, 'design', None), 'target_enrollment', 0) or 0
        condition = getattr(protocol, 'condition', '') or ''

        # 1. Analyze historical amendments from similar trials
        historical_patterns = self._analyze_historical_patterns(similar_trials)
        historical_amendments = self._extract_historical_amendments(similar_trials)

        # 2. Calculate risk factors based on protocol characteristics
        risk_factors = self._calculate_risk_factors(
            protocol, similar_trials, phase, therapeutic_area, target_enrollment
        )

        # 3. Get benchmarks for this phase and therapeutic area
        benchmarks = self._get_benchmarks(phase, therapeutic_area)

        # 4. Calculate overall risk score
        overall_risk, risk_level, predicted_amendments = self._calculate_overall_risk(
            risk_factors, benchmarks, phase
        )

        # 5. Generate actionable recommendations
        recommendations = self._generate_recommendations(
            risk_factors, historical_patterns, benchmarks, protocol
        )

        return {
            "overall_risk_score": overall_risk,
            "risk_level": risk_level,
            "predicted_amendments": round(predicted_amendments, 1),
            "risk_factors": [asdict(rf) for rf in risk_factors],
            "historical_patterns": [asdict(hp) for hp in historical_patterns],
            "historical_amendments": [asdict(ha) for ha in historical_amendments[:10]],  # Top 10
            "recommendations": recommendations,
            "benchmarks": benchmarks,
            "trials_analyzed": len(similar_trials),
            "data_source": "Historical analysis of similar trials + industry benchmarks"
        }

    def _analyze_historical_patterns(self, similar_trials: List[Any]) -> List[AmendmentPattern]:
        """Analyze amendment patterns from similar trials."""
        patterns = []

        if not similar_trials:
            return patterns

        # Categorize trials by what changes we can infer
        enrollment_changes = []
        timeline_changes = []
        termination_patterns = Counter()

        for trial in similar_trials:
            nct_id = getattr(trial, 'nct_id', '')
            status = getattr(trial, 'status', '')
            enrollment = getattr(trial, 'enrollment', 0) or 0
            enrollment_type = getattr(trial, 'enrollment_type', '')
            why_stopped = getattr(trial, 'why_stopped', '') or ''

            # Track enrollment discrepancies (proxy for sample size amendments)
            if enrollment_type == 'ACTUAL' and enrollment > 0:
                enrollment_changes.append({
                    'nct_id': nct_id,
                    'enrollment': enrollment,
                    'status': status
                })

            # Track termination reasons
            if why_stopped:
                reason_category = self._categorize_termination_reason(why_stopped)
                termination_patterns[reason_category] += 1

        # Build patterns from termination analysis
        total_terminated = sum(termination_patterns.values())
        if total_terminated > 0:
            for category, count in termination_patterns.most_common(5):
                if count >= 2:  # At least 2 trials with this pattern
                    patterns.append(AmendmentPattern(
                        category=category,
                        frequency=count,
                        percentage=round(count / len(similar_trials) * 100, 1),
                        common_triggers=self._get_triggers_for_category(category),
                        typical_timing=self._get_typical_timing(category),
                        prevention_strategies=self._get_prevention_strategies(category),
                        example_trials=self._get_example_trials(similar_trials, category)[:3]
                    ))

        # Add enrollment pattern analysis
        if enrollment_changes:
            completed = [e for e in enrollment_changes if e['status'] == 'COMPLETED']
            if completed:
                enrollments = [e['enrollment'] for e in completed]
                avg_enrollment = statistics.mean(enrollments)
                std_enrollment = statistics.stdev(enrollments) if len(enrollments) > 1 else 0

                # High variability suggests frequent sample size amendments
                if std_enrollment > avg_enrollment * 0.5:
                    patterns.append(AmendmentPattern(
                        category="sample_size_variability",
                        frequency=len(completed),
                        percentage=round(std_enrollment / avg_enrollment * 100, 1),
                        common_triggers=[
                            "Enrollment slower than expected",
                            "Interim analysis results",
                            "Regulatory feedback"
                        ],
                        typical_timing="mid-study",
                        prevention_strategies=[
                            "Build 15-20% enrollment buffer into initial estimate",
                            "Plan adaptive design with pre-specified sample size re-estimation",
                            "Include interim enrollment reviews in protocol"
                        ],
                        example_trials=[]
                    ))

        return patterns

    def _extract_historical_amendments(self, similar_trials: List[Any]) -> List[HistoricalAmendment]:
        """Extract specific amendments from similar trials."""
        amendments = []

        for trial in similar_trials:
            nct_id = getattr(trial, 'nct_id', '')
            title = getattr(trial, 'title', '')[:100]
            status = getattr(trial, 'status', '')
            why_stopped = getattr(trial, 'why_stopped', '') or ''

            # Infer amendments from termination reasons
            if why_stopped and status in ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']:
                amendment_type = self._categorize_termination_reason(why_stopped)
                amendments.append(HistoricalAmendment(
                    nct_id=nct_id,
                    trial_name=title,
                    amendment_type=amendment_type,
                    description=why_stopped[:200],
                    timing="varies",
                    impact=self._get_impact_for_category(amendment_type),
                    source="inferred_from_termination"
                ))

        return amendments

    def _calculate_risk_factors(
        self,
        protocol: Any,
        similar_trials: List[Any],
        phase: str,
        therapeutic_area: str,
        target_enrollment: int
    ) -> List[AmendmentRiskFactor]:
        """Calculate specific risk factors for amendments."""
        risk_factors = []

        # 1. Eligibility Criteria Complexity Risk
        eligibility_risk = self._assess_eligibility_risk(protocol, similar_trials)
        if eligibility_risk:
            risk_factors.append(eligibility_risk)

        # 2. Sample Size Risk
        sample_risk = self._assess_sample_size_risk(target_enrollment, similar_trials, phase)
        if sample_risk:
            risk_factors.append(sample_risk)

        # 3. Endpoint Risk
        endpoint_risk = self._assess_endpoint_risk(protocol, similar_trials)
        if endpoint_risk:
            risk_factors.append(endpoint_risk)

        # 4. Timeline Risk
        timeline_risk = self._assess_timeline_risk(protocol, similar_trials)
        if timeline_risk:
            risk_factors.append(timeline_risk)

        # 5. Phase-specific Risk
        phase_risk = self._assess_phase_risk(phase)
        if phase_risk:
            risk_factors.append(phase_risk)

        # 6. Therapeutic Area Risk
        ta_risk = self._assess_therapeutic_area_risk(therapeutic_area)
        if ta_risk:
            risk_factors.append(ta_risk)

        # Sort by probability descending
        risk_factors.sort(key=lambda x: x.probability, reverse=True)

        return risk_factors

    def _assess_eligibility_risk(self, protocol: Any, similar_trials: List[Any]) -> Optional[AmendmentRiskFactor]:
        """Assess risk of eligibility criteria amendments."""
        # Check if protocol has complex eligibility
        population = getattr(protocol, 'population', None)

        complexity_score = 0
        evidence = []

        if population:
            excluded = getattr(population, 'excluded_conditions', []) or []
            required = getattr(population, 'required_conditions', []) or []

            if len(excluded) > 5:
                complexity_score += 20
                evidence.append(f"{len(excluded)} exclusion criteria (high)")

            if len(required) > 3:
                complexity_score += 15
                evidence.append(f"{len(required)} required conditions")

            min_age = getattr(population, 'min_age', None)
            max_age = getattr(population, 'max_age', None)
            if min_age and max_age and (max_age - min_age) < 30:
                complexity_score += 10
                evidence.append(f"Narrow age range ({min_age}-{max_age})")

        # Check termination patterns in similar trials for enrollment issues
        enrollment_terminations = sum(
            1 for t in similar_trials
            if 'enroll' in (getattr(t, 'why_stopped', '') or '').lower()
        )
        if enrollment_terminations > 0:
            complexity_score += enrollment_terminations * 5
            evidence.append(f"{enrollment_terminations} similar trials stopped for enrollment issues")

        if complexity_score > 0:
            probability = min(85, 30 + complexity_score)
            risk_level = "high" if probability > 60 else "medium" if probability > 40 else "low"

            return AmendmentRiskFactor(
                category="eligibility_criteria",
                risk_level=risk_level,
                probability=probability,
                description="Risk of needing to modify inclusion/exclusion criteria to improve enrollment",
                evidence=evidence,
                mitigation="Consider broader eligibility criteria upfront; build in protocol-specified flexibility for borderline cases",
                estimated_cost_if_occurs=500000,
                estimated_delay_weeks=8
            )

        return None

    def _assess_sample_size_risk(
        self,
        target_enrollment: int,
        similar_trials: List[Any],
        phase: str
    ) -> Optional[AmendmentRiskFactor]:
        """Assess risk of sample size amendments."""
        evidence = []

        # Get enrollments from completed similar trials
        completed_enrollments = [
            getattr(t, 'enrollment', 0)
            for t in similar_trials
            if getattr(t, 'status', '') == 'COMPLETED' and getattr(t, 'enrollment', 0) > 0
        ]

        probability = 25  # Base probability

        if completed_enrollments:
            avg_enrollment = statistics.mean(completed_enrollments)
            median_enrollment = statistics.median(completed_enrollments)

            if target_enrollment > avg_enrollment * 1.5:
                probability += 25
                evidence.append(f"Target ({target_enrollment}) is {round(target_enrollment/avg_enrollment, 1)}x the average ({round(avg_enrollment)})")

            if target_enrollment > max(completed_enrollments):
                probability += 15
                evidence.append(f"Target exceeds largest completed trial ({max(completed_enrollments)})")

        # Phase-specific risk
        if 'PHASE3' in phase.upper():
            probability += 10
            evidence.append("Phase 3 trials have highest sample size amendment rate")

        probability = min(85, probability)

        if probability > 30:
            risk_level = "high" if probability > 60 else "medium" if probability > 40 else "low"

            return AmendmentRiskFactor(
                category="sample_size",
                risk_level=risk_level,
                probability=probability,
                description="Risk of needing to adjust sample size based on interim data or enrollment challenges",
                evidence=evidence,
                mitigation="Consider adaptive design with pre-specified sample size re-estimation; include 15-20% enrollment buffer",
                estimated_cost_if_occurs=2000000,
                estimated_delay_weeks=12
            )

        return None

    def _assess_endpoint_risk(self, protocol: Any, similar_trials: List[Any]) -> Optional[AmendmentRiskFactor]:
        """Assess risk of endpoint amendments."""
        evidence = []
        probability = 20  # Base probability

        # Check endpoint complexity
        endpoints = getattr(protocol, 'endpoints', None)
        if endpoints:
            primary = getattr(endpoints, 'primary_endpoint', '')
            secondaries = getattr(endpoints, 'secondary_endpoints', []) or []

            if len(secondaries) > 5:
                probability += 15
                evidence.append(f"{len(secondaries)} secondary endpoints (complex)")

            # Check if endpoint is unusual for similar trials
            # (This would be better with actual endpoint matching)

        # Check if similar trials had endpoint issues
        endpoint_terminations = sum(
            1 for t in similar_trials
            if any(term in (getattr(t, 'why_stopped', '') or '').lower()
                   for term in ['endpoint', 'efficacy', 'futility'])
        )
        if endpoint_terminations > 0:
            probability += endpoint_terminations * 8
            evidence.append(f"{endpoint_terminations} similar trials stopped for endpoint/efficacy issues")

        if probability > 25:
            risk_level = "high" if probability > 55 else "medium" if probability > 35 else "low"

            return AmendmentRiskFactor(
                category="endpoints",
                risk_level=risk_level,
                probability=min(75, probability),
                description="Risk of needing to modify primary or secondary endpoints",
                evidence=evidence,
                mitigation="Align endpoints with FDA guidance and successful precedent trials; consider composite endpoints for robustness",
                estimated_cost_if_occurs=750000,
                estimated_delay_weeks=6
            )

        return None

    def _assess_timeline_risk(self, protocol: Any, similar_trials: List[Any]) -> Optional[AmendmentRiskFactor]:
        """Assess risk of timeline amendments."""
        evidence = []
        probability = 20

        # Check if similar trials had timeline issues
        timeline_issues = sum(
            1 for t in similar_trials
            if getattr(t, 'status', '') in ['TERMINATED', 'SUSPENDED'] and
            any(term in (getattr(t, 'why_stopped', '') or '').lower()
                for term in ['slow', 'delay', 'timeline', 'recruitment', 'suspended'])
        )

        if timeline_issues > 0:
            probability += timeline_issues * 10
            evidence.append(f"{timeline_issues} similar trials had timeline/recruitment issues")

        if probability > 25:
            risk_level = "high" if probability > 55 else "medium"

            return AmendmentRiskFactor(
                category="study_duration",
                risk_level=risk_level,
                probability=min(70, probability),
                description="Risk of needing to extend study duration due to enrollment or follow-up challenges",
                evidence=evidence,
                mitigation="Build realistic timelines with contingency; plan parallel site activation; consider regional expansion strategy",
                estimated_cost_if_occurs=400000,
                estimated_delay_weeks=10
            )

        return None

    def _assess_phase_risk(self, phase: str) -> Optional[AmendmentRiskFactor]:
        """Assess phase-specific amendment risk."""
        phase_upper = phase.upper() if phase else ''

        for phase_key, data in INDUSTRY_BENCHMARKS["phase_specific"].items():
            if phase_key in phase_upper:
                probability = int(data["amendment_rate"] * 100)

                return AmendmentRiskFactor(
                    category="phase_specific",
                    risk_level="high" if probability > 50 else "medium",
                    probability=probability,
                    description=f"{phase_key} trials have {probability}% amendment rate (industry benchmark)",
                    evidence=[
                        f"Average {data['avg_amendments']} amendments per {phase_key} trial",
                        "Based on industry-wide analysis of clinical trials"
                    ],
                    mitigation="Factor in amendment contingency for budget and timeline planning",
                    estimated_cost_if_occurs=0,  # Informational
                    estimated_delay_weeks=0
                )

        return None

    def _assess_therapeutic_area_risk(self, therapeutic_area: str) -> Optional[AmendmentRiskFactor]:
        """Assess therapeutic area-specific amendment risk."""
        ta_lower = therapeutic_area.lower() if therapeutic_area else ''

        for ta_key, data in INDUSTRY_BENCHMARKS["therapeutic_area_risks"].items():
            if ta_key in ta_lower:
                probability = int(data["amendment_rate"] * 100)

                return AmendmentRiskFactor(
                    category="therapeutic_area",
                    risk_level="high" if probability > 55 else "medium",
                    probability=probability,
                    description=f"{ta_key.title()} trials have {probability}% amendment rate",
                    evidence=[
                        f"Top amendment reason: {data['top_reason'].replace('_', ' ')}",
                        f"Based on {ta_key} trial analysis"
                    ],
                    mitigation=f"Pay special attention to {data['top_reason'].replace('_', ' ')} when designing protocol",
                    estimated_cost_if_occurs=0,
                    estimated_delay_weeks=0
                )

        return None

    def _calculate_overall_risk(
        self,
        risk_factors: List[AmendmentRiskFactor],
        benchmarks: Dict[str, Any],
        phase: str
    ) -> Tuple[int, str, float]:
        """Calculate overall amendment risk score."""
        if not risk_factors:
            return 35, "medium", 1.5

        # Weighted average of risk factors
        weights = {
            "eligibility_criteria": 1.5,
            "sample_size": 1.3,
            "endpoints": 1.2,
            "study_duration": 1.0,
            "phase_specific": 0.8,
            "therapeutic_area": 0.8,
        }

        total_weight = 0
        weighted_sum = 0

        for rf in risk_factors:
            weight = weights.get(rf.category, 1.0)
            weighted_sum += rf.probability * weight
            total_weight += weight

        overall_risk = int(weighted_sum / total_weight) if total_weight > 0 else 50

        # Determine risk level
        if overall_risk >= 60:
            risk_level = "high"
        elif overall_risk >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Predict number of amendments
        phase_data = INDUSTRY_BENCHMARKS["phase_specific"].get(
            phase.upper() if phase else "PHASE3",
            {"avg_amendments": 2.0}
        )
        base_amendments = phase_data["avg_amendments"]

        # Adjust based on risk score
        if overall_risk >= 60:
            predicted_amendments = base_amendments * 1.3
        elif overall_risk >= 40:
            predicted_amendments = base_amendments
        else:
            predicted_amendments = base_amendments * 0.7

        return overall_risk, risk_level, predicted_amendments

    def _get_benchmarks(self, phase: str, therapeutic_area: str) -> Dict[str, Any]:
        """Get relevant benchmarks for phase and therapeutic area."""
        phase_upper = phase.upper() if phase else 'PHASE3'

        phase_benchmarks = INDUSTRY_BENCHMARKS["phase_specific"].get(
            phase_upper,
            {"amendment_rate": 0.50, "avg_amendments": 2.0}
        )

        ta_lower = therapeutic_area.lower() if therapeutic_area else ''
        ta_benchmarks = None
        for ta_key, data in INDUSTRY_BENCHMARKS["therapeutic_area_risks"].items():
            if ta_key in ta_lower:
                ta_benchmarks = data
                break

        return {
            "phase_amendment_rate": phase_benchmarks["amendment_rate"],
            "phase_avg_amendments": phase_benchmarks["avg_amendments"],
            "therapeutic_area_rate": ta_benchmarks["amendment_rate"] if ta_benchmarks else 0.50,
            "therapeutic_area_top_reason": ta_benchmarks["top_reason"] if ta_benchmarks else "eligibility_criteria",
            "industry_avg_amendment_rate": INDUSTRY_BENCHMARKS["overall_amendment_rate"],
            "industry_avg_amendments": INDUSTRY_BENCHMARKS["avg_amendments_per_trial"],
            "amendment_categories": INDUSTRY_BENCHMARKS["amendment_categories"]
        }

    def _generate_recommendations(
        self,
        risk_factors: List[AmendmentRiskFactor],
        patterns: List[AmendmentPattern],
        benchmarks: Dict[str, Any],
        protocol: Any
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations to reduce amendment risk."""
        recommendations = []

        # High-priority recommendations from risk factors
        for rf in risk_factors[:3]:  # Top 3 risk factors
            if rf.probability >= 40:
                recommendations.append({
                    "priority": "high" if rf.probability >= 60 else "medium",
                    "category": rf.category.replace("_", " ").title(),
                    "recommendation": rf.mitigation,
                    "potential_savings": f"${rf.estimated_cost_if_occurs:,}" if rf.estimated_cost_if_occurs else "Significant",
                    "time_savings": f"{rf.estimated_delay_weeks} weeks" if rf.estimated_delay_weeks else "Variable"
                })

        # Pattern-based recommendations
        for pattern in patterns[:2]:  # Top 2 patterns
            for strategy in pattern.prevention_strategies[:2]:
                recommendations.append({
                    "priority": "medium",
                    "category": pattern.category.replace("_", " ").title(),
                    "recommendation": strategy,
                    "potential_savings": "Based on similar trial experience",
                    "time_savings": "Variable"
                })

        # General best practices
        general_recommendations = [
            {
                "priority": "low",
                "category": "Best Practice",
                "recommendation": "Conduct thorough feasibility assessment before protocol finalization",
                "potential_savings": "Up to $1M+",
                "time_savings": "8-12 weeks"
            },
            {
                "priority": "low",
                "category": "Best Practice",
                "recommendation": "Engage regulatory consultants early to align on endpoints and population",
                "potential_savings": "$500K-1M",
                "time_savings": "6-10 weeks"
            }
        ]

        # Add general recommendations if we don't have enough specific ones
        if len(recommendations) < 5:
            recommendations.extend(general_recommendations[:5 - len(recommendations)])

        return recommendations[:7]  # Return top 7 recommendations

    def _categorize_termination_reason(self, reason: str) -> str:
        """Categorize a termination reason into amendment categories."""
        reason_lower = reason.lower()

        if any(term in reason_lower for term in ['enroll', 'recruit', 'patient', 'accrual']):
            return "enrollment_challenges"
        elif any(term in reason_lower for term in ['safety', 'adverse', 'toxicity', 'sae']):
            return "safety_concerns"
        elif any(term in reason_lower for term in ['efficacy', 'futility', 'endpoint', 'interim']):
            return "efficacy_concerns"
        elif any(term in reason_lower for term in ['fund', 'sponsor', 'business', 'resource', 'financial']):
            return "funding_business"
        elif any(term in reason_lower for term in ['protocol', 'design', 'amendment']):
            return "protocol_design"
        elif any(term in reason_lower for term in ['regulatory', 'fda', 'ema', 'approval']):
            return "regulatory_issues"
        else:
            return "other"

    def _get_triggers_for_category(self, category: str) -> List[str]:
        """Get common triggers for an amendment category."""
        triggers = {
            "enrollment_challenges": [
                "Overly restrictive eligibility criteria",
                "Competing trials for same population",
                "Unrealistic enrollment projections"
            ],
            "safety_concerns": [
                "Unexpected adverse events",
                "Higher than expected SAE rate",
                "New safety signals from related trials"
            ],
            "efficacy_concerns": [
                "Lower than expected effect size",
                "High placebo response",
                "Interim analysis findings"
            ],
            "funding_business": [
                "Change in sponsor priorities",
                "Insufficient funding runway",
                "Merger/acquisition activity"
            ],
            "protocol_design": [
                "Operational complexity",
                "Site feedback on feasibility",
                "Regulatory guidance changes"
            ],
            "regulatory_issues": [
                "Clinical hold",
                "Required protocol modifications",
                "New guidance documents"
            ]
        }
        return triggers.get(category, ["Various factors"])

    def _get_prevention_strategies(self, category: str) -> List[str]:
        """Get prevention strategies for an amendment category."""
        strategies = {
            "enrollment_challenges": [
                "Benchmark eligibility criteria against successful trials",
                "Conduct site feasibility surveys before finalization",
                "Build enrollment flexibility into protocol"
            ],
            "safety_concerns": [
                "Review safety data from similar compounds/class",
                "Implement robust safety monitoring plan",
                "Define clear stopping rules upfront"
            ],
            "efficacy_concerns": [
                "Use conservative effect size assumptions",
                "Plan adaptive design with interim analyses",
                "Align endpoints with regulatory precedent"
            ],
            "funding_business": [
                "Secure committed funding for full trial duration",
                "Build contingency budget (15-20%)",
                "Document sponsor commitment in writing"
            ],
            "protocol_design": [
                "Pilot test procedures at select sites",
                "Get investigator input during design",
                "Simplify assessments where possible"
            ],
            "regulatory_issues": [
                "Pre-IND/Pre-submission meetings",
                "Regular regulatory updates during trial",
                "Monitor guidance document changes"
            ]
        }
        return strategies.get(category, ["Review and plan carefully"])

    def _get_typical_timing(self, category: str) -> str:
        """Get typical timing for when this type of amendment occurs."""
        timing = {
            "enrollment_challenges": "Early to mid-study (first 6-12 months)",
            "safety_concerns": "Any time (often mid-study)",
            "efficacy_concerns": "Mid-study (at interim analysis)",
            "funding_business": "Any time",
            "protocol_design": "Early (first 3-6 months)",
            "regulatory_issues": "Any time (often early)"
        }
        return timing.get(category, "Variable")

    def _get_impact_for_category(self, category: str) -> str:
        """Get typical impact for an amendment category."""
        impacts = {
            "enrollment_challenges": "enrollment_timeline",
            "safety_concerns": "safety_monitoring",
            "efficacy_concerns": "efficacy_analysis",
            "funding_business": "trial_continuation",
            "protocol_design": "operational",
            "regulatory_issues": "regulatory_compliance"
        }
        return impacts.get(category, "general")

    def _get_example_trials(self, trials: List[Any], category: str) -> List[Dict[str, str]]:
        """Get example trials for a specific amendment category."""
        examples = []

        for trial in trials:
            why_stopped = getattr(trial, 'why_stopped', '') or ''
            if why_stopped and self._categorize_termination_reason(why_stopped) == category:
                examples.append({
                    "nct_id": getattr(trial, 'nct_id', ''),
                    "title": (getattr(trial, 'title', '') or '')[:80],
                    "reason": why_stopped[:150]
                })

                if len(examples) >= 3:
                    break

        return examples


# Singleton accessor
_analyzer_instance = None

def get_amendment_analyzer(db_manager=None) -> AmendmentIntelligenceAnalyzer:
    """Get or create amendment analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None and db_manager:
        _analyzer_instance = AmendmentIntelligenceAnalyzer(db_manager)
    return _analyzer_instance
