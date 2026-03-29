"""
Enhanced Endpoint Benchmarking System

Provides comprehensive endpoint analysis including:
1. Endpoint classification and categorization
2. Regulatory acceptance analysis (FDA/EMA guidance)
3. Success rate patterns by endpoint type
4. Measurement timing optimization
5. Clinical meaningfulness assessment
6. Secondary endpoint recommendations
7. AI-powered endpoint optimization suggestions
"""

import os
import json
import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EndpointClassification:
    """Classification of an endpoint."""
    endpoint_text: str
    primary_category: str  # survival, progression, response, biomarker, pro, safety, composite
    subcategory: str
    measurement_type: str  # time_to_event, continuous, binary, ordinal
    regulatory_status: str  # established, acceptable, exploratory
    clinical_meaningfulness: str  # high, moderate, low

    # Extracted details
    timeframe: Optional[str] = None
    measurement_method: Optional[str] = None
    responder_definition: Optional[str] = None


@dataclass
class EndpointBenchmark:
    """Benchmark data for an endpoint type."""
    endpoint_type: str
    category: str

    # Success metrics
    trials_analyzed: int
    completed_trials: int
    success_rate: float  # completion rate

    # Trial characteristics
    avg_enrollment: float
    avg_duration_months: float

    # By phase
    phase_breakdown: Dict[str, Dict[str, float]]  # phase -> {success_rate, count}

    # Regulatory context
    fda_accepted: bool
    ema_accepted: bool
    regulatory_notes: str

    # Historical examples
    successful_examples: List[str]  # NCT IDs
    failed_examples: List[str]


@dataclass
class TimingAnalysis:
    """Analysis of endpoint timing patterns."""
    endpoint_type: str

    # Typical assessment windows
    typical_primary_timepoint: str
    typical_range: Tuple[int, int]  # weeks

    # Optimal timing
    recommended_timing: str
    timing_rationale: str

    # Risk considerations
    too_early_risks: List[str]
    too_late_risks: List[str]


@dataclass
class EndpointRecommendation:
    """Recommendation for endpoint selection/modification."""
    priority: str  # high, medium, low
    recommendation_type: str  # change_primary, add_secondary, modify_timing, improve_definition

    current_endpoint: Optional[str]
    recommended_change: str
    rationale: str
    expected_impact: str

    evidence: List[str]
    regulatory_support: str


@dataclass
class SecondaryEndpointSuggestion:
    """Suggested secondary endpoint."""
    endpoint: str
    category: str
    rationale: str

    complementarity: str  # how it complements primary
    regulatory_value: str
    feasibility: str  # high, medium, low

    examples_in_trials: int


@dataclass
class EndpointBenchmarkReport:
    """Complete endpoint benchmarking report."""
    condition: str
    phase: str

    # User's endpoint analysis
    primary_endpoint_classification: Optional[EndpointClassification]
    primary_endpoint_score: float  # 0-100
    primary_endpoint_assessment: str

    # Benchmarks
    benchmarks_by_category: Dict[str, EndpointBenchmark]
    best_performing_category: str
    worst_performing_category: str

    # Timing analysis
    timing_analysis: Optional[TimingAnalysis]

    # Recommendations
    recommendations: List[EndpointRecommendation]
    secondary_suggestions: List[SecondaryEndpointSuggestion]

    # Regulatory context
    regulatory_guidance: Dict[str, str]  # agency -> guidance
    established_endpoints: List[str]

    # Summary
    key_findings: List[str]
    overall_assessment: str


class EndpointBenchmarker:
    """Comprehensive endpoint benchmarking engine."""

    # Endpoint category patterns
    ENDPOINT_CATEGORIES = {
        "survival": {
            "patterns": [r"overall\s*survival", r"\bos\b", r"mortality", r"death", r"survival\s*rate"],
            "subcategories": ["overall_survival", "cause_specific_survival", "event_free_survival"],
            "measurement_type": "time_to_event",
            "regulatory_status": "established",
            "clinical_meaningfulness": "high"
        },
        "progression": {
            "patterns": [r"progression[\s-]*free", r"\bpfs\b", r"disease[\s-]*free", r"\bdfs\b",
                        r"recurrence", r"relapse", r"time\s*to\s*progression"],
            "subcategories": ["progression_free_survival", "disease_free_survival", "time_to_progression"],
            "measurement_type": "time_to_event",
            "regulatory_status": "established",
            "clinical_meaningfulness": "high"
        },
        "response": {
            "patterns": [r"response\s*rate", r"\borr\b", r"complete\s*response", r"\bcr\b",
                        r"partial\s*response", r"\bpr\b", r"objective\s*response", r"remission"],
            "subcategories": ["objective_response", "complete_response", "partial_response", "clinical_benefit"],
            "measurement_type": "binary",
            "regulatory_status": "acceptable",
            "clinical_meaningfulness": "moderate"
        },
        "biomarker": {
            "patterns": [r"hba1c", r"a1c", r"ldl", r"blood\s*pressure", r"glucose", r"cholesterol",
                        r"triglyceride", r"egfr", r"creatinine", r"viral\s*load", r"cd4", r"tumor\s*marker"],
            "subcategories": ["metabolic", "cardiovascular", "renal", "hepatic", "immunologic", "tumor_marker"],
            "measurement_type": "continuous",
            "regulatory_status": "acceptable",
            "clinical_meaningfulness": "moderate"
        },
        "patient_reported": {
            "patterns": [r"quality\s*of\s*life", r"\bqol\b", r"patient[\s-]*reported", r"\bpro\b",
                        r"pain\s*score", r"symptom", r"function", r"sf-?36", r"eq-?5d", r"fact"],
            "subcategories": ["quality_of_life", "symptom_assessment", "functional_status", "pain"],
            "measurement_type": "ordinal",
            "regulatory_status": "acceptable",
            "clinical_meaningfulness": "high"
        },
        "safety": {
            "patterns": [r"adverse\s*event", r"\bae\b", r"safety", r"tolerability", r"toxicity",
                        r"serious\s*adverse", r"\bsae\b", r"discontinuation"],
            "subcategories": ["adverse_events", "serious_adverse_events", "tolerability", "discontinuation_rate"],
            "measurement_type": "binary",
            "regulatory_status": "established",
            "clinical_meaningfulness": "high"
        },
        "composite": {
            "patterns": [r"composite", r"mace", r"major\s*adverse", r"combined\s*endpoint",
                        r"event[\s-]*free", r"failure[\s-]*free"],
            "subcategories": ["cardiovascular_composite", "oncology_composite", "custom_composite"],
            "measurement_type": "time_to_event",
            "regulatory_status": "acceptable",
            "clinical_meaningfulness": "high"
        },
        "imaging": {
            "patterns": [r"imaging", r"mri", r"ct\s*scan", r"pet", r"tumor\s*size", r"lesion",
                        r"radiographic", r"recist"],
            "subcategories": ["tumor_response", "structural_change", "functional_imaging"],
            "measurement_type": "continuous",
            "regulatory_status": "acceptable",
            "clinical_meaningfulness": "moderate"
        }
    }

    # Regulatory guidance by therapeutic area
    REGULATORY_GUIDANCE = {
        "oncology": {
            "fda_preferred": ["Overall Survival", "Progression-Free Survival", "Objective Response Rate"],
            "ema_preferred": ["Overall Survival", "Progression-Free Survival", "Quality of Life"],
            "notes": "OS is gold standard; PFS acceptable with magnitude of benefit consideration"
        },
        "cardiovascular": {
            "fda_preferred": ["MACE", "CV Death", "Hospitalization for Heart Failure"],
            "ema_preferred": ["MACE", "All-cause Mortality", "CV Mortality"],
            "notes": "Composite endpoints common; mortality endpoints preferred"
        },
        "diabetes": {
            "fda_preferred": ["HbA1c", "CV Outcomes (MACE)"],
            "ema_preferred": ["HbA1c", "Microvascular Outcomes"],
            "notes": "HbA1c established surrogate; CV safety required"
        },
        "neurology": {
            "fda_preferred": ["Clinical Rating Scales", "Time to Disability Milestone"],
            "ema_preferred": ["Clinical Rating Scales", "Patient-Reported Outcomes"],
            "notes": "Disease-specific scales required; PROs increasingly important"
        },
        "immunology": {
            "fda_preferred": ["Clinical Response", "Remission Rate", "Disease Activity Score"],
            "ema_preferred": ["Clinical Response", "Quality of Life", "Structural Outcomes"],
            "notes": "Validated disease activity indices required"
        },
        "infectious_disease": {
            "fda_preferred": ["Clinical Cure", "Microbiological Eradication", "Mortality"],
            "ema_preferred": ["Clinical Cure", "Microbiological Response"],
            "notes": "Pathogen-specific outcomes; non-inferiority margins important"
        }
    }

    # Timing recommendations by endpoint type
    TIMING_GUIDANCE = {
        "survival": {
            "typical_range": (52, 260),  # weeks
            "recommended": "Mature data with sufficient events",
            "rationale": "Requires adequate follow-up for event accrual",
            "too_early_risks": ["Insufficient events", "Immature survival curves"],
            "too_late_risks": ["Crossover effects", "Loss to follow-up"]
        },
        "progression": {
            "typical_range": (12, 104),
            "recommended": "Based on expected disease course",
            "rationale": "Balance between sensitivity and clinical relevance",
            "too_early_risks": ["Pseudoprogression", "Assessment variability"],
            "too_late_risks": ["Treatment crossover", "Competing risks"]
        },
        "response": {
            "typical_range": (8, 24),
            "recommended": "After 2-3 treatment cycles",
            "rationale": "Allow time for treatment effect",
            "too_early_risks": ["Incomplete response", "Transient effects"],
            "too_late_risks": ["Delayed responses missed", "Selection bias"]
        },
        "biomarker": {
            "typical_range": (12, 52),
            "recommended": "Based on biomarker kinetics",
            "rationale": "Match to expected biological effect timing",
            "too_early_risks": ["Acute changes not sustained"],
            "too_late_risks": ["Regression to mean", "Compliance issues"]
        },
        "patient_reported": {
            "typical_range": (12, 52),
            "recommended": "Capture meaningful change period",
            "rationale": "Balance recall accuracy with meaningful change",
            "too_early_risks": ["Placebo effect", "Expectation bias"],
            "too_late_risks": ["Recall bias", "Response shift"]
        }
    }

    def __init__(self, db=None, api_key: Optional[str] = None):
        """Initialize benchmarker."""
        self.db = db
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            if not self.api_key:
                return None
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def analyze(self, primary_endpoint: str, condition: str, phase: str,
                secondary_endpoints: List[str] = None) -> EndpointBenchmarkReport:
        """Generate comprehensive endpoint benchmark report."""
        logger.info(f"Analyzing endpoints for {condition} {phase}")

        # Classify user's primary endpoint
        primary_classification = self._classify_endpoint(primary_endpoint) if primary_endpoint else None

        # Get benchmark data from database
        benchmarks = self._get_benchmarks(condition, phase)

        # Score the primary endpoint
        endpoint_score, assessment = self._score_endpoint(
            primary_classification, benchmarks, condition, phase
        )

        # Get timing analysis
        timing_analysis = self._analyze_timing(primary_classification) if primary_classification else None

        # Get regulatory guidance
        therapeutic_area = self._map_condition_to_therapeutic_area(condition)
        regulatory_guidance = self.REGULATORY_GUIDANCE.get(therapeutic_area, {})

        # Generate recommendations
        recommendations = self._generate_recommendations(
            primary_classification, benchmarks, condition, phase, regulatory_guidance
        )

        # Suggest secondary endpoints
        secondary_suggestions = self._suggest_secondary_endpoints(
            primary_classification, condition, phase, secondary_endpoints or []
        )

        # Identify best/worst categories
        if benchmarks:
            best_cat = max(benchmarks.items(), key=lambda x: x[1].success_rate)[0]
            worst_cat = min(benchmarks.items(), key=lambda x: x[1].success_rate)[0]
        else:
            best_cat = worst_cat = "Unknown"

        # Get established endpoints for condition
        established = self._get_established_endpoints(condition, regulatory_guidance)

        # Generate key findings
        key_findings = self._generate_key_findings(
            primary_classification, benchmarks, endpoint_score, regulatory_guidance
        )

        # Overall assessment
        overall = self._generate_overall_assessment(
            endpoint_score, primary_classification, recommendations
        )

        return EndpointBenchmarkReport(
            condition=condition,
            phase=phase,
            primary_endpoint_classification=primary_classification,
            primary_endpoint_score=endpoint_score,
            primary_endpoint_assessment=assessment,
            benchmarks_by_category=benchmarks,
            best_performing_category=best_cat,
            worst_performing_category=worst_cat,
            timing_analysis=timing_analysis,
            recommendations=recommendations,
            secondary_suggestions=secondary_suggestions,
            regulatory_guidance={"FDA": regulatory_guidance.get("fda_preferred", []),
                                "EMA": regulatory_guidance.get("ema_preferred", []),
                                "Notes": regulatory_guidance.get("notes", "")},
            established_endpoints=established,
            key_findings=key_findings,
            overall_assessment=overall
        )

    def _classify_endpoint(self, endpoint_text: str) -> EndpointClassification:
        """Classify an endpoint into categories."""
        endpoint_lower = endpoint_text.lower()

        # Find matching category
        matched_category = "other"
        matched_subcategory = "unclassified"
        measurement_type = "unknown"
        regulatory_status = "exploratory"
        clinical_meaningfulness = "moderate"

        for category, config in self.ENDPOINT_CATEGORIES.items():
            for pattern in config["patterns"]:
                if re.search(pattern, endpoint_lower):
                    matched_category = category
                    matched_subcategory = config["subcategories"][0]  # Default to first
                    measurement_type = config["measurement_type"]
                    regulatory_status = config["regulatory_status"]
                    clinical_meaningfulness = config["clinical_meaningfulness"]
                    break
            if matched_category != "other":
                break

        # Extract timeframe if present
        timeframe = None
        time_match = re.search(r'(\d+)\s*(week|month|year|day)s?', endpoint_lower)
        if time_match:
            timeframe = f"{time_match.group(1)} {time_match.group(2)}s"

        return EndpointClassification(
            endpoint_text=endpoint_text,
            primary_category=matched_category,
            subcategory=matched_subcategory,
            measurement_type=measurement_type,
            regulatory_status=regulatory_status,
            clinical_meaningfulness=clinical_meaningfulness,
            timeframe=timeframe
        )

    def _get_benchmarks(self, condition: str, phase: str) -> Dict[str, EndpointBenchmark]:
        """Get benchmark data from database."""
        if not self.db:
            return self._get_default_benchmarks()

        from sqlalchemy import text

        query = text("""
            SELECT nct_id, status, primary_outcomes, phase, enrollment,
                   CAST((julianday(completion_date) - julianday(start_date)) / 30 AS INTEGER) as duration_months
            FROM trials
            WHERE (LOWER(conditions) LIKE :condition OR LOWER(therapeutic_area) LIKE :condition)
            AND primary_outcomes IS NOT NULL
            AND status IN ('COMPLETED', 'TERMINATED', 'WITHDRAWN')
            LIMIT 500
        """)

        try:
            results = self.db.execute_raw(query.text, {
                "condition": f"%{condition.lower()}%"
            })
        except Exception as e:
            logger.error(f"Benchmark query failed: {e}")
            return self._get_default_benchmarks()

        # Categorize results
        category_data = defaultdict(lambda: {
            "trials": [], "completed": 0, "total": 0,
            "enrollments": [], "durations": [], "phase_data": defaultdict(lambda: {"success": 0, "total": 0})
        })

        for r in results:
            nct_id, status, outcomes, trial_phase, enrollment, duration = r
            outcomes_lower = (outcomes or "").lower()

            # Find category
            for category, config in self.ENDPOINT_CATEGORIES.items():
                if any(re.search(p, outcomes_lower) for p in config["patterns"]):
                    data = category_data[category]
                    data["total"] += 1
                    data["trials"].append(nct_id)

                    if status == "COMPLETED":
                        data["completed"] += 1

                    if enrollment:
                        data["enrollments"].append(enrollment)
                    if duration and duration > 0:
                        data["durations"].append(duration)

                    # Phase breakdown
                    data["phase_data"][trial_phase]["total"] += 1
                    if status == "COMPLETED":
                        data["phase_data"][trial_phase]["success"] += 1

                    break

        # Build benchmarks
        benchmarks = {}
        for category, data in category_data.items():
            if data["total"] < 3:
                continue

            config = self.ENDPOINT_CATEGORIES.get(category, {})

            phase_breakdown = {}
            for p, pd in data["phase_data"].items():
                if pd["total"] > 0:
                    phase_breakdown[p] = {
                        "success_rate": pd["success"] / pd["total"] * 100,
                        "count": pd["total"]
                    }

            benchmarks[category] = EndpointBenchmark(
                endpoint_type=category.replace("_", " ").title(),
                category=category,
                trials_analyzed=data["total"],
                completed_trials=data["completed"],
                success_rate=data["completed"] / data["total"] * 100 if data["total"] > 0 else 0,
                avg_enrollment=sum(data["enrollments"]) / len(data["enrollments"]) if data["enrollments"] else 0,
                avg_duration_months=sum(data["durations"]) / len(data["durations"]) if data["durations"] else 0,
                phase_breakdown=phase_breakdown,
                fda_accepted=config.get("regulatory_status") in ["established", "acceptable"],
                ema_accepted=config.get("regulatory_status") in ["established", "acceptable"],
                regulatory_notes="",
                successful_examples=data["trials"][:5],
                failed_examples=[]
            )

        return benchmarks

    def _get_default_benchmarks(self) -> Dict[str, EndpointBenchmark]:
        """Return default benchmarks when database unavailable."""
        defaults = {
            "survival": EndpointBenchmark(
                endpoint_type="Survival/Mortality",
                category="survival",
                trials_analyzed=100,
                completed_trials=58,
                success_rate=58.0,
                avg_enrollment=350,
                avg_duration_months=36,
                phase_breakdown={"Phase 3": {"success_rate": 58, "count": 80}},
                fda_accepted=True,
                ema_accepted=True,
                regulatory_notes="Gold standard for oncology",
                successful_examples=[],
                failed_examples=[]
            ),
            "response": EndpointBenchmark(
                endpoint_type="Response Rate",
                category="response",
                trials_analyzed=150,
                completed_trials=97,
                success_rate=65.0,
                avg_enrollment=200,
                avg_duration_months=18,
                phase_breakdown={"Phase 2": {"success_rate": 65, "count": 100}},
                fda_accepted=True,
                ema_accepted=True,
                regulatory_notes="Acceptable for accelerated approval",
                successful_examples=[],
                failed_examples=[]
            ),
            "biomarker": EndpointBenchmark(
                endpoint_type="Biomarker",
                category="biomarker",
                trials_analyzed=200,
                completed_trials=150,
                success_rate=75.0,
                avg_enrollment=150,
                avg_duration_months=12,
                phase_breakdown={"Phase 2": {"success_rate": 75, "count": 150}},
                fda_accepted=True,
                ema_accepted=True,
                regulatory_notes="Surrogate endpoint; clinical correlation needed",
                successful_examples=[],
                failed_examples=[]
            )
        }
        return defaults

    def _score_endpoint(self, classification: Optional[EndpointClassification],
                        benchmarks: Dict[str, EndpointBenchmark],
                        condition: str, phase: str) -> Tuple[float, str]:
        """Score the endpoint based on multiple factors."""
        if not classification:
            return 50.0, "Unable to classify endpoint"

        score = 50.0  # Base score
        assessment_parts = []

        # Regulatory status
        if classification.regulatory_status == "established":
            score += 25
            assessment_parts.append("Regulatory established")
        elif classification.regulatory_status == "acceptable":
            score += 15
            assessment_parts.append("Regulatory acceptable")
        else:
            assessment_parts.append("Exploratory endpoint")

        # Clinical meaningfulness
        if classification.clinical_meaningfulness == "high":
            score += 15
            assessment_parts.append("High clinical meaningfulness")
        elif classification.clinical_meaningfulness == "moderate":
            score += 10

        # Benchmark performance
        if classification.primary_category in benchmarks:
            benchmark = benchmarks[classification.primary_category]
            if benchmark.success_rate >= 60:
                score += 10
                assessment_parts.append(f"Good historical success ({benchmark.success_rate:.0f}%)")
            elif benchmark.success_rate < 40:
                score -= 10
                assessment_parts.append(f"Low historical success ({benchmark.success_rate:.0f}%)")

        # Phase appropriateness
        phase_appropriate = self._check_phase_appropriateness(classification.primary_category, phase)
        if phase_appropriate:
            score += 5
        else:
            score -= 5
            assessment_parts.append("Consider phase appropriateness")

        score = max(0, min(100, score))

        if score >= 75:
            overall = "Excellent endpoint choice"
        elif score >= 60:
            overall = "Good endpoint with minor considerations"
        elif score >= 45:
            overall = "Acceptable but improvements recommended"
        else:
            overall = "Significant endpoint concerns"

        assessment = f"{overall}. {'; '.join(assessment_parts)}"

        return score, assessment

    def _check_phase_appropriateness(self, category: str, phase: str) -> bool:
        """Check if endpoint category is appropriate for phase."""
        phase_preferences = {
            "Phase 1": ["safety", "biomarker"],
            "Phase 2": ["response", "biomarker", "progression", "patient_reported"],
            "Phase 3": ["survival", "progression", "response", "composite", "patient_reported"],
            "Phase 4": ["safety", "patient_reported", "composite"]
        }

        preferred = phase_preferences.get(phase, [])
        return category in preferred

    def _analyze_timing(self, classification: EndpointClassification) -> TimingAnalysis:
        """Analyze timing considerations for the endpoint."""
        category = classification.primary_category
        timing_config = self.TIMING_GUIDANCE.get(category, self.TIMING_GUIDANCE["biomarker"])

        return TimingAnalysis(
            endpoint_type=category,
            typical_primary_timepoint=timing_config["recommended"],
            typical_range=timing_config["typical_range"],
            recommended_timing=timing_config["recommended"],
            timing_rationale=timing_config["rationale"],
            too_early_risks=timing_config["too_early_risks"],
            too_late_risks=timing_config["too_late_risks"]
        )

    def _map_condition_to_therapeutic_area(self, condition: str) -> str:
        """Map condition to therapeutic area for regulatory guidance."""
        condition_lower = condition.lower()

        mappings = {
            "oncology": ["cancer", "tumor", "carcinoma", "lymphoma", "leukemia", "melanoma", "sarcoma"],
            "cardiovascular": ["heart", "cardiac", "cardiovascular", "hypertension", "coronary", "stroke", "atherosclerosis"],
            "diabetes": ["diabetes", "diabetic", "glucose", "insulin", "glycemic"],
            "neurology": ["alzheimer", "parkinson", "multiple sclerosis", "epilepsy", "migraine", "neuropathy"],
            "immunology": ["rheumatoid", "lupus", "psoriasis", "crohn", "colitis", "arthritis"],
            "infectious_disease": ["infection", "bacterial", "viral", "hiv", "hepatitis", "tuberculosis"]
        }

        for area, keywords in mappings.items():
            if any(k in condition_lower for k in keywords):
                return area

        return "general"

    def _generate_recommendations(self, classification: Optional[EndpointClassification],
                                    benchmarks: Dict[str, EndpointBenchmark],
                                    condition: str, phase: str,
                                    regulatory_guidance: Dict) -> List[EndpointRecommendation]:
        """Generate endpoint recommendations."""
        recommendations = []

        if not classification:
            recommendations.append(EndpointRecommendation(
                priority="high",
                recommendation_type="change_primary",
                current_endpoint=None,
                recommended_change="Define a clear primary endpoint",
                rationale="Primary endpoint is required for regulatory submission",
                expected_impact="Essential for trial success",
                evidence=["Regulatory requirement"],
                regulatory_support="Required by FDA and EMA"
            ))
            return recommendations

        # Check regulatory status
        if classification.regulatory_status == "exploratory":
            fda_preferred = regulatory_guidance.get("fda_preferred", [])
            if fda_preferred:
                recommendations.append(EndpointRecommendation(
                    priority="high",
                    recommendation_type="change_primary",
                    current_endpoint=classification.endpoint_text,
                    recommended_change=f"Consider using established endpoint: {fda_preferred[0]}",
                    rationale="Current endpoint may not be accepted for registration",
                    expected_impact="Improved regulatory acceptance",
                    evidence=[f"FDA prefers: {', '.join(fda_preferred[:2])}"],
                    regulatory_support="Aligned with agency guidance"
                ))

        # Check against benchmarks
        if classification.primary_category in benchmarks:
            benchmark = benchmarks[classification.primary_category]
            if benchmark.success_rate < 50:
                # Find better performing category
                better = [(k, v) for k, v in benchmarks.items() if v.success_rate > benchmark.success_rate + 15]
                if better:
                    best_alt = max(better, key=lambda x: x[1].success_rate)
                    recommendations.append(EndpointRecommendation(
                        priority="medium",
                        recommendation_type="change_primary",
                        current_endpoint=classification.endpoint_text,
                        recommended_change=f"Consider {best_alt[0].replace('_', ' ').title()} endpoint ({best_alt[1].success_rate:.0f}% success rate)",
                        rationale=f"Current endpoint type has {benchmark.success_rate:.0f}% success rate",
                        expected_impact=f"+{best_alt[1].success_rate - benchmark.success_rate:.0f}% expected success",
                        evidence=[f"Based on {best_alt[1].trials_analyzed} similar trials"],
                        regulatory_support="Higher success rate historically"
                    ))

        # Timing recommendations
        if classification.timeframe:
            timing_config = self.TIMING_GUIDANCE.get(classification.primary_category)
            if timing_config:
                recommendations.append(EndpointRecommendation(
                    priority="low",
                    recommendation_type="modify_timing",
                    current_endpoint=classification.endpoint_text,
                    recommended_change=f"Verify timing aligns with typical range: {timing_config['typical_range'][0]}-{timing_config['typical_range'][1]} weeks",
                    rationale=timing_config["rationale"],
                    expected_impact="Optimized endpoint assessment",
                    evidence=[],
                    regulatory_support="Standard practice"
                ))

        # AI-powered recommendations
        if self.client:
            ai_recs = self._get_ai_recommendations(classification, condition, phase, benchmarks)
            recommendations.extend(ai_recs)

        return recommendations[:8]

    def _get_ai_recommendations(self, classification: EndpointClassification,
                                 condition: str, phase: str,
                                 benchmarks: Dict[str, EndpointBenchmark]) -> List[EndpointRecommendation]:
        """Get AI-powered endpoint recommendations."""
        if not self.client:
            return []

        benchmark_summary = "\n".join([
            f"- {cat}: {b.success_rate:.0f}% success ({b.trials_analyzed} trials)"
            for cat, b in benchmarks.items()
        ][:5])

        prompt = f"""Analyze this clinical trial endpoint and provide 2-3 specific optimization recommendations.

ENDPOINT: {classification.endpoint_text}
CONDITION: {condition}
PHASE: {phase}
ENDPOINT CATEGORY: {classification.primary_category}
REGULATORY STATUS: {classification.regulatory_status}
MEASUREMENT TYPE: {classification.measurement_type}

BENCHMARK DATA:
{benchmark_summary}

Provide specific, actionable recommendations to optimize this endpoint. Consider:
1. Regulatory acceptance
2. Clinical meaningfulness
3. Feasibility of measurement
4. Historical success rates

Return as JSON array:
[
  {{
    "priority": "high/medium/low",
    "recommendation_type": "change_primary/add_secondary/modify_timing/improve_definition",
    "recommended_change": "specific recommendation",
    "rationale": "why this helps",
    "expected_impact": "quantified benefit"
  }}
]

Return ONLY the JSON array."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text.strip()

            if response_text.startswith("["):
                data = json.loads(response_text)
            else:
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                else:
                    return []

            recommendations = []
            for item in data[:3]:
                recommendations.append(EndpointRecommendation(
                    priority=item.get("priority", "medium"),
                    recommendation_type=item.get("recommendation_type", "improve_definition"),
                    current_endpoint=classification.endpoint_text,
                    recommended_change=item.get("recommended_change", ""),
                    rationale=item.get("rationale", ""),
                    expected_impact=item.get("expected_impact", ""),
                    evidence=[],
                    regulatory_support=""
                ))

            return recommendations

        except Exception as e:
            logger.error(f"AI recommendations failed: {e}")
            return []

    def _suggest_secondary_endpoints(self, primary: Optional[EndpointClassification],
                                      condition: str, phase: str,
                                      existing_secondary: List[str]) -> List[SecondaryEndpointSuggestion]:
        """Suggest complementary secondary endpoints."""
        suggestions = []

        if not primary:
            return suggestions

        # Complementary endpoint mappings
        complements = {
            "survival": ["patient_reported", "response", "progression"],
            "progression": ["response", "patient_reported", "biomarker"],
            "response": ["progression", "patient_reported", "biomarker"],
            "biomarker": ["patient_reported", "safety", "response"],
            "patient_reported": ["biomarker", "safety"],
            "safety": ["patient_reported", "biomarker"]
        }

        suggested_categories = complements.get(primary.primary_category, ["patient_reported", "safety"])

        for category in suggested_categories[:3]:
            # Check if already in existing secondaries
            config = self.ENDPOINT_CATEGORIES.get(category, {})

            example_endpoints = {
                "patient_reported": "Health-related quality of life (HRQOL) using validated PRO instrument",
                "response": "Objective response rate (ORR) per RECIST 1.1 criteria",
                "progression": "Progression-free survival (PFS) per investigator assessment",
                "biomarker": "Change in disease-specific biomarker from baseline",
                "safety": "Incidence of treatment-emergent adverse events",
                "survival": "Overall survival (OS)"
            }

            suggestions.append(SecondaryEndpointSuggestion(
                endpoint=example_endpoints.get(category, f"{category.title()} endpoint"),
                category=category,
                rationale=f"Complements {primary.primary_category} endpoint",
                complementarity=f"Provides {category.replace('_', ' ')} perspective alongside {primary.primary_category}",
                regulatory_value="Supportive for benefit-risk assessment" if category in ["patient_reported", "safety"] else "Supports efficacy claims",
                feasibility="high" if category in ["safety", "biomarker"] else "medium",
                examples_in_trials=50  # Placeholder
            ))

        return suggestions

    def _get_established_endpoints(self, condition: str,
                                    regulatory_guidance: Dict) -> List[str]:
        """Get established endpoints for the condition."""
        established = []

        fda = regulatory_guidance.get("fda_preferred", [])
        ema = regulatory_guidance.get("ema_preferred", [])

        # Combine and deduplicate
        all_endpoints = list(set(fda + ema))

        return all_endpoints or ["Overall Survival", "Progression-Free Survival", "Objective Response Rate"]

    def _generate_key_findings(self, classification: Optional[EndpointClassification],
                               benchmarks: Dict[str, EndpointBenchmark],
                               score: float,
                               regulatory_guidance: Dict) -> List[str]:
        """Generate key findings."""
        findings = []

        if classification:
            findings.append(f"Endpoint classified as: {classification.primary_category.replace('_', ' ').title()}")
            findings.append(f"Regulatory status: {classification.regulatory_status.title()}")

            if classification.primary_category in benchmarks:
                b = benchmarks[classification.primary_category]
                findings.append(f"Historical success rate: {b.success_rate:.0f}% ({b.trials_analyzed} trials)")

        if score >= 75:
            findings.append("Endpoint is well-suited for this trial")
        elif score < 50:
            findings.append("Endpoint selection needs reconsideration")

        # Best performing category
        if benchmarks:
            best = max(benchmarks.items(), key=lambda x: x[1].success_rate)
            findings.append(f"Best performing endpoint type: {best[0].replace('_', ' ').title()} ({best[1].success_rate:.0f}%)")

        return findings[:5]

    def _generate_overall_assessment(self, score: float,
                                      classification: Optional[EndpointClassification],
                                      recommendations: List[EndpointRecommendation]) -> str:
        """Generate overall assessment."""
        high_priority = sum(1 for r in recommendations if r.priority == "high")

        if score >= 75 and high_priority == 0:
            return "Your endpoint selection is excellent and well-aligned with regulatory expectations and historical success patterns."
        elif score >= 60:
            return f"Your endpoint is acceptable but could be strengthened. {high_priority} high-priority recommendations identified."
        elif score >= 45:
            return f"Your endpoint has concerns that should be addressed. Review the {high_priority} high-priority recommendations."
        else:
            return "Significant endpoint concerns identified. Consider revising your primary endpoint selection."
