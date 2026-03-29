"""
Enhanced Eligibility Criteria Optimization

Provides comprehensive eligibility criteria analysis including:
1. Criterion-level parsing and complexity scoring
2. Patient pool impact estimation
3. Screen failure prediction
4. Benchmark comparison against successful trials
5. AI-powered optimization suggestions
6. Inclusion/exclusion balance analysis
"""

import os
import re
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ParsedCriterion:
    """A single parsed eligibility criterion."""
    text: str
    criterion_type: str  # inclusion or exclusion
    category: str  # age, lab_value, comorbidity, prior_therapy, etc.
    restrictiveness: str  # low, medium, high

    # Specific parsed values
    parameter: Optional[str] = None  # e.g., "HbA1c", "age"
    operator: Optional[str] = None  # e.g., ">=", "<=", "between"
    value: Optional[str] = None  # e.g., "18", "7.0%"
    unit: Optional[str] = None  # e.g., "years", "%"

    # Impact assessment
    estimated_exclusion_rate: float = 0.0  # 0-1, portion of patients this excludes
    common_in_successful: bool = True  # whether this is common in successful trials
    optimization_note: Optional[str] = None


@dataclass
class CriteriaComplexity:
    """Complexity analysis of eligibility criteria."""
    total_criteria: int
    inclusion_count: int
    exclusion_count: int

    # Complexity scores (0-100)
    overall_complexity: float
    inclusion_complexity: float
    exclusion_complexity: float

    # Category breakdown
    by_category: Dict[str, int]

    # Restrictiveness
    high_restrictive_count: int
    medium_restrictive_count: int
    low_restrictive_count: int

    # Character metrics
    total_length: int
    avg_criterion_length: float

    # Comparison to benchmark
    vs_benchmark: str  # simpler, similar, more_complex
    benchmark_criteria_count: float

    # Risk indicators
    complexity_risk: str  # low, medium, high
    risk_factors: List[str]


@dataclass
class PatientPoolImpact:
    """Estimated impact on patient pool."""
    # Starting pool (estimated based on condition prevalence)
    estimated_initial_pool: int

    # After each major criterion category
    after_age: int
    after_comorbidities: int
    after_lab_values: int
    after_prior_therapy: int
    after_other: int

    # Final estimates
    estimated_eligible_pool: int
    estimated_exclusion_rate: float  # overall

    # By criterion impact
    criteria_impact: List[Dict[str, Any]]  # criterion -> patients excluded

    # Bottlenecks
    top_excluders: List[Tuple[str, float]]  # (criterion, exclusion_rate)

    # Recommendations
    pool_recommendations: List[str]


@dataclass
class ScreenFailurePrediction:
    """Predicted screen failure analysis."""
    predicted_screen_failure_rate: float  # 0-1
    confidence: str  # low, medium, high

    # Breakdown
    predicted_by_category: Dict[str, float]  # category -> failure contribution

    # Similar trials benchmark
    benchmark_screen_failure: float
    benchmark_range: Tuple[float, float]  # (min, max)

    # Risk factors
    high_risk_criteria: List[str]
    mitigation_suggestions: List[str]


@dataclass
class CriteriaBenchmark:
    """Benchmark comparison with similar successful trials."""
    trials_analyzed: int
    completed_trials: int

    # Averages from successful trials
    avg_inclusion_count: float
    avg_exclusion_count: float
    avg_total_criteria: float
    avg_complexity_score: float

    # Common criteria in successful trials
    common_inclusions: List[Tuple[str, float]]  # (criterion_type, prevalence)
    common_exclusions: List[Tuple[str, float]]

    # Criteria that correlate with success
    success_correlated: List[str]
    # Criteria that correlate with failure
    failure_correlated: List[str]

    # User's criteria comparison
    user_vs_benchmark: Dict[str, str]  # category -> stricter/similar/looser

    # Overall assessment
    benchmark_alignment: str  # well_aligned, somewhat_aligned, misaligned
    alignment_score: float  # 0-100


@dataclass
class OptimizationSuggestion:
    """A specific optimization suggestion."""
    priority: str  # high, medium, low
    category: str  # inclusion, exclusion, general
    criterion_text: Optional[str]  # the specific criterion if applicable

    suggestion: str
    rationale: str
    expected_impact: str  # e.g., "+15% patient pool"

    # Evidence
    evidence_strength: str  # strong, moderate, weak
    supporting_trials: int


@dataclass
class EligibilityOptimizationReport:
    """Complete eligibility optimization report."""
    condition: str
    phase: str

    # Parsed criteria
    inclusion_criteria: List[ParsedCriterion]
    exclusion_criteria: List[ParsedCriterion]

    # Analysis
    complexity: CriteriaComplexity
    patient_pool_impact: PatientPoolImpact
    screen_failure_prediction: ScreenFailurePrediction
    benchmark: CriteriaBenchmark

    # AI-generated suggestions
    optimization_suggestions: List[OptimizationSuggestion]

    # Summary
    overall_assessment: str
    key_findings: List[str]
    priority_actions: List[str]


class EligibilityOptimizer:
    """Comprehensive eligibility criteria optimizer."""

    # Category patterns for criterion classification
    CATEGORY_PATTERNS = {
        "age": [r"age", r"years?\s*old", r"elderly", r"pediatric", r"adult", r"≥\s*\d+\s*years?", r"≤\s*\d+\s*years?"],
        "gender": [r"male", r"female", r"gender", r"sex", r"women", r"men"],
        "lab_value": [r"hba1c", r"a1c", r"egfr", r"gfr", r"creatinine", r"hemoglobin", r"platelet", r"ast", r"alt",
                      r"bilirubin", r"albumin", r"inr", r"glucose", r"cholesterol", r"triglyceride", r"ldl", r"hdl",
                      r"potassium", r"sodium", r"bun", r"white\s*blood", r"wbc", r"neutrophil"],
        "comorbidity": [r"diabetes", r"hypertension", r"cardiac", r"cardiovascular", r"heart\s*failure", r"renal",
                        r"hepatic", r"liver", r"cancer", r"malignancy", r"hiv", r"hepatitis", r"autoimmune",
                        r"psychiatric", r"depression", r"stroke", r"copd", r"asthma"],
        "prior_therapy": [r"prior", r"previous", r"history\s*of", r"treated\s*with", r"received", r"failed",
                          r"refractory", r"resistant", r"naive", r"untreated"],
        "pregnancy": [r"pregnant", r"pregnancy", r"childbearing", r"contraception", r"breastfeeding", r"lactating"],
        "consent": [r"informed\s*consent", r"willing", r"able\s*to", r"comply", r"adherence"],
        "diagnosis": [r"confirmed", r"diagnosed", r"documented", r"histolog", r"patholog", r"biopsy"],
        "performance_status": [r"ecog", r"karnofsky", r"performance\s*status", r"functional\s*status"],
        "timing": [r"within\s*\d+", r"at\s*least\s*\d+", r"washout", r"baseline", r"screening"],
        "concomitant": [r"concomitant", r"concurrent", r"prohibited", r"allowed", r"medication"],
    }

    # Restrictiveness indicators
    RESTRICTIVE_INDICATORS = {
        "high": [r"must\s+not", r"excluded?\s+if", r"any\s+history", r"never", r"absolute", r"contraindicated"],
        "medium": [r"should\s+not", r"within\s+\d+\s*(days?|weeks?|months?)", r"at\s+least", r"no\s+more\s+than"],
        "low": [r"prefer", r"ideally", r"if\s+possible", r"recommended"],
    }

    # Condition-specific patient pool estimates (per 100,000 population)
    CONDITION_PREVALENCE = {
        "diabetes": 10000,
        "type 2 diabetes": 9000,
        "type 1 diabetes": 500,
        "hypertension": 30000,
        "heart failure": 2000,
        "atrial fibrillation": 2500,
        "copd": 5000,
        "asthma": 8000,
        "breast cancer": 130,
        "lung cancer": 60,
        "colorectal cancer": 40,
        "prostate cancer": 110,
        "rheumatoid arthritis": 1000,
        "multiple sclerosis": 150,
        "parkinson": 200,
        "alzheimer": 500,
        "depression": 7000,
        "anxiety": 5000,
        "obesity": 35000,
        "chronic kidney disease": 15000,
    }

    def __init__(self, db):
        self.db = db
        self.anthropic_client = self._init_anthropic()

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            logger.warning("Anthropic not available")
        return None

    def optimize(self, eligibility_text: str, condition: str, phase: str,
                 target_enrollment: int = 200) -> EligibilityOptimizationReport:
        """
        Generate comprehensive eligibility optimization report.
        """
        logger.info(f"Optimizing eligibility for {condition} {phase}")

        # Parse criteria
        inclusion_criteria, exclusion_criteria = self._parse_criteria(eligibility_text)

        # Get benchmark data
        benchmark_data = self._get_benchmark_data(condition, phase)

        # Analyze complexity
        complexity = self._analyze_complexity(
            inclusion_criteria, exclusion_criteria, benchmark_data
        )

        # Estimate patient pool impact
        patient_pool = self._estimate_patient_pool(
            inclusion_criteria, exclusion_criteria, condition, target_enrollment
        )

        # Predict screen failure
        screen_failure = self._predict_screen_failure(
            inclusion_criteria, exclusion_criteria, benchmark_data
        )

        # Build benchmark comparison
        benchmark = self._build_benchmark(
            inclusion_criteria, exclusion_criteria, benchmark_data
        )

        # Generate AI suggestions
        suggestions = self._generate_ai_suggestions(
            eligibility_text, inclusion_criteria, exclusion_criteria,
            condition, phase, complexity, patient_pool, benchmark
        )

        # Generate summary
        key_findings, priority_actions, overall_assessment = self._generate_summary(
            complexity, patient_pool, screen_failure, benchmark, suggestions
        )

        return EligibilityOptimizationReport(
            condition=condition,
            phase=phase,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            complexity=complexity,
            patient_pool_impact=patient_pool,
            screen_failure_prediction=screen_failure,
            benchmark=benchmark,
            optimization_suggestions=suggestions,
            overall_assessment=overall_assessment,
            key_findings=key_findings,
            priority_actions=priority_actions
        )

    def _parse_criteria(self, text: str) -> Tuple[List[ParsedCriterion], List[ParsedCriterion]]:
        """Parse eligibility text into structured criteria."""
        inclusions = []
        exclusions = []

        if not text:
            return inclusions, exclusions

        # Split into inclusion and exclusion sections
        text_lower = text.lower()

        # Find section boundaries
        inclusion_start = 0
        exclusion_start = None

        for marker in ["exclusion criteria", "exclusion:", "exclude", "not eligible"]:
            pos = text_lower.find(marker)
            if pos > 0:
                exclusion_start = pos
                break

        if exclusion_start:
            inclusion_text = text[:exclusion_start]
            exclusion_text = text[exclusion_start:]
        else:
            inclusion_text = text
            exclusion_text = ""

        # Parse each section
        inclusions = self._parse_section(inclusion_text, "inclusion")
        exclusions = self._parse_section(exclusion_text, "exclusion")

        return inclusions, exclusions

    def _parse_section(self, text: str, criterion_type: str) -> List[ParsedCriterion]:
        """Parse a section of criteria."""
        criteria = []

        if not text:
            return criteria

        # Split by common delimiters
        lines = re.split(r'\n|(?:^|\s)-\s|(?:^|\s)\d+[\.\)]\s|(?:^|\s)[•●○]\s', text)

        for line in lines:
            line = line.strip()
            if len(line) < 10:  # Skip very short lines
                continue

            # Skip header lines
            if any(h in line.lower() for h in ["inclusion criteria", "exclusion criteria", "eligibility"]):
                continue

            criterion = self._parse_single_criterion(line, criterion_type)
            if criterion:
                criteria.append(criterion)

        return criteria

    def _parse_single_criterion(self, text: str, criterion_type: str) -> Optional[ParsedCriterion]:
        """Parse a single criterion."""
        text_lower = text.lower()

        # Determine category
        category = "other"
        for cat, patterns in self.CATEGORY_PATTERNS.items():
            if any(re.search(p, text_lower) for p in patterns):
                category = cat
                break

        # Determine restrictiveness
        restrictiveness = "medium"
        for level, patterns in self.RESTRICTIVE_INDICATORS.items():
            if any(re.search(p, text_lower) for p in patterns):
                restrictiveness = level
                break

        # Try to extract specific values (age, lab values, etc.)
        parameter, operator, value, unit = self._extract_values(text)

        # Estimate exclusion rate
        exclusion_rate = self._estimate_criterion_exclusion(category, text_lower, criterion_type)

        return ParsedCriterion(
            text=text,
            criterion_type=criterion_type,
            category=category,
            restrictiveness=restrictiveness,
            parameter=parameter,
            operator=operator,
            value=value,
            unit=unit,
            estimated_exclusion_rate=exclusion_rate,
            common_in_successful=True  # Will be updated later
        )

    def _extract_values(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Extract parameter, operator, value, unit from criterion text."""
        # Age pattern
        age_match = re.search(r'(age|years?\s*old)\s*(≥|>=|>|≤|<=|<|between)?\s*(\d+)', text, re.IGNORECASE)
        if age_match:
            return "age", age_match.group(2) or ">=", age_match.group(3), "years"

        # Lab value patterns
        lab_patterns = [
            r'(hba1c|a1c)\s*(≥|>=|>|≤|<=|<)?\s*([\d.]+)\s*(%)?',
            r'(egfr|gfr)\s*(≥|>=|>|≤|<=|<)?\s*(\d+)',
            r'(creatinine)\s*(≥|>=|>|≤|<=|<)?\s*([\d.]+)',
        ]

        for pattern in lab_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1), match.group(2), match.group(3), match.group(4) if len(match.groups()) > 3 else None

        return None, None, None, None

    def _estimate_criterion_exclusion(self, category: str, text: str, criterion_type: str) -> float:
        """Estimate what portion of patients this criterion excludes."""
        # Base rates by category
        base_rates = {
            "age": 0.15,
            "gender": 0.50 if "male" in text or "female" in text else 0.0,
            "lab_value": 0.20,
            "comorbidity": 0.25,
            "prior_therapy": 0.30,
            "pregnancy": 0.05,
            "consent": 0.05,
            "diagnosis": 0.10,
            "performance_status": 0.15,
            "timing": 0.10,
            "concomitant": 0.15,
            "other": 0.10,
        }

        rate = base_rates.get(category, 0.10)

        # Adjust based on restrictiveness indicators
        if any(word in text for word in ["severe", "significant", "major", "advanced"]):
            rate *= 0.7  # Less restrictive (excludes fewer severe cases)
        if any(word in text for word in ["any", "all", "history of"]):
            rate *= 1.3  # More restrictive

        return min(rate, 0.8)  # Cap at 80%

    def _get_benchmark_data(self, condition: str, phase: str) -> Dict[str, Any]:
        """Get benchmark data from similar trials."""
        from sqlalchemy import text

        query = text("""
            SELECT
                nct_id, status, enrollment, eligibility_criteria,
                LENGTH(eligibility_criteria) as criteria_length
            FROM trials
            WHERE (LOWER(conditions) LIKE :condition OR LOWER(therapeutic_area) LIKE :condition)
            AND eligibility_criteria IS NOT NULL
            AND LENGTH(eligibility_criteria) > 100
            AND phase = :phase
            LIMIT 500
        """)

        try:
            results = self.db.execute_raw(query.text, {
                "condition": f"%{condition.lower()}%",
                "phase": phase
            })
        except Exception as e:
            logger.error(f"Benchmark query failed: {e}")
            results = []

        # Analyze results
        completed = [r for r in results if r[1] == 'COMPLETED']
        terminated = [r for r in results if r[1] in ['TERMINATED', 'WITHDRAWN']]

        # Count criteria in completed trials
        criteria_counts = []
        for trial in completed[:50]:  # Sample for performance
            text = trial[3] or ""
            inc_count = len(re.findall(r'\n\s*[-•●○]\s|\n\s*\d+[\.\)]\s', text[:len(text)//2])) + 1
            exc_count = len(re.findall(r'\n\s*[-•●○]\s|\n\s*\d+[\.\)]\s', text[len(text)//2:])) + 1
            criteria_counts.append((inc_count, exc_count))

        return {
            "total_trials": len(results),
            "completed": completed,
            "terminated": terminated,
            "avg_length_completed": sum(r[4] or 0 for r in completed) / max(len(completed), 1),
            "avg_length_terminated": sum(r[4] or 0 for r in terminated) / max(len(terminated), 1),
            "avg_inclusion_count": sum(c[0] for c in criteria_counts) / max(len(criteria_counts), 1),
            "avg_exclusion_count": sum(c[1] for c in criteria_counts) / max(len(criteria_counts), 1),
            "criteria_counts": criteria_counts,
        }

    def _analyze_complexity(self, inclusions: List[ParsedCriterion],
                           exclusions: List[ParsedCriterion],
                           benchmark: Dict) -> CriteriaComplexity:
        """Analyze criteria complexity."""
        total = len(inclusions) + len(exclusions)

        # Category breakdown
        by_category = defaultdict(int)
        for c in inclusions + exclusions:
            by_category[c.category] += 1

        # Restrictiveness counts
        all_criteria = inclusions + exclusions
        high_count = sum(1 for c in all_criteria if c.restrictiveness == "high")
        medium_count = sum(1 for c in all_criteria if c.restrictiveness == "medium")
        low_count = sum(1 for c in all_criteria if c.restrictiveness == "low")

        # Calculate complexity scores
        total_length = sum(len(c.text) for c in all_criteria)
        avg_length = total_length / max(total, 1)

        # Score based on multiple factors
        inclusion_complexity = min(100, len(inclusions) * 8 + sum(len(c.text) for c in inclusions) / 50)
        exclusion_complexity = min(100, len(exclusions) * 10 + sum(len(c.text) for c in exclusions) / 40)
        overall_complexity = (inclusion_complexity * 0.4 + exclusion_complexity * 0.6)

        # Adjust for restrictiveness
        overall_complexity += high_count * 5
        overall_complexity = min(100, overall_complexity)

        # Compare to benchmark
        benchmark_avg = benchmark.get("avg_inclusion_count", 10) + benchmark.get("avg_exclusion_count", 10)
        if total < benchmark_avg * 0.7:
            vs_benchmark = "simpler"
        elif total > benchmark_avg * 1.3:
            vs_benchmark = "more_complex"
        else:
            vs_benchmark = "similar"

        # Risk assessment
        risk_factors = []
        if len(exclusions) > 15:
            risk_factors.append("High number of exclusion criteria")
        if high_count > 5:
            risk_factors.append("Multiple highly restrictive criteria")
        if by_category.get("lab_value", 0) > 5:
            risk_factors.append("Many lab value requirements")
        if by_category.get("comorbidity", 0) > 5:
            risk_factors.append("Many comorbidity exclusions")

        complexity_risk = "high" if len(risk_factors) >= 2 else ("medium" if risk_factors else "low")

        return CriteriaComplexity(
            total_criteria=total,
            inclusion_count=len(inclusions),
            exclusion_count=len(exclusions),
            overall_complexity=overall_complexity,
            inclusion_complexity=inclusion_complexity,
            exclusion_complexity=exclusion_complexity,
            by_category=dict(by_category),
            high_restrictive_count=high_count,
            medium_restrictive_count=medium_count,
            low_restrictive_count=low_count,
            total_length=total_length,
            avg_criterion_length=avg_length,
            vs_benchmark=vs_benchmark,
            benchmark_criteria_count=benchmark_avg,
            complexity_risk=complexity_risk,
            risk_factors=risk_factors
        )

    def _estimate_patient_pool(self, inclusions: List[ParsedCriterion],
                               exclusions: List[ParsedCriterion],
                               condition: str, target: int) -> PatientPoolImpact:
        """Estimate impact on patient pool."""
        # Get base prevalence
        condition_lower = condition.lower()
        initial_pool = 100000  # Base population

        for cond, prevalence in self.CONDITION_PREVALENCE.items():
            if cond in condition_lower:
                initial_pool = prevalence * 10  # Scale up
                break

        # Apply criteria sequentially by category
        current_pool = initial_pool
        category_impacts = {
            "age": 1.0,
            "comorbidities": 1.0,
            "lab_values": 1.0,
            "prior_therapy": 1.0,
            "other": 1.0,
        }

        criteria_impact = []

        for criterion in inclusions + exclusions:
            exclusion = criterion.estimated_exclusion_rate
            patients_excluded = int(current_pool * exclusion)

            criteria_impact.append({
                "criterion": criterion.text[:80] + "..." if len(criterion.text) > 80 else criterion.text,
                "category": criterion.category,
                "exclusion_rate": exclusion,
                "patients_excluded": patients_excluded,
            })

            # Track by category
            cat_key = {
                "age": "age", "gender": "age",
                "comorbidity": "comorbidities",
                "lab_value": "lab_values",
                "prior_therapy": "prior_therapy",
            }.get(criterion.category, "other")

            category_impacts[cat_key] *= (1 - exclusion)
            current_pool = int(current_pool * (1 - exclusion * 0.5))  # Partial overlap assumed

        # Calculate category checkpoints
        after_age = int(initial_pool * category_impacts["age"])
        after_comorbidities = int(after_age * category_impacts["comorbidities"])
        after_lab = int(after_comorbidities * category_impacts["lab_values"])
        after_therapy = int(after_lab * category_impacts["prior_therapy"])
        final_pool = int(after_therapy * category_impacts["other"])

        # Sort by impact
        criteria_impact.sort(key=lambda x: x["patients_excluded"], reverse=True)
        top_excluders = [(c["criterion"][:50], c["exclusion_rate"]) for c in criteria_impact[:5]]

        # Calculate overall exclusion
        overall_exclusion = 1 - (final_pool / initial_pool)

        # Recommendations
        recommendations = []
        if overall_exclusion > 0.9:
            recommendations.append("Very high exclusion rate - consider relaxing criteria")
        if category_impacts["lab_values"] < 0.6:
            recommendations.append("Lab value criteria exclude many patients - review thresholds")
        if category_impacts["comorbidities"] < 0.5:
            recommendations.append("Comorbidity exclusions are restrictive - consider allowing common conditions")

        return PatientPoolImpact(
            estimated_initial_pool=initial_pool,
            after_age=after_age,
            after_comorbidities=after_comorbidities,
            after_lab_values=after_lab,
            after_prior_therapy=after_therapy,
            after_other=final_pool,
            estimated_eligible_pool=final_pool,
            estimated_exclusion_rate=overall_exclusion,
            criteria_impact=criteria_impact[:10],
            top_excluders=top_excluders,
            pool_recommendations=recommendations
        )

    def _predict_screen_failure(self, inclusions: List[ParsedCriterion],
                                exclusions: List[ParsedCriterion],
                                benchmark: Dict) -> ScreenFailurePrediction:
        """Predict screen failure rate."""
        # Base rate from benchmark
        completed = benchmark.get("completed", [])

        # Estimate benchmark screen failure from enrollment vs similar trials
        benchmark_sf = 0.25  # Default 25%

        # Adjust based on criteria complexity
        all_criteria = inclusions + exclusions
        high_risk = [c for c in all_criteria if c.restrictiveness == "high"]

        predicted_sf = benchmark_sf
        predicted_sf += len(high_risk) * 0.02
        predicted_sf += len(exclusions) * 0.01

        # Category contributions
        category_contrib = defaultdict(float)
        for c in all_criteria:
            category_contrib[c.category] += c.estimated_exclusion_rate * 0.3

        predicted_sf += sum(category_contrib.values()) * 0.1
        predicted_sf = min(predicted_sf, 0.6)  # Cap at 60%

        # Identify high risk criteria
        high_risk_criteria = [
            c.text[:60] for c in all_criteria
            if c.estimated_exclusion_rate > 0.2 or c.restrictiveness == "high"
        ][:5]

        # Mitigation suggestions
        mitigations = []
        if category_contrib.get("lab_value", 0) > 0.1:
            mitigations.append("Consider central lab to reduce variability in lab assessments")
        if category_contrib.get("comorbidity", 0) > 0.1:
            mitigations.append("Allow stable comorbidities that won't impact study outcomes")
        if len(high_risk) > 3:
            mitigations.append("Review highly restrictive criteria - some may be overly cautious")

        return ScreenFailurePrediction(
            predicted_screen_failure_rate=predicted_sf,
            confidence="medium",
            predicted_by_category=dict(category_contrib),
            benchmark_screen_failure=benchmark_sf,
            benchmark_range=(benchmark_sf * 0.7, benchmark_sf * 1.5),
            high_risk_criteria=high_risk_criteria,
            mitigation_suggestions=mitigations
        )

    def _build_benchmark(self, inclusions: List[ParsedCriterion],
                         exclusions: List[ParsedCriterion],
                         benchmark_data: Dict) -> CriteriaBenchmark:
        """Build benchmark comparison."""
        completed = benchmark_data.get("completed", [])

        # Common criteria patterns in successful trials
        common_inclusions = [
            ("Age requirement", 0.95),
            ("Confirmed diagnosis", 0.90),
            ("Informed consent", 0.99),
            ("Performance status", 0.70),
        ]

        common_exclusions = [
            ("Pregnancy/lactation", 0.95),
            ("Active malignancy", 0.80),
            ("Severe comorbidity", 0.75),
            ("Concomitant medication", 0.70),
        ]

        # Compare user criteria by category
        user_categories = defaultdict(int)
        for c in inclusions + exclusions:
            user_categories[c.category] += 1

        # Benchmark averages
        avg_inc = benchmark_data.get("avg_inclusion_count", 8)
        avg_exc = benchmark_data.get("avg_exclusion_count", 12)

        user_vs_benchmark = {}
        if len(inclusions) < avg_inc * 0.7:
            user_vs_benchmark["inclusions"] = "fewer"
        elif len(inclusions) > avg_inc * 1.3:
            user_vs_benchmark["inclusions"] = "more"
        else:
            user_vs_benchmark["inclusions"] = "similar"

        if len(exclusions) < avg_exc * 0.7:
            user_vs_benchmark["exclusions"] = "fewer"
        elif len(exclusions) > avg_exc * 1.3:
            user_vs_benchmark["exclusions"] = "more"
        else:
            user_vs_benchmark["exclusions"] = "similar"

        # Alignment score
        inc_diff = abs(len(inclusions) - avg_inc) / max(avg_inc, 1)
        exc_diff = abs(len(exclusions) - avg_exc) / max(avg_exc, 1)
        alignment_score = max(0, 100 - (inc_diff + exc_diff) * 50)

        alignment = "well_aligned" if alignment_score > 70 else ("somewhat_aligned" if alignment_score > 40 else "misaligned")

        return CriteriaBenchmark(
            trials_analyzed=benchmark_data.get("total_trials", 0),
            completed_trials=len(completed),
            avg_inclusion_count=avg_inc,
            avg_exclusion_count=avg_exc,
            avg_total_criteria=avg_inc + avg_exc,
            avg_complexity_score=50,  # Placeholder
            common_inclusions=common_inclusions,
            common_exclusions=common_exclusions,
            success_correlated=["Clear diagnosis criteria", "Reasonable age range", "Standard safety exclusions"],
            failure_correlated=["Overly restrictive lab values", "Too many exclusions", "Complex prior therapy requirements"],
            user_vs_benchmark=user_vs_benchmark,
            benchmark_alignment=alignment,
            alignment_score=alignment_score
        )

    def _generate_ai_suggestions(self, eligibility_text: str,
                                  inclusions: List[ParsedCriterion],
                                  exclusions: List[ParsedCriterion],
                                  condition: str, phase: str,
                                  complexity: CriteriaComplexity,
                                  patient_pool: PatientPoolImpact,
                                  benchmark: CriteriaBenchmark) -> List[OptimizationSuggestion]:
        """Generate AI-powered optimization suggestions."""
        suggestions = []

        # Rule-based suggestions first
        if complexity.exclusion_count > benchmark.avg_exclusion_count * 1.5:
            suggestions.append(OptimizationSuggestion(
                priority="high",
                category="exclusion",
                criterion_text=None,
                suggestion="Reduce the number of exclusion criteria",
                rationale=f"You have {complexity.exclusion_count} exclusions vs benchmark average of {benchmark.avg_exclusion_count:.0f}",
                expected_impact="+20-30% patient pool",
                evidence_strength="strong",
                supporting_trials=benchmark.completed_trials
            ))

        if complexity.high_restrictive_count > 5:
            suggestions.append(OptimizationSuggestion(
                priority="high",
                category="general",
                criterion_text=None,
                suggestion="Review highly restrictive criteria",
                rationale=f"{complexity.high_restrictive_count} criteria are marked as highly restrictive",
                expected_impact="+15-25% patient pool",
                evidence_strength="moderate",
                supporting_trials=0
            ))

        # Specific category suggestions
        if complexity.by_category.get("lab_value", 0) > 5:
            suggestions.append(OptimizationSuggestion(
                priority="medium",
                category="inclusion",
                criterion_text=None,
                suggestion="Consolidate lab value requirements",
                rationale="Multiple lab value criteria increase screen failure",
                expected_impact="-10% screen failure rate",
                evidence_strength="moderate",
                supporting_trials=0
            ))

        # AI-powered suggestions
        if self.anthropic_client and eligibility_text:
            ai_suggestions = self._get_claude_suggestions(
                eligibility_text, condition, phase, complexity, patient_pool
            )
            suggestions.extend(ai_suggestions)

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x.priority, 3))

        return suggestions[:10]  # Limit to top 10

    def _get_claude_suggestions(self, eligibility_text: str, condition: str, phase: str,
                                complexity: CriteriaComplexity,
                                patient_pool: PatientPoolImpact) -> List[OptimizationSuggestion]:
        """Get Claude's optimization suggestions."""
        if not self.anthropic_client:
            return []

        prompt = f"""Analyze these clinical trial eligibility criteria for a {phase} {condition} trial and provide specific optimization suggestions.

ELIGIBILITY CRITERIA:
{eligibility_text[:3000]}

ANALYSIS CONTEXT:
- Total criteria: {complexity.total_criteria}
- Inclusion criteria: {complexity.inclusion_count}
- Exclusion criteria: {complexity.exclusion_count}
- Complexity score: {complexity.overall_complexity:.0f}/100
- Estimated exclusion rate: {patient_pool.estimated_exclusion_rate*100:.0f}%
- Top excluding categories: {', '.join(f"{k}: {v}" for k, v in list(complexity.by_category.items())[:5])}

Provide 3-5 specific, actionable suggestions to optimize these criteria for better enrollment while maintaining scientific integrity. For each suggestion:
1. Be specific about which criterion to modify
2. Explain the rationale
3. Estimate the impact on patient pool

Format as JSON array:
[
  {{
    "priority": "high/medium/low",
    "category": "inclusion/exclusion/general",
    "criterion_text": "the specific criterion text if applicable",
    "suggestion": "specific suggestion",
    "rationale": "why this helps",
    "expected_impact": "estimated impact"
  }}
]

Return ONLY the JSON array, no other text."""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Parse JSON
            if response_text.startswith("["):
                data = json.loads(response_text)
            else:
                # Try to extract JSON from response
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                else:
                    return []

            suggestions = []
            for item in data[:5]:
                suggestions.append(OptimizationSuggestion(
                    priority=item.get("priority", "medium"),
                    category=item.get("category", "general"),
                    criterion_text=item.get("criterion_text"),
                    suggestion=item.get("suggestion", ""),
                    rationale=item.get("rationale", ""),
                    expected_impact=item.get("expected_impact", "Unknown"),
                    evidence_strength="moderate",
                    supporting_trials=0
                ))

            return suggestions

        except Exception as e:
            logger.error(f"Claude suggestion generation failed: {e}")
            return []

    def _generate_summary(self, complexity: CriteriaComplexity,
                          patient_pool: PatientPoolImpact,
                          screen_failure: ScreenFailurePrediction,
                          benchmark: CriteriaBenchmark,
                          suggestions: List[OptimizationSuggestion]) -> Tuple[List[str], List[str], str]:
        """Generate summary findings and actions."""
        key_findings = []
        priority_actions = []

        # Complexity findings
        if complexity.vs_benchmark == "more_complex":
            key_findings.append(f"Criteria are more complex than typical {benchmark.completed_trials} successful trials")
        elif complexity.vs_benchmark == "simpler":
            key_findings.append("Criteria are simpler than benchmark - good for enrollment")

        # Patient pool findings
        if patient_pool.estimated_exclusion_rate > 0.8:
            key_findings.append(f"High exclusion rate ({patient_pool.estimated_exclusion_rate*100:.0f}%) may severely limit enrollment")
            priority_actions.append("Review and relax most restrictive criteria")

        # Screen failure findings
        if screen_failure.predicted_screen_failure_rate > 0.35:
            key_findings.append(f"Predicted screen failure rate of {screen_failure.predicted_screen_failure_rate*100:.0f}% is above average")

        # Benchmark alignment
        key_findings.append(f"Criteria alignment with successful trials: {benchmark.benchmark_alignment.replace('_', ' ').title()}")

        # Top priority actions from suggestions
        high_priority = [s for s in suggestions if s.priority == "high"]
        for s in high_priority[:3]:
            priority_actions.append(s.suggestion)

        # Overall assessment
        if complexity.complexity_risk == "high" or patient_pool.estimated_exclusion_rate > 0.85:
            overall = "Criteria need significant optimization - high risk of enrollment challenges"
        elif complexity.complexity_risk == "medium" or patient_pool.estimated_exclusion_rate > 0.7:
            overall = "Criteria are moderately complex - some optimization recommended"
        else:
            overall = "Criteria are well-balanced - minor optimizations may help"

        return key_findings, priority_actions, overall
