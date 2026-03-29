"""
Endpoint Benchmarking Intelligence

This module analyzes endpoints (outcomes) from historical trials to help
sponsors select appropriate endpoints for their protocols.

Key insights this provides:
1. What primary endpoints work for specific indications?
2. What timeframes are typical?
3. Which endpoints have led to FDA approval vs. failure?
4. What secondary endpoints are commonly paired with primaries?

This addresses a critical protocol design question that Jeeva doesn't
currently help with.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class EndpointPattern:
    """A pattern of endpoint usage across trials."""
    measure: str  # Normalized endpoint name
    raw_examples: List[str]  # Original endpoint text examples
    frequency: int  # How many trials use this
    typical_timeframes: List[str]  # Common measurement timeframes
    associated_conditions: List[str]  # What conditions use this endpoint
    phases: List[str]  # What phases typically use this
    success_indicators: Dict[str, int]  # completed vs terminated counts


@dataclass
class EndpointRecommendation:
    """A recommended endpoint for a protocol."""
    endpoint: str
    confidence: str  # "high", "medium", "low"
    rationale: str
    typical_timeframe: str
    similar_trials: List[str]  # NCT IDs of trials using this endpoint
    considerations: List[str]  # Things to consider


@dataclass
class EndpointAnalysis:
    """Complete endpoint analysis for an indication."""
    condition: str
    total_trials_analyzed: int
    primary_endpoints: List[EndpointPattern]
    secondary_endpoints: List[EndpointPattern]
    recommendations: List[EndpointRecommendation]
    regulatory_insights: List[str]


class EndpointBenchmarker:
    """
    Analyzes historical trial endpoints to provide intelligence on endpoint selection.
    
    Key capabilities:
    1. Identify most common endpoints by indication
    2. Analyze endpoint success/failure patterns
    3. Recommend endpoints based on historical precedent
    4. Flag potentially problematic endpoint choices
    """
    
    # Endpoint normalization patterns
    ENDPOINT_PATTERNS = {
        # Efficacy endpoints
        r"overall.*survival|os\b": "overall_survival",
        r"progression.?free.*survival|pfs\b": "progression_free_survival",
        r"disease.?free.*survival|dfs\b": "disease_free_survival",
        r"event.?free.*survival|efs\b": "event_free_survival",
        r"objective.*response.*rate|orr\b|overall.*response": "objective_response_rate",
        r"complete.*response|cr\b": "complete_response_rate",
        r"partial.*response|pr\b": "partial_response_rate",
        r"pathologic.*complete.*response|pcr\b": "pathologic_complete_response",
        r"duration.*response|dor\b": "duration_of_response",
        r"time.*progression|ttp\b": "time_to_progression",
        r"clinical.*benefit": "clinical_benefit_rate",
        
        # Diabetes endpoints
        r"hba1c|glycated.*haemoglobin|glycated.*hemoglobin|a1c": "hba1c_change",
        r"fasting.*plasma.*glucose|fpg\b|fasting.*glucose": "fasting_glucose",
        r"hypoglyc": "hypoglycemia_events",
        r"body.*weight|weight.*change|weight.*loss": "body_weight_change",
        
        # Cardiovascular endpoints
        r"mace\b|major.*adverse.*cardiovascular": "mace",
        r"myocardial.*infarction|mi\b|heart.*attack": "myocardial_infarction",
        r"stroke\b": "stroke",
        r"cardiovascular.*death": "cardiovascular_death",
        r"hospitalization.*heart.*failure|hf.*hospitalization": "heart_failure_hospitalization",
        r"blood.*pressure|bp\b|systolic|diastolic": "blood_pressure_change",
        r"ldl|cholesterol": "ldl_cholesterol_change",
        
        # Safety endpoints
        r"adverse.*event|ae\b|safety": "adverse_events",
        r"serious.*adverse.*event|sae\b": "serious_adverse_events",
        r"treatment.*emergent": "treatment_emergent_ae",
        r"discontinu.*adverse|adverse.*discontinu": "discontinuation_due_to_ae",
        
        # Quality of life
        r"quality.*life|qol|hrqol": "quality_of_life",
        r"patient.*reported|pro\b": "patient_reported_outcomes",
        
        # Neurology endpoints
        r"adas.?cog|cognitive.*assessment": "adas_cog",
        r"mmse\b|mini.*mental": "mmse",
        r"cdr.?sb|clinical.*dementia": "cdr_sb",
        r"adcs.?adl|activities.*daily.*living": "adl_score",
    }
    
    # Timeframe normalization
    TIMEFRAME_PATTERNS = {
        r"(\d+)\s*weeks?": lambda m: f"{m.group(1)} weeks",
        r"(\d+)\s*months?": lambda m: f"{m.group(1)} months",
        r"(\d+)\s*years?": lambda m: f"{int(m.group(1)) * 12} months",
        r"(\d+)\s*days?": lambda m: f"{int(m.group(1)) // 7} weeks",
    }
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the benchmarker.

        Args:
            db_session: SQLAlchemy session for database queries.
                       If provided, will use pre-aggregated data from database.
        """
        self.db_session = db_session
        self._endpoint_repo = None
        self.endpoint_data: Dict[str, Dict[str, EndpointPattern]] = defaultdict(dict)
        self._trial_endpoint_map: Dict[str, List[str]] = defaultdict(list)
        self._loaded_from_db = False

    @property
    def endpoint_repo(self):
        """Lazy-load endpoint repository."""
        if self._endpoint_repo is None and self.db_session is not None:
            from ..database import EndpointRepository
            self._endpoint_repo = EndpointRepository(self.db_session)
        return self._endpoint_repo

    def load_from_database(self) -> int:
        """
        Load endpoint data from database.

        Returns:
            Number of endpoints loaded
        """
        if self.endpoint_repo is None:
            logger.warning("No database session available")
            return 0

        from ..database.models import Endpoint
        import json

        try:
            db_endpoints = self.db_session.query(Endpoint).all()

            for ep in db_endpoints:
                # Determine condition from therapeutic areas
                therapeutic_areas = json.loads(ep.therapeutic_areas) if ep.therapeutic_areas else []
                condition = self._normalize_condition(therapeutic_areas) if therapeutic_areas else "other"

                endpoint_key = f"primary_{ep.measure_normalized}" if ep.as_primary > 0 else f"secondary_{ep.measure_normalized}"

                self.endpoint_data[condition][endpoint_key] = EndpointPattern(
                    measure=ep.measure_normalized,
                    raw_examples=json.loads(ep.raw_examples) if ep.raw_examples else [],
                    frequency=ep.frequency,
                    typical_timeframes=json.loads(ep.typical_timeframes) if ep.typical_timeframes else [],
                    associated_conditions=therapeutic_areas,
                    phases=json.loads(ep.phases) if ep.phases else [],
                    success_indicators={
                        "completed": ep.trials_completed,
                        "terminated": ep.trials_terminated,
                    },
                )

            self._loaded_from_db = True
            logger.info(f"Loaded {len(db_endpoints)} endpoints from database")
            return len(db_endpoints)

        except Exception as e:
            logger.error(f"Error loading endpoints from database: {e}")
            return 0
    
    def process_trial(self, trial_data: Dict[str, Any]) -> None:
        """
        Process a trial to extract endpoint information.
        
        Args:
            trial_data: Dict containing:
                - nct_id: str
                - conditions: List[str]
                - phase: List[str]
                - status: str
                - primary_outcomes: List[Dict] with measure, timeFrame
                - secondary_outcomes: List[Dict] with measure, timeFrame
        """
        nct_id = trial_data.get("nct_id", "")
        conditions = trial_data.get("conditions", [])
        phases = trial_data.get("phase", [])
        status = trial_data.get("status", "")
        
        # Normalize condition
        condition_key = self._normalize_condition(conditions)
        
        # Process primary outcomes
        for outcome in trial_data.get("primary_outcomes", []):
            self._process_outcome(
                outcome, condition_key, phases, status, nct_id, is_primary=True
            )
        
        # Process secondary outcomes
        for outcome in trial_data.get("secondary_outcomes", []):
            self._process_outcome(
                outcome, condition_key, phases, status, nct_id, is_primary=False
            )
    
    def _process_outcome(
        self,
        outcome: Dict[str, Any],
        condition: str,
        phases: List[str],
        status: str,
        nct_id: str,
        is_primary: bool,
    ) -> None:
        """Process a single outcome/endpoint."""
        measure = outcome.get("measure", "")
        timeframe = outcome.get("timeFrame", "")
        
        if not measure:
            return
        
        # Normalize endpoint
        normalized = self._normalize_endpoint(measure)
        if not normalized:
            normalized = self._create_generic_key(measure)
        
        # Get or create endpoint pattern
        endpoint_key = f"{'primary' if is_primary else 'secondary'}_{normalized}"
        
        if endpoint_key not in self.endpoint_data[condition]:
            self.endpoint_data[condition][endpoint_key] = EndpointPattern(
                measure=normalized,
                raw_examples=[],
                frequency=0,
                typical_timeframes=[],
                associated_conditions=[condition],
                phases=[],
                success_indicators={"completed": 0, "terminated": 0},
            )
        
        pattern = self.endpoint_data[condition][endpoint_key]
        
        # Update pattern
        if nct_id not in self._trial_endpoint_map[endpoint_key]:
            self._trial_endpoint_map[endpoint_key].append(nct_id)
            pattern.frequency += 1
            
            # Add raw example (keep limited set)
            if len(pattern.raw_examples) < 5:
                pattern.raw_examples.append(measure[:200])
            
            # Add timeframe
            normalized_tf = self._normalize_timeframe(timeframe)
            if normalized_tf and normalized_tf not in pattern.typical_timeframes:
                pattern.typical_timeframes.append(normalized_tf)
            
            # Add phase
            for phase in phases:
                if phase not in pattern.phases:
                    pattern.phases.append(phase)
            
            # Update success indicators
            if status == "COMPLETED":
                pattern.success_indicators["completed"] += 1
            elif status in ["TERMINATED", "WITHDRAWN"]:
                pattern.success_indicators["terminated"] += 1
    
    def _normalize_endpoint(self, measure: str) -> Optional[str]:
        """Normalize endpoint measure to standard category."""
        measure_lower = measure.lower()
        
        for pattern, normalized in self.ENDPOINT_PATTERNS.items():
            if re.search(pattern, measure_lower):
                return normalized
        
        return None
    
    def _create_generic_key(self, measure: str) -> str:
        """Create a generic key for unrecognized endpoints."""
        # Extract key words
        words = re.findall(r'\b[a-z]{4,}\b', measure.lower())
        if words:
            return "_".join(words[:3])
        return "other_endpoint"
    
    def _normalize_condition(self, conditions: List[str]) -> str:
        """Normalize condition list to single category."""
        if not conditions:
            return "other"
        
        combined = " ".join(conditions).lower()
        
        if any(term in combined for term in ["diabet", "glucose", "hba1c"]):
            return "diabetes"
        elif any(term in combined for term in ["breast cancer"]):
            return "breast_cancer"
        elif any(term in combined for term in ["lung cancer", "nsclc", "sclc"]):
            return "lung_cancer"
        elif any(term in combined for term in ["cancer", "tumor", "carcinoma", "lymphoma"]):
            return "oncology_other"
        elif any(term in combined for term in ["heart failure", "hf"]):
            return "heart_failure"
        elif any(term in combined for term in ["coronary", "acs", "mi"]):
            return "coronary_artery_disease"
        elif any(term in combined for term in ["alzheimer", "dementia"]):
            return "alzheimer"
        elif any(term in combined for term in ["parkinson"]):
            return "parkinson"
        elif any(term in combined for term in ["rheumatoid", "ra "]):
            return "rheumatoid_arthritis"
        elif any(term in combined for term in ["psoria"]):
            return "psoriasis"
        
        return "other"
    
    def _normalize_timeframe(self, timeframe: str) -> Optional[str]:
        """Normalize timeframe to consistent format."""
        if not timeframe:
            return None
        
        tf_lower = timeframe.lower()
        
        for pattern, converter in self.TIMEFRAME_PATTERNS.items():
            match = re.search(pattern, tf_lower)
            if match:
                return converter(match)
        
        return None
    
    def analyze_condition(
        self,
        condition: str,
        phase_filter: Optional[List[str]] = None,
    ) -> EndpointAnalysis:
        """
        Analyze endpoints for a specific condition.

        Args:
            condition: Indication to analyze (e.g., "diabetes", "breast_cancer")
            phase_filter: Only include specific phases (e.g., ["PHASE3"])

        Returns:
            EndpointAnalysis with patterns and recommendations
        """
        # Try to get analysis from database first
        if self.endpoint_repo is not None and not self._loaded_from_db:
            try:
                db_analysis = self.endpoint_repo.get_endpoint_analysis(condition)
                if db_analysis and db_analysis.get("total_trials_analyzed", 0) > 0:
                    # Convert database results to EndpointPattern objects
                    primary_patterns = []
                    for ep_dict in db_analysis.get("primary_endpoints", []):
                        primary_patterns.append(EndpointPattern(
                            measure=ep_dict["measure_normalized"],
                            raw_examples=ep_dict.get("raw_examples", []),
                            frequency=ep_dict["frequency"],
                            typical_timeframes=ep_dict.get("typical_timeframes", []),
                            associated_conditions=ep_dict.get("therapeutic_areas", []),
                            phases=ep_dict.get("phases", []),
                            success_indicators={
                                "completed": ep_dict.get("trials_completed", 0),
                                "terminated": ep_dict.get("trials_terminated", 0),
                            },
                        ))

                    secondary_patterns = []
                    for ep_dict in db_analysis.get("secondary_endpoints", []):
                        secondary_patterns.append(EndpointPattern(
                            measure=ep_dict["measure_normalized"],
                            raw_examples=ep_dict.get("raw_examples", []),
                            frequency=ep_dict["frequency"],
                            typical_timeframes=ep_dict.get("typical_timeframes", []),
                            associated_conditions=ep_dict.get("therapeutic_areas", []),
                            phases=ep_dict.get("phases", []),
                            success_indicators={
                                "completed": ep_dict.get("trials_completed", 0),
                                "terminated": ep_dict.get("trials_terminated", 0),
                            },
                        ))

                    # Generate recommendations from database data
                    recommendations = self._generate_recommendations(
                        condition, primary_patterns, secondary_patterns
                    )
                    regulatory_insights = self._generate_regulatory_insights(
                        condition, primary_patterns
                    )

                    return EndpointAnalysis(
                        condition=condition,
                        total_trials_analyzed=db_analysis["total_trials_analyzed"],
                        primary_endpoints=primary_patterns,
                        secondary_endpoints=secondary_patterns,
                        recommendations=recommendations,
                        regulatory_insights=regulatory_insights,
                    )

            except Exception as e:
                logger.warning(f"Database endpoint analysis failed: {e}")

        # Fall back to in-memory data
        condition_data = self.endpoint_data.get(condition, {})
        
        # Separate primary and secondary
        primary_patterns = []
        secondary_patterns = []
        
        for key, pattern in condition_data.items():
            # Apply phase filter if specified
            if phase_filter:
                matching_phases = [p for p in pattern.phases if p in phase_filter]
                if not matching_phases:
                    continue
            
            if key.startswith("primary_"):
                primary_patterns.append(pattern)
            else:
                secondary_patterns.append(pattern)
        
        # Sort by frequency
        primary_patterns.sort(key=lambda x: x.frequency, reverse=True)
        secondary_patterns.sort(key=lambda x: x.frequency, reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            condition, primary_patterns, secondary_patterns
        )
        
        # Generate regulatory insights
        regulatory_insights = self._generate_regulatory_insights(
            condition, primary_patterns
        )
        
        return EndpointAnalysis(
            condition=condition,
            total_trials_analyzed=sum(p.frequency for p in primary_patterns),
            primary_endpoints=primary_patterns[:10],
            secondary_endpoints=secondary_patterns[:10],
            recommendations=recommendations,
            regulatory_insights=regulatory_insights,
        )
    
    def _generate_recommendations(
        self,
        condition: str,
        primary_patterns: List[EndpointPattern],
        secondary_patterns: List[EndpointPattern],
    ) -> List[EndpointRecommendation]:
        """Generate endpoint recommendations based on patterns."""
        recommendations = []
        
        for pattern in primary_patterns[:5]:
            # Calculate success rate
            total = pattern.success_indicators["completed"] + pattern.success_indicators["terminated"]
            success_rate = pattern.success_indicators["completed"] / max(total, 1)
            
            # Determine confidence
            if pattern.frequency >= 20 and success_rate >= 0.7:
                confidence = "high"
            elif pattern.frequency >= 10 and success_rate >= 0.5:
                confidence = "medium"
            else:
                confidence = "low"
            
            # Build rationale
            rationale = f"Used in {pattern.frequency} {condition} trials"
            if success_rate > 0:
                rationale += f" with {success_rate:.0%} completion rate"
            
            # Get typical timeframe
            typical_tf = pattern.typical_timeframes[0] if pattern.typical_timeframes else "Not specified"
            
            # Considerations
            considerations = []
            if pattern.success_indicators["terminated"] > pattern.success_indicators["completed"]:
                considerations.append("Higher termination rate than completion - review carefully")
            if "PHASE3" in pattern.phases and len(pattern.phases) == 1:
                considerations.append("Primarily used in Phase 3 - appropriate for pivotal trials")
            if pattern.frequency < 10:
                considerations.append("Limited historical precedent - consider additional validation")
            
            recommendations.append(EndpointRecommendation(
                endpoint=pattern.measure,
                confidence=confidence,
                rationale=rationale,
                typical_timeframe=typical_tf,
                similar_trials=self._trial_endpoint_map.get(f"primary_{pattern.measure}", [])[:5],
                considerations=considerations,
            ))
        
        return recommendations
    
    def _generate_regulatory_insights(
        self,
        condition: str,
        primary_patterns: List[EndpointPattern],
    ) -> List[str]:
        """Generate regulatory insights based on endpoint patterns."""
        insights = []
        
        # Condition-specific insights
        condition_insights = {
            "diabetes": [
                "HbA1c is the gold standard primary endpoint for diabetes trials",
                "FDA now requires cardiovascular safety outcomes for diabetes drugs",
                "Body weight change increasingly important as secondary endpoint",
            ],
            "breast_cancer": [
                "Pathologic complete response (pCR) accepted for accelerated approval in neoadjuvant setting",
                "Overall survival remains preferred for full approval",
                "PFS accepted in metastatic setting with adequate magnitude of effect",
            ],
            "lung_cancer": [
                "Overall survival preferred for full approval",
                "PFS acceptable for accelerated approval with strong magnitude of effect",
                "ORR used for tissue-agnostic approvals",
            ],
            "alzheimer": [
                "Dual endpoints required: cognitive + functional",
                "ADAS-Cog + ADCS-ADL most commonly used combination",
                "Recent approvals based on amyloid reduction as biomarker endpoint",
            ],
            "heart_failure": [
                "CV death + HF hospitalization is standard composite primary endpoint",
                "NT-proBNP acceptable as biomarker endpoint for certain indications",
                "Functional endpoints (6MWT, KCCQ) important for symptom claims",
            ],
        }
        
        if condition in condition_insights:
            insights.extend(condition_insights[condition])
        
        # Pattern-based insights
        if primary_patterns:
            top_endpoint = primary_patterns[0]
            if top_endpoint.frequency >= 50:
                insights.append(
                    f"'{top_endpoint.measure}' is the dominant endpoint with strong precedent"
                )
        
        return insights
    
    def compare_endpoints(
        self,
        endpoint1: str,
        endpoint2: str,
        condition: str,
    ) -> Dict[str, Any]:
        """
        Compare two endpoints for a condition.
        
        Returns comparative statistics.
        """
        condition_data = self.endpoint_data.get(condition, {})
        
        pattern1 = None
        pattern2 = None
        
        for key, pattern in condition_data.items():
            if pattern.measure == endpoint1:
                pattern1 = pattern
            elif pattern.measure == endpoint2:
                pattern2 = pattern
        
        comparison = {
            "endpoint1": {
                "name": endpoint1,
                "frequency": pattern1.frequency if pattern1 else 0,
                "success_rate": (
                    pattern1.success_indicators["completed"] / 
                    max(sum(pattern1.success_indicators.values()), 1)
                ) if pattern1 else 0,
                "typical_timeframe": pattern1.typical_timeframes[0] if pattern1 and pattern1.typical_timeframes else "N/A",
            },
            "endpoint2": {
                "name": endpoint2,
                "frequency": pattern2.frequency if pattern2 else 0,
                "success_rate": (
                    pattern2.success_indicators["completed"] /
                    max(sum(pattern2.success_indicators.values()), 1)
                ) if pattern2 else 0,
                "typical_timeframe": pattern2.typical_timeframes[0] if pattern2 and pattern2.typical_timeframes else "N/A",
            },
            "recommendation": "",
        }
        
        # Generate recommendation
        if comparison["endpoint1"]["frequency"] > comparison["endpoint2"]["frequency"] * 2:
            comparison["recommendation"] = f"{endpoint1} has much stronger precedent"
        elif comparison["endpoint2"]["frequency"] > comparison["endpoint1"]["frequency"] * 2:
            comparison["recommendation"] = f"{endpoint2} has much stronger precedent"
        else:
            comparison["recommendation"] = "Both endpoints have comparable precedent"
        
        return comparison


def build_endpoint_benchmarks(trial_list: List[Dict[str, Any]]) -> EndpointBenchmarker:
    """
    Build endpoint benchmarks from a list of trials.

    Args:
        trial_list: List of trial dicts with outcome data

    Returns:
        Populated EndpointBenchmarker instance
    """
    benchmarker = EndpointBenchmarker()

    for trial in trial_list:
        benchmarker.process_trial(trial)

    return benchmarker


def create_benchmarker_with_db() -> EndpointBenchmarker:
    """
    Create EndpointBenchmarker with database connection.

    Automatically loads endpoint data from the database.

    Usage:
        benchmarker = create_benchmarker_with_db()
        analysis = benchmarker.analyze_condition("diabetes")
    """
    from ..database import DatabaseManager

    db = DatabaseManager.get_instance()
    session = db.get_session()

    benchmarker = EndpointBenchmarker(db_session=session)
    benchmarker.load_from_database()

    return benchmarker


if __name__ == "__main__":
    # Example with sample data
    sample_trials = [
        {
            "nct_id": "NCT001",
            "conditions": ["Type 2 Diabetes"],
            "phase": ["PHASE3"],
            "status": "COMPLETED",
            "primary_outcomes": [
                {"measure": "Change in HbA1c from baseline", "timeFrame": "52 weeks"},
            ],
            "secondary_outcomes": [
                {"measure": "Body weight change", "timeFrame": "52 weeks"},
                {"measure": "Fasting plasma glucose", "timeFrame": "52 weeks"},
            ],
        },
        {
            "nct_id": "NCT002",
            "conditions": ["Type 2 Diabetes"],
            "phase": ["PHASE3"],
            "status": "COMPLETED",
            "primary_outcomes": [
                {"measure": "Change in HbA1c", "timeFrame": "24 weeks"},
            ],
            "secondary_outcomes": [
                {"measure": "Incidence of hypoglycemia", "timeFrame": "24 weeks"},
            ],
        },
        {
            "nct_id": "NCT003",
            "conditions": ["Breast Cancer"],
            "phase": ["PHASE3"],
            "status": "COMPLETED",
            "primary_outcomes": [
                {"measure": "Progression-free survival", "timeFrame": "36 months"},
            ],
            "secondary_outcomes": [
                {"measure": "Overall survival", "timeFrame": "60 months"},
                {"measure": "Objective response rate", "timeFrame": "12 weeks"},
            ],
        },
    ]
    
    benchmarker = build_endpoint_benchmarks(sample_trials)
    
    print("=" * 60)
    print("ENDPOINT BENCHMARKING")
    print("=" * 60)
    
    # Analyze diabetes endpoints
    diabetes_analysis = benchmarker.analyze_condition("diabetes")
    
    print(f"\nDiabetes Endpoint Analysis:")
    print(f"Trials analyzed: {diabetes_analysis.total_trials_analyzed}")
    
    print("\nTop Primary Endpoints:")
    for pattern in diabetes_analysis.primary_endpoints[:3]:
        print(f"  - {pattern.measure}: {pattern.frequency} trials")
        if pattern.typical_timeframes:
            print(f"    Typical timeframe: {pattern.typical_timeframes[0]}")
    
    print("\nRecommendations:")
    for rec in diabetes_analysis.recommendations[:2]:
        print(f"\n  Endpoint: {rec.endpoint}")
        print(f"  Confidence: {rec.confidence}")
        print(f"  Rationale: {rec.rationale}")
        print(f"  Typical timeframe: {rec.typical_timeframe}")
    
    print("\nRegulatory Insights:")
    for insight in diabetes_analysis.regulatory_insights[:3]:
        print(f"  • {insight}")
