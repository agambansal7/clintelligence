"""
Trial Similarity Engine

Finds individual trials most similar to a given protocol or trial,
enabling precise benchmarking against specific historical examples
rather than just therapeutic area averages.

Similarity factors:
- Condition/indication match
- Phase match
- Eligibility criteria similarity (NLP-based)
- Endpoint similarity
- Enrollment target similarity
- Geographic overlap
- Sponsor type match
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
import json

logger = logging.getLogger(__name__)


@dataclass
class SimilarTrial:
    """A trial identified as similar to the query."""
    nct_id: str
    title: str
    similarity_score: float  # 0-100
    status: str
    phase: str
    sponsor: str
    enrollment: int
    duration_months: Optional[float]
    num_sites: int
    completion_date: Optional[str]
    why_stopped: Optional[str]  # If terminated

    # Similarity breakdown
    condition_match: float
    phase_match: float
    eligibility_similarity: float
    endpoint_similarity: float
    enrollment_similarity: float

    # Key learnings from this trial
    primary_endpoints: List[str] = field(default_factory=list)
    eligibility_criteria: Optional[str] = None
    amendments_count: int = 0
    enrollment_rate: Optional[float] = None  # patients per site per month


@dataclass
class TrialComparisonResult:
    """Side-by-side comparison of two trials."""
    query_trial: Dict[str, Any]
    comparison_trial: Dict[str, Any]

    # Differences highlighted
    eligibility_differences: List[Dict[str, Any]] = field(default_factory=list)
    endpoint_differences: List[Dict[str, Any]] = field(default_factory=list)
    design_differences: List[Dict[str, Any]] = field(default_factory=list)

    # Outcome comparison (if comparison trial completed)
    outcome_insights: List[str] = field(default_factory=list)
    risk_factors_from_comparison: List[str] = field(default_factory=list)


class TrialSimilarityEngine:
    """
    Engine for finding and comparing similar trials.

    Usage:
        engine = TrialSimilarityEngine(db_manager)

        # Find similar trials
        similar = engine.find_similar_trials(
            condition="Type 2 Diabetes",
            phase="PHASE3",
            eligibility_criteria="Adults 18-75 with HbA1c 7-10%...",
            primary_endpoint="Change in HbA1c",
            enrollment_target=500
        )

        # Compare specific trials
        comparison = engine.compare_trials("NCT03689374", "NCT04123456")
    """

    # Condition synonyms for better matching
    CONDITION_SYNONYMS = {
        "diabetes": ["diabetes mellitus", "type 2 diabetes", "t2dm", "type 1 diabetes", "t1dm", "diabetic"],
        "breast cancer": ["breast neoplasm", "breast carcinoma", "breast tumor", "mammary cancer"],
        "lung cancer": ["nsclc", "sclc", "non-small cell lung cancer", "small cell lung cancer", "lung carcinoma"],
        "heart failure": ["hf", "chf", "congestive heart failure", "cardiac failure", "hfref", "hfpef"],
        "hypertension": ["htn", "high blood pressure", "elevated blood pressure"],
        "depression": ["mdd", "major depressive disorder", "depressive disorder", "major depression"],
        "alzheimer": ["alzheimer's disease", "ad", "alzheimer disease", "dementia"],
        "rheumatoid arthritis": ["ra", "rheumatoid", "inflammatory arthritis"],
        "multiple sclerosis": ["ms", "relapsing ms", "rrms", "progressive ms"],
        "copd": ["chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis"],
    }

    # Eligibility criteria keywords for similarity matching
    ELIGIBILITY_KEYWORDS = [
        # Age-related
        r"(\d+)\s*(?:years?|yrs?)?\s*(?:of age|old)?",
        r"(?:age|aged)\s*(?:>=?|<=?|between)?\s*(\d+)",
        # Lab values
        r"hba1c\s*(?:>=?|<=?|of)?\s*([\d.]+)",
        r"egfr\s*(?:>=?|<=?)\s*(\d+)",
        r"bmi\s*(?:>=?|<=?|between)?\s*([\d.]+)",
        r"ldl\s*(?:>=?|<=?)\s*(\d+)",
        # Disease severity
        r"nyha\s*(?:class)?\s*([i-iv]+|\d)",
        r"ecog\s*(?:performance status|ps)?\s*(?:of|<=?)?\s*(\d)",
        r"child.?pugh\s*(?:class|score)?\s*([a-c]|\d+)",
        # Treatment history
        r"(?:prior|previous)\s+(?:treatment|therapy|medication)",
        r"treatment.?na[ií]ve",
        r"(?:failed|inadequate response to)\s+\d+\s+(?:prior|previous)",
        # Exclusions
        r"(?:no|without)\s+(?:prior|history of)\s+(\w+)",
        r"(?:exclude|excluded|exclusion).*?(\w+\s+\w+)",
    ]

    # Endpoint categories for matching
    ENDPOINT_CATEGORIES = {
        "efficacy_primary": [
            "overall survival", "progression-free survival", "disease-free survival",
            "objective response rate", "complete response", "partial response",
            "hba1c", "blood pressure", "ldl", "pain score", "symptom improvement"
        ],
        "safety": [
            "adverse events", "serious adverse events", "treatment-emergent",
            "mortality", "hospitalization", "discontinuation"
        ],
        "qol": [
            "quality of life", "patient-reported", "functional status",
            "health-related quality", "sf-36", "eq-5d"
        ],
    }

    def __init__(self, db_manager):
        """Initialize with database connection."""
        self.db = db_manager

    def find_similar_trials(
        self,
        condition: str,
        phase: Optional[str] = None,
        eligibility_criteria: Optional[str] = None,
        primary_endpoint: Optional[str] = None,
        enrollment_target: Optional[int] = None,
        sponsor_type: Optional[str] = None,
        exclude_nct_ids: Optional[List[str]] = None,
        limit: int = 10,
        include_terminated: bool = True,
    ) -> List[SimilarTrial]:
        """
        Find trials most similar to the given parameters.

        Args:
            condition: Target condition/indication
            phase: Trial phase (PHASE1, PHASE2, PHASE3, etc.)
            eligibility_criteria: Full eligibility criteria text
            primary_endpoint: Primary endpoint description
            enrollment_target: Target enrollment number
            sponsor_type: INDUSTRY, ACADEMIC, etc.
            exclude_nct_ids: NCT IDs to exclude from results
            limit: Maximum similar trials to return
            include_terminated: Include terminated trials (valuable for learning)

        Returns:
            List of SimilarTrial objects sorted by similarity score
        """
        exclude_nct_ids = exclude_nct_ids or []

        # Step 1: Get candidate trials from database
        candidates = self._get_candidate_trials(
            condition=condition,
            phase=phase,
            include_terminated=include_terminated,
            limit=500  # Get more candidates for scoring
        )

        if not candidates:
            logger.warning(f"No candidate trials found for condition: {condition}")
            return []

        # Step 2: Score each candidate
        scored_trials = []
        for trial in candidates:
            if trial.get("nct_id") in exclude_nct_ids:
                continue

            similarity = self._calculate_similarity(
                trial=trial,
                query_condition=condition,
                query_phase=phase,
                query_eligibility=eligibility_criteria,
                query_endpoint=primary_endpoint,
                query_enrollment=enrollment_target,
            )

            if similarity["total_score"] > 20:  # Minimum threshold
                scored_trials.append((trial, similarity))

        # Step 3: Sort by similarity and return top results
        scored_trials.sort(key=lambda x: x[1]["total_score"], reverse=True)

        results = []
        for trial, similarity in scored_trials[:limit]:
            # Calculate enrollment rate if we have the data
            enrollment_rate = None
            if trial.get("enrollment") and trial.get("num_sites") and trial.get("duration_months"):
                if trial["num_sites"] > 0 and trial["duration_months"] > 0:
                    enrollment_rate = trial["enrollment"] / trial["num_sites"] / trial["duration_months"]

            # Parse primary endpoints
            primary_endpoints = []
            try:
                outcomes = trial.get("primary_outcomes")
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                if outcomes:
                    primary_endpoints = [o.get("measure", "") for o in outcomes if o.get("measure")]
            except (json.JSONDecodeError, TypeError):
                pass

            results.append(SimilarTrial(
                nct_id=trial.get("nct_id", ""),
                title=trial.get("title", ""),
                similarity_score=similarity["total_score"],
                status=trial.get("status", ""),
                phase=trial.get("phase", ""),
                sponsor=trial.get("sponsor", ""),
                enrollment=trial.get("enrollment", 0),
                duration_months=trial.get("duration_months"),
                num_sites=trial.get("num_sites", 0),
                completion_date=trial.get("completion_date"),
                why_stopped=trial.get("why_stopped"),
                condition_match=similarity["condition_score"],
                phase_match=similarity["phase_score"],
                eligibility_similarity=similarity["eligibility_score"],
                endpoint_similarity=similarity["endpoint_score"],
                enrollment_similarity=similarity["enrollment_score"],
                primary_endpoints=primary_endpoints,
                eligibility_criteria=trial.get("eligibility_criteria"),
                enrollment_rate=enrollment_rate,
            ))

        return results

    def _get_candidate_trials(
        self,
        condition: str,
        phase: Optional[str],
        include_terminated: bool,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get candidate trials from database based on condition."""
        # Expand condition with synonyms
        search_terms = [condition.lower()]
        for key, synonyms in self.CONDITION_SYNONYMS.items():
            if condition.lower() in key or key in condition.lower():
                search_terms.extend(synonyms)
            for syn in synonyms:
                if condition.lower() in syn or syn in condition.lower():
                    search_terms.append(key)
                    search_terms.extend(synonyms)
                    break

        search_terms = list(set(search_terms))

        # Build status filter
        statuses = ["COMPLETED"]
        if include_terminated:
            statuses.extend(["TERMINATED", "WITHDRAWN", "SUSPENDED"])

        # Query database
        with self.db.get_session() as session:
            from src.database.models import Trial
            from sqlalchemy import or_, func

            # Build condition filter
            condition_filters = []
            for term in search_terms:
                condition_filters.append(
                    func.lower(Trial.conditions).contains(term)
                )

            query = session.query(Trial).filter(
                or_(*condition_filters),
                Trial.status.in_(statuses),
                Trial.study_type == "INTERVENTIONAL"
            )

            if phase:
                query = query.filter(Trial.phase.contains(phase.replace("PHASE", "")))

            # Order by enrollment (larger trials often more informative)
            query = query.order_by(Trial.enrollment.desc().nullslast())

            trials = query.limit(limit).all()

            results = []
            for t in trials:
                # Calculate duration in months
                duration_months = None
                if t.start_date and t.completion_date:
                    try:
                        from datetime import datetime
                        start = datetime.strptime(str(t.start_date)[:10], "%Y-%m-%d")
                        end = datetime.strptime(str(t.completion_date)[:10], "%Y-%m-%d")
                        duration_months = (end - start).days / 30.44
                    except (ValueError, TypeError):
                        pass

                results.append({
                    "nct_id": t.nct_id,
                    "title": t.title,
                    "status": t.status,
                    "phase": t.phase,
                    "conditions": t.conditions,
                    "sponsor": t.sponsor,
                    "enrollment": t.enrollment,
                    "num_sites": t.num_sites,
                    "start_date": str(t.start_date) if t.start_date else None,
                    "completion_date": str(t.completion_date) if t.completion_date else None,
                    "duration_months": duration_months,
                    "eligibility_criteria": t.eligibility_criteria,
                    "primary_outcomes": t.primary_outcomes,
                    "secondary_outcomes": t.secondary_outcomes,
                    "why_stopped": t.why_stopped,
                    "therapeutic_area": t.therapeutic_area,
                })

            return results

    def _calculate_similarity(
        self,
        trial: Dict[str, Any],
        query_condition: str,
        query_phase: Optional[str],
        query_eligibility: Optional[str],
        query_endpoint: Optional[str],
        query_enrollment: Optional[int],
    ) -> Dict[str, float]:
        """Calculate similarity scores between query and candidate trial."""
        scores = {
            "condition_score": 0.0,
            "phase_score": 0.0,
            "eligibility_score": 0.0,
            "endpoint_score": 0.0,
            "enrollment_score": 0.0,
            "total_score": 0.0,
        }

        # Weights for each factor
        weights = {
            "condition": 30,
            "phase": 20,
            "eligibility": 25,
            "endpoint": 15,
            "enrollment": 10,
        }

        # 1. Condition similarity (30%)
        trial_conditions = (trial.get("conditions") or "").lower()
        query_lower = query_condition.lower()

        if query_lower in trial_conditions:
            scores["condition_score"] = 100
        else:
            # Check for synonym matches
            matched = False
            for key, synonyms in self.CONDITION_SYNONYMS.items():
                if query_lower in key or key in query_lower:
                    for syn in synonyms:
                        if syn in trial_conditions:
                            scores["condition_score"] = 80
                            matched = True
                            break
                if matched:
                    break

            if not matched:
                # Partial word overlap
                query_words = set(query_lower.split())
                trial_words = set(trial_conditions.split())
                overlap = len(query_words & trial_words)
                if overlap > 0:
                    scores["condition_score"] = min(60, overlap * 20)

        # 2. Phase similarity (20%)
        if query_phase:
            trial_phase = trial.get("phase", "")
            if query_phase in trial_phase or trial_phase in query_phase:
                scores["phase_score"] = 100
            elif self._phases_adjacent(query_phase, trial_phase):
                scores["phase_score"] = 60
        else:
            scores["phase_score"] = 50  # Neutral if no phase specified

        # 3. Eligibility criteria similarity (25%)
        if query_eligibility and trial.get("eligibility_criteria"):
            scores["eligibility_score"] = self._calculate_eligibility_similarity(
                query_eligibility, trial["eligibility_criteria"]
            )
        else:
            scores["eligibility_score"] = 30  # Neutral

        # 4. Endpoint similarity (15%)
        if query_endpoint:
            trial_endpoints = self._extract_endpoints(trial)
            scores["endpoint_score"] = self._calculate_endpoint_similarity(
                query_endpoint, trial_endpoints
            )
        else:
            scores["endpoint_score"] = 30  # Neutral

        # 5. Enrollment similarity (10%)
        if query_enrollment and trial.get("enrollment"):
            trial_enrollment = trial["enrollment"]
            ratio = min(query_enrollment, trial_enrollment) / max(query_enrollment, trial_enrollment)
            scores["enrollment_score"] = ratio * 100
        else:
            scores["enrollment_score"] = 50  # Neutral

        # Calculate weighted total
        scores["total_score"] = (
            scores["condition_score"] * weights["condition"] +
            scores["phase_score"] * weights["phase"] +
            scores["eligibility_score"] * weights["eligibility"] +
            scores["endpoint_score"] * weights["endpoint"] +
            scores["enrollment_score"] * weights["enrollment"]
        ) / 100

        return scores

    def _phases_adjacent(self, phase1: str, phase2: str) -> bool:
        """Check if two phases are adjacent (e.g., Phase 2 and Phase 3)."""
        phase_order = ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]

        def normalize_phase(p):
            p = p.upper().replace(" ", "")
            for i, phase in enumerate(phase_order):
                if phase in p:
                    return i
            return -1

        idx1 = normalize_phase(phase1)
        idx2 = normalize_phase(phase2)

        if idx1 >= 0 and idx2 >= 0:
            return abs(idx1 - idx2) == 1
        return False

    def _calculate_eligibility_similarity(self, query: str, trial: str) -> float:
        """Calculate similarity between eligibility criteria texts."""
        query_lower = query.lower()
        trial_lower = trial.lower()

        # Extract key criteria from both
        query_criteria = self._extract_eligibility_features(query_lower)
        trial_criteria = self._extract_eligibility_features(trial_lower)

        if not query_criteria and not trial_criteria:
            return 50  # Can't compare

        # Compare extracted features
        matches = 0
        total = len(query_criteria)

        for criterion_type, query_value in query_criteria.items():
            if criterion_type in trial_criteria:
                trial_value = trial_criteria[criterion_type]

                # Numeric comparison with tolerance
                if isinstance(query_value, (int, float)) and isinstance(trial_value, (int, float)):
                    if query_value > 0:
                        ratio = min(query_value, trial_value) / max(query_value, trial_value)
                        if ratio > 0.7:
                            matches += 1
                        elif ratio > 0.5:
                            matches += 0.5
                else:
                    # String comparison
                    if query_value == trial_value:
                        matches += 1

        if total == 0:
            return 50

        return (matches / total) * 100

    def _extract_eligibility_features(self, text: str) -> Dict[str, Any]:
        """Extract structured features from eligibility text."""
        features = {}

        # Age range
        age_match = re.search(r"(?:age[ds]?|between)\s*(\d+)\s*(?:and|to|-)\s*(\d+)", text)
        if age_match:
            features["age_min"] = int(age_match.group(1))
            features["age_max"] = int(age_match.group(2))
        else:
            age_min = re.search(r"(?:>=?|at least|minimum)\s*(\d+)\s*years?", text)
            age_max = re.search(r"(?:<=?|up to|maximum)\s*(\d+)\s*years?", text)
            if age_min:
                features["age_min"] = int(age_min.group(1))
            if age_max:
                features["age_max"] = int(age_max.group(1))

        # HbA1c
        hba1c = re.search(r"hba1c\s*(?:>=?|<=?|between|of)?\s*([\d.]+)(?:\s*(?:and|to|-)\s*([\d.]+))?", text)
        if hba1c:
            features["hba1c_min"] = float(hba1c.group(1))
            if hba1c.group(2):
                features["hba1c_max"] = float(hba1c.group(2))

        # eGFR
        egfr = re.search(r"egfr\s*(?:>=?|>)\s*(\d+)", text)
        if egfr:
            features["egfr_min"] = int(egfr.group(1))

        # BMI
        bmi = re.search(r"bmi\s*(?:>=?|<=?|between|of)?\s*([\d.]+)(?:\s*(?:and|to|-)\s*([\d.]+))?", text)
        if bmi:
            features["bmi_min"] = float(bmi.group(1))
            if bmi.group(2):
                features["bmi_max"] = float(bmi.group(2))

        # ECOG
        ecog = re.search(r"ecog\s*(?:performance status|ps)?\s*(?:of|<=?)?\s*(\d)", text)
        if ecog:
            features["ecog_max"] = int(ecog.group(1))

        # Treatment history
        if "treatment" in text and "naive" in text.replace("-", " ").replace("ï", "i"):
            features["treatment_naive"] = True
        if re.search(r"(?:failed|inadequate).+(?:prior|previous)", text):
            features["prior_failure_required"] = True

        # Count inclusion/exclusion criteria
        features["inclusion_count"] = text.count("inclusion") + len(re.findall(r"(?:must|should|required to)", text))
        features["exclusion_count"] = text.count("exclusion") + len(re.findall(r"(?:must not|should not|excluded|no prior|no history)", text))

        return features

    def _extract_endpoints(self, trial: Dict[str, Any]) -> List[str]:
        """Extract endpoint strings from trial data."""
        endpoints = []

        for outcome_field in ["primary_outcomes", "secondary_outcomes"]:
            outcomes = trial.get(outcome_field)
            if outcomes:
                try:
                    if isinstance(outcomes, str):
                        outcomes = json.loads(outcomes)
                    for outcome in outcomes:
                        measure = outcome.get("measure", "")
                        if measure:
                            endpoints.append(measure.lower())
                except (json.JSONDecodeError, TypeError):
                    pass

        return endpoints

    def _calculate_endpoint_similarity(self, query: str, trial_endpoints: List[str]) -> float:
        """Calculate similarity between query endpoint and trial endpoints."""
        if not trial_endpoints:
            return 30

        query_lower = query.lower()

        # Direct match
        for endpoint in trial_endpoints:
            if query_lower in endpoint or endpoint in query_lower:
                return 100

        # Category match
        query_category = self._categorize_endpoint(query_lower)
        trial_categories = [self._categorize_endpoint(e) for e in trial_endpoints]

        if query_category and query_category in trial_categories:
            return 70

        # Word overlap
        query_words = set(query_lower.split())
        max_overlap = 0
        for endpoint in trial_endpoints:
            endpoint_words = set(endpoint.split())
            overlap = len(query_words & endpoint_words)
            max_overlap = max(max_overlap, overlap)

        if max_overlap > 0:
            return min(60, max_overlap * 20)

        return 20

    def _categorize_endpoint(self, endpoint: str) -> Optional[str]:
        """Categorize an endpoint into predefined categories."""
        for category, keywords in self.ENDPOINT_CATEGORIES.items():
            for keyword in keywords:
                if keyword in endpoint:
                    return category
        return None

    def compare_trials(
        self,
        nct_id_1: str,
        nct_id_2: str,
    ) -> TrialComparisonResult:
        """
        Compare two specific trials side-by-side.

        Args:
            nct_id_1: First trial (often the query/draft protocol)
            nct_id_2: Second trial (historical comparison)

        Returns:
            TrialComparisonResult with detailed differences
        """
        # Fetch both trials
        trial1 = self._get_trial_by_nct_id(nct_id_1)
        trial2 = self._get_trial_by_nct_id(nct_id_2)

        if not trial1 or not trial2:
            raise ValueError(f"Could not find one or both trials: {nct_id_1}, {nct_id_2}")

        result = TrialComparisonResult(
            query_trial=trial1,
            comparison_trial=trial2,
        )

        # Compare eligibility criteria
        result.eligibility_differences = self._compare_eligibility(
            trial1.get("eligibility_criteria", ""),
            trial2.get("eligibility_criteria", "")
        )

        # Compare endpoints
        result.endpoint_differences = self._compare_endpoints(trial1, trial2)

        # Compare design elements
        result.design_differences = self._compare_design(trial1, trial2)

        # Generate outcome insights if comparison trial is completed/terminated
        if trial2.get("status") in ["COMPLETED", "TERMINATED", "WITHDRAWN"]:
            result.outcome_insights = self._generate_outcome_insights(trial2)
            result.risk_factors_from_comparison = self._identify_risk_factors(trial1, trial2)

        return result

    def _get_trial_by_nct_id(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a trial by NCT ID."""
        with self.db.get_session() as session:
            from src.database.models import Trial

            trial = session.query(Trial).filter(Trial.nct_id == nct_id).first()
            if not trial:
                return None

            return {
                "nct_id": trial.nct_id,
                "title": trial.title,
                "status": trial.status,
                "phase": trial.phase,
                "conditions": trial.conditions,
                "sponsor": trial.sponsor,
                "enrollment": trial.enrollment,
                "num_sites": trial.num_sites,
                "eligibility_criteria": trial.eligibility_criteria,
                "primary_outcomes": trial.primary_outcomes,
                "secondary_outcomes": trial.secondary_outcomes,
                "why_stopped": trial.why_stopped,
                "start_date": str(trial.start_date) if trial.start_date else None,
                "completion_date": str(trial.completion_date) if trial.completion_date else None,
            }

    def _compare_eligibility(self, criteria1: str, criteria2: str) -> List[Dict[str, Any]]:
        """Compare eligibility criteria and identify differences."""
        differences = []

        features1 = self._extract_eligibility_features((criteria1 or "").lower())
        features2 = self._extract_eligibility_features((criteria2 or "").lower())

        # Compare numeric criteria
        numeric_criteria = [
            ("age_min", "Minimum Age", "years"),
            ("age_max", "Maximum Age", "years"),
            ("hba1c_min", "Minimum HbA1c", "%"),
            ("hba1c_max", "Maximum HbA1c", "%"),
            ("egfr_min", "Minimum eGFR", "mL/min"),
            ("bmi_min", "Minimum BMI", "kg/m²"),
            ("bmi_max", "Maximum BMI", "kg/m²"),
            ("ecog_max", "Maximum ECOG", ""),
        ]

        for key, label, unit in numeric_criteria:
            val1 = features1.get(key)
            val2 = features2.get(key)

            if val1 is not None or val2 is not None:
                if val1 != val2:
                    differences.append({
                        "criterion": label,
                        "trial_1_value": f"{val1} {unit}".strip() if val1 else "Not specified",
                        "trial_2_value": f"{val2} {unit}".strip() if val2 else "Not specified",
                        "significance": self._assess_criterion_significance(key, val1, val2),
                    })

        # Compare boolean criteria
        if features1.get("treatment_naive") != features2.get("treatment_naive"):
            differences.append({
                "criterion": "Treatment History",
                "trial_1_value": "Treatment-naive required" if features1.get("treatment_naive") else "Prior treatment allowed",
                "trial_2_value": "Treatment-naive required" if features2.get("treatment_naive") else "Prior treatment allowed",
                "significance": "high",
            })

        # Compare complexity
        inc_diff = abs(features1.get("inclusion_count", 0) - features2.get("inclusion_count", 0))
        exc_diff = abs(features1.get("exclusion_count", 0) - features2.get("exclusion_count", 0))

        if inc_diff > 5 or exc_diff > 5:
            differences.append({
                "criterion": "Criteria Complexity",
                "trial_1_value": f"{features1.get('inclusion_count', 0)} inclusion, {features1.get('exclusion_count', 0)} exclusion items",
                "trial_2_value": f"{features2.get('inclusion_count', 0)} inclusion, {features2.get('exclusion_count', 0)} exclusion items",
                "significance": "medium",
            })

        return differences

    def _assess_criterion_significance(self, criterion: str, val1: Any, val2: Any) -> str:
        """Assess the significance of a criterion difference."""
        if val1 is None or val2 is None:
            return "medium"

        try:
            v1, v2 = float(val1), float(val2)
            if v2 != 0:
                pct_diff = abs(v1 - v2) / v2 * 100
                if pct_diff > 30:
                    return "high"
                elif pct_diff > 15:
                    return "medium"
        except (ValueError, TypeError):
            pass

        return "low"

    def _compare_endpoints(self, trial1: Dict, trial2: Dict) -> List[Dict[str, Any]]:
        """Compare endpoints between trials."""
        differences = []

        endpoints1 = self._extract_endpoints(trial1)
        endpoints2 = self._extract_endpoints(trial2)

        # Find unique to trial 1
        for ep in endpoints1:
            if not any(ep in ep2 or ep2 in ep for ep2 in endpoints2):
                differences.append({
                    "type": "unique_to_trial_1",
                    "endpoint": ep,
                    "note": "Present in trial 1, no match in trial 2",
                })

        # Find unique to trial 2
        for ep in endpoints2:
            if not any(ep in ep1 or ep1 in ep for ep1 in endpoints1):
                differences.append({
                    "type": "unique_to_trial_2",
                    "endpoint": ep,
                    "note": "Present in trial 2, could be relevant for trial 1",
                })

        return differences

    def _compare_design(self, trial1: Dict, trial2: Dict) -> List[Dict[str, Any]]:
        """Compare trial design elements."""
        differences = []

        # Enrollment comparison
        enroll1 = trial1.get("enrollment", 0)
        enroll2 = trial2.get("enrollment", 0)
        if enroll1 and enroll2:
            ratio = enroll1 / enroll2 if enroll2 > 0 else 0
            if ratio < 0.5 or ratio > 2:
                differences.append({
                    "element": "Enrollment Target",
                    "trial_1": enroll1,
                    "trial_2": enroll2,
                    "note": f"Trial 1 targets {ratio:.1f}x the enrollment of trial 2",
                })

        # Site count comparison
        sites1 = trial1.get("num_sites", 0)
        sites2 = trial2.get("num_sites", 0)
        if sites1 and sites2:
            ratio = sites1 / sites2 if sites2 > 0 else 0
            if ratio < 0.5 or ratio > 2:
                differences.append({
                    "element": "Number of Sites",
                    "trial_1": sites1,
                    "trial_2": sites2,
                    "note": f"Different site strategy ({sites1} vs {sites2} sites)",
                })

        # Enrollment per site
        if enroll1 and sites1 and enroll2 and sites2:
            eps1 = enroll1 / sites1
            eps2 = enroll2 / sites2
            if abs(eps1 - eps2) > 10:
                differences.append({
                    "element": "Patients per Site",
                    "trial_1": f"{eps1:.1f}",
                    "trial_2": f"{eps2:.1f}",
                    "note": "Different enrollment burden per site",
                })

        return differences

    def _generate_outcome_insights(self, completed_trial: Dict) -> List[str]:
        """Generate insights from a completed/terminated trial."""
        insights = []

        status = completed_trial.get("status", "")

        if status == "COMPLETED":
            insights.append(f"Trial completed successfully with {completed_trial.get('enrollment', 'N/A')} patients enrolled")
            if completed_trial.get("num_sites"):
                insights.append(f"Used {completed_trial['num_sites']} sites to achieve enrollment")

        elif status in ["TERMINATED", "WITHDRAWN"]:
            why_stopped = completed_trial.get("why_stopped", "")
            if why_stopped:
                insights.append(f"Trial was {status.lower()}: {why_stopped}")
            else:
                insights.append(f"Trial was {status.lower()} - reason not specified")

            # Add learnings
            insights.append("Consider whether your protocol addresses the issues that led to this trial's termination")

        return insights

    def _identify_risk_factors(self, query_trial: Dict, comparison_trial: Dict) -> List[str]:
        """Identify risk factors based on comparison."""
        risks = []

        # If comparison was terminated and query is similar, flag risks
        if comparison_trial.get("status") in ["TERMINATED", "WITHDRAWN"]:
            why_stopped = (comparison_trial.get("why_stopped") or "").lower()

            if "enrollment" in why_stopped or "recruitment" in why_stopped:
                risks.append("Similar trial terminated due to enrollment issues - verify your enrollment assumptions")

            if "efficacy" in why_stopped or "futility" in why_stopped:
                risks.append("Similar trial stopped for efficacy concerns - review your endpoint selection")

            if "safety" in why_stopped:
                risks.append("Similar trial had safety concerns - ensure adequate safety monitoring")

            if "funding" in why_stopped or "business" in why_stopped:
                risks.append("Similar trial had operational/funding issues - ensure adequate resources")

        # Check for restrictive criteria in query vs successful comparison
        if comparison_trial.get("status") == "COMPLETED":
            query_features = self._extract_eligibility_features(
                (query_trial.get("eligibility_criteria") or "").lower()
            )
            comp_features = self._extract_eligibility_features(
                (comparison_trial.get("eligibility_criteria") or "").lower()
            )

            # More restrictive age range
            if query_features.get("age_max", 100) < comp_features.get("age_max", 100):
                risks.append(f"Your age range is more restrictive than the successful trial ({query_features.get('age_max')} vs {comp_features.get('age_max')} max age)")

            # More exclusion criteria
            if query_features.get("exclusion_count", 0) > comp_features.get("exclusion_count", 0) + 5:
                risks.append("Your protocol has significantly more exclusion criteria than the successful trial")

        return risks


def get_similar_trials(
    db_manager,
    condition: str,
    phase: Optional[str] = None,
    eligibility_criteria: Optional[str] = None,
    primary_endpoint: Optional[str] = None,
    enrollment_target: Optional[int] = None,
    limit: int = 10,
) -> List[SimilarTrial]:
    """Convenience function to find similar trials."""
    engine = TrialSimilarityEngine(db_manager)
    return engine.find_similar_trials(
        condition=condition,
        phase=phase,
        eligibility_criteria=eligibility_criteria,
        primary_endpoint=primary_endpoint,
        enrollment_target=enrollment_target,
        limit=limit,
    )
