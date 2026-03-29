"""
Multi-Dimensional Trial Scorer

Computes similarity between a protocol and trials across multiple dimensions,
allowing for nuanced matching and filtering.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from src.analysis.enhanced_extractor import ExtractedProtocolV2
from src.analysis.medical_ontology import get_ontology

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """Score for a single dimension with explanation."""
    score: float  # 0-100
    confidence: float  # 0-1, how confident we are in this score
    explanation: str = ""
    matched_terms: List[str] = field(default_factory=list)


@dataclass
class MultiDimensionalScore:
    """Complete multi-dimensional similarity score."""
    # Individual dimension scores
    condition_score: DimensionScore = field(default_factory=lambda: DimensionScore(0, 0))
    intervention_score: DimensionScore = field(default_factory=lambda: DimensionScore(0, 0))
    endpoint_score: DimensionScore = field(default_factory=lambda: DimensionScore(0, 0))
    population_score: DimensionScore = field(default_factory=lambda: DimensionScore(0, 0))
    design_score: DimensionScore = field(default_factory=lambda: DimensionScore(0, 0))

    # Hard filter results
    has_exclusion_conflict: bool = False
    exclusion_reasons: List[str] = field(default_factory=list)

    # Final combined score
    overall_score: float = 0.0

    # Metadata
    nct_id: str = ""
    trial_title: str = ""

    def compute_overall(self, weights: Dict[str, float] = None) -> float:
        """Compute weighted overall score."""
        if weights is None:
            weights = {
                "condition": 0.35,
                "intervention": 0.25,
                "endpoint": 0.15,
                "population": 0.15,
                "design": 0.10,
            }

        # If there's an exclusion conflict, heavily penalize
        if self.has_exclusion_conflict:
            self.overall_score = min(30, self.condition_score.score * 0.3)
            return self.overall_score

        weighted_sum = (
            self.condition_score.score * weights.get("condition", 0.35) +
            self.intervention_score.score * weights.get("intervention", 0.25) +
            self.endpoint_score.score * weights.get("endpoint", 0.15) +
            self.population_score.score * weights.get("population", 0.15) +
            self.design_score.score * weights.get("design", 0.10)
        )

        self.overall_score = weighted_sum
        return self.overall_score

    def to_dict(self) -> Dict:
        return {
            "nct_id": self.nct_id,
            "overall_score": self.overall_score,
            "condition_score": self.condition_score.score,
            "intervention_score": self.intervention_score.score,
            "endpoint_score": self.endpoint_score.score,
            "population_score": self.population_score.score,
            "design_score": self.design_score.score,
            "has_exclusion_conflict": self.has_exclusion_conflict,
            "exclusion_reasons": self.exclusion_reasons,
        }


class TrialScorer:
    """
    Compute multi-dimensional similarity between protocol and trials.
    """

    def __init__(self):
        self.ontology = get_ontology()

    def score_trial(
        self,
        protocol: ExtractedProtocolV2,
        trial: Dict[str, Any],
        vector_score: float = 0.5
    ) -> MultiDimensionalScore:
        """
        Score a single trial against the protocol.

        Args:
            protocol: Extracted protocol information
            trial: Trial data dictionary
            vector_score: Pre-computed vector similarity (0-1)

        Returns:
            MultiDimensionalScore with all dimension scores
        """
        result = MultiDimensionalScore(
            nct_id=trial.get("nct_id", ""),
            trial_title=trial.get("title", ""),
        )

        # Score each dimension
        result.condition_score = self._score_condition(protocol, trial, vector_score)
        result.intervention_score = self._score_intervention(protocol, trial)
        result.endpoint_score = self._score_endpoint(protocol, trial)
        result.population_score = self._score_population(protocol, trial)
        result.design_score = self._score_design(protocol, trial)

        # Check for hard exclusion conflicts
        conflicts = self._check_exclusion_conflicts(protocol, trial)
        if conflicts:
            result.has_exclusion_conflict = True
            result.exclusion_reasons = conflicts

        # Compute overall
        result.compute_overall()

        return result

    def _score_condition(
        self,
        protocol: ExtractedProtocolV2,
        trial: Dict[str, Any],
        vector_score: float
    ) -> DimensionScore:
        """Score condition/disease match."""
        score = 0.0
        confidence = 0.5
        matched = []

        trial_conditions = (trial.get("conditions") or "").lower()
        trial_title = (trial.get("title") or "").lower()
        trial_ta = (trial.get("therapeutic_area") or "").lower()
        trial_text = f"{trial_conditions} {trial_title} {trial_ta}"

        protocol_condition = protocol.condition.lower()

        # Direct match (highest value)
        if protocol_condition in trial_conditions:
            score += 50
            matched.append(f"Direct: {protocol_condition}")
            confidence = 0.9

        # Synonym matches
        for syn in protocol.condition_synonyms[:10]:
            if syn.lower() in trial_text:
                score += 20
                matched.append(f"Synonym: {syn}")
                confidence = max(confidence, 0.8)
                break

        # Ontology expansion
        canonical, expanded = self.ontology.normalize_condition(protocol_condition)
        for term in expanded[:5]:
            if term.lower() in trial_text and term.lower() not in matched:
                score += 15
                matched.append(f"Ontology: {term}")
                confidence = max(confidence, 0.7)
                break

        # Category match (lower value but still relevant)
        if protocol.condition_category:
            if protocol.condition_category.lower() in trial_text:
                score += 10
                matched.append(f"Category: {protocol.condition_category}")

        # Therapeutic area match
        if protocol.therapeutic_area:
            if protocol.therapeutic_area.lower() in trial_ta:
                score += 5
                matched.append(f"TA: {protocol.therapeutic_area}")

        # Incorporate vector score (semantic similarity)
        vector_contribution = vector_score * 30  # Up to 30 points from vector
        score += vector_contribution

        explanation = f"Condition match: {', '.join(matched[:3])}" if matched else "No direct condition match"

        return DimensionScore(
            score=min(score, 100),
            confidence=confidence,
            explanation=explanation,
            matched_terms=matched,
        )

    def _score_intervention(
        self,
        protocol: ExtractedProtocolV2,
        trial: Dict[str, Any]
    ) -> DimensionScore:
        """Score intervention/drug match."""
        score = 0.0
        confidence = 0.3
        matched = []

        trial_interventions = (trial.get("interventions") or "").lower()
        trial_title = (trial.get("title") or "").lower()
        trial_text = f"{trial_interventions} {trial_title}"

        intv = protocol.intervention

        # Drug name match (if known)
        if intv.drug_name and intv.drug_name.lower() != "investigational":
            if intv.drug_name.lower() in trial_text:
                score += 40
                matched.append(f"Drug: {intv.drug_name}")
                confidence = 0.95

        # Drug class match
        if intv.drug_class:
            class_lower = intv.drug_class.lower()
            if class_lower in trial_text:
                score += 35
                matched.append(f"Class: {intv.drug_class}")
                confidence = max(confidence, 0.85)
            else:
                # Check for class keywords
                class_keywords = self._get_class_keywords(intv.drug_class)
                for kw in class_keywords:
                    if kw.lower() in trial_text:
                        score += 25
                        matched.append(f"Class keyword: {kw}")
                        confidence = max(confidence, 0.7)
                        break

        # Similar known drugs match
        for similar_drug in intv.similar_known_drugs[:5]:
            if similar_drug.lower() in trial_text:
                score += 30
                matched.append(f"Similar drug: {similar_drug}")
                confidence = max(confidence, 0.8)
                break

        # Search terms match
        for term in intv.search_terms[:8]:
            if len(term) >= 4 and term.lower() in trial_text:
                score += 10
                matched.append(f"Term: {term}")
                confidence = max(confidence, 0.6)
                if len(matched) >= 3:
                    break

        # Intervention type match
        if intv.intervention_type:
            type_lower = intv.intervention_type.lower()
            if type_lower in trial_interventions:
                score += 5
                matched.append(f"Type: {intv.intervention_type}")

        # Route match
        if intv.route:
            route_lower = intv.route.lower()
            if route_lower in trial_text or self._route_matches(route_lower, trial_text):
                score += 5
                matched.append(f"Route: {intv.route}")

        explanation = f"Intervention match: {', '.join(matched[:3])}" if matched else "No intervention match"

        return DimensionScore(
            score=min(score, 100),
            confidence=confidence,
            explanation=explanation,
            matched_terms=matched,
        )

    def _score_endpoint(
        self,
        protocol: ExtractedProtocolV2,
        trial: Dict[str, Any]
    ) -> DimensionScore:
        """Score endpoint/outcome match."""
        score = 0.0
        confidence = 0.3
        matched = []

        trial_outcomes = (trial.get("primary_outcomes") or "").lower()
        trial_title = (trial.get("title") or "").lower()
        trial_text = f"{trial_outcomes} {trial_title}"

        endp = protocol.endpoints

        # Check endpoint categories
        if endp.has_weight_endpoint:
            weight_terms = ["weight", "body weight", "bmi", "waist circumference", "weight loss", "weight reduction"]
            for term in weight_terms:
                if term in trial_text:
                    score += 30
                    matched.append(f"Weight endpoint: {term}")
                    confidence = max(confidence, 0.8)
                    break

        if endp.has_glycemic_endpoint:
            glycemic_terms = ["hba1c", "a1c", "glycemic", "glucose", "fasting plasma glucose", "fpg"]
            for term in glycemic_terms:
                if term in trial_text:
                    score += 25
                    matched.append(f"Glycemic endpoint: {term}")
                    confidence = max(confidence, 0.8)
                    break

        if endp.has_survival_endpoint:
            survival_terms = ["overall survival", "progression-free survival", "pfs", "os", "mortality", "death"]
            for term in survival_terms:
                if term in trial_text:
                    score += 30
                    matched.append(f"Survival endpoint: {term}")
                    confidence = max(confidence, 0.8)
                    break

        if endp.has_cv_endpoint:
            cv_terms = ["cardiovascular", "mace", "heart failure", "myocardial infarction", "stroke"]
            for term in cv_terms:
                if term in trial_text:
                    score += 25
                    matched.append(f"CV endpoint: {term}")
                    confidence = max(confidence, 0.8)
                    break

        if endp.has_response_rate_endpoint:
            response_terms = ["response rate", "orr", "objective response", "complete response", "partial response"]
            for term in response_terms:
                if term in trial_text:
                    score += 25
                    matched.append(f"Response endpoint: {term}")
                    confidence = max(confidence, 0.8)
                    break

        # Primary endpoint text match
        if endp.primary_endpoint:
            endpoint_words = [w for w in endp.primary_endpoint.lower().split() if len(w) >= 4]
            word_matches = sum(1 for w in endpoint_words if w in trial_text)
            if word_matches >= 2:
                score += 20
                matched.append(f"Endpoint words: {word_matches} matches")
                confidence = max(confidence, 0.6)

        # Endpoint measure type match
        if endp.primary_endpoint_measure:
            measure_lower = endp.primary_endpoint_measure.lower()
            if measure_lower in trial_text:
                score += 10
                matched.append(f"Measure: {endp.primary_endpoint_measure}")

        explanation = f"Endpoint match: {', '.join(matched[:3])}" if matched else "No endpoint match"

        return DimensionScore(
            score=min(score, 100),
            confidence=confidence,
            explanation=explanation,
            matched_terms=matched,
        )

    def _score_population(
        self,
        protocol: ExtractedProtocolV2,
        trial: Dict[str, Any]
    ) -> DimensionScore:
        """Score population/eligibility match."""
        score = 50  # Start at neutral
        confidence = 0.5
        matched = []

        pop = protocol.population
        trial_elig = (trial.get("eligibility_criteria") or "").lower()
        trial_conditions = (trial.get("conditions") or "").lower()

        # Age overlap - ensure integer comparison
        trial_min_age = trial.get("min_age")
        trial_max_age = trial.get("max_age")

        # Convert ages to integers if they're strings (e.g., "18 Years" -> 18)
        def parse_age(age_val):
            if age_val is None:
                return None
            if isinstance(age_val, int):
                return age_val
            if isinstance(age_val, str):
                import re
                match = re.search(r'(\d+)', str(age_val))
                return int(match.group(1)) if match else None
            return None

        trial_min_age = parse_age(trial_min_age)
        trial_max_age = parse_age(trial_max_age)
        protocol_min_age = parse_age(pop.min_age) if pop.min_age else None
        protocol_max_age = parse_age(pop.max_age) if pop.max_age else None

        if protocol_min_age and protocol_max_age and trial_min_age and trial_max_age:
            # Calculate overlap
            overlap_min = max(protocol_min_age, trial_min_age)
            overlap_max = min(protocol_max_age, trial_max_age)

            if overlap_min <= overlap_max:
                # There is overlap
                protocol_range = protocol_max_age - protocol_min_age
                overlap_range = overlap_max - overlap_min
                overlap_pct = overlap_range / protocol_range if protocol_range > 0 else 0

                if overlap_pct >= 0.8:
                    score += 15
                    matched.append(f"Age overlap: {overlap_pct:.0%}")
                    confidence = max(confidence, 0.8)
                elif overlap_pct >= 0.5:
                    score += 10
                    matched.append(f"Age partial overlap: {overlap_pct:.0%}")
            else:
                # No age overlap - penalize
                score -= 20
                matched.append("Age: no overlap")

        # BMI criteria match (for obesity trials)
        if pop.bmi_min:
            if "bmi" in trial_elig:
                score += 10
                matched.append("BMI criteria present")
                confidence = max(confidence, 0.7)

        # Required conditions alignment
        for req_cond in pop.required_conditions[:3]:
            if req_cond.lower() in trial_conditions or req_cond.lower() in trial_elig:
                score += 10
                matched.append(f"Requires: {req_cond}")

        # Sex alignment
        if pop.sex != "all":
            trial_sex = (trial.get("sex") or "all").lower()
            if trial_sex == "all" or trial_sex == pop.sex.lower():
                score += 5
                matched.append(f"Sex: compatible")
            else:
                score -= 15
                matched.append(f"Sex: incompatible")

        explanation = f"Population match: {', '.join(matched[:3])}" if matched else "Population baseline match"

        return DimensionScore(
            score=min(max(score, 0), 100),
            confidence=confidence,
            explanation=explanation,
            matched_terms=matched,
        )

    def _score_design(
        self,
        protocol: ExtractedProtocolV2,
        trial: Dict[str, Any]
    ) -> DimensionScore:
        """Score study design match."""
        score = 50  # Start at neutral
        confidence = 0.5
        matched = []

        des = protocol.design
        trial_phase = trial.get("phase") or ""
        trial_status = trial.get("status") or ""
        trial_enrollment = trial.get("enrollment") or 0

        # Phase match
        if des.phase and trial_phase:
            if des.phase == trial_phase:
                score += 25
                matched.append(f"Phase: exact match ({des.phase})")
                confidence = 0.9
            elif self._phases_adjacent(des.phase, trial_phase):
                score += 15
                matched.append(f"Phase: adjacent ({trial_phase})")
                confidence = 0.7
            else:
                score -= 10
                matched.append(f"Phase: different ({trial_phase})")

        # Enrollment similarity
        target_enrollment = des.target_enrollment or 0
        if target_enrollment > 0 and trial_enrollment and trial_enrollment > 0:
            ratio = min(target_enrollment, trial_enrollment) / max(target_enrollment, trial_enrollment)
            if ratio >= 0.5:
                score += 10
                matched.append(f"Enrollment similar ({trial_enrollment})")
            elif ratio >= 0.2:
                score += 5
                matched.append(f"Enrollment somewhat similar ({trial_enrollment})")

        # Status preference (completed trials are more useful for benchmarking)
        if trial_status == "COMPLETED":
            score += 15
            matched.append("Status: completed")
            confidence = max(confidence, 0.8)
        elif trial_status in ["TERMINATED", "WITHDRAWN"]:
            score += 5
            matched.append("Status: terminated (has learnings)")

        # Control type match
        if des.control_type:
            control_lower = des.control_type.lower()
            if "placebo" in control_lower:
                trial_title = (trial.get("title") or "").lower()
                trial_elig = (trial.get("eligibility_criteria") or "").lower()
                if "placebo" in trial_title or "placebo" in trial_elig:
                    score += 5
                    matched.append("Control: placebo-controlled")

        explanation = f"Design match: {', '.join(matched[:3])}" if matched else "Design baseline match"

        return DimensionScore(
            score=min(max(score, 0), 100),
            confidence=confidence,
            explanation=explanation,
            matched_terms=matched,
        )

    def _check_exclusion_conflicts(
        self,
        protocol: ExtractedProtocolV2,
        trial: Dict[str, Any]
    ) -> List[str]:
        """
        Check for hard conflicts between protocol exclusions and trial requirements.

        Returns list of conflict reasons, empty if no conflicts.
        """
        conflicts = []

        trial_conditions = (trial.get("conditions") or "").lower()
        trial_elig = (trial.get("eligibility_criteria") or "").lower()
        trial_title = (trial.get("title") or "").lower()
        trial_text = f"{trial_conditions} {trial_title}"

        # Check each excluded condition
        for excluded in protocol.population.excluded_conditions:
            excluded_lower = excluded.lower()

            # Check if trial REQUIRES this condition
            # Pattern: condition appears in trial conditions or as requirement
            if excluded_lower in trial_conditions:
                # Check if it's the main condition (not just mentioned)
                # Trial conditions field usually lists primary conditions
                conditions_list = [c.strip() for c in trial_conditions.split(",")]
                for cond in conditions_list:
                    if excluded_lower in cond or cond in excluded_lower:
                        conflicts.append(f"Trial requires '{excluded}' but protocol excludes it")
                        break

            # Check for diabetes specifically (common exclusion)
            if "diabetes" in excluded_lower:
                diabetes_terms = ["type 2 diabetes", "type 1 diabetes", "t2dm", "t1dm", "diabetes mellitus"]
                for term in diabetes_terms:
                    if term in trial_conditions and "obesity" not in trial_conditions:
                        # It's primarily a diabetes trial
                        conflicts.append(f"Trial is for diabetes patients but protocol excludes diabetes")
                        break

        return conflicts

    def _get_class_keywords(self, drug_class: str) -> List[str]:
        """Get keywords associated with a drug class."""
        class_keywords = {
            "glp-1 receptor agonist": ["glp-1", "glucagon-like peptide", "incretin", "semaglutide",
                                        "liraglutide", "tirzepatide", "dulaglutide", "exenatide"],
            "sglt2 inhibitor": ["sglt2", "sodium glucose", "gliflozin", "empagliflozin",
                                "dapagliflozin", "canagliflozin"],
            "pd-1 inhibitor": ["pd-1", "pd-l1", "checkpoint", "pembrolizumab", "nivolumab",
                              "atezolizumab", "immunotherapy"],
            "tnf inhibitor": ["tnf", "anti-tnf", "adalimumab", "infliximab", "etanercept"],
            "jak inhibitor": ["jak", "janus kinase", "tofacitinib", "baricitinib", "upadacitinib"],
        }

        class_lower = drug_class.lower()
        for key, keywords in class_keywords.items():
            if key in class_lower or class_lower in key:
                return keywords

        # Return words from the class name itself
        return [w for w in drug_class.lower().split() if len(w) >= 4]

    def _route_matches(self, protocol_route: str, trial_text: str) -> bool:
        """Check if routes match with common variations."""
        route_synonyms = {
            "subcutaneous": ["sc", "subq", "s.c.", "subcutaneously"],
            "intravenous": ["iv", "i.v.", "intravenously", "infusion"],
            "oral": ["orally", "po", "p.o.", "tablet", "capsule"],
            "intramuscular": ["im", "i.m.", "intramuscularly"],
        }

        for standard, synonyms in route_synonyms.items():
            if protocol_route in [standard] + synonyms:
                if any(s in trial_text for s in [standard] + synonyms):
                    return True
        return False

    def _phases_adjacent(self, phase1: str, phase2: str) -> bool:
        """Check if two phases are adjacent."""
        phase_order = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
        try:
            idx1 = phase_order.index(phase1)
            idx2 = phase_order.index(phase2)
            return abs(idx1 - idx2) == 1
        except ValueError:
            return False


# Singleton
_scorer = None

def get_scorer() -> TrialScorer:
    global _scorer
    if _scorer is None:
        _scorer = TrialScorer()
    return _scorer
