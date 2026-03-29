"""
Structured Eligibility Criteria Parser

Parses eligibility criteria text into structured components
for precise numerical comparison.
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class AgeRange:
    """Age eligibility criteria."""
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    unit: str = "years"

    def overlaps(self, other: 'AgeRange') -> float:
        """Calculate overlap percentage with another age range."""
        if not self.min_age or not self.max_age or not other.min_age or not other.max_age:
            return 0.5  # Unknown, assume partial overlap

        # Calculate overlap
        overlap_min = max(self.min_age, other.min_age)
        overlap_max = min(self.max_age, other.max_age)

        if overlap_min > overlap_max:
            return 0.0  # No overlap

        self_range = self.max_age - self.min_age
        other_range = other.max_age - other.min_age
        overlap_range = overlap_max - overlap_min

        if self_range == 0 or other_range == 0:
            return 0.5

        # Return average of overlap percentages
        return (overlap_range / self_range + overlap_range / other_range) / 2


@dataclass
class LabValue:
    """A lab value criterion."""
    name: str
    operator: str  # >, <, >=, <=, =, range
    value: float
    value_max: Optional[float] = None  # For ranges
    unit: str = ""

    def compatible(self, other: 'LabValue') -> float:
        """Check if lab criteria are compatible (0-1 score)."""
        if self.name.lower() != other.name.lower():
            return 0.0

        # Same lab, check value overlap
        self_min = self.value if self.operator in ['>', '>=', 'range'] else float('-inf')
        self_max = self.value_max if self.operator == 'range' else (self.value if self.operator in ['<', '<='] else float('inf'))

        other_min = other.value if other.operator in ['>', '>=', 'range'] else float('-inf')
        other_max = other.value_max if other.operator == 'range' else (other.value if other.operator in ['<', '<='] else float('inf'))

        # Check overlap
        overlap_min = max(self_min, other_min)
        overlap_max = min(self_max, other_max)

        if overlap_min > overlap_max:
            return 0.0  # No overlap

        return 0.8  # Good overlap


@dataclass
class ParsedEligibility:
    """Structured eligibility criteria."""
    age_range: Optional[AgeRange] = None
    sex: str = "All"  # All, Male, Female

    # Lab values
    lab_values: List[LabValue] = field(default_factory=list)

    # Conditions (required or excluded)
    required_conditions: List[str] = field(default_factory=list)
    excluded_conditions: List[str] = field(default_factory=list)

    # Prior treatments
    required_prior_treatments: List[str] = field(default_factory=list)
    excluded_prior_treatments: List[str] = field(default_factory=list)

    # Other criteria
    performance_status: Optional[str] = None  # ECOG, Karnofsky
    pregnant_excluded: bool = True
    healthy_volunteers: bool = False

    # Raw text for reference
    raw_inclusion: str = ""
    raw_exclusion: str = ""

    def calculate_similarity(self, other: 'ParsedEligibility') -> float:
        """Calculate similarity score with another eligibility criteria (0-100)."""
        scores = []
        weights = []

        # Age overlap (high weight)
        if self.age_range and other.age_range:
            age_score = self.age_range.overlaps(other.age_range) * 100
            scores.append(age_score)
            weights.append(3)

        # Sex match
        if self.sex == other.sex or self.sex == "All" or other.sex == "All":
            scores.append(100)
        else:
            scores.append(0)
        weights.append(2)

        # Lab values overlap
        if self.lab_values and other.lab_values:
            lab_matches = 0
            for lab1 in self.lab_values:
                for lab2 in other.lab_values:
                    if lab1.name.lower() == lab2.name.lower():
                        lab_matches += lab1.compatible(lab2)
                        break

            lab_score = (lab_matches / max(len(self.lab_values), 1)) * 100
            scores.append(lab_score)
            weights.append(2)

        # Required conditions overlap
        if self.required_conditions and other.required_conditions:
            self_conds = set(c.lower() for c in self.required_conditions)
            other_conds = set(c.lower() for c in other.required_conditions)

            if self_conds and other_conds:
                overlap = len(self_conds.intersection(other_conds))
                union = len(self_conds.union(other_conds))
                scores.append((overlap / union) * 100 if union > 0 else 50)
                weights.append(3)

        # Excluded conditions overlap
        if self.excluded_conditions and other.excluded_conditions:
            self_excl = set(c.lower() for c in self.excluded_conditions)
            other_excl = set(c.lower() for c in other.excluded_conditions)

            if self_excl and other_excl:
                overlap = len(self_excl.intersection(other_excl))
                union = len(self_excl.union(other_excl))
                scores.append((overlap / union) * 100 if union > 0 else 50)
                weights.append(1)

        if not scores:
            return 50.0  # No data to compare

        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return weighted_sum / total_weight


class EligibilityParser:
    """
    Parses eligibility criteria text into structured format.
    """

    # Age patterns
    AGE_PATTERNS = [
        r'(?:age[s]?|aged?)[\s:]*(\d+)\s*(?:to|-|–)\s*(\d+)\s*(?:years?|yrs?)?',
        r'(\d+)\s*(?:to|-|–)\s*(\d+)\s*years?\s*(?:of\s*age|old)?',
        r'(?:≥|>=|at least|minimum)\s*(\d+)\s*years?',
        r'(?:≤|<=|up to|maximum)\s*(\d+)\s*years?',
        r'(?:over|older than|>)\s*(\d+)\s*years?',
        r'(?:under|younger than|<)\s*(\d+)\s*years?',
        r'adults?\s*(?:\()?(\d+)(?:\s*(?:to|-|–)\s*(\d+))?\s*years?',
    ]

    # Lab value patterns
    LAB_PATTERNS = [
        (r'HbA1c\s*(?:≥|>=|>|of)?\s*([\d.]+)\s*(?:to|-|–)?\s*([\d.]+)?(?:\s*%)?', 'HbA1c', '%'),
        (r'eGFR\s*(?:≥|>=|>)?\s*([\d.]+)', 'eGFR', 'mL/min/1.73m²'),
        (r'(?:BMI|body mass index)\s*(?:≥|>=|>|of)?\s*([\d.]+)\s*(?:to|-|–)?\s*([\d.]+)?', 'BMI', 'kg/m²'),
        (r'LDL(?:-C)?\s*(?:≥|>=|>)?\s*([\d.]+)', 'LDL', 'mg/dL'),
        (r'(?:creatinine|Cr)\s*(?:≤|<=|<)?\s*([\d.]+)', 'creatinine', 'mg/dL'),
        (r'(?:hemoglobin|Hgb|Hb)\s*(?:≥|>=|>)?\s*([\d.]+)', 'hemoglobin', 'g/dL'),
        (r'(?:platelet|plt)\s*(?:count)?\s*(?:≥|>=|>)?\s*([\d,]+)', 'platelets', '×10⁹/L'),
        (r'(?:AST|ALT)\s*(?:≤|<=|<)?\s*([\d.]+)\s*(?:×|x)?\s*(?:ULN)?', 'AST/ALT', 'xULN'),
        (r'(?:bilirubin)\s*(?:≤|<=|<)?\s*([\d.]+)', 'bilirubin', 'mg/dL'),
        (r'(?:ejection fraction|EF|LVEF)\s*(?:≥|>=|>)?\s*([\d.]+)(?:\s*%)?', 'LVEF', '%'),
        (r'(?:blood pressure|BP)\s*(?:≤|<=|<)?\s*(\d+)/(\d+)', 'blood_pressure', 'mmHg'),
    ]

    # Sex patterns
    SEX_PATTERNS = [
        (r'\b(?:male|men)\s*(?:only|and female|or female)?\b', 'Male'),
        (r'\b(?:female|women)\s*(?:only|and male|or male)?\b', 'Female'),
        (r'\b(?:both sexes|male and female|female and male|all genders?)\b', 'All'),
    ]

    # Performance status patterns
    PS_PATTERNS = [
        r'ECOG\s*(?:performance status|PS)?\s*(?:≤|<=|of)?\s*(\d)',
        r'Karnofsky\s*(?:performance status|PS|score)?\s*(?:≥|>=)?\s*(\d+)',
    ]

    def parse(self, eligibility_text: str, is_inclusion: bool = None) -> ParsedEligibility:
        """
        Parse eligibility criteria text into structured format.

        Args:
            eligibility_text: Raw eligibility criteria text
            is_inclusion: If True, treat as inclusion criteria; if False, exclusion

        Returns:
            ParsedEligibility object
        """
        if not eligibility_text:
            return ParsedEligibility()

        text = eligibility_text.lower()
        result = ParsedEligibility()
        result.raw_inclusion = eligibility_text if is_inclusion else ""
        result.raw_exclusion = eligibility_text if is_inclusion is False else ""

        # Parse age
        result.age_range = self._parse_age(text)

        # Parse sex
        result.sex = self._parse_sex(text)

        # Parse lab values
        result.lab_values = self._parse_lab_values(text)

        # Parse conditions
        result.required_conditions, result.excluded_conditions = self._parse_conditions(text)

        # Parse performance status
        result.performance_status = self._parse_performance_status(text)

        # Check pregnancy exclusion
        result.pregnant_excluded = bool(re.search(r'pregnan|breast.?feed|lactat|child.?bearing', text))

        # Check healthy volunteers
        result.healthy_volunteers = bool(re.search(r'healthy\s*volunteer', text))

        return result

    def _parse_age(self, text: str) -> Optional[AgeRange]:
        """Extract age range from text."""
        for pattern in self.AGE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2 and groups[1]:
                    return AgeRange(min_age=int(groups[0]), max_age=int(groups[1]))
                elif len(groups) >= 1:
                    # Single age - determine if min or max based on context
                    age = int(groups[0])
                    if '≥' in text[:match.start()+20] or '>=' in text[:match.start()+20] or 'at least' in text[:match.start()+20]:
                        return AgeRange(min_age=age, max_age=100)
                    elif '≤' in text[:match.start()+20] or '<=' in text[:match.start()+20] or 'up to' in text[:match.start()+20]:
                        return AgeRange(min_age=18, max_age=age)
                    else:
                        return AgeRange(min_age=age, max_age=age+57)  # Assume adult range

        return None

    def _parse_sex(self, text: str) -> str:
        """Extract sex criteria from text."""
        for pattern, sex in self.SEX_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return sex
        return "All"

    def _parse_lab_values(self, text: str) -> List[LabValue]:
        """Extract lab values from text."""
        lab_values = []

        for pattern, name, unit in self.LAB_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                try:
                    value = float(groups[0].replace(',', ''))
                    value_max = float(groups[1].replace(',', '')) if len(groups) > 1 and groups[1] else None

                    # Determine operator
                    context = text[max(0, match.start()-10):match.start()]
                    if '≥' in context or '>=' in context:
                        operator = '>='
                    elif '≤' in context or '<=' in context:
                        operator = '<='
                    elif '>' in context:
                        operator = '>'
                    elif '<' in context:
                        operator = '<'
                    elif value_max:
                        operator = 'range'
                    else:
                        operator = '='

                    lab_values.append(LabValue(
                        name=name,
                        operator=operator,
                        value=value,
                        value_max=value_max,
                        unit=unit
                    ))
                except (ValueError, IndexError):
                    pass

        return lab_values

    def _parse_conditions(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract required and excluded conditions."""
        required = []
        excluded = []

        # Common condition keywords
        condition_keywords = [
            'diabetes', 'cancer', 'heart failure', 'hypertension', 'obesity',
            'asthma', 'copd', 'depression', 'anxiety', 'arthritis', 'hepatitis',
            'hiv', 'stroke', 'alzheimer', 'parkinson', 'epilepsy', 'migraine'
        ]

        # Check for "history of X" or "diagnosis of X" (required)
        for cond in condition_keywords:
            if re.search(rf'(?:diagnosis|history|confirmed)\s+(?:of\s+)?{cond}', text):
                required.append(cond)

        # Check for "no history of X" or "without X" (excluded)
        for cond in condition_keywords:
            if re.search(rf'(?:no|without|exclude|absence of)\s+(?:history of\s+)?{cond}', text):
                excluded.append(cond)

        return required, excluded

    def _parse_performance_status(self, text: str) -> Optional[str]:
        """Extract performance status criteria."""
        for pattern in self.PS_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def compare_eligibility(self, criteria1: ParsedEligibility, criteria2: ParsedEligibility) -> Dict[str, Any]:
        """
        Compare two eligibility criteria and return detailed comparison.

        Returns:
            Dict with overall similarity and breakdown
        """
        overall_similarity = criteria1.calculate_similarity(criteria2)

        breakdown = {
            "overall_similarity": overall_similarity,
            "components": {}
        }

        # Age comparison
        if criteria1.age_range and criteria2.age_range:
            age_overlap = criteria1.age_range.overlaps(criteria2.age_range)
            breakdown["components"]["age"] = {
                "score": age_overlap * 100,
                "criteria1": f"{criteria1.age_range.min_age}-{criteria1.age_range.max_age} years",
                "criteria2": f"{criteria2.age_range.min_age}-{criteria2.age_range.max_age} years",
            }

        # Sex comparison
        sex_match = criteria1.sex == criteria2.sex or "All" in [criteria1.sex, criteria2.sex]
        breakdown["components"]["sex"] = {
            "score": 100 if sex_match else 0,
            "criteria1": criteria1.sex,
            "criteria2": criteria2.sex,
        }

        # Lab values comparison
        if criteria1.lab_values or criteria2.lab_values:
            lab_scores = []
            for lab1 in criteria1.lab_values:
                for lab2 in criteria2.lab_values:
                    if lab1.name.lower() == lab2.name.lower():
                        lab_scores.append(lab1.compatible(lab2) * 100)

            breakdown["components"]["lab_values"] = {
                "score": sum(lab_scores) / len(lab_scores) if lab_scores else 50,
                "matched_count": len(lab_scores),
            }

        return breakdown


# Singleton parser
_parser = None

def get_parser() -> EligibilityParser:
    """Get singleton parser instance."""
    global _parser
    if _parser is None:
        _parser = EligibilityParser()
    return _parser
