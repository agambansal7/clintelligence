"""
Eligibility Matcher Module

Uses Claude AI to evaluate patient eligibility against trial criteria.
Provides detailed matching with explanations.
"""

import os
import json
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import anthropic

logger = logging.getLogger(__name__)


@dataclass
class CriterionMatch:
    """Match result for a single eligibility criterion."""
    criterion: str
    status: str  # 'met', 'not_met', 'unknown', 'needs_verification'
    explanation: str
    patient_value: str = ''


@dataclass
class TrialMatch:
    """Complete match result for a trial."""
    nct_id: str
    title: str
    phase: str
    status: str
    match_score: float  # 0-100
    match_level: str  # 'excellent', 'good', 'possible', 'unlikely'
    criteria_met: List[CriterionMatch]
    criteria_not_met: List[CriterionMatch]
    criteria_unknown: List[CriterionMatch]
    summary: str
    nearest_site: Optional[Dict[str, Any]] = None
    distance_miles: Optional[float] = None


class EligibilityMatcher:
    """
    Matches patient profiles to clinical trial eligibility criteria.

    Uses Claude AI for intelligent criterion-by-criterion evaluation.
    """

    def __init__(self):
        """Initialize with Anthropic client."""
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = "claude-sonnet-4-20250514"

    def match_patient_to_trials(
        self,
        patient_answers: Dict[str, Any],
        trials: List[Any],
        patient_location: Tuple[float, float] = None,
        max_trials: int = 20
    ) -> List[TrialMatch]:
        """
        Match a patient to multiple trials.

        Args:
            patient_answers: Dictionary of patient's screening answers
            trials: List of candidate trials to evaluate
            patient_location: Optional (lat, lon) tuple for distance calculation
            max_trials: Maximum number of trials to return

        Returns:
            List of TrialMatch results, sorted by match score
        """

        matches = []

        for trial in trials[:max_trials * 2]:  # Process extra to account for filtering
            match = self._evaluate_trial(patient_answers, trial, patient_location)
            if match:
                matches.append(match)

        # Sort by match score descending
        matches.sort(key=lambda x: x.match_score, reverse=True)

        return matches[:max_trials]

    def _evaluate_trial(
        self,
        patient_answers: Dict[str, Any],
        trial: Any,
        patient_location: Tuple[float, float] = None
    ) -> Optional[TrialMatch]:
        """Evaluate a single trial for a patient."""

        eligibility_criteria = getattr(trial, 'eligibility_criteria', '') or ''

        if not eligibility_criteria or len(eligibility_criteria) < 50:
            # No meaningful criteria to evaluate
            return self._create_basic_match(trial, patient_answers, patient_location)

        # Create patient profile summary
        patient_summary = self._format_patient_summary(patient_answers)

        prompt = f"""You are evaluating whether a patient may be eligible for a clinical trial.

PATIENT PROFILE:
{patient_summary}

CLINICAL TRIAL:
NCT ID: {trial.nct_id}
Title: {trial.title}
Phase: {trial.phase}
Condition: {trial.condition}

ELIGIBILITY CRITERIA:
{eligibility_criteria[:4000]}

TASK:
Evaluate each major eligibility criterion against the patient's profile.
For each criterion, determine if it is:
- "met": Patient clearly meets this criterion
- "not_met": Patient clearly does NOT meet this criterion
- "unknown": Not enough information to determine
- "needs_verification": Likely met but should be verified with doctor

Return your evaluation as JSON with this structure:
{{
  "overall_match_score": <0-100>,
  "match_level": "<excellent|good|possible|unlikely>",
  "summary": "<1-2 sentence summary of match>",
  "criteria_evaluations": [
    {{
      "criterion": "<the criterion being evaluated>",
      "status": "<met|not_met|unknown|needs_verification>",
      "explanation": "<patient-friendly explanation>",
      "patient_value": "<relevant value from patient profile if applicable>"
    }}
  ]
}}

SCORING GUIDELINES:
- 90-100 (excellent): Meets all known criteria, no exclusions
- 70-89 (good): Meets most criteria, minor unknowns
- 50-69 (possible): Several unknowns but no clear exclusions
- 0-49 (unlikely): Doesn't meet key criteria or has exclusions

Be conservative - patient safety is paramount. If any exclusion criterion is met, score should be low.

Return ONLY the JSON object."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Parse JSON response
            if response_text.startswith('{'):
                result = json.loads(response_text)
            else:
                import re
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    logger.warning(f"Could not parse evaluation for {trial.nct_id}")
                    return self._create_basic_match(trial, patient_answers, patient_location)

            # Convert to TrialMatch
            criteria_met = []
            criteria_not_met = []
            criteria_unknown = []

            for eval_item in result.get('criteria_evaluations', []):
                criterion_match = CriterionMatch(
                    criterion=eval_item.get('criterion', ''),
                    status=eval_item.get('status', 'unknown'),
                    explanation=eval_item.get('explanation', ''),
                    patient_value=eval_item.get('patient_value', '')
                )

                if criterion_match.status == 'met':
                    criteria_met.append(criterion_match)
                elif criterion_match.status == 'not_met':
                    criteria_not_met.append(criterion_match)
                else:
                    criteria_unknown.append(criterion_match)

            # Calculate nearest site
            nearest_site, distance = self._find_nearest_site(trial, patient_location)

            return TrialMatch(
                nct_id=trial.nct_id,
                title=trial.title,
                phase=trial.phase or 'Not specified',
                status=trial.status,
                match_score=result.get('overall_match_score', 50),
                match_level=result.get('match_level', 'possible'),
                criteria_met=criteria_met,
                criteria_not_met=criteria_not_met,
                criteria_unknown=criteria_unknown,
                summary=result.get('summary', ''),
                nearest_site=nearest_site,
                distance_miles=distance
            )

        except Exception as e:
            logger.error(f"Error evaluating trial {trial.nct_id}: {e}")
            return self._create_basic_match(trial, patient_answers, patient_location)

    def _format_patient_summary(self, answers: Dict[str, Any]) -> str:
        """Format patient answers into a readable summary."""

        lines = []

        # Standard fields
        if 'age' in answers:
            lines.append(f"Age: {answers['age']}")
        if 'sex' in answers:
            lines.append(f"Sex: {answers['sex']}")
        if 'location' in answers:
            lines.append(f"Location: {answers['location']}")

        # All other answers
        for key, value in answers.items():
            if key not in ['age', 'sex', 'location', 'travel_distance']:
                if isinstance(value, list):
                    lines.append(f"{key.replace('_', ' ').title()}: {', '.join(value)}")
                elif isinstance(value, bool):
                    lines.append(f"{key.replace('_', ' ').title()}: {'Yes' if value else 'No'}")
                else:
                    lines.append(f"{key.replace('_', ' ').title()}: {value}")

        return '\n'.join(lines)

    def _create_basic_match(
        self,
        trial: Any,
        patient_answers: Dict[str, Any],
        patient_location: Tuple[float, float] = None
    ) -> TrialMatch:
        """Create a basic match when detailed evaluation isn't possible."""

        # Check basic criteria
        criteria_met = []
        criteria_not_met = []
        criteria_unknown = []

        # Age check
        patient_age = patient_answers.get('age')
        if patient_age:
            min_age = getattr(trial, 'min_age', None)
            max_age = getattr(trial, 'max_age', None)

            if min_age and patient_age < min_age:
                criteria_not_met.append(CriterionMatch(
                    criterion=f"Minimum age {min_age}",
                    status="not_met",
                    explanation=f"Trial requires age {min_age}+, you are {patient_age}",
                    patient_value=str(patient_age)
                ))
            elif max_age and patient_age > max_age:
                criteria_not_met.append(CriterionMatch(
                    criterion=f"Maximum age {max_age}",
                    status="not_met",
                    explanation=f"Trial requires age up to {max_age}, you are {patient_age}",
                    patient_value=str(patient_age)
                ))
            else:
                criteria_met.append(CriterionMatch(
                    criterion="Age requirement",
                    status="met",
                    explanation="Your age is within the trial's range",
                    patient_value=str(patient_age)
                ))

        # Sex check
        patient_sex = patient_answers.get('sex')
        trial_sex = getattr(trial, 'sex', 'All')
        if patient_sex and trial_sex and trial_sex != 'All':
            if patient_sex.lower() != trial_sex.lower():
                criteria_not_met.append(CriterionMatch(
                    criterion=f"Sex: {trial_sex}",
                    status="not_met",
                    explanation=f"Trial is only for {trial_sex} participants",
                    patient_value=patient_sex
                ))

        # Calculate score
        if criteria_not_met:
            score = 20
            level = 'unlikely'
        elif criteria_met:
            score = 70
            level = 'possible'
        else:
            score = 50
            level = 'possible'

        # Find nearest site
        nearest_site, distance = self._find_nearest_site(trial, patient_location)

        return TrialMatch(
            nct_id=trial.nct_id,
            title=trial.title,
            phase=trial.phase or 'Not specified',
            status=trial.status,
            match_score=score,
            match_level=level,
            criteria_met=criteria_met,
            criteria_not_met=criteria_not_met,
            criteria_unknown=[CriterionMatch(
                criterion="Detailed eligibility criteria",
                status="unknown",
                explanation="Full eligibility criteria should be reviewed with your doctor"
            )],
            summary="Basic eligibility check completed. Please review full criteria with your healthcare provider.",
            nearest_site=nearest_site,
            distance_miles=distance
        )

    def _find_nearest_site(
        self,
        trial: Any,
        patient_location: Tuple[float, float] = None
    ) -> Tuple[Optional[Dict], Optional[float]]:
        """Find the nearest trial site to the patient."""

        locations = getattr(trial, 'locations', []) or []

        if not locations:
            return None, None

        if not patient_location:
            # Return first location if no patient location
            return locations[0] if locations else None, None

        patient_lat, patient_lon = patient_location
        nearest = None
        min_distance = float('inf')

        for loc in locations:
            site_lat = loc.get('latitude')
            site_lon = loc.get('longitude')

            if site_lat and site_lon:
                distance = self._haversine_distance(
                    patient_lat, patient_lon,
                    float(site_lat), float(site_lon)
                )
                if distance < min_distance:
                    min_distance = distance
                    nearest = loc

        if nearest and min_distance != float('inf'):
            return nearest, round(min_distance, 1)

        return locations[0] if locations else None, None

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in miles."""

        R = 3959  # Earth's radius in miles

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c
