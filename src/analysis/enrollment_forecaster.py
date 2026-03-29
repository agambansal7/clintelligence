"""
Enhanced Enrollment Forecaster

Provides realistic enrollment projections using:
1. S-curve enrollment model (slow start, ramp-up, plateau)
2. Site activation modeling (staggered site initiation)
3. Risk-adjusted projections based on protocol complexity
4. Screen failure and dropout modeling
5. Milestone tracking and scenario comparison
6. Historical benchmarking from similar trials
"""

import os
import json
import logging
import math
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EnrollmentMilestone:
    """A milestone checkpoint for enrollment tracking."""
    month: int
    target_patients: int
    cumulative_percent: float
    status: str = "pending"  # pending, on_track, at_risk, behind
    notes: str = ""


@dataclass
class SiteActivationPlan:
    """Site activation schedule."""
    total_sites: int
    activation_months: int  # Time to activate all sites
    sites_per_month: List[int] = field(default_factory=list)
    cumulative_active: List[int] = field(default_factory=list)


@dataclass
class EnrollmentScenario:
    """A single enrollment projection scenario."""
    name: str
    description: str
    total_months: float
    monthly_enrollment: List[float]
    cumulative_enrollment: List[float]
    milestones: List[EnrollmentMilestone]
    monthly_rate: float  # Average patients/month at steady state
    site_rate: float  # Patients/site/month at steady state
    confidence_level: str  # optimistic, expected, pessimistic, conservative


@dataclass
class RiskFactors:
    """Risk factors affecting enrollment."""
    eligibility_complexity: float  # 0-1, higher = more complex
    competition_level: float  # 0-1, higher = more competition
    condition_prevalence: str  # rare, uncommon, common
    site_experience: float  # 0-1, higher = more experienced
    geographic_spread: float  # 0-1, higher = more countries

    def calculate_risk_multiplier(self) -> float:
        """Calculate overall risk multiplier (lower = slower enrollment)."""
        base = 1.0

        # Eligibility complexity reduces enrollment
        base -= self.eligibility_complexity * 0.3

        # Competition reduces enrollment
        base -= self.competition_level * 0.2

        # Rare conditions are harder to enroll
        if self.condition_prevalence == "rare":
            base -= 0.3
        elif self.condition_prevalence == "uncommon":
            base -= 0.15

        # Experienced sites enroll faster
        base += self.site_experience * 0.15

        # Geographic spread can help or hurt
        if self.geographic_spread > 0.5:
            base += 0.1  # More countries = larger patient pool

        return max(0.3, min(1.5, base))


@dataclass
class ScreeningMetrics:
    """Screen failure and dropout modeling."""
    screen_failure_rate: float = 0.25  # % who fail screening
    early_dropout_rate: float = 0.10  # % who drop out early
    randomization_rate: float = 0.65  # % of screened who are randomized

    def patients_to_screen(self, target_randomized: int) -> int:
        """Calculate how many patients need to be screened."""
        return int(target_randomized / self.randomization_rate)


@dataclass
class EnrollmentForecast:
    """Complete enrollment forecast with multiple scenarios."""
    target_enrollment: int
    num_sites: int
    condition: str
    phase: str

    # Historical benchmarks
    historical_rate_median: float
    historical_rate_p25: float
    historical_rate_p75: float
    historical_duration_median: float
    similar_trials_count: int

    # Scenarios
    optimistic: EnrollmentScenario
    expected: EnrollmentScenario
    pessimistic: EnrollmentScenario
    risk_adjusted: EnrollmentScenario

    # Site plan
    site_activation: SiteActivationPlan

    # Risk factors
    risk_factors: RiskFactors
    risk_multiplier: float

    # Screening
    screening: ScreeningMetrics
    patients_to_screen: int

    # Key insights
    key_insights: List[str]
    recommendations: List[str]


class EnrollmentForecaster:
    """
    Advanced enrollment forecasting with S-curve modeling.
    """

    # S-curve parameters for different phases
    PHASE_PARAMS = {
        "PHASE1": {"ramp_months": 2, "plateau_factor": 0.9, "typical_sites": 5},
        "PHASE2": {"ramp_months": 3, "plateau_factor": 0.85, "typical_sites": 20},
        "PHASE3": {"ramp_months": 4, "plateau_factor": 0.8, "typical_sites": 80},
        "PHASE4": {"ramp_months": 3, "plateau_factor": 0.85, "typical_sites": 50},
    }

    # Condition prevalence estimates
    PREVALENCE_MAP = {
        "rare": ["orphan", "rare disease", "ultra-rare"],
        "uncommon": ["als", "huntington", "cystic fibrosis", "multiple sclerosis"],
        "common": ["diabetes", "hypertension", "heart failure", "copd", "asthma",
                   "depression", "anxiety", "obesity", "cancer", "breast cancer",
                   "lung cancer", "colorectal"]
    }

    def __init__(self, db_manager, api_key: Optional[str] = None):
        """Initialize forecaster."""
        self.db = db_manager
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _get_historical_rates(
        self,
        condition: str,
        phase: str,
        limit: int = 500
    ) -> Tuple[List[float], List[Dict]]:
        """Get historical enrollment rates from similar trials."""
        from sqlalchemy import text

        query = text("""
            SELECT
                nct_id, enrollment, num_sites, sponsor,
                start_date, completion_date,
                julianday(completion_date) - julianday(start_date) as duration_days
            FROM trials
            WHERE (LOWER(conditions) LIKE :condition OR LOWER(therapeutic_area) LIKE :condition)
            AND phase = :phase
            AND status = 'COMPLETED'
            AND enrollment > 0 AND num_sites > 0
            AND start_date IS NOT NULL AND completion_date IS NOT NULL
            AND julianday(completion_date) > julianday(start_date)
            ORDER BY completion_date DESC
            LIMIT :limit
        """)

        results = self.db.execute_raw(query.text, {
            "condition": f"%{condition.lower()}%",
            "phase": phase,
            "limit": limit
        })

        rates = []
        trial_data = []

        for r in results:
            nct_id, enrollment, sites, sponsor, start, completion, duration_days = r
            if duration_days and duration_days > 30:
                duration_months = duration_days / 30.44
                rate = enrollment / sites / duration_months
                # Filter outliers
                if 0.05 < rate < 30:
                    rates.append(rate)
                    trial_data.append({
                        "nct_id": nct_id,
                        "enrollment": enrollment,
                        "sites": sites,
                        "duration_months": duration_months,
                        "rate": rate,
                        "sponsor": sponsor
                    })

        return rates, trial_data

    def _estimate_prevalence(self, condition: str) -> str:
        """Estimate condition prevalence category."""
        condition_lower = condition.lower()

        for category, terms in self.PREVALENCE_MAP.items():
            for term in terms:
                if term in condition_lower:
                    return category

        # Default to uncommon
        return "uncommon"

    def _estimate_eligibility_complexity(self, eligibility_criteria: str) -> float:
        """Estimate eligibility complexity (0-1)."""
        if not eligibility_criteria:
            return 0.5

        criteria_lower = eligibility_criteria.lower()
        complexity = 0.3  # Base complexity

        # Count exclusion criteria indicators
        exclusion_words = ["exclude", "exclusion", "not eligible", "contraindicated"]
        for word in exclusion_words:
            complexity += criteria_lower.count(word) * 0.02

        # Lab value requirements increase complexity
        lab_indicators = ["hba1c", "egfr", "creatinine", "hemoglobin", "platelet",
                         "liver function", "alt", "ast", "bilirubin"]
        for indicator in lab_indicators:
            if indicator in criteria_lower:
                complexity += 0.03

        # Prior therapy requirements
        if "prior" in criteria_lower or "previous" in criteria_lower:
            complexity += 0.1

        # Age restrictions
        if "65" in criteria_lower or "70" in criteria_lower or "75" in criteria_lower:
            complexity += 0.05

        # Length of criteria text
        if len(eligibility_criteria) > 2000:
            complexity += 0.1
        elif len(eligibility_criteria) > 4000:
            complexity += 0.2

        return min(1.0, complexity)

    def _calculate_s_curve(
        self,
        target: int,
        monthly_rate: float,
        total_months: int,
        ramp_months: int = 3,
        plateau_factor: float = 0.85
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate S-curve enrollment using logistic function.

        The S-curve models:
        1. Slow start (site activation, learning curve)
        2. Ramp-up phase (sites at full capacity)
        3. Plateau (harder to find remaining patients)
        """
        monthly = []
        cumulative = []
        enrolled = 0

        for month in range(total_months + 1):
            if month == 0:
                monthly.append(0)
                cumulative.append(0)
                continue

            # S-curve factor using logistic function
            # Slow start, then accelerate, then plateau
            midpoint = total_months / 2
            steepness = 0.3

            # Logistic growth rate factor
            t = month
            s_factor = 1 / (1 + math.exp(-steepness * (t - ramp_months)))

            # Plateau factor (enrollment slows near end)
            remaining_ratio = (target - enrolled) / target if target > 0 else 0
            plateau = plateau_factor + (1 - plateau_factor) * remaining_ratio

            # Calculate monthly enrollment
            base_rate = monthly_rate * s_factor * plateau

            # Don't exceed target
            new_patients = min(base_rate, target - enrolled)
            new_patients = max(0, new_patients)

            enrolled += new_patients
            monthly.append(new_patients)
            cumulative.append(enrolled)

            if enrolled >= target:
                break

        return monthly, cumulative

    def _create_site_activation_plan(
        self,
        num_sites: int,
        phase: str
    ) -> SiteActivationPlan:
        """Create realistic site activation schedule."""
        params = self.PHASE_PARAMS.get(phase, self.PHASE_PARAMS["PHASE2"])

        # Sites activate over time, not all at once
        if num_sites <= 5:
            activation_months = 1
        elif num_sites <= 20:
            activation_months = 2
        elif num_sites <= 50:
            activation_months = 3
        else:
            activation_months = 4

        # Calculate sites per month (front-loaded)
        sites_per_month = []
        cumulative = []
        remaining = num_sites

        for month in range(activation_months):
            # More sites activated early
            if month == 0:
                activate = int(num_sites * 0.4)
            elif month == 1:
                activate = int(num_sites * 0.3)
            else:
                activate = remaining

            activate = min(activate, remaining)
            sites_per_month.append(activate)
            remaining -= activate

            cum = sum(sites_per_month)
            cumulative.append(cum)

        # Ensure we got all sites
        if sum(sites_per_month) < num_sites:
            sites_per_month[-1] += num_sites - sum(sites_per_month)
            cumulative[-1] = num_sites

        return SiteActivationPlan(
            total_sites=num_sites,
            activation_months=activation_months,
            sites_per_month=sites_per_month,
            cumulative_active=cumulative
        )

    def _create_milestones(
        self,
        target: int,
        cumulative: List[float],
        scenario_name: str
    ) -> List[EnrollmentMilestone]:
        """Create enrollment milestones."""
        milestones = []

        checkpoints = [0.25, 0.50, 0.75, 1.0]  # 25%, 50%, 75%, 100%

        for pct in checkpoints:
            target_patients = int(target * pct)

            # Find month when this is reached
            month = None
            for m, cum in enumerate(cumulative):
                if cum >= target_patients:
                    month = m
                    break

            if month is not None:
                milestones.append(EnrollmentMilestone(
                    month=month,
                    target_patients=target_patients,
                    cumulative_percent=pct * 100,
                    notes=f"{int(pct*100)}% enrollment target"
                ))

        return milestones

    def _generate_insights(
        self,
        forecast: EnrollmentForecast,
        historical_trials: List[Dict]
    ) -> Tuple[List[str], List[str]]:
        """Generate key insights and recommendations."""
        insights = []
        recommendations = []

        # Compare to historical
        if forecast.historical_rate_median > 0:
            expected_rate = forecast.expected.site_rate
            if expected_rate < forecast.historical_rate_p25:
                insights.append(
                    f"Your projected rate ({expected_rate:.2f}/site/month) is below the 25th percentile "
                    f"({forecast.historical_rate_p25:.2f}). Consider adding more sites or relaxing criteria."
                )
            elif expected_rate > forecast.historical_rate_p75:
                insights.append(
                    f"Your projected rate ({expected_rate:.2f}/site/month) is ambitious - "
                    f"above the 75th percentile ({forecast.historical_rate_p75:.2f})."
                )
            else:
                insights.append(
                    f"Your projected rate ({expected_rate:.2f}/site/month) is in line with "
                    f"historical median ({forecast.historical_rate_median:.2f})."
                )

        # Risk factor insights
        if forecast.risk_multiplier < 0.7:
            insights.append(
                f"High risk factors detected (multiplier: {forecast.risk_multiplier:.2f}). "
                "Enrollment may be 30%+ slower than baseline."
            )
            recommendations.append("Consider expanding eligibility criteria to increase patient pool.")

        # Screening insights
        if forecast.screening.screen_failure_rate > 0.3:
            insights.append(
                f"High screen failure rate ({forecast.screening.screen_failure_rate*100:.0f}%) expected. "
                f"You'll need to screen ~{forecast.patients_to_screen:,} patients."
            )
            recommendations.append("Implement pre-screening to reduce screen failures.")

        # Timeline insights
        expected_months = forecast.expected.total_months
        pessimistic_months = forecast.pessimistic.total_months

        if pessimistic_months > expected_months * 1.5:
            insights.append(
                f"Wide uncertainty range: {expected_months:.0f}-{pessimistic_months:.0f} months. "
                "Consider contingency planning."
            )
            recommendations.append("Build in buffer time for delays. Have backup sites ready.")

        # Site recommendations
        if forecast.num_sites < 20 and forecast.target_enrollment > 200:
            recommendations.append(
                f"With {forecast.num_sites} sites targeting {forecast.target_enrollment} patients, "
                "consider adding more sites to reduce timeline."
            )

        # Competition
        if forecast.risk_factors.competition_level > 0.5:
            recommendations.append(
                "High competition detected. Differentiate your protocol or target underserved geographies."
            )

        return insights, recommendations

    def forecast(
        self,
        target_enrollment: int,
        num_sites: int,
        condition: str,
        phase: str,
        eligibility_criteria: str = "",
        competition_level: float = 0.3,
        site_experience: float = 0.5,
        geographic_spread: float = 0.3,
        screen_failure_rate: float = 0.25
    ) -> EnrollmentForecast:
        """
        Generate comprehensive enrollment forecast.

        Args:
            target_enrollment: Target number of patients to enroll
            num_sites: Number of planned sites
            condition: Medical condition being studied
            phase: Trial phase (PHASE1, PHASE2, PHASE3, PHASE4)
            eligibility_criteria: Eligibility criteria text
            competition_level: 0-1, level of competition (0.5 = moderate)
            site_experience: 0-1, experience level of sites
            geographic_spread: 0-1, geographic diversity
            screen_failure_rate: Expected screen failure rate

        Returns:
            EnrollmentForecast with multiple scenarios
        """
        logger.info(f"Generating enrollment forecast for {condition} {phase}")

        # Get historical data
        historical_rates, historical_trials = self._get_historical_rates(condition, phase)

        if len(historical_rates) >= 5:
            historical_rates.sort()
            median_rate = historical_rates[len(historical_rates) // 2]
            p25_rate = historical_rates[int(len(historical_rates) * 0.25)]
            p75_rate = historical_rates[int(len(historical_rates) * 0.75)]
            median_duration = None  # Could calculate from trials
        else:
            # Default rates by phase
            default_rates = {
                "PHASE1": 1.0,
                "PHASE2": 1.5,
                "PHASE3": 2.0,
                "PHASE4": 2.5
            }
            median_rate = default_rates.get(phase, 1.5)
            p25_rate = median_rate * 0.6
            p75_rate = median_rate * 1.4
            median_duration = None

        # Calculate risk factors
        prevalence = self._estimate_prevalence(condition)
        complexity = self._estimate_eligibility_complexity(eligibility_criteria)

        risk_factors = RiskFactors(
            eligibility_complexity=complexity,
            competition_level=competition_level,
            condition_prevalence=prevalence,
            site_experience=site_experience,
            geographic_spread=geographic_spread
        )
        risk_multiplier = risk_factors.calculate_risk_multiplier()

        # Screening metrics
        screening = ScreeningMetrics(
            screen_failure_rate=screen_failure_rate,
            early_dropout_rate=0.10,
            randomization_rate=1 - screen_failure_rate - 0.10
        )
        patients_to_screen = screening.patients_to_screen(target_enrollment)

        # Site activation plan
        site_activation = self._create_site_activation_plan(num_sites, phase)

        # Get phase parameters
        params = self.PHASE_PARAMS.get(phase, self.PHASE_PARAMS["PHASE2"])

        # Calculate scenarios
        scenarios = {}

        for scenario_name, rate_multiplier, confidence in [
            ("optimistic", 1.3, "optimistic"),
            ("expected", 1.0, "expected"),
            ("pessimistic", 0.7, "pessimistic"),
            ("risk_adjusted", risk_multiplier, "conservative")
        ]:
            # Calculate monthly rate
            base_rate = median_rate * rate_multiplier
            monthly_rate = base_rate * num_sites

            # Estimate total months
            estimated_months = int(target_enrollment / monthly_rate) + params["ramp_months"] + 2
            estimated_months = max(6, min(60, estimated_months))  # Reasonable bounds

            # Generate S-curve
            monthly, cumulative = self._calculate_s_curve(
                target=target_enrollment,
                monthly_rate=monthly_rate,
                total_months=estimated_months,
                ramp_months=params["ramp_months"],
                plateau_factor=params["plateau_factor"]
            )

            # Find actual completion month
            completion_month = len(cumulative) - 1
            for m, cum in enumerate(cumulative):
                if cum >= target_enrollment:
                    completion_month = m
                    break

            # Create milestones
            milestones = self._create_milestones(target_enrollment, cumulative, scenario_name)

            scenarios[scenario_name] = EnrollmentScenario(
                name=scenario_name.replace("_", " ").title(),
                description=f"{confidence.title()} scenario based on {rate_multiplier:.0%} of historical rate",
                total_months=completion_month,
                monthly_enrollment=monthly,
                cumulative_enrollment=cumulative,
                milestones=milestones,
                monthly_rate=monthly_rate,
                site_rate=base_rate,
                confidence_level=confidence
            )

        # Generate insights
        forecast = EnrollmentForecast(
            target_enrollment=target_enrollment,
            num_sites=num_sites,
            condition=condition,
            phase=phase,
            historical_rate_median=median_rate,
            historical_rate_p25=p25_rate,
            historical_rate_p75=p75_rate,
            historical_duration_median=median_duration,
            similar_trials_count=len(historical_trials),
            optimistic=scenarios["optimistic"],
            expected=scenarios["expected"],
            pessimistic=scenarios["pessimistic"],
            risk_adjusted=scenarios["risk_adjusted"],
            site_activation=site_activation,
            risk_factors=risk_factors,
            risk_multiplier=risk_multiplier,
            screening=screening,
            patients_to_screen=patients_to_screen,
            key_insights=[],
            recommendations=[]
        )

        insights, recommendations = self._generate_insights(forecast, historical_trials)
        forecast.key_insights = insights
        forecast.recommendations = recommendations

        return forecast

    def generate_scenario_comparison(
        self,
        base_forecast: EnrollmentForecast,
        scenarios: List[Dict[str, Any]]
    ) -> List[EnrollmentForecast]:
        """
        Generate "what-if" scenario comparisons.

        Example scenarios:
        - {"name": "Add 10 sites", "num_sites": base + 10}
        - {"name": "Relax criteria", "eligibility_complexity": 0.3}
        """
        comparisons = []

        for scenario in scenarios:
            # Start with base parameters
            params = {
                "target_enrollment": base_forecast.target_enrollment,
                "num_sites": base_forecast.num_sites,
                "condition": base_forecast.condition,
                "phase": base_forecast.phase,
                "competition_level": base_forecast.risk_factors.competition_level,
                "site_experience": base_forecast.risk_factors.site_experience,
                "geographic_spread": base_forecast.risk_factors.geographic_spread,
                "screen_failure_rate": base_forecast.screening.screen_failure_rate
            }

            # Apply scenario changes
            for key, value in scenario.items():
                if key in params:
                    params[key] = value

            # Generate new forecast
            new_forecast = self.forecast(**params)
            comparisons.append(new_forecast)

        return comparisons

    def get_monthly_targets(
        self,
        forecast: EnrollmentForecast,
        scenario: str = "expected"
    ) -> List[Dict[str, Any]]:
        """Get monthly enrollment targets for tracking."""
        scenario_data = getattr(forecast, scenario)

        targets = []
        for month, (monthly, cumulative) in enumerate(zip(
            scenario_data.monthly_enrollment,
            scenario_data.cumulative_enrollment
        )):
            targets.append({
                "month": month,
                "monthly_target": round(monthly, 0),
                "cumulative_target": round(cumulative, 0),
                "percent_complete": round(cumulative / forecast.target_enrollment * 100, 1)
            })

        return targets
