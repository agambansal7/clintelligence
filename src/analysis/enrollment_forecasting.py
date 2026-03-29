"""
Enrollment Forecasting Module

Predicts enrollment timelines based on historical data from similar individual trials.
Provides:
- Days to target enrollment projections
- Early warning indicators when enrollment lags
- Site-level enrollment velocity predictions
- Confidence intervals based on historical variance

Unlike therapeutic-area averages, this uses individual similar trials for precision.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
import json

logger = logging.getLogger(__name__)


@dataclass
class EnrollmentProjection:
    """Projected enrollment timeline for a trial."""
    target_enrollment: int
    num_sites: int

    # Time projections
    projected_days_to_target: float
    projected_completion_date: str
    confidence_level: str  # high, medium, low

    # Confidence interval (days)
    optimistic_days: float  # 25th percentile
    pessimistic_days: float  # 75th percentile

    # Rate projections
    projected_monthly_rate: float  # patients/month total
    projected_rate_per_site: float  # patients/site/month

    # Risk indicators
    enrollment_risk_score: float  # 0-100, higher = more risk
    risk_factors: List[str] = field(default_factory=list)

    # Comparison to similar trials
    similar_trials_used: int = 0
    benchmark_trials: List[Dict[str, Any]] = field(default_factory=list)

    # Milestones
    projected_milestones: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EnrollmentAlert:
    """Alert when enrollment is lagging or at risk."""
    alert_type: str  # "lag", "slow_start", "site_underperformance", "projection_miss"
    severity: str  # "high", "medium", "low"
    message: str
    current_value: float
    expected_value: float
    recommendation: str
    affected_sites: List[str] = field(default_factory=list)


@dataclass
class SiteEnrollmentForecast:
    """Enrollment forecast for a specific site."""
    site_name: str
    city: str
    state: str
    country: str

    # Projections
    projected_enrollment: int
    projected_rate_per_month: float
    confidence_level: str

    # Historical performance
    historical_trials: int
    historical_avg_enrollment: float
    historical_completion_rate: float

    # Risk assessment
    risk_score: float
    risk_factors: List[str] = field(default_factory=list)


class EnrollmentForecaster:
    """
    Forecasts enrollment timelines using individual similar trial data.

    Usage:
        forecaster = EnrollmentForecaster(db_manager, similarity_engine)

        # Get enrollment projection
        projection = forecaster.forecast_enrollment(
            condition="Type 2 Diabetes",
            phase="PHASE3",
            target_enrollment=500,
            num_sites=50,
            eligibility_criteria="..."
        )

        # Check for enrollment alerts
        alerts = forecaster.check_enrollment_health(
            trial_id="NCT...",
            current_enrollment=150,
            days_since_start=90,
            target_enrollment=500
        )

        # Get site-level forecasts
        site_forecasts = forecaster.forecast_by_site(
            condition="Type 2 Diabetes",
            sites=[{"name": "...", "city": "..."}],
            enrollment_per_site=10
        )
    """

    # Enrollment velocity benchmarks by phase (patients/site/month)
    PHASE_VELOCITY_BENCHMARKS = {
        "PHASE1": {"median": 0.8, "p25": 0.4, "p75": 1.5},
        "PHASE2": {"median": 1.2, "p25": 0.6, "p75": 2.0},
        "PHASE3": {"median": 1.5, "p25": 0.8, "p75": 2.5},
        "PHASE4": {"median": 2.0, "p25": 1.0, "p75": 3.5},
    }

    # Risk multipliers for eligibility restrictions
    ELIGIBILITY_RISK_FACTORS = {
        "narrow_age_range": 1.3,  # <30 years span
        "strict_lab_values": 1.2,
        "treatment_naive_only": 1.25,
        "rare_biomarker": 1.5,
        "many_exclusions": 1.15,  # >15 exclusion criteria
    }

    def __init__(self, db_manager, similarity_engine=None):
        """
        Initialize forecaster.

        Args:
            db_manager: Database manager instance
            similarity_engine: Optional TrialSimilarityEngine instance
        """
        self.db = db_manager
        self.similarity_engine = similarity_engine

    def forecast_enrollment(
        self,
        condition: str,
        phase: str,
        target_enrollment: int,
        num_sites: int,
        eligibility_criteria: Optional[str] = None,
        primary_endpoint: Optional[str] = None,
        countries: Optional[List[str]] = None,
        start_date: Optional[str] = None,
    ) -> EnrollmentProjection:
        """
        Forecast enrollment timeline based on similar trials.

        Args:
            condition: Target condition/indication
            phase: Trial phase
            target_enrollment: Target number of patients
            num_sites: Number of planned sites
            eligibility_criteria: Eligibility criteria text
            primary_endpoint: Primary endpoint
            countries: Target countries
            start_date: Planned start date (YYYY-MM-DD)

        Returns:
            EnrollmentProjection with detailed timeline forecasts
        """
        # Get similar trials for benchmarking
        similar_trials = self._get_similar_trials_with_enrollment_data(
            condition=condition,
            phase=phase,
            eligibility_criteria=eligibility_criteria,
            primary_endpoint=primary_endpoint,
            target_enrollment=target_enrollment,
        )

        # Calculate enrollment velocities from similar trials
        velocities = self._extract_enrollment_velocities(similar_trials)

        # Get phase-based fallback if limited similar trials
        if len(velocities) < 3:
            phase_key = phase.upper().replace(" ", "")
            if phase_key in self.PHASE_VELOCITY_BENCHMARKS:
                benchmark = self.PHASE_VELOCITY_BENCHMARKS[phase_key]
                # Add synthetic data points if we have few similar trials
                if len(velocities) == 0:
                    velocities = [benchmark["p25"], benchmark["median"], benchmark["p75"]]
                elif len(velocities) < 3:
                    velocities.append(benchmark["median"])

        # Calculate statistics
        if velocities:
            median_velocity = statistics.median(velocities)
            try:
                p25 = statistics.quantiles(velocities, n=4)[0] if len(velocities) >= 4 else min(velocities)
                p75 = statistics.quantiles(velocities, n=4)[2] if len(velocities) >= 4 else max(velocities)
            except statistics.StatisticsError:
                p25 = min(velocities)
                p75 = max(velocities)
        else:
            # Absolute fallback
            median_velocity = 1.0
            p25 = 0.5
            p75 = 2.0

        # Apply eligibility risk multipliers
        risk_multiplier, risk_factors = self._assess_eligibility_risks(eligibility_criteria)
        adjusted_velocity = median_velocity / risk_multiplier

        # Calculate projections
        enrollment_per_site = target_enrollment / num_sites
        days_to_target = (enrollment_per_site / adjusted_velocity) * 30.44  # Convert months to days

        optimistic_days = (enrollment_per_site / p75) * 30.44
        pessimistic_days = (enrollment_per_site / p25) * 30.44 * risk_multiplier

        # Monthly rate
        monthly_rate = adjusted_velocity * num_sites

        # Calculate completion date
        start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now()
        completion_date = start + timedelta(days=days_to_target)

        # Determine confidence level
        confidence = self._determine_confidence_level(
            num_similar_trials=len(similar_trials),
            velocity_variance=statistics.stdev(velocities) if len(velocities) > 1 else 1.0,
            median_velocity=median_velocity,
        )

        # Calculate risk score
        risk_score = self._calculate_enrollment_risk_score(
            velocities=velocities,
            risk_multiplier=risk_multiplier,
            target_enrollment=target_enrollment,
            num_sites=num_sites,
            num_similar_trials=len(similar_trials),
        )

        # Generate milestones
        milestones = self._generate_milestones(
            target_enrollment=target_enrollment,
            monthly_rate=monthly_rate,
            start_date=start,
        )

        # Format benchmark trials
        benchmark_trials = []
        for trial in similar_trials[:5]:
            benchmark_trials.append({
                "nct_id": trial.get("nct_id"),
                "title": trial.get("title", "")[:100],
                "enrollment": trial.get("enrollment"),
                "num_sites": trial.get("num_sites"),
                "duration_months": trial.get("duration_months"),
                "velocity": trial.get("velocity"),
                "status": trial.get("status"),
            })

        return EnrollmentProjection(
            target_enrollment=target_enrollment,
            num_sites=num_sites,
            projected_days_to_target=round(days_to_target, 1),
            projected_completion_date=completion_date.strftime("%Y-%m-%d"),
            confidence_level=confidence,
            optimistic_days=round(optimistic_days, 1),
            pessimistic_days=round(pessimistic_days, 1),
            projected_monthly_rate=round(monthly_rate, 1),
            projected_rate_per_site=round(adjusted_velocity, 2),
            enrollment_risk_score=round(risk_score, 1),
            risk_factors=risk_factors,
            similar_trials_used=len(similar_trials),
            benchmark_trials=benchmark_trials,
            projected_milestones=milestones,
        )

    def _get_similar_trials_with_enrollment_data(
        self,
        condition: str,
        phase: str,
        eligibility_criteria: Optional[str],
        primary_endpoint: Optional[str],
        target_enrollment: int,
    ) -> List[Dict[str, Any]]:
        """Get similar trials that have enrollment velocity data."""
        # Use similarity engine if available
        if self.similarity_engine:
            from src.analysis.trial_similarity import TrialSimilarityEngine
            similar = self.similarity_engine.find_similar_trials(
                condition=condition,
                phase=phase,
                eligibility_criteria=eligibility_criteria,
                primary_endpoint=primary_endpoint,
                enrollment_target=target_enrollment,
                limit=50,
                include_terminated=True,
            )

            trials = []
            for s in similar:
                if s.enrollment and s.num_sites and s.duration_months and s.duration_months > 0:
                    velocity = s.enrollment / s.num_sites / s.duration_months
                    trials.append({
                        "nct_id": s.nct_id,
                        "title": s.title,
                        "enrollment": s.enrollment,
                        "num_sites": s.num_sites,
                        "duration_months": s.duration_months,
                        "velocity": velocity,
                        "status": s.status,
                        "similarity_score": s.similarity_score,
                    })
            return trials

        # Fallback: query database directly
        return self._query_similar_trials_direct(condition, phase, target_enrollment)

    def _query_similar_trials_direct(
        self,
        condition: str,
        phase: str,
        target_enrollment: int,
    ) -> List[Dict[str, Any]]:
        """Query database directly for similar trials."""
        with self.db.get_session() as session:
            from src.database.models import Trial
            from sqlalchemy import func

            query = session.query(Trial).filter(
                func.lower(Trial.conditions).contains(condition.lower()),
                Trial.status.in_(["COMPLETED", "TERMINATED"]),
                Trial.enrollment.isnot(None),
                Trial.num_sites.isnot(None),
                Trial.num_sites > 0,
                Trial.start_date.isnot(None),
                Trial.completion_date.isnot(None),
            )

            if phase:
                query = query.filter(Trial.phase.contains(phase.replace("PHASE", "")))

            trials = query.limit(100).all()

            results = []
            for t in trials:
                try:
                    start = datetime.strptime(str(t.start_date)[:10], "%Y-%m-%d")
                    end = datetime.strptime(str(t.completion_date)[:10], "%Y-%m-%d")
                    duration_months = (end - start).days / 30.44

                    if duration_months > 0 and t.enrollment > 0:
                        velocity = t.enrollment / t.num_sites / duration_months
                        results.append({
                            "nct_id": t.nct_id,
                            "title": t.title,
                            "enrollment": t.enrollment,
                            "num_sites": t.num_sites,
                            "duration_months": duration_months,
                            "velocity": velocity,
                            "status": t.status,
                        })
                except (ValueError, TypeError):
                    continue

            return results

    def _extract_enrollment_velocities(self, trials: List[Dict[str, Any]]) -> List[float]:
        """Extract enrollment velocities from trials."""
        velocities = []
        for trial in trials:
            velocity = trial.get("velocity")
            if velocity and 0.1 < velocity < 20:  # Sanity check
                velocities.append(velocity)
        return velocities

    def _assess_eligibility_risks(self, eligibility_criteria: Optional[str]) -> Tuple[float, List[str]]:
        """Assess enrollment risks from eligibility criteria."""
        if not eligibility_criteria:
            return 1.0, []

        criteria_lower = eligibility_criteria.lower()
        risk_multiplier = 1.0
        risk_factors = []

        # Check for narrow age range
        import re
        age_match = re.search(r"(\d+)\s*(?:to|-|and)\s*(\d+)\s*years?", criteria_lower)
        if age_match:
            age_range = int(age_match.group(2)) - int(age_match.group(1))
            if age_range < 30:
                risk_multiplier *= self.ELIGIBILITY_RISK_FACTORS["narrow_age_range"]
                risk_factors.append(f"Narrow age range ({age_range} years) may limit patient pool")

        # Check for strict lab values
        lab_patterns = [
            r"hba1c\s*(?:>=?|<=?)\s*[\d.]+\s*(?:and|to)\s*<=?\s*[\d.]+",
            r"egfr\s*>=?\s*\d+",
            r"(?:alt|ast)\s*<=?\s*\d+\s*x?\s*(?:uln|upper)",
        ]
        strict_labs = sum(1 for p in lab_patterns if re.search(p, criteria_lower))
        if strict_labs >= 2:
            risk_multiplier *= self.ELIGIBILITY_RISK_FACTORS["strict_lab_values"]
            risk_factors.append("Multiple strict laboratory value requirements")

        # Check for treatment-naive requirement
        if "treatment" in criteria_lower and ("naive" in criteria_lower or "naïve" in criteria_lower):
            risk_multiplier *= self.ELIGIBILITY_RISK_FACTORS["treatment_naive_only"]
            risk_factors.append("Treatment-naive requirement limits eligible population")

        # Check for rare biomarkers
        rare_markers = ["her2", "egfr mutation", "alk", "braf", "pd-l1", "microsatellite"]
        if any(marker in criteria_lower for marker in rare_markers):
            risk_multiplier *= self.ELIGIBILITY_RISK_FACTORS["rare_biomarker"]
            risk_factors.append("Biomarker requirement may significantly limit patient pool")

        # Count exclusion criteria
        exclusion_count = criteria_lower.count("exclusion") + len(re.findall(r"(?:must not|cannot|excluded|no (?:prior|history))", criteria_lower))
        if exclusion_count > 15:
            risk_multiplier *= self.ELIGIBILITY_RISK_FACTORS["many_exclusions"]
            risk_factors.append(f"High number of exclusion criteria ({exclusion_count}+)")

        return risk_multiplier, risk_factors

    def _determine_confidence_level(
        self,
        num_similar_trials: int,
        velocity_variance: float,
        median_velocity: float,
    ) -> str:
        """Determine confidence level of the forecast."""
        # Calculate coefficient of variation
        cv = velocity_variance / median_velocity if median_velocity > 0 else 1.0

        if num_similar_trials >= 10 and cv < 0.3:
            return "high"
        elif num_similar_trials >= 5 and cv < 0.5:
            return "medium"
        else:
            return "low"

    def _calculate_enrollment_risk_score(
        self,
        velocities: List[float],
        risk_multiplier: float,
        target_enrollment: int,
        num_sites: int,
        num_similar_trials: int,
    ) -> float:
        """Calculate overall enrollment risk score (0-100)."""
        risk_score = 0

        # Base risk from eligibility restrictions
        risk_score += (risk_multiplier - 1.0) * 50

        # Risk from enrollment per site target
        enrollment_per_site = target_enrollment / num_sites if num_sites > 0 else target_enrollment
        if enrollment_per_site > 15:
            risk_score += min(30, (enrollment_per_site - 15) * 2)

        # Risk from low similar trial count (uncertainty)
        if num_similar_trials < 3:
            risk_score += 20
        elif num_similar_trials < 10:
            risk_score += 10

        # Risk from high velocity variance in similar trials
        if velocities and len(velocities) > 1:
            try:
                cv = statistics.stdev(velocities) / statistics.mean(velocities)
                if cv > 0.5:
                    risk_score += 15
                elif cv > 0.3:
                    risk_score += 8
            except statistics.StatisticsError:
                pass

        return min(100, max(0, risk_score))

    def _generate_milestones(
        self,
        target_enrollment: int,
        monthly_rate: float,
        start_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Generate enrollment milestones."""
        milestones = []

        milestone_pcts = [0.25, 0.50, 0.75, 1.0]

        for pct in milestone_pcts:
            patients = int(target_enrollment * pct)
            months_needed = patients / monthly_rate if monthly_rate > 0 else 0
            milestone_date = start_date + timedelta(days=months_needed * 30.44)

            milestones.append({
                "percentage": int(pct * 100),
                "patients": patients,
                "projected_date": milestone_date.strftime("%Y-%m-%d"),
                "months_from_start": round(months_needed, 1),
            })

        return milestones

    def check_enrollment_health(
        self,
        current_enrollment: int,
        days_since_start: int,
        target_enrollment: int,
        num_sites: int,
        condition: str,
        phase: str,
    ) -> List[EnrollmentAlert]:
        """
        Check if enrollment is on track and generate alerts.

        Args:
            current_enrollment: Current number of enrolled patients
            days_since_start: Days since first patient enrolled
            target_enrollment: Target enrollment
            num_sites: Number of active sites
            condition: Trial condition
            phase: Trial phase

        Returns:
            List of EnrollmentAlert objects
        """
        alerts = []

        if days_since_start <= 0:
            return alerts

        # Calculate current rate
        months_elapsed = days_since_start / 30.44
        current_monthly_rate = current_enrollment / months_elapsed if months_elapsed > 0 else 0
        current_rate_per_site = current_monthly_rate / num_sites if num_sites > 0 else 0

        # Get expected rate from similar trials
        similar_trials = self._query_similar_trials_direct(condition, phase, target_enrollment)
        velocities = self._extract_enrollment_velocities(similar_trials)

        if velocities:
            expected_rate_per_site = statistics.median(velocities)
        else:
            phase_key = phase.upper().replace(" ", "")
            benchmark = self.PHASE_VELOCITY_BENCHMARKS.get(phase_key, {"median": 1.0})
            expected_rate_per_site = benchmark["median"]

        expected_monthly_rate = expected_rate_per_site * num_sites
        expected_enrollment = expected_monthly_rate * months_elapsed

        # Alert 1: Overall enrollment lag
        if current_enrollment < expected_enrollment * 0.7:
            lag_pct = (1 - current_enrollment / expected_enrollment) * 100 if expected_enrollment > 0 else 0
            severity = "high" if lag_pct > 40 else "medium" if lag_pct > 20 else "low"

            alerts.append(EnrollmentAlert(
                alert_type="lag",
                severity=severity,
                message=f"Enrollment is {lag_pct:.0f}% behind expected pace",
                current_value=current_enrollment,
                expected_value=round(expected_enrollment, 0),
                recommendation="Review site performance, consider adding sites or adjusting eligibility criteria",
            ))

        # Alert 2: Slow start
        if days_since_start < 90 and current_enrollment < expected_enrollment * 0.5:
            alerts.append(EnrollmentAlert(
                alert_type="slow_start",
                severity="medium",
                message="Slow enrollment start - first 90 days critical for momentum",
                current_value=current_enrollment,
                expected_value=round(expected_enrollment, 0),
                recommendation="Engage with sites to understand barriers, provide additional training/support",
            ))

        # Alert 3: Low site productivity
        if current_rate_per_site < expected_rate_per_site * 0.5:
            alerts.append(EnrollmentAlert(
                alert_type="site_underperformance",
                severity="medium",
                message=f"Site productivity ({current_rate_per_site:.2f}/month) below benchmark ({expected_rate_per_site:.2f}/month)",
                current_value=current_rate_per_site,
                expected_value=expected_rate_per_site,
                recommendation="Identify underperforming sites, consider site visits or replacing non-enrolling sites",
            ))

        # Alert 4: Projection miss risk
        remaining_enrollment = target_enrollment - current_enrollment
        if remaining_enrollment > 0:
            months_to_target = remaining_enrollment / current_monthly_rate if current_monthly_rate > 0 else float('inf')

            # If it would take > 2x expected time
            expected_total_months = target_enrollment / expected_monthly_rate if expected_monthly_rate > 0 else 0
            if months_to_target > expected_total_months * 2 and months_elapsed > 3:
                alerts.append(EnrollmentAlert(
                    alert_type="projection_miss",
                    severity="high",
                    message=f"At current rate, enrollment will take {months_to_target:.0f} months (expected: {expected_total_months:.0f})",
                    current_value=months_to_target,
                    expected_value=expected_total_months,
                    recommendation="Consider protocol amendment, additional sites, or revised enrollment strategy",
                ))

        return alerts

    def forecast_by_site(
        self,
        condition: str,
        phase: str,
        sites: List[Dict[str, Any]],
        enrollment_per_site: int,
    ) -> List[SiteEnrollmentForecast]:
        """
        Generate enrollment forecasts for specific sites.

        Args:
            condition: Trial condition
            phase: Trial phase
            sites: List of sites with name, city, state, country
            enrollment_per_site: Target enrollment per site

        Returns:
            List of SiteEnrollmentForecast objects
        """
        forecasts = []

        for site in sites:
            site_name = site.get("name", "")
            city = site.get("city", "")
            state = site.get("state", "")
            country = site.get("country", "USA")

            # Get historical performance for this site
            site_history = self._get_site_history(site_name, city, state, condition)

            if site_history:
                # Use historical data
                projected_rate = site_history["avg_velocity"]
                historical_trials = site_history["trial_count"]
                historical_avg = site_history["avg_enrollment"]
                completion_rate = site_history["completion_rate"]
                confidence = "high" if historical_trials >= 3 else "medium"
            else:
                # Use regional/phase benchmarks
                phase_key = phase.upper().replace(" ", "")
                benchmark = self.PHASE_VELOCITY_BENCHMARKS.get(phase_key, {"median": 1.0})
                projected_rate = benchmark["median"]
                historical_trials = 0
                historical_avg = 0
                completion_rate = 0
                confidence = "low"

            # Calculate risk factors
            risk_factors = []
            risk_score = 30  # Base uncertainty

            if historical_trials == 0:
                risk_factors.append("No historical data for this site")
                risk_score += 20
            elif completion_rate < 0.7:
                risk_factors.append(f"Lower completion rate ({completion_rate:.0%})")
                risk_score += 15

            if historical_avg < enrollment_per_site * 0.5:
                risk_factors.append(f"Historical enrollment ({historical_avg:.0f}) below target ({enrollment_per_site})")
                risk_score += 10

            forecasts.append(SiteEnrollmentForecast(
                site_name=site_name,
                city=city,
                state=state,
                country=country,
                projected_enrollment=enrollment_per_site,
                projected_rate_per_month=round(projected_rate, 2),
                confidence_level=confidence,
                historical_trials=historical_trials,
                historical_avg_enrollment=round(historical_avg, 1),
                historical_completion_rate=round(completion_rate, 2),
                risk_score=min(100, risk_score),
                risk_factors=risk_factors,
            ))

        # Sort by risk score (lower risk first)
        forecasts.sort(key=lambda x: x.risk_score)

        return forecasts

    def _get_site_history(
        self,
        site_name: str,
        city: str,
        state: str,
        condition: str,
    ) -> Optional[Dict[str, Any]]:
        """Get historical performance data for a site."""
        with self.db.get_session() as session:
            from src.database.models import Site

            # Try to find site by name and location
            query = session.query(Site)

            if site_name:
                query = query.filter(Site.facility_name.ilike(f"%{site_name}%"))
            if city:
                query = query.filter(Site.city.ilike(f"%{city}%"))
            if state:
                query = query.filter(Site.state.ilike(f"%{state}%"))

            site = query.first()

            if site and site.total_trials > 0:
                # Calculate velocity from historical data
                avg_velocity = site.enrollment_velocity if site.enrollment_velocity else 1.0

                return {
                    "trial_count": site.total_trials,
                    "avg_enrollment": site.avg_enrollment or 0,
                    "avg_velocity": avg_velocity,
                    "completion_rate": site.completion_rate or 0,
                }

            return None


def forecast_trial_enrollment(
    db_manager,
    condition: str,
    phase: str,
    target_enrollment: int,
    num_sites: int,
    eligibility_criteria: Optional[str] = None,
    start_date: Optional[str] = None,
) -> EnrollmentProjection:
    """Convenience function for enrollment forecasting."""
    forecaster = EnrollmentForecaster(db_manager)
    return forecaster.forecast_enrollment(
        condition=condition,
        phase=phase,
        target_enrollment=target_enrollment,
        num_sites=num_sites,
        eligibility_criteria=eligibility_criteria,
        start_date=start_date or datetime.now().strftime("%Y-%m-%d"),
    )
