"""
Site Performance Leaderboard

Provides ranked site performance with individual trial-level metrics:
- Sites ranked by performance in specific therapeutic areas
- Individual trial performance history for each site
- Head-to-head site comparisons
- Availability status and current trial load
- Geographic and diversity scoring
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class TrialPerformance:
    """Performance metrics for a single trial at a site."""
    nct_id: str
    title: str
    condition: str
    phase: str
    status: str
    enrollment: int
    enrollment_target: Optional[int]
    start_date: Optional[str]
    completion_date: Optional[str]
    duration_months: Optional[float]
    enrollment_rate: Optional[float]  # patients/month
    completed_on_time: Optional[bool]
    role: str  # "lead", "participating"


@dataclass
class SiteRanking:
    """Complete site ranking with performance details."""
    rank: int
    site_id: int
    facility_name: str
    city: str
    state: str
    country: str

    # Performance scores (0-100)
    overall_score: float
    enrollment_score: float
    completion_score: float
    speed_score: float
    experience_score: float

    # Key metrics
    total_trials: int
    completed_trials: int
    active_trials: int
    therapeutic_experience: int  # trials in specific therapeutic area

    # Enrollment metrics
    total_enrollment: int
    avg_enrollment_per_trial: float
    enrollment_velocity: float  # patients/site/month

    # Performance ratios
    completion_rate: float
    on_time_rate: float

    # Availability
    availability_status: str  # "available", "busy", "at_capacity"
    current_trial_load: int

    # Diversity
    diversity_score: float

    # Individual trial history
    trial_history: List[TrialPerformance] = field(default_factory=list)

    # Strengths and considerations
    strengths: List[str] = field(default_factory=list)
    considerations: List[str] = field(default_factory=list)


@dataclass
class SiteComparison:
    """Head-to-head comparison of two sites."""
    site_a: SiteRanking
    site_b: SiteRanking

    # Winner by category
    better_enrollment: str  # site_a or site_b
    better_completion: str
    better_speed: str
    better_experience: str

    # Detailed comparisons
    metric_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    recommendation: str = ""


class SiteLeaderboard:
    """
    Site performance leaderboard with trial-level metrics.

    Usage:
        leaderboard = SiteLeaderboard(db_manager)

        # Get top sites for a therapeutic area
        rankings = leaderboard.get_rankings(
            therapeutic_area="oncology",
            phase="PHASE3",
            country="United States",
            limit=20
        )

        # Compare two sites
        comparison = leaderboard.compare_sites(site_id_1, site_id_2)

        # Get site's trial history
        history = leaderboard.get_site_trial_history(site_id)
    """

    # Availability thresholds
    AVAILABILITY_THRESHOLDS = {
        "available": (0, 2),  # 0-2 active trials
        "busy": (3, 5),  # 3-5 active trials
        "at_capacity": (6, float('inf')),  # 6+ active trials
    }

    # Diversity regions (US states with diverse populations)
    HIGH_DIVERSITY_STATES = ["CA", "TX", "FL", "NY", "GA", "IL", "NJ", "AZ", "NC"]
    MEDIUM_DIVERSITY_STATES = ["PA", "OH", "MI", "VA", "MD", "MA", "CO", "TN", "IN"]

    def __init__(self, db_manager):
        """Initialize with database connection."""
        self.db = db_manager

    def get_rankings(
        self,
        therapeutic_area: Optional[str] = None,
        condition: Optional[str] = None,
        phase: Optional[str] = None,
        country: Optional[str] = None,
        state: Optional[str] = None,
        min_trials: int = 3,
        include_trial_history: bool = True,
        limit: int = 50,
    ) -> List[SiteRanking]:
        """
        Get ranked list of sites based on performance.

        Args:
            therapeutic_area: Filter by therapeutic area
            condition: Filter by specific condition
            phase: Filter by trial phase
            country: Filter by country
            state: Filter by state (US only)
            min_trials: Minimum trials required to be ranked
            include_trial_history: Include individual trial history
            limit: Maximum sites to return

        Returns:
            List of SiteRanking objects sorted by overall_score
        """
        # Get sites from database
        sites = self._get_sites_from_db(
            therapeutic_area=therapeutic_area,
            country=country,
            state=state,
            min_trials=min_trials,
        )

        rankings = []
        for site_data in sites:
            # Get trial history for scoring
            trial_history = []
            if include_trial_history:
                trial_history = self._get_site_trials(
                    site_id=site_data["id"],
                    therapeutic_area=therapeutic_area,
                    condition=condition,
                    phase=phase,
                )

            # Calculate scores
            scores = self._calculate_site_scores(site_data, trial_history)

            # Determine availability status
            active_count = site_data.get("active_trials", 0)
            availability = "available"
            for status, (min_t, max_t) in self.AVAILABILITY_THRESHOLDS.items():
                if min_t <= active_count <= max_t:
                    availability = status
                    break

            # Calculate diversity score
            diversity_score = self._calculate_diversity_score(
                site_data.get("state", ""),
                site_data.get("country", "")
            )

            # Identify strengths and considerations
            strengths, considerations = self._identify_strengths_considerations(
                scores, site_data, trial_history
            )

            # Calculate therapeutic experience
            therapeutic_exp = 0
            if therapeutic_area:
                therapeutic_exp = len([t for t in trial_history
                                      if therapeutic_area.lower() in (t.condition or "").lower()])

            rankings.append(SiteRanking(
                rank=0,  # Will be set after sorting
                site_id=site_data["id"],
                facility_name=site_data.get("facility_name", "Unknown"),
                city=site_data.get("city", ""),
                state=site_data.get("state", ""),
                country=site_data.get("country", ""),
                overall_score=scores["overall"],
                enrollment_score=scores["enrollment"],
                completion_score=scores["completion"],
                speed_score=scores["speed"],
                experience_score=scores["experience"],
                total_trials=site_data.get("total_trials", 0),
                completed_trials=site_data.get("completed_trials", 0),
                active_trials=active_count,
                therapeutic_experience=therapeutic_exp,
                total_enrollment=site_data.get("total_enrollment", 0),
                avg_enrollment_per_trial=site_data.get("avg_enrollment", 0),
                enrollment_velocity=site_data.get("enrollment_velocity", 0),
                completion_rate=site_data.get("completion_rate", 0),
                on_time_rate=self._calculate_on_time_rate(trial_history),
                availability_status=availability,
                current_trial_load=active_count,
                diversity_score=diversity_score,
                trial_history=trial_history if include_trial_history else [],
                strengths=strengths,
                considerations=considerations,
            ))

        # Sort by overall score and assign ranks
        rankings.sort(key=lambda x: x.overall_score, reverse=True)
        for i, ranking in enumerate(rankings[:limit], 1):
            ranking.rank = i

        return rankings[:limit]

    def _get_sites_from_db(
        self,
        therapeutic_area: Optional[str],
        country: Optional[str],
        state: Optional[str],
        min_trials: int,
    ) -> List[Dict[str, Any]]:
        """Get sites from database with filters."""
        with self.db.get_session() as session:
            from src.database.models import Site

            query = session.query(Site).filter(
                Site.total_trials >= min_trials
            )

            if country:
                query = query.filter(Site.country.ilike(f"%{country}%"))
            if state:
                query = query.filter(Site.state.ilike(f"%{state}%"))
            if therapeutic_area:
                query = query.filter(
                    Site.therapeutic_areas.ilike(f"%{therapeutic_area}%")
                )

            query = query.order_by(Site.experience_score.desc().nullslast())

            sites = query.limit(500).all()

            return [{
                "id": s.id,
                "facility_name": s.facility_name,
                "city": s.city,
                "state": s.state,
                "country": s.country,
                "total_trials": s.total_trials,
                "completed_trials": s.completed_trials,
                "terminated_trials": s.terminated_trials,
                "active_trials": s.active_trials,
                "total_enrollment": s.total_enrollment,
                "avg_enrollment": s.avg_enrollment,
                "enrollment_velocity": s.enrollment_velocity,
                "completion_rate": s.completion_rate,
                "experience_score": s.experience_score,
                "therapeutic_areas": s.therapeutic_areas,
            } for s in sites]

    def _get_site_trials(
        self,
        site_id: int,
        therapeutic_area: Optional[str],
        condition: Optional[str],
        phase: Optional[str],
    ) -> List[TrialPerformance]:
        """Get individual trial performance for a site."""
        with self.db.get_session() as session:
            from src.database.models import Trial, Site
            from sqlalchemy import func

            # Get site name for matching
            site = session.query(Site).filter(Site.id == site_id).first()
            if not site:
                return []

            # Query trials that include this site
            # Note: This is approximate since we store locations as JSON
            query = session.query(Trial).filter(
                Trial.locations.ilike(f"%{site.facility_name}%") |
                (Trial.locations.ilike(f"%{site.city}%") &
                 Trial.locations.ilike(f"%{site.state}%"))
            )

            if therapeutic_area:
                query = query.filter(
                    func.lower(Trial.therapeutic_area).contains(therapeutic_area.lower())
                )
            if condition:
                query = query.filter(
                    func.lower(Trial.conditions).contains(condition.lower())
                )
            if phase:
                query = query.filter(Trial.phase.contains(phase.replace("PHASE", "")))

            trials = query.order_by(Trial.start_date.desc()).limit(50).all()

            trial_performances = []
            for t in trials:
                # Calculate duration
                duration_months = None
                if t.start_date and t.completion_date:
                    try:
                        start = datetime.strptime(str(t.start_date)[:10], "%Y-%m-%d")
                        end = datetime.strptime(str(t.completion_date)[:10], "%Y-%m-%d")
                        duration_months = (end - start).days / 30.44
                    except (ValueError, TypeError):
                        pass

                # Calculate enrollment rate
                enrollment_rate = None
                if t.enrollment and duration_months and duration_months > 0:
                    enrollment_rate = t.enrollment / duration_months

                trial_performances.append(TrialPerformance(
                    nct_id=t.nct_id,
                    title=t.title[:100] if t.title else "",
                    condition=t.conditions[:50] if t.conditions else "",
                    phase=t.phase or "",
                    status=t.status or "",
                    enrollment=t.enrollment or 0,
                    enrollment_target=t.enrollment,
                    start_date=str(t.start_date)[:10] if t.start_date else None,
                    completion_date=str(t.completion_date)[:10] if t.completion_date else None,
                    duration_months=duration_months,
                    enrollment_rate=enrollment_rate,
                    completed_on_time=t.status == "COMPLETED",
                    role="participating",  # Would need more data to determine lead
                ))

            return trial_performances

    def _calculate_site_scores(
        self,
        site_data: Dict[str, Any],
        trial_history: List[TrialPerformance]
    ) -> Dict[str, float]:
        """Calculate performance scores for a site."""
        scores = {
            "enrollment": 50.0,
            "completion": 50.0,
            "speed": 50.0,
            "experience": 50.0,
            "overall": 50.0,
        }

        # Enrollment score (0-100)
        avg_enrollment = site_data.get("avg_enrollment") or 0
        if avg_enrollment >= 20:
            scores["enrollment"] = min(100, 60 + avg_enrollment)
        elif avg_enrollment >= 10:
            scores["enrollment"] = 40 + avg_enrollment * 2
        else:
            scores["enrollment"] = avg_enrollment * 4

        # Completion score (0-100)
        completion_rate = site_data.get("completion_rate") or 0
        scores["completion"] = completion_rate * 100

        # Speed score (based on enrollment velocity)
        velocity = site_data.get("enrollment_velocity") or 0
        if velocity >= 2.0:
            scores["speed"] = min(100, 70 + velocity * 10)
        elif velocity >= 1.0:
            scores["speed"] = 50 + velocity * 20
        else:
            scores["speed"] = velocity * 50

        # Experience score (based on total trials and recency)
        total_trials = site_data.get("total_trials", 0)
        if total_trials >= 20:
            scores["experience"] = min(100, 70 + total_trials)
        elif total_trials >= 10:
            scores["experience"] = 50 + total_trials * 2
        else:
            scores["experience"] = total_trials * 5

        # Adjust based on trial history specifics
        if trial_history:
            successful_trials = len([t for t in trial_history if t.status == "COMPLETED"])
            if successful_trials >= 5:
                scores["completion"] = min(100, scores["completion"] + 10)

        # Overall score (weighted average)
        weights = {
            "enrollment": 0.30,
            "completion": 0.25,
            "speed": 0.25,
            "experience": 0.20,
        }
        scores["overall"] = sum(
            scores[k] * w for k, w in weights.items()
        )

        return {k: round(v, 1) for k, v in scores.items()}

    def _calculate_on_time_rate(self, trial_history: List[TrialPerformance]) -> float:
        """Calculate percentage of trials completed on time."""
        if not trial_history:
            return 0.0

        completed = [t for t in trial_history if t.status == "COMPLETED"]
        if not completed:
            return 0.0

        on_time = len([t for t in completed if t.completed_on_time])
        return round(on_time / len(completed), 2)

    def _calculate_diversity_score(self, state: str, country: str) -> float:
        """Calculate diversity score based on location."""
        if country.lower() not in ["united states", "usa", "us"]:
            return 50.0  # Neutral for non-US

        state_upper = state.upper()
        if state_upper in self.HIGH_DIVERSITY_STATES:
            return 90.0
        elif state_upper in self.MEDIUM_DIVERSITY_STATES:
            return 70.0
        else:
            return 50.0

    def _identify_strengths_considerations(
        self,
        scores: Dict[str, float],
        site_data: Dict[str, Any],
        trial_history: List[TrialPerformance]
    ) -> Tuple[List[str], List[str]]:
        """Identify site strengths and considerations."""
        strengths = []
        considerations = []

        # Score-based strengths
        if scores["enrollment"] >= 80:
            strengths.append("High enrollment performer")
        if scores["completion"] >= 80:
            strengths.append("Excellent trial completion rate")
        if scores["speed"] >= 80:
            strengths.append("Fast enrollment velocity")
        if scores["experience"] >= 80:
            strengths.append("Highly experienced site")

        # History-based strengths
        if trial_history:
            completed_count = len([t for t in trial_history if t.status == "COMPLETED"])
            if completed_count >= 10:
                strengths.append(f"{completed_count} completed trials")

        # Score-based considerations
        if scores["enrollment"] < 40:
            considerations.append("Lower than average enrollment")
        if scores["completion"] < 60:
            considerations.append("Below average completion rate")
        if site_data.get("active_trials", 0) >= 5:
            considerations.append("Currently managing multiple active trials")

        # History-based considerations
        terminated = len([t for t in trial_history if t.status in ["TERMINATED", "WITHDRAWN"]])
        if terminated > 0:
            considerations.append(f"{terminated} trials terminated at this site")

        return strengths, considerations

    def compare_sites(
        self,
        site_id_a: int,
        site_id_b: int,
        therapeutic_area: Optional[str] = None,
    ) -> SiteComparison:
        """
        Compare two sites head-to-head.

        Args:
            site_id_a: First site ID
            site_id_b: Second site ID
            therapeutic_area: Optional filter for comparison

        Returns:
            SiteComparison with detailed breakdown
        """
        # Get rankings for both sites
        rankings = self.get_rankings(
            therapeutic_area=therapeutic_area,
            include_trial_history=True,
            limit=1000,
        )

        site_a = next((r for r in rankings if r.site_id == site_id_a), None)
        site_b = next((r for r in rankings if r.site_id == site_id_b), None)

        if not site_a or not site_b:
            raise ValueError("Could not find one or both sites")

        # Determine winners by category
        better_enrollment = "site_a" if site_a.enrollment_score > site_b.enrollment_score else "site_b"
        better_completion = "site_a" if site_a.completion_score > site_b.completion_score else "site_b"
        better_speed = "site_a" if site_a.speed_score > site_b.speed_score else "site_b"
        better_experience = "site_a" if site_a.experience_score > site_b.experience_score else "site_b"

        # Build metric comparisons
        metric_comparisons = [
            {
                "metric": "Overall Score",
                "site_a_value": site_a.overall_score,
                "site_b_value": site_b.overall_score,
                "winner": "site_a" if site_a.overall_score > site_b.overall_score else "site_b",
            },
            {
                "metric": "Enrollment Score",
                "site_a_value": site_a.enrollment_score,
                "site_b_value": site_b.enrollment_score,
                "winner": better_enrollment,
            },
            {
                "metric": "Completion Rate",
                "site_a_value": f"{site_a.completion_rate:.0%}",
                "site_b_value": f"{site_b.completion_rate:.0%}",
                "winner": better_completion,
            },
            {
                "metric": "Enrollment Velocity",
                "site_a_value": f"{site_a.enrollment_velocity:.1f}/month",
                "site_b_value": f"{site_b.enrollment_velocity:.1f}/month",
                "winner": better_speed,
            },
            {
                "metric": "Total Trials",
                "site_a_value": site_a.total_trials,
                "site_b_value": site_b.total_trials,
                "winner": "site_a" if site_a.total_trials > site_b.total_trials else "site_b",
            },
            {
                "metric": "Availability",
                "site_a_value": site_a.availability_status,
                "site_b_value": site_b.availability_status,
                "winner": self._compare_availability(site_a.availability_status, site_b.availability_status),
            },
        ]

        # Generate recommendation
        score_diff = site_a.overall_score - site_b.overall_score
        if abs(score_diff) < 5:
            recommendation = f"Sites are closely matched. Consider {site_a.facility_name}'s {site_a.strengths[0] if site_a.strengths else 'performance'} vs {site_b.facility_name}'s {site_b.strengths[0] if site_b.strengths else 'performance'}."
        elif score_diff > 0:
            recommendation = f"{site_a.facility_name} is the stronger choice overall ({site_a.overall_score:.0f} vs {site_b.overall_score:.0f} score)."
        else:
            recommendation = f"{site_b.facility_name} is the stronger choice overall ({site_b.overall_score:.0f} vs {site_a.overall_score:.0f} score)."

        return SiteComparison(
            site_a=site_a,
            site_b=site_b,
            better_enrollment=better_enrollment,
            better_completion=better_completion,
            better_speed=better_speed,
            better_experience=better_experience,
            metric_comparisons=metric_comparisons,
            recommendation=recommendation,
        )

    def _compare_availability(self, status_a: str, status_b: str) -> str:
        """Compare availability statuses."""
        order = {"available": 0, "busy": 1, "at_capacity": 2}
        if order.get(status_a, 2) < order.get(status_b, 2):
            return "site_a"
        elif order.get(status_a, 2) > order.get(status_b, 2):
            return "site_b"
        return "tie"

    def get_site_detail(self, site_id: int) -> Optional[SiteRanking]:
        """Get detailed information for a specific site."""
        rankings = self.get_rankings(
            include_trial_history=True,
            limit=1000,
        )
        return next((r for r in rankings if r.site_id == site_id), None)

    def get_top_sites_by_metric(
        self,
        metric: str,
        therapeutic_area: Optional[str] = None,
        limit: int = 10,
    ) -> List[SiteRanking]:
        """
        Get top sites by a specific metric.

        Args:
            metric: One of "enrollment", "completion", "speed", "experience", "diversity"
            therapeutic_area: Optional therapeutic area filter
            limit: Number of sites to return

        Returns:
            List of SiteRanking sorted by the specified metric
        """
        rankings = self.get_rankings(
            therapeutic_area=therapeutic_area,
            include_trial_history=False,
            limit=200,
        )

        metric_map = {
            "enrollment": lambda x: x.enrollment_score,
            "completion": lambda x: x.completion_score,
            "speed": lambda x: x.speed_score,
            "experience": lambda x: x.experience_score,
            "diversity": lambda x: x.diversity_score,
            "overall": lambda x: x.overall_score,
        }

        sort_key = metric_map.get(metric, metric_map["overall"])
        rankings.sort(key=sort_key, reverse=True)

        # Re-assign ranks
        for i, r in enumerate(rankings[:limit], 1):
            r.rank = i

        return rankings[:limit]


def get_site_leaderboard(
    db_manager,
    therapeutic_area: Optional[str] = None,
    country: str = "United States",
    limit: int = 20,
) -> List[SiteRanking]:
    """Convenience function for getting site leaderboard."""
    leaderboard = SiteLeaderboard(db_manager)
    return leaderboard.get_rankings(
        therapeutic_area=therapeutic_area,
        country=country,
        include_trial_history=True,
        limit=limit,
    )
