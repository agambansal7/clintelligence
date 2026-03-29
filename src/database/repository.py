"""Repository layer for data access in TrialIntel."""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from sqlalchemy import func, and_, or_, desc, text
from sqlalchemy.orm import Session

from .models import Trial, Site, Investigator, Endpoint, TrialBenchmark
from ..utils.condition_normalizer import (
    normalize_condition,
    get_condition_variants,
    get_search_terms,
)

logger = logging.getLogger(__name__)


def _build_condition_filter(column, condition: str):
    """Build a filter that matches any variant of a condition."""
    variants = get_condition_variants(condition)
    search_terms = get_search_terms(condition)

    # Build OR conditions for all variants and search terms
    conditions = []
    for variant in variants:
        conditions.append(column.ilike(f"%{variant}%"))
    for term in search_terms:
        if term not in [v.lower() for v in variants]:
            conditions.append(column.ilike(f"%{term}%"))

    return or_(*conditions) if conditions else column.ilike(f"%{condition}%")


class TrialRepository:
    """Repository for trial data access."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_nct_id(self, nct_id: str) -> Optional[Trial]:
        """Get a trial by NCT ID."""
        return self.session.query(Trial).filter(Trial.nct_id == nct_id).first()

    def get_many(
        self,
        nct_ids: List[str] = None,
        conditions: List[str] = None,
        therapeutic_area: str = None,
        phase: str = None,
        status: str = None,
        sponsor: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trial]:
        """Query trials with filters."""
        query = self.session.query(Trial)

        if nct_ids:
            query = query.filter(Trial.nct_id.in_(nct_ids))

        if conditions:
            # Search for any of the conditions
            conditions_filter = or_(*[
                Trial.conditions.ilike(f"%{cond}%") for cond in conditions
            ])
            query = query.filter(conditions_filter)

        if therapeutic_area:
            query = query.filter(Trial.therapeutic_area.ilike(f"%{therapeutic_area}%"))

        if phase:
            query = query.filter(Trial.phase.ilike(f"%{phase}%"))

        if status:
            query = query.filter(Trial.status == status)

        if sponsor:
            query = query.filter(Trial.sponsor.ilike(f"%{sponsor}%"))

        return query.offset(offset).limit(limit).all()

    def find_similar_trials(
        self,
        condition: str,
        phase: str,
        status: str = "COMPLETED",
        limit: int = 10,
    ) -> List[Trial]:
        """Find similar completed trials for benchmarking."""
        # Build condition filter that matches all variants
        condition_filter = or_(
            _build_condition_filter(Trial.conditions, condition),
            _build_condition_filter(Trial.therapeutic_area, condition),
        )

        query = self.session.query(Trial).filter(
            and_(
                condition_filter,
                Trial.phase.ilike(f"%{phase}%"),
                Trial.status == status,
                Trial.enrollment.isnot(None),
                Trial.enrollment > 0,
            )
        ).order_by(desc(Trial.completion_date)).limit(limit)

        return query.all()

    def get_benchmark_trials(
        self,
        condition: str,
        phase: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get trials for benchmarking with summary stats."""
        trials = self.find_similar_trials(condition, phase, limit=limit)

        return [
            {
                "nct_id": t.nct_id,
                "title": t.title,
                "enrollment": t.enrollment,
                "num_sites": t.num_sites,
                "start_date": t.start_date,
                "completion_date": t.completion_date,
                "status": t.status,
                "sponsor": t.sponsor,
            }
            for t in trials
        ]

    def get_historical_stats(
        self,
        therapeutic_area: str = None,
        phase: str = None,
    ) -> Dict[str, Any]:
        """Get historical statistics for a therapeutic area and phase."""
        query = self.session.query(Trial)

        if therapeutic_area:
            condition_filter = or_(
                _build_condition_filter(Trial.therapeutic_area, therapeutic_area),
                _build_condition_filter(Trial.conditions, therapeutic_area),
            )
            query = query.filter(condition_filter)

        if phase:
            query = query.filter(Trial.phase.ilike(f"%{phase}%"))

        # Get counts by status
        total = query.count()
        completed = query.filter(Trial.status == "COMPLETED").count()
        terminated = query.filter(Trial.status.in_(["TERMINATED", "WITHDRAWN", "SUSPENDED"])).count()
        active = query.filter(Trial.status.in_(["RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"])).count()

        # Get enrollment stats
        enrollment_stats = self.session.query(
            func.avg(Trial.enrollment).label("avg_enrollment"),
            func.min(Trial.enrollment).label("min_enrollment"),
            func.max(Trial.enrollment).label("max_enrollment"),
        ).filter(
            Trial.enrollment.isnot(None),
            Trial.enrollment > 0,
        )

        if therapeutic_area:
            enrollment_condition_filter = or_(
                _build_condition_filter(Trial.therapeutic_area, therapeutic_area),
                _build_condition_filter(Trial.conditions, therapeutic_area),
            )
            enrollment_stats = enrollment_stats.filter(enrollment_condition_filter)
        if phase:
            enrollment_stats = enrollment_stats.filter(Trial.phase.ilike(f"%{phase}%"))

        enrollment_result = enrollment_stats.first()

        # Calculate rates
        termination_rate = terminated / total if total > 0 else 0.0
        completion_rate = completed / total if total > 0 else 0.0

        return {
            "total_trials": total,
            "completed_trials": completed,
            "terminated_trials": terminated,
            "active_trials": active,
            "termination_rate": round(termination_rate, 3),
            "completion_rate": round(completion_rate, 3),
            "avg_enrollment": round(enrollment_result.avg_enrollment or 0, 1),
            "min_enrollment": enrollment_result.min_enrollment or 0,
            "max_enrollment": enrollment_result.max_enrollment or 0,
        }

    def get_terminated_trials_with_reasons(
        self,
        therapeutic_area: str = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get terminated trials with their stop reasons."""
        query = self.session.query(Trial).filter(
            Trial.status.in_(["TERMINATED", "WITHDRAWN", "SUSPENDED"]),
            Trial.why_stopped.isnot(None),
            Trial.why_stopped != "",
        )

        if therapeutic_area:
            condition_filter = or_(
                _build_condition_filter(Trial.therapeutic_area, therapeutic_area),
                _build_condition_filter(Trial.conditions, therapeutic_area),
            )
            query = query.filter(condition_filter)

        trials = query.limit(limit).all()

        return [
            {
                "nct_id": t.nct_id,
                "title": t.title,
                "status": t.status,
                "why_stopped": t.why_stopped,
                "phase": t.phase,
                "enrollment": t.enrollment,
                "sponsor": t.sponsor,
            }
            for t in trials
        ]

    def count_by_criteria(self, criteria: Dict[str, Any]) -> int:
        """Count trials matching given criteria."""
        query = self.session.query(func.count(Trial.nct_id))

        for key, value in criteria.items():
            if hasattr(Trial, key) and value is not None:
                query = query.filter(getattr(Trial, key) == value)

        return query.scalar() or 0

    def bulk_insert(self, trials: List[Dict[str, Any]]) -> int:
        """Bulk insert or update trials."""
        count = 0
        for trial_data in trials:
            existing = self.get_by_nct_id(trial_data.get("nct_id"))
            if existing:
                for key, value in trial_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                trial = Trial(**trial_data)
                self.session.add(trial)
            count += 1

        self.session.flush()
        return count


class SiteRepository:
    """Repository for site data access."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, site_id: int) -> Optional[Site]:
        """Get a site by ID."""
        return self.session.query(Site).filter(Site.id == site_id).first()

    def find_by_location(
        self,
        facility_name: str = None,
        city: str = None,
        country: str = None,
    ) -> List[Site]:
        """Find sites by location."""
        query = self.session.query(Site)

        if facility_name:
            query = query.filter(Site.facility_name.ilike(f"%{facility_name}%"))
        if city:
            query = query.filter(Site.city.ilike(f"%{city}%"))
        if country:
            query = query.filter(Site.country.ilike(f"%{country}%"))

        return query.all()

    def get_top_sites(
        self,
        therapeutic_area: str = None,
        country: str = None,
        min_trials: int = 5,
        limit: int = 20,
    ) -> List[Site]:
        """Get top performing sites."""
        query = self.session.query(Site).filter(
            Site.total_trials >= min_trials
        )

        if therapeutic_area:
            query = query.filter(_build_condition_filter(Site.therapeutic_areas, therapeutic_area))

        if country:
            query = query.filter(Site.country.ilike(f"%{country}%"))

        return query.order_by(desc(Site.experience_score)).limit(limit).all()

    def recommend_sites(
        self,
        therapeutic_area: str,
        target_enrollment: int,
        countries: List[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Recommend sites for a new trial."""
        query = self.session.query(Site).filter(
            Site.total_trials >= 3,
            _build_condition_filter(Site.therapeutic_areas, therapeutic_area),
        )

        if countries:
            query = query.filter(Site.country.in_(countries))

        # Order by experience score and enrollment velocity
        sites = query.order_by(
            desc(Site.experience_score),
            desc(Site.enrollment_velocity),
        ).limit(limit * 2).all()

        recommendations = []
        for site in sites[:limit]:
            # Calculate match score
            match_score = 0

            # Therapeutic area match
            if therapeutic_area.lower() in (site.therapeutic_areas or "").lower():
                match_score += 40

            # Experience bonus
            if site.total_trials >= 10:
                match_score += 25
            elif site.total_trials >= 5:
                match_score += 15

            # Completion rate bonus
            if site.completion_rate and site.completion_rate > 0.7:
                match_score += 20

            # Velocity bonus
            if site.enrollment_velocity and site.enrollment_velocity > 2:
                match_score += 15

            # Project enrollment
            velocity = site.enrollment_velocity or 1.0
            projected_months = target_enrollment / (velocity * 10) if velocity > 0 else 24

            recommendations.append({
                "site": site.to_dict(),
                "match_score": min(match_score, 100),
                "projected_enrollment_months": round(projected_months, 1),
                "reasons": self._generate_recommendation_reasons(site, therapeutic_area),
            })

        # Sort by match score
        recommendations.sort(key=lambda x: x["match_score"], reverse=True)
        return recommendations

    def _generate_recommendation_reasons(self, site: Site, therapeutic_area: str) -> List[str]:
        """Generate human-readable reasons for site recommendation."""
        reasons = []

        if site.total_trials >= 10:
            reasons.append(f"Extensive experience with {site.total_trials} trials")
        elif site.total_trials >= 5:
            reasons.append(f"Solid experience with {site.total_trials} trials")

        if site.completion_rate and site.completion_rate > 0.8:
            reasons.append(f"High completion rate ({site.completion_rate:.0%})")

        if site.enrollment_velocity and site.enrollment_velocity > 2:
            reasons.append(f"Strong enrollment velocity ({site.enrollment_velocity:.1f} patients/month)")

        if therapeutic_area.lower() in (site.therapeutic_areas or "").lower():
            reasons.append(f"Experience in {therapeutic_area}")

        return reasons

    def aggregate_from_trials(self, trials: List[Trial]) -> int:
        """Aggregate site data from trial locations."""
        site_data = defaultdict(lambda: {
            "total_trials": 0,
            "completed_trials": 0,
            "terminated_trials": 0,
            "total_enrollment": 0,
            "therapeutic_areas": set(),
        })

        for trial in trials:
            locations = trial.locations_list
            if not locations:
                continue

            enrollment_per_site = (trial.enrollment or 0) / max(len(locations), 1)

            for loc in locations:
                key = (
                    loc.get("facility", "Unknown"),
                    loc.get("city", ""),
                    loc.get("country", ""),
                )

                site_data[key]["total_trials"] += 1
                site_data[key]["total_enrollment"] += enrollment_per_site
                site_data[key]["city"] = loc.get("city", "")
                site_data[key]["state"] = loc.get("state", "")
                site_data[key]["country"] = loc.get("country", "")

                if trial.status == "COMPLETED":
                    site_data[key]["completed_trials"] += 1
                elif trial.status in ["TERMINATED", "WITHDRAWN", "SUSPENDED"]:
                    site_data[key]["terminated_trials"] += 1

                if trial.therapeutic_area:
                    # Store both normalized and original for better matching
                    normalized = normalize_condition(trial.therapeutic_area)
                    site_data[key]["therapeutic_areas"].add(normalized)
                    site_data[key]["therapeutic_areas"].add(trial.therapeutic_area.lower())

        # Upsert sites
        count = 0
        for (facility, city, country), data in site_data.items():
            if not facility or facility == "Unknown":
                continue

            existing = self.session.query(Site).filter(
                Site.facility_name == facility,
                Site.city == city,
                Site.country == country,
            ).first()

            total_trials = data["total_trials"]
            completed = data["completed_trials"]
            completion_rate = completed / total_trials if total_trials > 0 else 0
            avg_enrollment = data["total_enrollment"] / total_trials if total_trials > 0 else 0
            experience_score = min(100, (total_trials * 5) + (completion_rate * 50))

            if existing:
                existing.total_trials = total_trials
                existing.completed_trials = completed
                existing.terminated_trials = data["terminated_trials"]
                existing.total_enrollment = int(data["total_enrollment"])
                existing.avg_enrollment = avg_enrollment
                existing.completion_rate = completion_rate
                existing.experience_score = experience_score
                existing.therapeutic_areas = json.dumps(list(data["therapeutic_areas"]))
                existing.updated_at = datetime.utcnow()
            else:
                site = Site(
                    facility_name=facility,
                    city=city,
                    state=data.get("state", ""),
                    country=country,
                    total_trials=total_trials,
                    completed_trials=completed,
                    terminated_trials=data["terminated_trials"],
                    total_enrollment=int(data["total_enrollment"]),
                    avg_enrollment=avg_enrollment,
                    completion_rate=completion_rate,
                    experience_score=experience_score,
                    therapeutic_areas=json.dumps(list(data["therapeutic_areas"])),
                )
                self.session.add(site)

            count += 1

        self.session.flush()
        return count


class EndpointRepository:
    """Repository for endpoint data access."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_measure(self, measure_normalized: str) -> Optional[Endpoint]:
        """Get an endpoint by normalized measure name."""
        return self.session.query(Endpoint).filter(
            Endpoint.measure_normalized == measure_normalized
        ).first()

    def get_top_endpoints(
        self,
        therapeutic_area: str = None,
        category: str = None,
        as_primary: bool = True,
        limit: int = 20,
    ) -> List[Endpoint]:
        """Get most frequently used endpoints."""
        query = self.session.query(Endpoint)

        if therapeutic_area:
            query = query.filter(_build_condition_filter(Endpoint.therapeutic_areas, therapeutic_area))

        if category:
            query = query.filter(Endpoint.measure_category == category)

        if as_primary:
            query = query.filter(Endpoint.as_primary > 0)
            query = query.order_by(desc(Endpoint.as_primary))
        else:
            query = query.order_by(desc(Endpoint.frequency))

        return query.limit(limit).all()

    def get_endpoint_analysis(
        self,
        condition: str,
    ) -> Dict[str, Any]:
        """Get comprehensive endpoint analysis for a condition."""
        # Get primary endpoints
        primary = self.get_top_endpoints(
            therapeutic_area=condition,
            as_primary=True,
            limit=10,
        )

        # Get secondary endpoints
        secondary = self.session.query(Endpoint).filter(
            _build_condition_filter(Endpoint.therapeutic_areas, condition),
            Endpoint.as_secondary > 0,
        ).order_by(desc(Endpoint.as_secondary)).limit(10).all()

        # Build recommendations
        recommendations = []
        for ep in primary[:5]:
            confidence = "high" if ep.success_rate and ep.success_rate > 0.7 else "medium"
            recommendations.append({
                "endpoint": ep.measure_normalized,
                "confidence": confidence,
                "frequency": ep.frequency,
                "success_rate": ep.success_rate,
                "typical_timeframes": json.loads(ep.typical_timeframes) if ep.typical_timeframes else [],
            })

        return {
            "condition": condition,
            "total_trials_analyzed": sum(ep.frequency for ep in primary),
            "primary_endpoints": [ep.to_dict() for ep in primary],
            "secondary_endpoints": [ep.to_dict() for ep in secondary],
            "recommendations": recommendations,
        }

    def aggregate_from_trials(self, trials: List[Trial]) -> int:
        """Aggregate endpoint data from trials."""
        endpoint_data = defaultdict(lambda: {
            "raw_examples": [],
            "frequency": 0,
            "as_primary": 0,
            "as_secondary": 0,
            "trials_completed": 0,
            "trials_terminated": 0,
            "therapeutic_areas": set(),
            "phases": set(),
            "timeframes": set(),
        })

        for trial in trials:
            # Process primary outcomes
            for outcome in trial.primary_outcomes_list:
                measure = outcome.get("measure", "")
                if not measure:
                    continue

                normalized = self._normalize_endpoint(measure)
                if not normalized:
                    continue

                data = endpoint_data[normalized]
                data["frequency"] += 1
                data["as_primary"] += 1

                if len(data["raw_examples"]) < 5:
                    data["raw_examples"].append(measure)

                if trial.status == "COMPLETED":
                    data["trials_completed"] += 1
                elif trial.status in ["TERMINATED", "WITHDRAWN"]:
                    data["trials_terminated"] += 1

                if trial.therapeutic_area:
                    # Store both normalized and original for better matching
                    normalized = normalize_condition(trial.therapeutic_area)
                    data["therapeutic_areas"].add(normalized)
                    data["therapeutic_areas"].add(trial.therapeutic_area.lower())
                if trial.phase:
                    data["phases"].add(trial.phase)
                if outcome.get("timeFrame"):
                    data["timeframes"].add(outcome["timeFrame"])

            # Process secondary outcomes
            for outcome in trial.secondary_outcomes_list:
                measure = outcome.get("measure", "")
                if not measure:
                    continue

                normalized = self._normalize_endpoint(measure)
                if not normalized:
                    continue

                data = endpoint_data[normalized]
                data["frequency"] += 1
                data["as_secondary"] += 1

                if len(data["raw_examples"]) < 5:
                    data["raw_examples"].append(measure)

                if trial.therapeutic_area:
                    # Store both normalized and original for better matching
                    normalized = normalize_condition(trial.therapeutic_area)
                    data["therapeutic_areas"].add(normalized)
                    data["therapeutic_areas"].add(trial.therapeutic_area.lower())
                if trial.phase:
                    data["phases"].add(trial.phase)
                if outcome.get("timeFrame"):
                    data["timeframes"].add(outcome["timeFrame"])

        # Upsert endpoints
        count = 0
        for normalized, data in endpoint_data.items():
            existing = self.get_by_measure(normalized)

            total = data["trials_completed"] + data["trials_terminated"]
            success_rate = data["trials_completed"] / total if total > 0 else None
            category = self._categorize_endpoint(normalized)

            if existing:
                existing.frequency = data["frequency"]
                existing.as_primary = data["as_primary"]
                existing.as_secondary = data["as_secondary"]
                existing.trials_completed = data["trials_completed"]
                existing.trials_terminated = data["trials_terminated"]
                existing.success_rate = success_rate
                existing.raw_examples = json.dumps(data["raw_examples"])
                existing.therapeutic_areas = json.dumps(list(data["therapeutic_areas"]))
                existing.phases = json.dumps(list(data["phases"]))
                existing.typical_timeframes = json.dumps(list(data["timeframes"])[:10])
                existing.updated_at = datetime.utcnow()
            else:
                endpoint = Endpoint(
                    measure_normalized=normalized,
                    measure_category=category,
                    frequency=data["frequency"],
                    as_primary=data["as_primary"],
                    as_secondary=data["as_secondary"],
                    trials_completed=data["trials_completed"],
                    trials_terminated=data["trials_terminated"],
                    success_rate=success_rate,
                    raw_examples=json.dumps(data["raw_examples"]),
                    therapeutic_areas=json.dumps(list(data["therapeutic_areas"])),
                    phases=json.dumps(list(data["phases"])),
                    typical_timeframes=json.dumps(list(data["timeframes"])[:10]),
                )
                self.session.add(endpoint)

            count += 1

        self.session.flush()
        return count

    def _normalize_endpoint(self, measure: str) -> Optional[str]:
        """Normalize an endpoint measure to a canonical form."""
        measure_lower = measure.lower()

        # Endpoint normalization patterns
        patterns = [
            (r"hba1c|hemoglobin a1c|glycated hemoglobin", "hba1c_change"),
            (r"fasting plasma glucose|fpg|fasting glucose", "fasting_glucose"),
            (r"overall survival|os\b", "overall_survival"),
            (r"progression.free survival|pfs", "progression_free_survival"),
            (r"disease.free survival|dfs", "disease_free_survival"),
            (r"event.free survival|efs", "event_free_survival"),
            (r"objective response rate|orr|overall response", "objective_response_rate"),
            (r"complete response|complete remission|\bcr\b", "complete_response"),
            (r"partial response|\bpr\b", "partial_response"),
            (r"pathological complete response|pcr", "pathological_complete_response"),
            (r"body weight|weight change|weight loss", "body_weight_change"),
            (r"blood pressure|systolic|diastolic|sbp|dbp", "blood_pressure"),
            (r"ldl.c|ldl cholesterol", "ldl_cholesterol"),
            (r"major adverse cardiovascular|mace", "mace"),
            (r"myocardial infarction|\bmi\b", "myocardial_infarction"),
            (r"adverse event|safety|tolerability|ae\b", "adverse_events"),
            (r"serious adverse event|sae", "serious_adverse_events"),
            (r"quality of life|qol|sf-36|eq-5d", "quality_of_life"),
            (r"pain score|vas|visual analog", "pain_score"),
            (r"acr20|acr50|acr70", "acr_response"),
            (r"pasi\s*\d+|psoriasis area", "pasi_response"),
            (r"edss|expanded disability", "edss_score"),
            (r"relapse rate|annualized relapse", "relapse_rate"),
        ]

        import re
        for pattern, normalized in patterns:
            if re.search(pattern, measure_lower):
                return normalized

        # If no pattern matches, return None (skip unknown endpoints)
        return None

    def _categorize_endpoint(self, normalized: str) -> str:
        """Categorize an endpoint."""
        efficacy = [
            "hba1c_change", "fasting_glucose", "overall_survival",
            "progression_free_survival", "disease_free_survival",
            "objective_response_rate", "complete_response", "partial_response",
            "pathological_complete_response", "body_weight_change",
            "blood_pressure", "ldl_cholesterol", "acr_response",
            "pasi_response", "edss_score", "relapse_rate",
        ]
        safety = [
            "adverse_events", "serious_adverse_events", "mace",
            "myocardial_infarction",
        ]
        qol = ["quality_of_life", "pain_score"]

        if normalized in efficacy:
            return "efficacy"
        elif normalized in safety:
            return "safety"
        elif normalized in qol:
            return "quality_of_life"
        else:
            return "other"


class BenchmarkRepository:
    """Repository for pre-computed benchmarks."""

    def __init__(self, session: Session):
        self.session = session

    def get_benchmark(
        self,
        therapeutic_area: str,
        phase: str,
    ) -> Optional[TrialBenchmark]:
        """Get benchmark for a therapeutic area and phase."""
        return self.session.query(TrialBenchmark).filter(
            TrialBenchmark.therapeutic_area == therapeutic_area,
            TrialBenchmark.phase == phase,
        ).first()

    def compute_benchmarks(self, trial_repo: TrialRepository) -> int:
        """Compute and store benchmarks from trial data."""
        # Get unique therapeutic areas and phases
        result = self.session.execute(text("""
            SELECT DISTINCT therapeutic_area, phase
            FROM trials
            WHERE therapeutic_area IS NOT NULL
            AND phase IS NOT NULL
            AND phase != ''
        """))

        combinations = result.fetchall()
        count = 0

        for therapeutic_area, phase in combinations:
            stats = trial_repo.get_historical_stats(therapeutic_area, phase)

            if stats["total_trials"] < 10:
                continue

            # Get sample NCT IDs
            samples = trial_repo.find_similar_trials(therapeutic_area, phase, limit=5)
            sample_ids = [t.nct_id for t in samples]

            existing = self.get_benchmark(therapeutic_area, phase)

            if existing:
                existing.total_trials = stats["total_trials"]
                existing.completed_trials = stats["completed_trials"]
                existing.terminated_trials = stats["terminated_trials"]
                existing.termination_rate = stats["termination_rate"]
                existing.avg_enrollment = stats["avg_enrollment"]
                existing.sample_nct_ids = json.dumps(sample_ids)
                existing.computed_at = datetime.utcnow()
            else:
                benchmark = TrialBenchmark(
                    therapeutic_area=therapeutic_area,
                    phase=phase,
                    total_trials=stats["total_trials"],
                    completed_trials=stats["completed_trials"],
                    terminated_trials=stats["terminated_trials"],
                    termination_rate=stats["termination_rate"],
                    avg_enrollment=stats["avg_enrollment"],
                    sample_nct_ids=json.dumps(sample_ids),
                )
                self.session.add(benchmark)

            count += 1

        self.session.flush()
        return count
