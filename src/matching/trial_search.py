"""
Trial Search Module

Searches for recruiting clinical trials by condition/intervention.
Filters to only actionable trials (RECRUITING, NOT_YET_RECRUITING).
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class TrialSearchResult:
    """A trial from search results."""
    nct_id: str
    title: str
    condition: str  # Maps to 'conditions' column in DB
    phase: str
    status: str
    enrollment: int
    interventions: str  # Maps to 'interventions' column in DB
    therapeutic_area: str  # Maps to 'therapeutic_area' column in DB
    eligibility_criteria: str
    min_age: Optional[int]
    max_age: Optional[int]
    sex: str
    locations: List[Dict[str, Any]] = field(default_factory=list)
    similarity_score: float = 0.0


class TrialSearcher:
    """
    Searches for recruiting clinical trials.

    Combines:
    1. SQL filtering (status, age, location)
    2. Semantic search (condition matching via embeddings)
    """

    # Only these statuses are actionable for patient matching
    RECRUITING_STATUSES = ('RECRUITING', 'NOT_YET_RECRUITING', 'ENROLLING_BY_INVITATION')

    def __init__(self, db_manager, vector_store=None):
        """
        Initialize the searcher.

        Args:
            db_manager: Database manager instance
            vector_store: Optional vector store for semantic search
        """
        self.db = db_manager
        self.vector_store = vector_store

    def search_by_condition(
        self,
        condition: str,
        max_results: int = 100,
        location_country: str = None,
        location_state: str = None,
        min_age: int = None,
        max_age: int = None,
        phase: str = None,
        use_semantic: bool = True
    ) -> List[TrialSearchResult]:
        """
        Search for recruiting trials matching a condition.

        Args:
            condition: The condition/disease to search for
            max_results: Maximum number of results to return
            location_country: Filter by country (e.g., "United States")
            location_state: Filter by state (e.g., "California")
            min_age: Patient's age (filter trials that accept this age)
            max_age: Patient's age (filter trials that accept this age)
            phase: Filter by phase (e.g., "PHASE3")
            use_semantic: Whether to use semantic search (if available)

        Returns:
            List of matching trials
        """

        # If we have a vector store and semantic search is enabled, use it
        if use_semantic and self.vector_store:
            return self._semantic_search(
                condition, max_results, location_country,
                location_state, min_age, max_age, phase
            )

        # Otherwise fall back to SQL search
        return self._sql_search(
            condition, max_results, location_country,
            location_state, min_age, max_age, phase
        )

    def _sql_search(
        self,
        condition: str,
        max_results: int,
        location_country: str = None,
        location_state: str = None,
        patient_age: int = None,
        max_age: int = None,
        phase: str = None
    ) -> List[TrialSearchResult]:
        """SQL-based search with LIKE matching."""

        # Build status list for IN clause
        status_list = "', '".join(self.RECRUITING_STATUSES)

        query = f"""
        SELECT DISTINCT
            t.nct_id,
            t.title,
            t.conditions,
            t.phase,
            t.status,
            t.enrollment,
            t.interventions,
            t.therapeutic_area,
            t.eligibility_criteria,
            t.min_age,
            t.max_age,
            t.sex
        FROM trials t
        WHERE t.status IN ('{status_list}')
        AND (
            LOWER(t.conditions) LIKE LOWER(:condition_pattern)
            OR LOWER(t.title) LIKE LOWER(:condition_pattern)
            OR LOWER(t.interventions) LIKE LOWER(:condition_pattern)
            OR LOWER(t.therapeutic_area) LIKE LOWER(:condition_pattern)
        )
        """

        params = {
            'condition_pattern': f'%{condition}%'
        }

        # Add age filter if provided
        if patient_age is not None:
            query += """
            AND (t.min_age IS NULL OR t.min_age <= :patient_age)
            AND (t.max_age IS NULL OR t.max_age >= :patient_age)
            """
            params['patient_age'] = patient_age

        # Add phase filter if provided
        if phase:
            query += " AND t.phase = :phase"
            params['phase'] = phase

        query += f" LIMIT {max_results}"

        with self.db.engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()

        trials = []
        for row in rows:
            trial = TrialSearchResult(
                nct_id=row[0],
                title=row[1],
                condition=row[2] or '',
                phase=row[3] or '',
                status=row[4],
                enrollment=row[5] or 0,
                interventions=row[6] or '',
                therapeutic_area=row[7] or '',
                eligibility_criteria=row[8] or '',
                min_age=row[9],
                max_age=row[10],
                sex=row[11] or 'All'
            )

            # Get locations for this trial
            trial.locations = self._get_trial_locations(trial.nct_id, location_country, location_state)

            # Only include if has matching locations (when location filter is applied)
            if location_country or location_state:
                if trial.locations:
                    trials.append(trial)
            else:
                trials.append(trial)

        return trials

    def _semantic_search(
        self,
        condition: str,
        max_results: int,
        location_country: str = None,
        location_state: str = None,
        patient_age: int = None,
        max_age: int = None,
        phase: str = None
    ) -> List[TrialSearchResult]:
        """Semantic search using vector embeddings."""

        try:
            # Search vector store - filter for recruiting statuses
            # Note: ChromaDB where clause syntax
            semantic_results = self.vector_store.search(
                query=condition,
                n_results=max_results * 3,
                filter_dict={"status": {"$in": list(self.RECRUITING_STATUSES)}}
            )

            if not semantic_results:
                logger.warning("No semantic results, falling back to SQL")
                return self._sql_search(
                    condition, max_results, location_country,
                    location_state, patient_age, max_age, phase
                )

            # Get full trial details from database
            # VectorSearchResult has .nct_id attribute
            nct_ids = [r.nct_id for r in semantic_results]
            trials = self._get_trials_by_ids(nct_ids)

            # Create a score lookup from semantic results
            score_lookup = {r.nct_id: r.score for r in semantic_results}

            # Apply additional filters
            filtered_trials = []
            for trial in trials:
                # Age filter
                if patient_age is not None:
                    if trial.min_age and trial.min_age > patient_age:
                        continue
                    if trial.max_age and trial.max_age < patient_age:
                        continue

                # Phase filter
                if phase and trial.phase != phase:
                    continue

                # Get locations
                trial.locations = self._get_trial_locations(
                    trial.nct_id, location_country, location_state
                )

                # Location filter
                if location_country or location_state:
                    if not trial.locations:
                        continue

                # Add similarity score from semantic search
                trial.similarity_score = score_lookup.get(trial.nct_id, 0)

                filtered_trials.append(trial)

                if len(filtered_trials) >= max_results:
                    break

            return filtered_trials

        except Exception as e:
            logger.error(f"Semantic search failed: {e}, falling back to SQL")
            return self._sql_search(
                condition, max_results, location_country,
                location_state, patient_age, max_age, phase
            )

    def _get_trials_by_ids(self, nct_ids: List[str]) -> List[TrialSearchResult]:
        """Get full trial details by NCT IDs."""

        if not nct_ids:
            return []

        placeholders = ','.join([f':id{i}' for i in range(len(nct_ids))])
        query = f"""
        SELECT
            nct_id, title, conditions, phase, status, enrollment,
            interventions, therapeutic_area, eligibility_criteria,
            min_age, max_age, sex
        FROM trials
        WHERE nct_id IN ({placeholders})
        """

        params = {f'id{i}': nct_id for i, nct_id in enumerate(nct_ids)}

        with self.db.engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()

        # Maintain order from semantic search
        trials_dict = {}
        for row in rows:
            trials_dict[row[0]] = TrialSearchResult(
                nct_id=row[0],
                title=row[1],
                condition=row[2] or '',
                phase=row[3] or '',
                status=row[4],
                enrollment=row[5] or 0,
                interventions=row[6] or '',
                therapeutic_area=row[7] or '',
                eligibility_criteria=row[8] or '',
                min_age=row[9],
                max_age=row[10],
                sex=row[11] or 'All'
            )

        return [trials_dict[nct_id] for nct_id in nct_ids if nct_id in trials_dict]

    def _get_trial_locations(
        self,
        nct_id: str,
        country: str = None,
        state: str = None
    ) -> List[Dict[str, Any]]:
        """Get locations for a trial, optionally filtered."""

        # Try with lat/lon first, fall back to without if columns don't exist
        queries = [
            """
            SELECT
                facility_name,
                city,
                state,
                country,
                zip_code,
                latitude,
                longitude,
                contact_name,
                contact_phone,
                contact_email
            FROM trial_locations
            WHERE nct_id = :nct_id
            """,
            """
            SELECT
                facility_name,
                city,
                state,
                country,
                zip_code,
                NULL as latitude,
                NULL as longitude,
                contact_name,
                contact_phone,
                contact_email
            FROM trial_locations
            WHERE nct_id = :nct_id
            """
        ]

        params = {'nct_id': nct_id}

        for query in queries:
            full_query = query
            if country:
                full_query += " AND LOWER(country) = LOWER(:country)"
                params['country'] = country

            if state:
                full_query += " AND LOWER(state) = LOWER(:state)"
                params['state'] = state

            try:
                with self.db.engine.connect() as conn:
                    result = conn.execute(text(full_query), params)
                    rows = result.fetchall()

                locations = []
                for row in rows:
                    locations.append({
                        'facility_name': row[0],
                        'city': row[1],
                        'state': row[2],
                        'country': row[3],
                        'zip_code': row[4],
                        'latitude': row[5],
                        'longitude': row[6],
                        'contact_name': row[7],
                        'contact_phone': row[8],
                        'contact_email': row[9]
                    })

                return locations

            except Exception as e:
                # If first query fails (likely missing columns), try the second
                if "latitude" in str(e) or "longitude" in str(e):
                    continue
                logger.warning(f"Could not get locations for {nct_id}: {e}")
                return []

        return []

    def get_recruiting_trial_count(self, condition: str = None) -> int:
        """Get count of recruiting trials, optionally filtered by condition."""

        status_list = "', '".join(self.RECRUITING_STATUSES)

        query = f"""
        SELECT COUNT(*) FROM trials
        WHERE status IN ('{status_list}')
        """
        params = {}

        if condition:
            query += " AND (LOWER(conditions) LIKE LOWER(:pattern) OR LOWER(title) LIKE LOWER(:pattern))"
            params['pattern'] = f'%{condition}%'

        with self.db.engine.connect() as conn:
            result = conn.execute(text(query), params)
            return result.scalar()

    def get_common_conditions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most common conditions in recruiting trials."""

        status_list = "', '".join(self.RECRUITING_STATUSES)

        query = f"""
        SELECT conditions, COUNT(*) as count
        FROM trials
        WHERE status IN ('{status_list}')
        AND conditions IS NOT NULL AND conditions != ''
        GROUP BY conditions
        ORDER BY count DESC
        LIMIT :limit
        """

        with self.db.engine.connect() as conn:
            result = conn.execute(text(query), {'limit': limit})
            rows = result.fetchall()

        return [{'condition': row[0], 'count': row[1]} for row in rows]
