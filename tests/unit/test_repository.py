"""Unit tests for database repository."""

import pytest
import json
from src.database import DatabaseManager, Trial
from src.database.repository import (
    TrialRepository,
    SiteRepository,
    EndpointRepository,
    _build_condition_filter,
)


class TestBuildConditionFilter:
    """Tests for _build_condition_filter helper."""

    def test_builds_or_conditions(self, db_session):
        """Test that filter builds OR conditions for variants."""
        filter_expr = _build_condition_filter(Trial.conditions, "diabetes")
        # Should be an OR expression
        assert filter_expr is not None


class TestTrialRepository:
    """Tests for TrialRepository."""

    def test_get_by_nct_id(self, populated_db):
        """Test getting trial by NCT ID."""
        with populated_db.session() as session:
            repo = TrialRepository(session)
            trial = repo.get_by_nct_id("NCT12345678")
            assert trial is not None
            assert trial.nct_id == "NCT12345678"
            assert trial.therapeutic_area == "diabetes"

    def test_get_by_nct_id_not_found(self, populated_db):
        """Test getting non-existent trial."""
        with populated_db.session() as session:
            repo = TrialRepository(session)
            trial = repo.get_by_nct_id("NCT00000000")
            assert trial is None

    def test_get_many_no_filters(self, populated_db):
        """Test getting trials without filters."""
        with populated_db.session() as session:
            repo = TrialRepository(session)
            trials = repo.get_many(limit=10)
            assert len(trials) == 2

    def test_get_many_by_therapeutic_area(self, populated_db):
        """Test filtering by therapeutic area."""
        with populated_db.session() as session:
            repo = TrialRepository(session)
            trials = repo.get_many(therapeutic_area="diabetes", limit=10)
            assert len(trials) >= 1
            assert any(t.therapeutic_area == "diabetes" for t in trials)

    def test_get_many_by_phase(self, populated_db):
        """Test filtering by phase."""
        with populated_db.session() as session:
            repo = TrialRepository(session)
            trials = repo.get_many(phase="PHASE3", limit=10)
            assert len(trials) == 2  # Both sample trials are PHASE3

    def test_get_many_by_status(self, populated_db):
        """Test filtering by status."""
        with populated_db.session() as session:
            repo = TrialRepository(session)

            completed = repo.get_many(status="COMPLETED", limit=10)
            assert len(completed) == 1
            assert completed[0].status == "COMPLETED"

            recruiting = repo.get_many(status="RECRUITING", limit=10)
            assert len(recruiting) == 1
            assert recruiting[0].status == "RECRUITING"

    def test_find_similar_trials(self, populated_db):
        """Test finding similar trials."""
        with populated_db.session() as session:
            repo = TrialRepository(session)
            trials = repo.find_similar_trials(
                condition="diabetes",
                phase="PHASE3",
                status="COMPLETED",
                limit=10,
            )
            assert len(trials) >= 1

    def test_find_similar_trials_with_variant(self, populated_db):
        """Test finding trials with condition variant."""
        with populated_db.session() as session:
            repo = TrialRepository(session)

            # Search with variant
            trials = repo.find_similar_trials(
                condition="Type 2 Diabetes",  # Variant
                phase="PHASE3",
                status="COMPLETED",
                limit=10,
            )
            assert len(trials) >= 1

    def test_find_similar_trials_oncology(self, populated_db):
        """Test finding similar oncology trials."""
        with populated_db.session() as session:
            repo = TrialRepository(session)

            # The oncology trial is RECRUITING, not COMPLETED
            trials = repo.find_similar_trials(
                condition="breast cancer",
                phase="PHASE3",
                status="RECRUITING",
                limit=10,
            )
            assert len(trials) >= 1

    def test_bulk_insert(self, test_db):
        """Test bulk inserting trials."""
        with test_db.session() as session:
            repo = TrialRepository(session)

            trials = [
                {
                    "nct_id": "NCT00000001",
                    "title": "Test Trial 1",
                    "status": "COMPLETED",
                    "phase": "PHASE2",
                    "therapeutic_area": "diabetes",
                },
                {
                    "nct_id": "NCT00000002",
                    "title": "Test Trial 2",
                    "status": "RECRUITING",
                    "phase": "PHASE3",
                    "therapeutic_area": "heart failure",
                },
            ]

            count = repo.bulk_insert(trials)
            assert count == 2
            session.commit()

            # Verify insertion
            t1 = repo.get_by_nct_id("NCT00000001")
            assert t1 is not None
            assert t1.title == "Test Trial 1"

    def test_get_historical_stats(self, populated_db):
        """Test getting historical statistics."""
        with populated_db.session() as session:
            repo = TrialRepository(session)
            stats = repo.get_historical_stats(therapeutic_area="diabetes")

            assert "total_trials" in stats
            assert "completed_trials" in stats
            assert "termination_rate" in stats
            assert stats["total_trials"] >= 1


class TestSiteRepository:
    """Tests for SiteRepository."""

    def test_find_by_location(self, test_db):
        """Test finding sites by location."""
        from src.database.models import Site

        with test_db.session() as session:
            # Add a test site
            site = Site(
                facility_name="Test Hospital",
                city="Boston",
                state="MA",
                country="United States",
                total_trials=10,
                therapeutic_areas='["diabetes", "heart failure"]',
            )
            session.add(site)
            session.commit()

            repo = SiteRepository(session)
            sites = repo.find_by_location(city="Boston")

            assert len(sites) >= 1
            assert any(s.city == "Boston" for s in sites)

    def test_get_top_sites(self, test_db):
        """Test getting top sites."""
        from src.database.models import Site

        with test_db.session() as session:
            # Add test sites
            for i in range(5):
                site = Site(
                    facility_name=f"Hospital {i}",
                    city="New York",
                    country="United States",
                    total_trials=10 + i,
                    experience_score=50 + i * 10,
                    therapeutic_areas='["diabetes"]',
                )
                session.add(site)
            session.commit()

            repo = SiteRepository(session)
            sites = repo.get_top_sites(
                therapeutic_area="diabetes",
                min_trials=5,
                limit=3,
            )

            assert len(sites) >= 1
            # Should be ordered by experience score
            assert sites[0].experience_score >= sites[-1].experience_score


class TestEndpointRepository:
    """Tests for EndpointRepository."""

    def test_get_by_measure(self, test_db):
        """Test getting endpoint by measure."""
        from src.database.models import Endpoint

        with test_db.session() as session:
            # Add a test endpoint
            endpoint = Endpoint(
                measure_normalized="hba1c_change",
                measure_category="efficacy",
                frequency=100,
                as_primary=80,
                as_secondary=20,
                therapeutic_areas='["diabetes"]',
            )
            session.add(endpoint)
            session.commit()

            repo = EndpointRepository(session)
            ep = repo.get_by_measure("hba1c_change")

            assert ep is not None
            assert ep.measure_normalized == "hba1c_change"

    def test_get_top_endpoints(self, test_db):
        """Test getting top endpoints."""
        from src.database.models import Endpoint

        with test_db.session() as session:
            # Add test endpoints
            endpoints = [
                Endpoint(
                    measure_normalized="hba1c_change",
                    measure_category="efficacy",
                    frequency=100,
                    as_primary=80,
                    therapeutic_areas='["diabetes"]',
                ),
                Endpoint(
                    measure_normalized="body_weight_change",
                    measure_category="efficacy",
                    frequency=50,
                    as_primary=30,
                    therapeutic_areas='["diabetes", "obesity"]',
                ),
            ]
            for ep in endpoints:
                session.add(ep)
            session.commit()

            repo = EndpointRepository(session)
            top = repo.get_top_endpoints(
                therapeutic_area="diabetes",
                as_primary=True,
                limit=5,
            )

            assert len(top) >= 1
            # Should be ordered by as_primary count
            assert top[0].as_primary >= top[-1].as_primary

    def test_normalize_endpoint(self, test_db):
        """Test endpoint normalization."""
        with test_db.session() as session:
            repo = EndpointRepository(session)

            # Test various endpoint normalizations
            assert repo._normalize_endpoint("Change in HbA1c") == "hba1c_change"
            assert repo._normalize_endpoint("Overall Survival") == "overall_survival"
            assert repo._normalize_endpoint("Progression-free Survival") == "progression_free_survival"
            assert repo._normalize_endpoint("Adverse Events") == "adverse_events"
            assert repo._normalize_endpoint("Quality of Life Score") == "quality_of_life"

    def test_categorize_endpoint(self, test_db):
        """Test endpoint categorization."""
        with test_db.session() as session:
            repo = EndpointRepository(session)

            assert repo._categorize_endpoint("hba1c_change") == "efficacy"
            assert repo._categorize_endpoint("overall_survival") == "efficacy"
            assert repo._categorize_endpoint("adverse_events") == "safety"
            assert repo._categorize_endpoint("serious_adverse_events") == "safety"
            assert repo._categorize_endpoint("quality_of_life") == "quality_of_life"
