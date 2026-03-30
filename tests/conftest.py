"""Pytest fixtures for TrialIntel tests."""

import pytest
import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.database import DatabaseManager, Trial, Site, Endpoint


@pytest.fixture(scope="session")
def test_db():
    """Create a test database for the session."""
    # Use a temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Reset singleton to create a new instance with test DB
    DatabaseManager.reset_instance()

    db_url = f"sqlite:///{db_path}"
    db = DatabaseManager.get_instance(database_url=db_url)
    db.create_tables()

    yield db

    # Cleanup
    DatabaseManager.reset_instance()
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def db_session(test_db):
    """Create a new database session for each test."""
    with test_db.session() as session:
        yield session
        session.rollback()


@pytest.fixture(scope="session")
def sample_trial_data():
    """Sample trial data for testing."""
    return {
        "nct_id": "NCT12345678",
        "title": "Test Trial for Diabetes",
        "status": "COMPLETED",
        "phase": "PHASE3",
        "study_type": "INTERVENTIONAL",
        "conditions": "Type 2 Diabetes Mellitus, Obesity",
        "interventions": "Drug: Test Drug",
        "therapeutic_area": "diabetes",
        "sponsor": "Test Pharma Inc",
        "sponsor_type": "INDUSTRY",
        "enrollment": 500,
        "start_date": "2020-01-01",
        "completion_date": "2022-12-31",
        "primary_completion_date": "2022-06-30",
        "eligibility_criteria": """
        Inclusion Criteria:
        - Age 18-65 years
        - HbA1c between 7.0% and 10.0%
        - BMI >= 25 kg/m2

        Exclusion Criteria:
        - Type 1 diabetes
        - Severe renal impairment
        """,
        "min_age": "18 Years",
        "max_age": "65 Years",
        "sex": "ALL",
        "primary_outcomes": '[{"measure": "Change in HbA1c", "timeFrame": "24 weeks"}]',
        "secondary_outcomes": '[{"measure": "Change in body weight", "timeFrame": "24 weeks"}]',
        "locations": '[{"facility": "Test Medical Center", "city": "Boston", "state": "MA", "country": "United States"}]',
        "num_sites": 50,
        "why_stopped": None,
        "has_results": True,
    }


@pytest.fixture(scope="session")
def sample_oncology_trial():
    """Sample oncology trial data."""
    return {
        "nct_id": "NCT87654321",
        "title": "Phase 3 Study of Test Drug in Breast Cancer",
        "status": "RECRUITING",
        "phase": "PHASE3",
        "study_type": "INTERVENTIONAL",
        "conditions": "Breast Neoplasms, HER2-positive Breast Cancer",
        "interventions": "Drug: Test Antibody",
        "therapeutic_area": "breast cancer",
        "sponsor": "Big Pharma Corp",
        "sponsor_type": "INDUSTRY",
        "enrollment": 800,
        "start_date": "2023-01-01",
        "completion_date": None,
        "primary_completion_date": "2025-12-31",
        "eligibility_criteria": """
        Inclusion Criteria:
        - HER2-positive breast cancer
        - ECOG performance status 0-1

        Exclusion Criteria:
        - Prior HER2-targeted therapy
        - Active brain metastases
        """,
        "min_age": "18 Years",
        "max_age": None,
        "sex": "FEMALE",
        "primary_outcomes": '[{"measure": "Progression-free survival", "timeFrame": "36 months"}]',
        "secondary_outcomes": '[{"measure": "Overall survival", "timeFrame": "60 months"}]',
        "locations": '[{"facility": "Cancer Center", "city": "New York", "state": "NY", "country": "United States"}]',
        "num_sites": 100,
        "why_stopped": None,
        "has_results": False,
    }


@pytest.fixture(scope="session")
def populated_db(test_db, sample_trial_data, sample_oncology_trial):
    """Database with sample data (populated once per session)."""
    with test_db.session() as session:
        # Check if data already exists
        existing = session.query(Trial).filter(Trial.nct_id == "NCT12345678").first()
        if existing is None:
            # Add sample trials
            trial1 = Trial(**sample_trial_data)
            trial2 = Trial(**sample_oncology_trial)
            session.add(trial1)
            session.add(trial2)
            session.commit()

    yield test_db


@pytest.fixture
def sample_protocol():
    """Sample protocol for risk scoring."""
    return {
        "condition": "diabetes",
        "phase": "PHASE3",
        "eligibility_criteria": """
        Inclusion Criteria:
        - Adults 18-75 years
        - Type 2 diabetes diagnosis
        - HbA1c 7.5-10.5%

        Exclusion Criteria:
        - eGFR < 30
        - NYHA Class III/IV heart failure
        """,
        "primary_endpoints": ["Change in HbA1c from baseline"],
        "target_enrollment": 500,
        "planned_sites": 50,
        "planned_duration_months": 24,
    }
