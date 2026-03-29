"""SQLAlchemy ORM models for TrialIntel."""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime,
    ForeignKey, Index, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()


class Trial(Base):
    """Clinical trial record from ClinicalTrials.gov."""

    __tablename__ = "trials"

    # Primary key
    nct_id = Column(String(20), primary_key=True, index=True)

    # Basic info
    title = Column(Text)
    status = Column(String(50), index=True)
    phase = Column(String(50), index=True)
    study_type = Column(String(50))

    # Conditions and interventions (stored as comma-separated)
    conditions = Column(Text)
    interventions = Column(Text)
    therapeutic_area = Column(String(100), index=True)

    # Sponsor info
    sponsor = Column(String(255), index=True)
    sponsor_type = Column(String(50))  # INDUSTRY, ACADEMIC, NIH, etc.

    # Enrollment
    enrollment = Column(Integer)
    enrollment_type = Column(String(20))  # ACTUAL, ESTIMATED

    # Dates
    start_date = Column(String(20))
    completion_date = Column(String(20))
    primary_completion_date = Column(String(20))

    # Eligibility
    eligibility_criteria = Column(Text)
    min_age = Column(String(20))
    max_age = Column(String(20))
    sex = Column(String(20))

    # Outcomes (stored as JSON)
    primary_outcomes = Column(Text)  # JSON array
    secondary_outcomes = Column(Text)  # JSON array

    # Locations (stored as JSON)
    locations = Column(Text)  # JSON array
    num_sites = Column(Integer)

    # Results and status
    why_stopped = Column(Text)
    has_results = Column(Boolean, default=False)

    # Raw data
    raw_json = Column(Text)

    # Metadata
    ingested_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_trial_status_phase', 'status', 'phase'),
        Index('idx_trial_therapeutic_area', 'therapeutic_area'),
        Index('idx_trial_sponsor', 'sponsor'),
    )

    @hybrid_property
    def conditions_list(self) -> List[str]:
        """Return conditions as a list."""
        if not self.conditions:
            return []
        return [c.strip() for c in self.conditions.split(",")]

    @hybrid_property
    def interventions_list(self) -> List[str]:
        """Return interventions as a list."""
        if not self.interventions:
            return []
        return [i.strip() for i in self.interventions.split(",")]

    @hybrid_property
    def primary_outcomes_list(self) -> List[Dict]:
        """Return primary outcomes as a list of dicts."""
        if not self.primary_outcomes:
            return []
        try:
            return json.loads(self.primary_outcomes)
        except json.JSONDecodeError:
            return []

    @hybrid_property
    def secondary_outcomes_list(self) -> List[Dict]:
        """Return secondary outcomes as a list of dicts."""
        if not self.secondary_outcomes:
            return []
        try:
            return json.loads(self.secondary_outcomes)
        except json.JSONDecodeError:
            return []

    @hybrid_property
    def locations_list(self) -> List[Dict]:
        """Return locations as a list of dicts."""
        if not self.locations:
            return []
        try:
            return json.loads(self.locations)
        except json.JSONDecodeError:
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "status": self.status,
            "phase": self.phase,
            "study_type": self.study_type,
            "conditions": self.conditions_list,
            "interventions": self.interventions_list,
            "therapeutic_area": self.therapeutic_area,
            "sponsor": self.sponsor,
            "sponsor_type": self.sponsor_type,
            "enrollment": self.enrollment,
            "start_date": self.start_date,
            "completion_date": self.completion_date,
            "eligibility_criteria": self.eligibility_criteria,
            "min_age": self.min_age,
            "max_age": self.max_age,
            "sex": self.sex,
            "primary_outcomes": self.primary_outcomes_list,
            "secondary_outcomes": self.secondary_outcomes_list,
            "locations": self.locations_list,
            "num_sites": self.num_sites,
            "why_stopped": self.why_stopped,
            "has_results": self.has_results,
        }


class Site(Base):
    """Aggregated site performance data."""

    __tablename__ = "sites"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Site identification (composite key for uniqueness)
    facility_name = Column(String(255), index=True)
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100), index=True)

    # Aggregated metrics
    total_trials = Column(Integer, default=0)
    completed_trials = Column(Integer, default=0)
    terminated_trials = Column(Integer, default=0)
    active_trials = Column(Integer, default=0)

    # Performance metrics
    avg_enrollment = Column(Float)
    total_enrollment = Column(Integer, default=0)
    enrollment_velocity = Column(Float)  # patients per month

    # Therapeutic areas (JSON array)
    therapeutic_areas = Column(Text)  # JSON array

    # Quality metrics
    completion_rate = Column(Float)
    experience_score = Column(Float)  # 0-100

    # Metadata
    last_trial_date = Column(String(20))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_site_location', 'facility_name', 'city', 'country'),
        Index('idx_site_country', 'country'),
    )

    @hybrid_property
    def therapeutic_areas_list(self) -> List[str]:
        if not self.therapeutic_areas:
            return []
        try:
            return json.loads(self.therapeutic_areas)
        except json.JSONDecodeError:
            return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "facility_name": self.facility_name,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "terminated_trials": self.terminated_trials,
            "active_trials": self.active_trials,
            "avg_enrollment": self.avg_enrollment,
            "enrollment_velocity": self.enrollment_velocity,
            "therapeutic_areas": self.therapeutic_areas_list,
            "completion_rate": self.completion_rate,
            "experience_score": self.experience_score,
        }


class Investigator(Base):
    """Aggregated investigator performance data."""

    __tablename__ = "investigators"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Investigator identification
    name = Column(String(255), index=True)
    name_normalized = Column(String(255), index=True)  # For deduplication

    # Affiliations (JSON array)
    affiliations = Column(Text)
    primary_affiliation = Column(String(255))

    # Aggregated metrics
    total_trials = Column(Integer, default=0)
    completed_trials = Column(Integer, default=0)
    as_lead = Column(Integer, default=0)  # Times as lead investigator

    # Performance
    avg_enrollment_rate = Column(Float)
    completion_rate = Column(Float)
    experience_score = Column(Float)  # 0-100

    # Therapeutic areas (JSON array)
    therapeutic_areas = Column(Text)

    # Metadata
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_investigator_name', 'name_normalized'),
    )

    @hybrid_property
    def therapeutic_areas_list(self) -> List[str]:
        if not self.therapeutic_areas:
            return []
        try:
            return json.loads(self.therapeutic_areas)
        except json.JSONDecodeError:
            return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "affiliations": json.loads(self.affiliations) if self.affiliations else [],
            "primary_affiliation": self.primary_affiliation,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "as_lead": self.as_lead,
            "avg_enrollment_rate": self.avg_enrollment_rate,
            "completion_rate": self.completion_rate,
            "experience_score": self.experience_score,
            "therapeutic_areas": self.therapeutic_areas_list,
        }


class Endpoint(Base):
    """Aggregated endpoint pattern data."""

    __tablename__ = "endpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Endpoint identification
    measure_normalized = Column(String(255), index=True)  # Normalized name (e.g., "hba1c_change")
    measure_category = Column(String(100), index=True)  # Category (efficacy, safety, qol)

    # Examples (JSON array of raw endpoint texts)
    raw_examples = Column(Text)

    # Usage statistics
    frequency = Column(Integer, default=0)  # Total times used
    as_primary = Column(Integer, default=0)
    as_secondary = Column(Integer, default=0)

    # Success metrics
    trials_completed = Column(Integer, default=0)
    trials_terminated = Column(Integer, default=0)
    success_rate = Column(Float)  # completed / (completed + terminated)

    # Associated data
    therapeutic_areas = Column(Text)  # JSON array
    phases = Column(Text)  # JSON array
    typical_timeframes = Column(Text)  # JSON array (e.g., ["12 weeks", "24 weeks"])

    # Metadata
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_endpoint_measure', 'measure_normalized'),
        Index('idx_endpoint_category', 'measure_category'),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "measure_normalized": self.measure_normalized,
            "measure_category": self.measure_category,
            "raw_examples": json.loads(self.raw_examples) if self.raw_examples else [],
            "frequency": self.frequency,
            "as_primary": self.as_primary,
            "as_secondary": self.as_secondary,
            "success_rate": self.success_rate,
            "therapeutic_areas": json.loads(self.therapeutic_areas) if self.therapeutic_areas else [],
            "phases": json.loads(self.phases) if self.phases else [],
            "typical_timeframes": json.loads(self.typical_timeframes) if self.typical_timeframes else [],
        }


class TrialBenchmark(Base):
    """Pre-computed benchmark statistics for faster queries."""

    __tablename__ = "trial_benchmarks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Grouping keys
    therapeutic_area = Column(String(100), index=True)
    phase = Column(String(50), index=True)

    # Trial counts
    total_trials = Column(Integer, default=0)
    completed_trials = Column(Integer, default=0)
    terminated_trials = Column(Integer, default=0)

    # Amendment/failure rates
    amendment_rate = Column(Float)  # Historical amendment rate
    termination_rate = Column(Float)
    delay_rate = Column(Float)

    # Enrollment benchmarks
    avg_enrollment = Column(Float)
    median_enrollment = Column(Float)
    avg_sites = Column(Float)
    avg_enrollment_per_site = Column(Float)

    # Duration benchmarks (in months)
    avg_duration_months = Column(Float)
    median_duration_months = Column(Float)

    # Metadata
    computed_at = Column(DateTime, default=datetime.utcnow)
    sample_nct_ids = Column(Text)  # JSON array of example NCT IDs

    __table_args__ = (
        Index('idx_benchmark_key', 'therapeutic_area', 'phase'),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "therapeutic_area": self.therapeutic_area,
            "phase": self.phase,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "terminated_trials": self.terminated_trials,
            "amendment_rate": self.amendment_rate,
            "termination_rate": self.termination_rate,
            "delay_rate": self.delay_rate,
            "avg_enrollment": self.avg_enrollment,
            "median_enrollment": self.median_enrollment,
            "avg_sites": self.avg_sites,
            "avg_enrollment_per_site": self.avg_enrollment_per_site,
            "avg_duration_months": self.avg_duration_months,
            "sample_nct_ids": json.loads(self.sample_nct_ids) if self.sample_nct_ids else [],
        }
