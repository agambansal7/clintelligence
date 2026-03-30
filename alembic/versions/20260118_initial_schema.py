"""Initial schema creation

Revision ID: 001
Revises: None
Create Date: 2026-01-18

This migration creates the initial TrialIntel database schema.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create trials table
    op.create_table(
        'trials',
        sa.Column('nct_id', sa.String(20), primary_key=True),
        sa.Column('title', sa.String(1000)),
        sa.Column('status', sa.String(50), index=True),
        sa.Column('phase', sa.String(50), index=True),
        sa.Column('study_type', sa.String(50)),
        sa.Column('conditions', sa.Text()),
        sa.Column('therapeutic_area', sa.String(200), index=True),
        sa.Column('interventions', sa.Text()),
        sa.Column('sponsor', sa.String(500)),
        sa.Column('sponsor_type', sa.String(50)),
        sa.Column('collaborators', sa.Text()),
        sa.Column('enrollment', sa.Integer()),
        sa.Column('start_date', sa.String(50)),
        sa.Column('completion_date', sa.String(50)),
        sa.Column('primary_completion_date', sa.String(50)),
        sa.Column('first_posted', sa.String(50)),
        sa.Column('last_updated', sa.String(50)),
        sa.Column('eligibility_criteria', sa.Text()),
        sa.Column('min_age', sa.String(20)),
        sa.Column('max_age', sa.String(20)),
        sa.Column('sex', sa.String(20)),
        sa.Column('healthy_volunteers', sa.String(10)),
        sa.Column('primary_outcomes', sa.Text()),
        sa.Column('secondary_outcomes', sa.Text()),
        sa.Column('locations', sa.Text()),
        sa.Column('num_sites', sa.Integer()),
        sa.Column('countries', sa.Text()),
        sa.Column('why_stopped', sa.Text()),
        sa.Column('results_first_posted', sa.String(50)),
        sa.Column('has_results', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Create composite index for common queries
    op.create_index(
        'ix_trials_therapeutic_status_phase',
        'trials',
        ['therapeutic_area', 'status', 'phase']
    )

    # Create sites table
    op.create_table(
        'sites',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('facility_name', sa.String(500)),
        sa.Column('city', sa.String(200)),
        sa.Column('state', sa.String(100)),
        sa.Column('country', sa.String(100)),
        sa.Column('total_trials', sa.Integer(), default=0),
        sa.Column('completed_trials', sa.Integer(), default=0),
        sa.Column('terminated_trials', sa.Integer(), default=0),
        sa.Column('recruiting_trials', sa.Integer(), default=0),
        sa.Column('completion_rate', sa.Float()),
        sa.Column('avg_enrollment', sa.Float()),
        sa.Column('enrollment_velocity', sa.Float()),
        sa.Column('therapeutic_areas', sa.Text()),
        sa.Column('experience_score', sa.Float()),
        sa.Column('diversity_score', sa.Float()),
        sa.Column('last_updated', sa.DateTime(), server_default=sa.func.now()),
    )

    # Create index for site lookups
    op.create_index(
        'ix_sites_location',
        'sites',
        ['facility_name', 'city', 'country']
    )

    # Create investigators table
    op.create_table(
        'investigators',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(200), index=True),
        sa.Column('affiliations', sa.Text()),
        sa.Column('total_trials', sa.Integer(), default=0),
        sa.Column('completed_trials', sa.Integer(), default=0),
        sa.Column('completion_rate', sa.Float()),
        sa.Column('therapeutic_areas', sa.Text()),
        sa.Column('experience_score', sa.Float()),
        sa.Column('last_updated', sa.DateTime(), server_default=sa.func.now()),
    )

    # Create endpoints table
    op.create_table(
        'endpoints',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('measure', sa.String(500)),
        sa.Column('measure_normalized', sa.String(200), index=True),
        sa.Column('endpoint_type', sa.String(20)),
        sa.Column('frequency', sa.Integer(), default=0),
        sa.Column('therapeutic_areas', sa.Text()),
        sa.Column('typical_timeframes', sa.Text()),
        sa.Column('success_rate', sa.Float()),
        sa.Column('last_updated', sa.DateTime(), server_default=sa.func.now()),
    )

    # Create trial_benchmarks table
    op.create_table(
        'trial_benchmarks',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('therapeutic_area', sa.String(200)),
        sa.Column('phase', sa.String(50)),
        sa.Column('total_trials', sa.Integer(), default=0),
        sa.Column('completed_trials', sa.Integer(), default=0),
        sa.Column('terminated_trials', sa.Integer(), default=0),
        sa.Column('avg_enrollment', sa.Float()),
        sa.Column('median_enrollment', sa.Float()),
        sa.Column('avg_duration_months', sa.Float()),
        sa.Column('median_duration_months', sa.Float()),
        sa.Column('avg_sites', sa.Float()),
        sa.Column('completion_rate', sa.Float()),
        sa.Column('termination_rate', sa.Float()),
        sa.Column('amendment_rate', sa.Float()),
        sa.Column('last_updated', sa.DateTime(), server_default=sa.func.now()),
    )

    # Create composite index for benchmark lookups
    op.create_index(
        'ix_benchmarks_area_phase',
        'trial_benchmarks',
        ['therapeutic_area', 'phase'],
        unique=True
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('trial_benchmarks')
    op.drop_table('endpoints')
    op.drop_table('investigators')
    op.drop_table('sites')
    op.drop_table('trials')
