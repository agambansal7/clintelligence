"""
Unified Data Pipeline for TrialIntel.

This module provides a complete data pipeline that:
1. Ingests trials from ClinicalTrials.gov
2. Stores them in the database
3. Aggregates site performance data
4. Aggregates endpoint patterns
5. Computes benchmarks

Usage:
    from src.ingestion.data_pipeline import DataPipeline

    pipeline = DataPipeline()
    pipeline.run_full_pipeline(therapeutic_areas=["diabetes", "breast cancer"])
"""

import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

from ..database import (
    DatabaseManager,
    Trial,
    TrialRepository,
    SiteRepository,
    EndpointRepository,
)
from ..database.repository import BenchmarkRepository
from .ctgov_client import ClinicalTrialsGovClient as ClinicalTrialsClient, TrialData

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    trials_fetched: int = 0
    trials_stored: int = 0
    sites_aggregated: int = 0
    endpoints_aggregated: int = 0
    benchmarks_computed: int = 0
    errors: int = 0
    start_time: datetime = None
    end_time: datetime = None

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trials_fetched": self.trials_fetched,
            "trials_stored": self.trials_stored,
            "sites_aggregated": self.sites_aggregated,
            "endpoints_aggregated": self.endpoints_aggregated,
            "benchmarks_computed": self.benchmarks_computed,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
        }


# Default therapeutic areas to ingest (None = no limit, fetch all available)
DEFAULT_THERAPEUTIC_AREAS = [
    # Oncology
    ("breast cancer", None),
    ("lung cancer", None),
    ("prostate cancer", None),
    ("colorectal cancer", None),
    ("melanoma", None),
    ("ovarian cancer", None),
    ("pancreatic cancer", None),
    ("liver cancer", None),
    ("gastric cancer", None),
    ("bladder cancer", None),
    ("kidney cancer", None),
    ("head and neck cancer", None),
    ("brain tumor", None),
    ("glioblastoma", None),
    ("lymphoma", None),
    ("leukemia", None),
    ("multiple myeloma", None),
    ("sarcoma", None),

    # Cardiovascular
    ("heart failure", None),
    ("hypertension", None),
    ("atrial fibrillation", None),
    ("coronary artery disease", None),
    ("myocardial infarction", None),
    ("stroke", None),
    ("peripheral artery disease", None),
    ("pulmonary hypertension", None),
    ("cardiomyopathy", None),

    # Metabolic/Endocrine
    ("diabetes", None),
    ("type 2 diabetes", None),
    ("type 1 diabetes", None),
    ("obesity", None),
    ("dyslipidemia", None),
    ("metabolic syndrome", None),
    ("thyroid disease", None),
    ("osteoporosis", None),

    # Neurology
    ("alzheimer", None),
    ("parkinson", None),
    ("multiple sclerosis", None),
    ("epilepsy", None),
    ("migraine", None),
    ("amyotrophic lateral sclerosis", None),
    ("huntington disease", None),
    ("neuropathy", None),
    ("spinal muscular atrophy", None),

    # Psychiatry
    ("depression", None),
    ("major depressive disorder", None),
    ("anxiety", None),
    ("schizophrenia", None),
    ("bipolar disorder", None),
    ("PTSD", None),
    ("ADHD", None),
    ("autism", None),
    ("substance use disorder", None),

    # Immunology/Rheumatology
    ("rheumatoid arthritis", None),
    ("psoriasis", None),
    ("psoriatic arthritis", None),
    ("lupus", None),
    ("inflammatory bowel disease", None),
    ("crohn disease", None),
    ("ulcerative colitis", None),
    ("ankylosing spondylitis", None),
    ("sjogren syndrome", None),
    ("dermatomyositis", None),

    # Respiratory
    ("asthma", None),
    ("COPD", None),
    ("pulmonary fibrosis", None),
    ("cystic fibrosis", None),
    ("pneumonia", None),
    ("acute respiratory distress syndrome", None),

    # Infectious Disease
    ("HIV", None),
    ("hepatitis B", None),
    ("hepatitis C", None),
    ("influenza", None),
    ("COVID-19", None),
    ("tuberculosis", None),
    ("sepsis", None),
    ("bacterial infection", None),

    # Nephrology
    ("chronic kidney disease", None),
    ("diabetic nephropathy", None),
    ("glomerulonephritis", None),
    ("polycystic kidney disease", None),

    # Gastroenterology/Hepatology
    ("non-alcoholic fatty liver disease", None),
    ("cirrhosis", None),
    ("gastroparesis", None),
    ("irritable bowel syndrome", None),

    # Hematology
    ("anemia", None),
    ("sickle cell disease", None),
    ("hemophilia", None),
    ("thrombocytopenia", None),
    ("myelodysplastic syndrome", None),

    # Ophthalmology
    ("macular degeneration", None),
    ("glaucoma", None),
    ("diabetic retinopathy", None),
    ("dry eye disease", None),

    # Dermatology
    ("atopic dermatitis", None),
    ("eczema", None),
    ("alopecia", None),
    ("vitiligo", None),

    # Rare/Genetic Diseases
    ("duchenne muscular dystrophy", None),
    ("fabry disease", None),
    ("gaucher disease", None),
    ("pompe disease", None),
    ("hemoglobin disorders", None),

    # Women's Health
    ("endometriosis", None),
    ("uterine fibroids", None),
    ("polycystic ovary syndrome", None),
    ("menopause", None),

    # Pain/Musculoskeletal
    ("chronic pain", None),
    ("osteoarthritis", None),
    ("fibromyalgia", None),
    ("gout", None),

    # Transplant/Immunosuppression
    ("transplant rejection", None),
    ("graft versus host disease", None),
]


class DataPipeline:
    """
    Unified data pipeline for TrialIntel.

    Handles ingestion from ClinicalTrials.gov and populates all database tables.
    """

    def __init__(
        self,
        db_manager: DatabaseManager = None,
        rate_limit_delay: float = 0.1,
    ):
        """
        Initialize the data pipeline.

        Args:
            db_manager: Database manager instance. If None, creates default.
            rate_limit_delay: Delay between API requests in seconds.
        """
        self.db = db_manager or DatabaseManager.get_instance()
        self.db.create_tables()
        self.client = ClinicalTrialsClient(rate_limit_delay=rate_limit_delay)
        self.stats = PipelineStats()

    def trial_data_to_model(self, data: TrialData, therapeutic_area: str) -> Dict[str, Any]:
        """Convert TrialData to Trial model dict."""
        # Determine sponsor type
        sponsor = data.sponsor or ""
        sponsor_type = "UNKNOWN"
        pharma_keywords = ["pfizer", "novartis", "roche", "merck", "johnson", "astrazeneca",
                          "bristol", "sanofi", "abbvie", "lilly", "gsk", "amgen", "gilead",
                          "biogen", "regeneron", "moderna", "biontech", "takeda", "boehringer"]
        academic_keywords = ["university", "hospital", "medical center", "institute", "college"]

        sponsor_lower = sponsor.lower()
        if any(kw in sponsor_lower for kw in pharma_keywords):
            sponsor_type = "INDUSTRY"
        elif any(kw in sponsor_lower for kw in academic_keywords):
            sponsor_type = "ACADEMIC"
        elif "nih" in sponsor_lower or "national" in sponsor_lower:
            sponsor_type = "NIH"

        # Count sites
        num_sites = len(data.locations) if data.locations else 0

        # Handle phase - could be string or list
        phase = data.phase
        if isinstance(phase, list):
            phase = ",".join(phase)

        return {
            "nct_id": data.nct_id,
            "title": data.title,
            "status": data.status,
            "phase": phase,
            "study_type": data.study_type,
            "conditions": ",".join(data.conditions) if data.conditions else "",
            "interventions": ",".join(data.interventions) if data.interventions else "",
            "therapeutic_area": therapeutic_area,
            "sponsor": data.sponsor,
            "sponsor_type": sponsor_type,
            "enrollment": data.enrollment,
            "start_date": data.start_date,
            "completion_date": data.completion_date,
            "primary_completion_date": data.primary_completion_date,
            "eligibility_criteria": data.eligibility_criteria,
            "min_age": data.minimum_age,
            "max_age": data.maximum_age,
            "sex": data.sex,
            "primary_outcomes": json.dumps([
                {"measure": o.get("measure", ""), "timeFrame": o.get("timeFrame", "")}
                for o in (data.primary_outcomes or [])
            ]),
            "secondary_outcomes": json.dumps([
                {"measure": o.get("measure", ""), "timeFrame": o.get("timeFrame", "")}
                for o in (data.secondary_outcomes or [])
            ]),
            "locations": json.dumps([
                {
                    "facility": loc.get("facility", ""),
                    "city": loc.get("city", ""),
                    "state": loc.get("state", ""),
                    "country": loc.get("country", ""),
                }
                for loc in (data.locations or [])[:50]  # Limit locations
            ]),
            "num_sites": num_sites,
            "why_stopped": data.why_stopped,
            "has_results": data.results_first_posted is not None,
            "ingested_at": datetime.utcnow(),
        }

    def ingest_therapeutic_area(
        self,
        condition: str,
        max_trials: int = 1000,
        batch_size: int = 100,
    ) -> int:
        """
        Ingest trials for a single therapeutic area.

        Args:
            condition: The condition/therapeutic area to search for
            max_trials: Maximum number of trials to fetch
            batch_size: Number of trials per API request

        Returns:
            Number of trials stored
        """
        limit_str = f"max: {max_trials}" if max_trials else "no limit"
        logger.info(f"Ingesting trials for: {condition} ({limit_str})")

        stored_count = 0

        with self.db.session() as session:
            trial_repo = TrialRepository(session)

            # Stream trials from API
            trial_generator = self.client.stream_all_trials(
                condition=condition,
                max_trials=max_trials,
            )

            batch = []
            for trial_data in trial_generator:
                self.stats.trials_fetched += 1

                try:
                    trial_dict = self.trial_data_to_model(trial_data, condition)
                    batch.append(trial_dict)

                    # Process in batches
                    if len(batch) >= batch_size:
                        stored = trial_repo.bulk_insert(batch)
                        stored_count += stored
                        self.stats.trials_stored += stored
                        session.commit()
                        batch = []
                        logger.debug(f"  Stored batch: {stored} trials")

                except Exception as e:
                    logger.error(f"Error processing trial {trial_data.nct_id}: {e}")
                    self.stats.errors += 1

            # Store remaining batch
            if batch:
                stored = trial_repo.bulk_insert(batch)
                stored_count += stored
                self.stats.trials_stored += stored
                session.commit()

        logger.info(f"Completed {condition}: {stored_count} trials stored")
        return stored_count

    def aggregate_sites(self, therapeutic_area: str = None) -> int:
        """
        Aggregate site data from stored trials.

        Args:
            therapeutic_area: Optional filter by therapeutic area

        Returns:
            Number of sites aggregated
        """
        logger.info("Aggregating site data...")

        with self.db.session() as session:
            trial_repo = TrialRepository(session)
            site_repo = SiteRepository(session)

            # Get trials with location data
            trials = trial_repo.get_many(
                therapeutic_area=therapeutic_area,
                limit=50000,  # Process in large batches
            )

            count = site_repo.aggregate_from_trials(trials)
            session.commit()

            self.stats.sites_aggregated = count
            logger.info(f"Aggregated {count} sites")

            return count

    def aggregate_endpoints(self, therapeutic_area: str = None) -> int:
        """
        Aggregate endpoint data from stored trials.

        Args:
            therapeutic_area: Optional filter by therapeutic area

        Returns:
            Number of endpoints aggregated
        """
        logger.info("Aggregating endpoint data...")

        with self.db.session() as session:
            trial_repo = TrialRepository(session)
            endpoint_repo = EndpointRepository(session)

            # Get trials with outcome data
            trials = trial_repo.get_many(
                therapeutic_area=therapeutic_area,
                limit=50000,
            )

            count = endpoint_repo.aggregate_from_trials(trials)
            session.commit()

            self.stats.endpoints_aggregated = count
            logger.info(f"Aggregated {count} endpoints")

            return count

    def compute_benchmarks(self) -> int:
        """
        Compute and store benchmark statistics.

        Returns:
            Number of benchmarks computed
        """
        logger.info("Computing benchmarks...")

        with self.db.session() as session:
            trial_repo = TrialRepository(session)
            benchmark_repo = BenchmarkRepository(session)

            count = benchmark_repo.compute_benchmarks(trial_repo)
            session.commit()

            self.stats.benchmarks_computed = count
            logger.info(f"Computed {count} benchmarks")

            return count

    def run_full_pipeline(
        self,
        therapeutic_areas: List[tuple] = None,
        skip_ingestion: bool = False,
    ) -> PipelineStats:
        """
        Run the complete data pipeline.

        Args:
            therapeutic_areas: List of (condition, max_trials) tuples
            skip_ingestion: If True, only aggregate existing data

        Returns:
            PipelineStats with results
        """
        self.stats = PipelineStats(start_time=datetime.utcnow())

        if therapeutic_areas is None:
            therapeutic_areas = DEFAULT_THERAPEUTIC_AREAS

        logger.info("=" * 60)
        logger.info("TRIALINTEL DATA PIPELINE")
        logger.info(f"Started: {self.stats.start_time}")
        logger.info(f"Therapeutic areas: {len(therapeutic_areas)}")
        logger.info("=" * 60)

        try:
            # Step 1: Ingest trials
            if not skip_ingestion:
                for condition, max_trials in therapeutic_areas:
                    try:
                        self.ingest_therapeutic_area(condition, max_trials)
                    except Exception as e:
                        logger.error(f"Error ingesting {condition}: {e}")
                        self.stats.errors += 1

            # Step 2: Aggregate sites
            try:
                self.aggregate_sites()
            except Exception as e:
                logger.error(f"Error aggregating sites: {e}")
                self.stats.errors += 1

            # Step 3: Aggregate endpoints
            try:
                self.aggregate_endpoints()
            except Exception as e:
                logger.error(f"Error aggregating endpoints: {e}")
                self.stats.errors += 1

            # Step 4: Compute benchmarks
            try:
                self.compute_benchmarks()
            except Exception as e:
                logger.error(f"Error computing benchmarks: {e}")
                self.stats.errors += 1

        finally:
            self.stats.end_time = datetime.utcnow()

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Trials fetched: {self.stats.trials_fetched}")
        logger.info(f"Trials stored: {self.stats.trials_stored}")
        logger.info(f"Sites aggregated: {self.stats.sites_aggregated}")
        logger.info(f"Endpoints aggregated: {self.stats.endpoints_aggregated}")
        logger.info(f"Benchmarks computed: {self.stats.benchmarks_computed}")
        logger.info(f"Errors: {self.stats.errors}")
        logger.info(f"Duration: {self.stats.duration_seconds:.1f} seconds")

        return self.stats

    def run_quick_demo(self, trials_per_area: int = 100) -> PipelineStats:
        """
        Run a quick demo pipeline with limited data.

        Args:
            trials_per_area: Number of trials to fetch per therapeutic area

        Returns:
            PipelineStats with results
        """
        demo_areas = [
            ("diabetes", trials_per_area),
            ("breast cancer", trials_per_area),
            ("lung cancer", trials_per_area),
        ]
        return self.run_full_pipeline(therapeutic_areas=demo_areas)


def main():
    """Command-line entry point."""
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="TrialIntel Data Pipeline")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick demo with limited data",
    )
    parser.add_argument(
        "--trials-per-area",
        type=int,
        default=100,
        help="Trials per therapeutic area (quick mode)",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing data, skip ingestion",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Database URL (defaults to SQLite in ./data)",
    )
    parser.add_argument(
        "--max-per-area",
        type=int,
        default=None,
        help="Optional max trials per therapeutic area (default: no limit)",
    )

    args = parser.parse_args()

    # Initialize pipeline
    db = DatabaseManager.get_instance(database_url=args.database)
    pipeline = DataPipeline(db_manager=db)

    # Run pipeline
    if args.quick:
        stats = pipeline.run_quick_demo(trials_per_area=args.trials_per_area)
    else:
        # Apply optional max per area limit
        if args.max_per_area:
            therapeutic_areas = [
                (condition, args.max_per_area)
                for condition, _ in DEFAULT_THERAPEUTIC_AREAS
            ]
        else:
            therapeutic_areas = DEFAULT_THERAPEUTIC_AREAS
        stats = pipeline.run_full_pipeline(
            therapeutic_areas=therapeutic_areas,
            skip_ingestion=args.aggregate_only
        )

    # Print final stats
    print("\n" + "=" * 60)
    print("FINAL DATABASE STATS")
    print("=" * 60)
    db_stats = db.get_stats()
    for key, value in db_stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in list(value.items())[:10]:
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
