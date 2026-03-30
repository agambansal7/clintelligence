#!/usr/bin/env python3
"""
TrialIntel CLI Runner

Convenience script to run various TrialIntel components.

Usage:
    python run.py api           # Start the API server
    python run.py dashboard     # Start the Streamlit dashboard
    python run.py ingest        # Run data ingestion pipeline
    python run.py ingest --quick # Quick demo ingestion (100 trials each)
    python run.py train-models  # Train ML risk models
    python run.py stats         # Show database statistics
    python run.py demo          # Run a demo analysis
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def run_api(args):
    """Start the API server."""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def run_dashboard(args):
    """Start the Streamlit dashboard."""
    import subprocess
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path, "--server.port", str(args.port)]
    if args.host:
        cmd.extend(["--server.address", args.host])
    subprocess.run(cmd)


def train_models(args):
    """Train ML risk models."""
    from src.database import DatabaseManager
    from src.analysis import MLRiskPredictor
    from src.utils.logging_config import setup_logging

    setup_logging(level="INFO")

    print("Training ML risk models...")
    print(f"Minimum samples required: {args.min_samples}")

    db = DatabaseManager.get_instance()
    predictor = MLRiskPredictor(db_session=db.get_session())

    metrics = predictor.train_models(min_samples=args.min_samples)

    if metrics:
        print("\n" + "=" * 60)
        print("MODEL TRAINING RESULTS")
        print("=" * 60)

        for name, m in metrics.items():
            print(f"\n{name.upper()} Model:")
            print(f"  Accuracy:  {m.accuracy:.3f}")
            print(f"  AUC-ROC:   {m.auc_roc:.3f}")
            print(f"  F1 Score:  {m.f1:.3f}")
            print(f"  Precision: {m.precision:.3f}")
            print(f"  Recall:    {m.recall:.3f}")
            print(f"  Samples:   {m.training_samples:,}")

            if m.feature_importance:
                print("  Top Features:")
                for feat, imp in list(m.feature_importance.items())[:5]:
                    print(f"    - {feat}: {imp:.3f}")

        print(f"\nModels saved to: {predictor.MODEL_DIR}/")
    else:
        print("\nTraining failed - not enough data.")
        print("Run 'python run.py ingest' first to populate the database.")


def run_ingest(args):
    """Run the data ingestion pipeline."""
    from src.ingestion.data_pipeline import DataPipeline, DEFAULT_THERAPEUTIC_AREAS
    from src.database import DatabaseManager
    from src.utils.logging_config import setup_logging

    setup_logging(level="INFO")

    db = DatabaseManager.get_instance()
    pipeline = DataPipeline(db_manager=db)

    if args.quick:
        print("Running quick demo ingestion (100 trials per area)...")
        stats = pipeline.run_quick_demo(trials_per_area=args.trials_per_area)
    elif args.aggregate_only:
        print("Running aggregation only (no new data fetch)...")
        stats = pipeline.run_full_pipeline(skip_ingestion=True)
    else:
        # Apply optional max per area limit
        if args.max_per_area:
            therapeutic_areas = [
                (condition, args.max_per_area)
                for condition, _ in DEFAULT_THERAPEUTIC_AREAS
            ]
            print(f"Running ingestion with {args.max_per_area} trials per area...")
        else:
            therapeutic_areas = DEFAULT_THERAPEUTIC_AREAS
            print(f"Running FULL ingestion ({len(DEFAULT_THERAPEUTIC_AREAS)} therapeutic areas, NO LIMIT)...")
            print("This will fetch ALL available trials from ClinicalTrials.gov.")
            print("Expected: 100,000+ trials. This may take 30-60+ minutes.\n")

        stats = pipeline.run_full_pipeline(therapeutic_areas=therapeutic_areas)

    print(f"\nPipeline completed in {stats.duration_seconds:.1f} seconds")
    print(f"Trials stored: {stats.trials_stored}")
    print(f"Sites aggregated: {stats.sites_aggregated}")
    print(f"Endpoints aggregated: {stats.endpoints_aggregated}")


def show_stats(args):
    """Show database statistics."""
    from src.database import DatabaseManager
    import json

    db = DatabaseManager.get_instance()
    stats = db.get_stats()

    print("\n" + "=" * 60)
    print("TRIALINTEL DATABASE STATISTICS")
    print("=" * 60)

    print(f"\nTotal Trials: {stats.get('total_trials', 0):,}")
    print(f"Total Sites: {stats.get('total_sites', 0):,}")
    print(f"Total Endpoints: {stats.get('total_endpoints', 0):,}")

    print("\nBy Status:")
    for status, count in list(stats.get('by_status', {}).items())[:8]:
        print(f"  {status}: {count:,}")

    print("\nBy Phase:")
    for phase, count in stats.get('by_phase', {}).items():
        print(f"  {phase}: {count:,}")

    print("\nBy Therapeutic Area (top 10):")
    for area, count in list(stats.get('by_therapeutic_area', {}).items())[:10]:
        print(f"  {area}: {count:,}")


def run_demo(args):
    """Run a demo analysis."""
    from src.analysis import create_scorer_with_db
    from src.database import DatabaseManager

    db = DatabaseManager.get_instance()
    stats = db.get_stats()

    print("\n" + "=" * 60)
    print("TRIALINTEL DEMO ANALYSIS")
    print("=" * 60)

    if stats.get('total_trials', 0) == 0:
        print("\nDatabase is empty! Run 'python run.py ingest --quick' first.")
        return

    print(f"\nDatabase contains {stats['total_trials']:,} trials")

    # Demo protocol scoring
    print("\n--- Demo: Protocol Risk Scoring ---")
    sample_criteria = """
    Inclusion Criteria:
    - Male or female, age 18-65 years
    - Diagnosed with type 2 diabetes >= 180 days prior
    - HbA1c between 7.5% and 10.0%
    - On stable metformin dose >= 1500mg for 90 days

    Exclusion Criteria:
    - History of pancreatitis
    - eGFR < 60 mL/min/1.73m2
    - Prior use of GLP-1 receptor agonists
    - NYHA Class III or IV heart failure
    """

    # Create scorer with database connection
    scorer = create_scorer_with_db()
    assessment = scorer.score_protocol(
        condition="diabetes",
        phase="PHASE3",
        eligibility_criteria=sample_criteria,
        primary_endpoints=["Change in HbA1c from baseline"],
        target_enrollment=500,
        planned_sites=50,
        planned_duration_months=24,
    )

    result = {
        "overall_risk": "low" if assessment.overall_risk_score < 30
            else "medium" if assessment.overall_risk_score < 60
            else "high",
        "risk_score": assessment.overall_risk_score,
        "amendment_probability": f"{assessment.amendment_probability:.0%}",
        "enrollment_delay_probability": f"{assessment.enrollment_delay_probability:.0%}",
        "top_recommendations": assessment.recommendations[:3],
        "benchmark_trials": assessment.benchmark_trials,
    }

    print(f"\nProtocol Risk Assessment:")
    print(f"  Overall Risk: {result['overall_risk'].upper()}")
    print(f"  Risk Score: {result['risk_score']}/100")
    print(f"  Amendment Probability: {result['amendment_probability']}")
    print(f"  Enrollment Delay Probability: {result['enrollment_delay_probability']}")
    print(f"\nBenchmark Trials: {', '.join(result['benchmark_trials'][:3])}")

    print("\nRecommendations:")
    for rec in result['top_recommendations']:
        print(f"  - {rec}")


def main():
    parser = argparse.ArgumentParser(
        description="TrialIntel CLI Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Start the Streamlit dashboard")
    dash_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    dash_parser.add_argument("--port", type=int, default=8501, help="Port to bind to")

    # Train models command
    train_parser = subparsers.add_parser("train-models", help="Train ML risk models")
    train_parser.add_argument("--min-samples", type=int, default=1000,
                             help="Minimum samples required for training")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Run data ingestion")
    ingest_parser.add_argument("--quick", action="store_true", help="Quick demo mode (3 areas, 100 each)")
    ingest_parser.add_argument("--trials-per-area", type=int, default=100, help="Trials per area in quick mode")
    ingest_parser.add_argument("--max-per-area", type=int, default=None, help="Cap trials per area (default: no limit)")
    ingest_parser.add_argument("--aggregate-only", action="store_true", help="Only run aggregation")

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # Demo command
    subparsers.add_parser("demo", help="Run demo analysis")

    args = parser.parse_args()

    if args.command == "api":
        run_api(args)
    elif args.command == "dashboard":
        run_dashboard(args)
    elif args.command == "train-models":
        train_models(args)
    elif args.command == "ingest":
        run_ingest(args)
    elif args.command == "stats":
        show_stats(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
