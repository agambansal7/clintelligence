#!/usr/bin/env python3
"""
TrialIntel Data Builder

This script demonstrates how to build the TrialIntel intelligence database
from ClinicalTrials.gov data. Run this to populate the system with real data.

Usage:
    python build_database.py --condition "diabetes" --limit 1000
    python build_database.py --all --limit 10000
"""

import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.ctgov_client import ClinicalTrialsGovClient, TrialData
from src.analysis.site_intelligence import SiteInvestigatorIntelligence, build_intelligence_from_trials
from src.analysis.endpoint_benchmarking import EndpointBenchmarker, build_endpoint_benchmarks


def fetch_trials_for_condition(
    condition: str,
    max_trials: int = 1000,
    phases: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch trials from ClinicalTrials.gov for a specific condition.
    
    Args:
        condition: Therapeutic area (e.g., "diabetes", "breast cancer")
        max_trials: Maximum number of trials to fetch
        phases: Filter by phases (default: all phases)
        
    Returns:
        List of trial data dictionaries
    """
    print(f"Fetching up to {max_trials} {condition} trials...")
    
    client = ClinicalTrialsGovClient(rate_limit_delay=0.2)
    
    trials = []
    for trial in client.stream_all_trials(
        condition=condition,
        phase=phases,
        max_trials=max_trials,
    ):
        # Convert TrialData to dict for storage
        trial_dict = {
            "nct_id": trial.nct_id,
            "title": trial.title,
            "status": trial.status,
            "phase": trial.phase,
            "conditions": trial.conditions,
            "interventions": trial.interventions,
            "sponsor": trial.sponsor,
            "collaborators": trial.collaborators,
            "enrollment": trial.enrollment,
            "start_date": trial.start_date,
            "completion_date": trial.completion_date,
            "primary_completion_date": trial.primary_completion_date,
            "study_type": trial.study_type,
            "eligibility_criteria": trial.eligibility_criteria,
            "minimum_age": trial.minimum_age,
            "maximum_age": trial.maximum_age,
            "sex": trial.sex,
            "primary_outcomes": trial.primary_outcomes,
            "secondary_outcomes": trial.secondary_outcomes,
            "locations": trial.locations,
            "why_stopped": trial.why_stopped,
        }
        trials.append(trial_dict)
        
        if len(trials) % 100 == 0:
            print(f"  Fetched {len(trials)} trials...")
    
    print(f"  Total fetched: {len(trials)} trials")
    return trials


def build_site_intelligence(trials: List[Dict[str, Any]]) -> SiteInvestigatorIntelligence:
    """Build site and investigator intelligence from trial data."""
    print("Building site & investigator intelligence...")
    intel = build_intelligence_from_trials(trials)
    print(f"  Sites tracked: {len(intel.sites)}")
    print(f"  Investigators tracked: {len(intel.investigators)}")
    return intel


def build_endpoint_intelligence(trials: List[Dict[str, Any]]) -> EndpointBenchmarker:
    """Build endpoint benchmarks from trial data."""
    print("Building endpoint benchmarks...")
    benchmarker = build_endpoint_benchmarks(trials)
    total_endpoints = sum(len(data) for data in benchmarker.endpoint_data.values())
    print(f"  Endpoint patterns tracked: {total_endpoints}")
    return benchmarker


def save_trials_to_file(trials: List[Dict[str, Any]], filename: str):
    """Save trials to JSON file for later use."""
    print(f"Saving {len(trials)} trials to {filename}...")
    with open(filename, 'w') as f:
        json.dump(trials, f, indent=2, default=str)
    print(f"  Saved to {filename}")


def generate_summary_report(
    trials: List[Dict[str, Any]],
    site_intel: SiteInvestigatorIntelligence,
    endpoint_intel: EndpointBenchmarker,
) -> str:
    """Generate a summary report of the intelligence built."""
    
    # Count by status
    status_counts = {}
    for trial in trials:
        status = trial.get("status", "Unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Count by phase
    phase_counts = {}
    for trial in trials:
        for phase in trial.get("phase", []):
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    # Top sponsors
    sponsor_counts = {}
    for trial in trials:
        sponsor = trial.get("sponsor", "Unknown")
        sponsor_counts[sponsor] = sponsor_counts.get(sponsor, 0) + 1
    top_sponsors = sorted(sponsor_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    report = f"""
================================================================================
TRIALINTEL DATABASE BUILD REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

TRIALS PROCESSED
----------------
Total trials: {len(trials)}

By Status:
"""
    for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        report += f"  {status}: {count}\n"
    
    report += "\nBy Phase:\n"
    for phase, count in sorted(phase_counts.items()):
        report += f"  {phase}: {count}\n"
    
    report += "\nTop 10 Sponsors:\n"
    for sponsor, count in top_sponsors:
        report += f"  {sponsor}: {count} trials\n"
    
    report += f"""
SITE INTELLIGENCE
-----------------
Total sites tracked: {len(site_intel.sites)}
Total investigators tracked: {len(site_intel.investigators)}

Top 5 Sites by Trial Count:
"""
    top_sites = sorted(site_intel.sites.values(), key=lambda x: x.total_trials, reverse=True)[:5]
    for site in top_sites:
        report += f"  {site.facility_name} ({site.city}, {site.country}): {site.total_trials} trials\n"
    
    report += f"""
ENDPOINT INTELLIGENCE
---------------------
Conditions analyzed: {len(endpoint_intel.endpoint_data)}

Endpoint patterns by condition:
"""
    for condition, endpoints in endpoint_intel.endpoint_data.items():
        primary_count = len([k for k in endpoints.keys() if k.startswith("primary_")])
        secondary_count = len([k for k in endpoints.keys() if k.startswith("secondary_")])
        report += f"  {condition}: {primary_count} primary, {secondary_count} secondary endpoints\n"
    
    report += """
================================================================================
Next Steps:
1. Use this data to train ML models for risk prediction
2. Build API endpoints around this intelligence
3. Create dashboard visualizations
4. Pitch to Jeeva!
================================================================================
"""
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Build TrialIntel intelligence database from ClinicalTrials.gov"
    )
    parser.add_argument(
        "--condition",
        type=str,
        help="Specific condition to fetch (e.g., 'diabetes', 'breast cancer')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch trials across multiple conditions"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum trials per condition (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for data files"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_trials = []
    
    if args.condition:
        # Fetch single condition
        trials = fetch_trials_for_condition(args.condition, args.limit)
        all_trials.extend(trials)
        save_trials_to_file(trials, f"{args.output_dir}/{args.condition.replace(' ', '_')}_trials.json")
    
    elif args.all:
        # Fetch multiple conditions
        conditions = [
            "diabetes",
            "breast cancer",
            "lung cancer",
            "heart failure",
            "alzheimer",
            "rheumatoid arthritis",
        ]
        
        for condition in conditions:
            print(f"\n{'='*50}")
            print(f"Processing: {condition}")
            print('='*50)
            
            trials = fetch_trials_for_condition(condition, args.limit)
            all_trials.extend(trials)
            save_trials_to_file(trials, f"{args.output_dir}/{condition.replace(' ', '_')}_trials.json")
    
    else:
        print("Please specify --condition or --all")
        print("Example: python build_database.py --condition 'diabetes' --limit 100")
        return
    
    if all_trials:
        # Build intelligence
        print(f"\n{'='*50}")
        print("Building Intelligence")
        print('='*50)
        
        site_intel = build_site_intelligence(all_trials)
        endpoint_intel = build_endpoint_intelligence(all_trials)
        
        # Generate and print report
        report = generate_summary_report(all_trials, site_intel, endpoint_intel)
        print(report)
        
        # Save report
        report_file = f"{args.output_dir}/build_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {report_file}")
        
        # Save combined data
        combined_file = f"{args.output_dir}/all_trials.json"
        save_trials_to_file(all_trials, combined_file)


if __name__ == "__main__":
    main()
