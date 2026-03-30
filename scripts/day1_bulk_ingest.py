#!/usr/bin/env python3
"""
BULK DATA INGESTION - Day 1 Sprint

This script downloads and processes 50,000+ trials from ClinicalTrials.gov
in a single day. Run overnight if needed.

Target: 50,000 trials across key therapeutic areas

Usage:
    python day1_bulk_ingest.py --output ./data
    
Estimated time: 4-8 hours (API rate limits)
"""

import requests
import json
import time
import os
import sqlite3
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import argparse


# ClinicalTrials.gov API v2
BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Fields we need for ML training
FIELDS = [
    "NCTId", "BriefTitle", "OfficialTitle", "OverallStatus", "Phase",
    "Condition", "InterventionName", "LeadSponsorName", "EnrollmentCount",
    "StartDate", "CompletionDate", "PrimaryCompletionDate", "StudyType",
    "EligibilityCriteria", "MinimumAge", "MaximumAge", "Sex",
    "PrimaryOutcomeMeasure", "PrimaryOutcomeTimeFrame",
    "SecondaryOutcomeMeasure", "SecondaryOutcomeTimeFrame",
    "LocationFacility", "LocationCity", "LocationState", "LocationCountry",
    "WhyStopped", "ResultsFirstPostDate", "LastUpdatePostDate",
    "DesignAllocation", "DesignInterventionModel", "DesignPrimaryPurpose",
    "DesignMasking", "NumberOfArms", "NumberOfGroups",
    "ResponsiblePartyInvestigatorFullName", "OverallOfficialName",
    "OverallOfficialAffiliation"
]

# Therapeutic areas to target (covers ~80% of industry trials)
THERAPEUTIC_AREAS = [
    ("diabetes", 8000),
    ("breast cancer", 6000),
    ("lung cancer", 5000),
    ("heart failure", 4000),
    ("hypertension", 3000),
    ("alzheimer", 2000),
    ("parkinson", 1500),
    ("rheumatoid arthritis", 2500),
    ("psoriasis", 2000),
    ("multiple sclerosis", 1500),
    ("hepatitis", 2000),
    ("HIV", 2500),
    ("asthma", 2500),
    ("COPD", 2000),
    ("depression", 3000),
    ("schizophrenia", 1500),
    ("obesity", 2500),
    ("chronic kidney disease", 1500),
    ("lymphoma", 2000),
    ("leukemia", 2000),
]


class BulkIngester:
    """High-speed bulk ingestion from ClinicalTrials.gov"""
    
    def __init__(self, output_dir: str, db_path: str = None):
        self.output_dir = output_dir
        self.db_path = db_path or os.path.join(output_dir, "trials.db")
        os.makedirs(output_dir, exist_ok=True)
        self._init_database()
        self.stats = {
            "total_fetched": 0,
            "total_stored": 0,
            "errors": 0,
            "start_time": datetime.now()
        }
    
    def _init_database(self):
        """Initialize SQLite database for fast storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main trials table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                nct_id TEXT PRIMARY KEY,
                title TEXT,
                status TEXT,
                phase TEXT,
                conditions TEXT,
                interventions TEXT,
                sponsor TEXT,
                enrollment INTEGER,
                start_date TEXT,
                completion_date TEXT,
                study_type TEXT,
                eligibility_criteria TEXT,
                min_age TEXT,
                max_age TEXT,
                sex TEXT,
                primary_outcomes TEXT,
                secondary_outcomes TEXT,
                locations TEXT,
                why_stopped TEXT,
                has_results INTEGER,
                therapeutic_area TEXT,
                raw_json TEXT,
                ingested_at TEXT
            )
        """)
        
        # Index for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON trials(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase ON trials(phase)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_therapeutic_area ON trials(therapeutic_area)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sponsor ON trials(sponsor)")
        
        conn.commit()
        conn.close()
        print(f"Database initialized: {self.db_path}")
    
    def fetch_trials(
        self, 
        condition: str, 
        max_trials: int = 1000,
        page_size: int = 100
    ) -> List[Dict]:
        """Fetch trials for a condition."""
        trials = []
        page_token = None
        
        while len(trials) < max_trials:
            params = {
                "format": "json",
                "pageSize": min(page_size, max_trials - len(trials)),
                "fields": "|".join(FIELDS),
                "query.cond": condition,
            }
            
            if page_token:
                params["pageToken"] = page_token
            
            try:
                response = requests.get(BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                studies = data.get("studies", [])
                if not studies:
                    break
                
                trials.extend(studies)
                page_token = data.get("nextPageToken")
                
                if not page_token:
                    break
                
                # Rate limiting - be nice to their servers
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  Error fetching {condition}: {e}")
                self.stats["errors"] += 1
                break
        
        return trials
    
    def parse_trial(self, study: Dict, therapeutic_area: str) -> Dict:
        """Parse raw API response into clean format."""
        protocol = study.get("protocolSection", {})
        
        identification = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        design_module = protocol.get("designModule", {})
        eligibility_module = protocol.get("eligibilityModule", {})
        outcomes_module = protocol.get("outcomesModule", {})
        contacts_module = protocol.get("contactsLocationsModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        interventions_module = protocol.get("armsInterventionsModule", {})
        
        # Extract fields
        nct_id = identification.get("nctId", "")
        
        # Interventions
        interventions = []
        for interv in interventions_module.get("interventions", []):
            if interv.get("name"):
                interventions.append(interv["name"])
        
        # Primary outcomes
        primary_outcomes = []
        for outcome in outcomes_module.get("primaryOutcomes", []):
            primary_outcomes.append({
                "measure": outcome.get("measure", ""),
                "timeFrame": outcome.get("timeFrame", ""),
            })
        
        # Secondary outcomes
        secondary_outcomes = []
        for outcome in outcomes_module.get("secondaryOutcomes", []):
            secondary_outcomes.append({
                "measure": outcome.get("measure", ""),
                "timeFrame": outcome.get("timeFrame", ""),
            })
        
        # Locations
        locations = []
        for loc in contacts_module.get("locations", []):
            locations.append({
                "facility": loc.get("facility", ""),
                "city": loc.get("city", ""),
                "state": loc.get("state", ""),
                "country": loc.get("country", ""),
            })
        
        return {
            "nct_id": nct_id,
            "title": identification.get("officialTitle") or identification.get("briefTitle", ""),
            "status": status_module.get("overallStatus", ""),
            "phase": ",".join(design_module.get("phases", [])),
            "conditions": ",".join(conditions_module.get("conditions", [])),
            "interventions": ",".join(interventions),
            "sponsor": sponsor_module.get("leadSponsor", {}).get("name", ""),
            "enrollment": design_module.get("enrollmentInfo", {}).get("count"),
            "start_date": status_module.get("startDateStruct", {}).get("date"),
            "completion_date": status_module.get("completionDateStruct", {}).get("date"),
            "study_type": design_module.get("studyType", ""),
            "eligibility_criteria": eligibility_module.get("eligibilityCriteria", ""),
            "min_age": eligibility_module.get("minimumAge", ""),
            "max_age": eligibility_module.get("maximumAge", ""),
            "sex": eligibility_module.get("sex", ""),
            "primary_outcomes": json.dumps(primary_outcomes),
            "secondary_outcomes": json.dumps(secondary_outcomes),
            "locations": json.dumps(locations[:50]),  # Limit to avoid huge rows
            "why_stopped": status_module.get("whyStopped", ""),
            "has_results": 1 if status_module.get("resultsFirstPostDateStruct") else 0,
            "therapeutic_area": therapeutic_area,
            "raw_json": json.dumps(study),
            "ingested_at": datetime.now().isoformat(),
        }
    
    def store_trials(self, trials: List[Dict]):
        """Store trials in SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for trial in trials:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO trials 
                    (nct_id, title, status, phase, conditions, interventions,
                     sponsor, enrollment, start_date, completion_date, study_type,
                     eligibility_criteria, min_age, max_age, sex, primary_outcomes,
                     secondary_outcomes, locations, why_stopped, has_results,
                     therapeutic_area, raw_json, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trial["nct_id"], trial["title"], trial["status"], trial["phase"],
                    trial["conditions"], trial["interventions"], trial["sponsor"],
                    trial["enrollment"], trial["start_date"], trial["completion_date"],
                    trial["study_type"], trial["eligibility_criteria"], trial["min_age"],
                    trial["max_age"], trial["sex"], trial["primary_outcomes"],
                    trial["secondary_outcomes"], trial["locations"], trial["why_stopped"],
                    trial["has_results"], trial["therapeutic_area"], trial["raw_json"],
                    trial["ingested_at"]
                ))
                self.stats["total_stored"] += 1
            except Exception as e:
                print(f"  Error storing {trial.get('nct_id')}: {e}")
                self.stats["errors"] += 1
        
        conn.commit()
        conn.close()
    
    def ingest_therapeutic_area(self, condition: str, max_trials: int) -> int:
        """Ingest trials for one therapeutic area."""
        print(f"\n{'='*50}")
        print(f"Ingesting: {condition} (target: {max_trials})")
        print('='*50)
        
        # Fetch from API
        raw_trials = self.fetch_trials(condition, max_trials)
        print(f"  Fetched: {len(raw_trials)} trials")
        self.stats["total_fetched"] += len(raw_trials)
        
        # Parse trials
        parsed_trials = []
        for study in raw_trials:
            try:
                parsed = self.parse_trial(study, condition)
                if parsed["nct_id"]:
                    parsed_trials.append(parsed)
            except Exception as e:
                self.stats["errors"] += 1
        
        print(f"  Parsed: {len(parsed_trials)} trials")
        
        # Store in database
        self.store_trials(parsed_trials)
        print(f"  Stored: {len(parsed_trials)} trials")
        
        return len(parsed_trials)
    
    def run_full_ingestion(self):
        """Run full ingestion across all therapeutic areas."""
        print("\n" + "="*60)
        print("TRIALINTEL BULK DATA INGESTION")
        print(f"Started: {self.stats['start_time']}")
        print(f"Target: {sum(t[1] for t in THERAPEUTIC_AREAS)} trials")
        print("="*60)
        
        for condition, target in THERAPEUTIC_AREAS:
            self.ingest_therapeutic_area(condition, target)
        
        # Print summary
        elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        print("\n" + "="*60)
        print("INGESTION COMPLETE")
        print("="*60)
        print(f"Total fetched: {self.stats['total_fetched']}")
        print(f"Total stored: {self.stats['total_stored']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"Database: {self.db_path}")
        print(f"Database size: {os.path.getsize(self.db_path) / 1024 / 1024:.1f} MB")
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total trials
        cursor.execute("SELECT COUNT(*) FROM trials")
        stats["total_trials"] = cursor.fetchone()[0]
        
        # By status
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM trials 
            GROUP BY status 
            ORDER BY COUNT(*) DESC
        """)
        stats["by_status"] = dict(cursor.fetchall())
        
        # By phase
        cursor.execute("""
            SELECT phase, COUNT(*) 
            FROM trials 
            GROUP BY phase 
            ORDER BY COUNT(*) DESC
        """)
        stats["by_phase"] = dict(cursor.fetchall())
        
        # By therapeutic area
        cursor.execute("""
            SELECT therapeutic_area, COUNT(*) 
            FROM trials 
            GROUP BY therapeutic_area 
            ORDER BY COUNT(*) DESC
        """)
        stats["by_therapeutic_area"] = dict(cursor.fetchall())
        
        # Top sponsors
        cursor.execute("""
            SELECT sponsor, COUNT(*) 
            FROM trials 
            GROUP BY sponsor 
            ORDER BY COUNT(*) DESC 
            LIMIT 20
        """)
        stats["top_sponsors"] = dict(cursor.fetchall())
        
        # Trials with results
        cursor.execute("SELECT COUNT(*) FROM trials WHERE has_results = 1")
        stats["trials_with_results"] = cursor.fetchone()[0]
        
        # Terminated trials (for failure analysis)
        cursor.execute("""
            SELECT COUNT(*) FROM trials 
            WHERE status IN ('TERMINATED', 'WITHDRAWN', 'SUSPENDED')
        """)
        stats["failed_trials"] = cursor.fetchone()[0]
        
        conn.close()
        return stats


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest ClinicalTrials.gov data")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test (100 trials each)")
    args = parser.parse_args()
    
    ingester = BulkIngester(args.output)
    
    if args.quick:
        # Quick test mode
        print("Running in QUICK TEST mode (100 trials per area)")
        for i, (condition, _) in enumerate(THERAPEUTIC_AREAS[:3]):
            ingester.ingest_therapeutic_area(condition, 100)
    else:
        # Full ingestion
        ingester.run_full_ingestion()
    
    # Print stats
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    stats = ingester.get_stats()
    print(f"Total trials: {stats['total_trials']}")
    print(f"Trials with results: {stats['trials_with_results']}")
    print(f"Failed trials: {stats['failed_trials']}")
    print(f"\nBy therapeutic area:")
    for area, count in list(stats['by_therapeutic_area'].items())[:10]:
        print(f"  {area}: {count}")


if __name__ == "__main__":
    main()
