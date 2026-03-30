#!/usr/bin/env python3
"""
Fetch ALL Trials from ClinicalTrials.gov

This script fetches all trials from ClinicalTrials.gov API and inserts them
into the TrialIntel database. It handles:
- Resumable fetching (tracks progress)
- Pagination
- Rate limiting
- Duplicate detection

Usage:
    python fetch_all_trials.py
    python fetch_all_trials.py --resume  # Resume from last checkpoint
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Set, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests
from sqlalchemy import text


def get_existing_nct_ids(db) -> Set[str]:
    """Get all NCT IDs already in the database."""
    print("Loading existing NCT IDs from database...")
    with db.session() as session:
        result = session.execute(text("SELECT nct_id FROM trials"))
        existing = {row[0] for row in result}
    print(f"Found {len(existing):,} existing trials")
    return existing


def fetch_trials_page(session: requests.Session, page_token: Optional[str] = None) -> Dict[str, Any]:
    """Fetch a page of trials from ClinicalTrials.gov API."""
    url = "https://clinicaltrials.gov/api/v2/studies"

    params = {
        "format": "json",
        "pageSize": 1000,
        "fields": "|".join([
            "NCTId", "BriefTitle", "OfficialTitle", "OverallStatus", "Phase",
            "Condition", "InterventionName", "InterventionType", "LeadSponsorName",
            "CollaboratorName", "EnrollmentCount", "EnrollmentType", "StartDate",
            "CompletionDate", "PrimaryCompletionDate", "StudyType", "EligibilityCriteria",
            "MinimumAge", "MaximumAge", "Sex", "PrimaryOutcomeMeasure", "PrimaryOutcomeTimeFrame",
            "SecondaryOutcomeMeasure", "SecondaryOutcomeTimeFrame", "LocationFacility",
            "LocationCity", "LocationState", "LocationCountry", "LastUpdatePostDate",
            "ResultsFirstPostDate", "WhyStopped", "OverallOfficialName",
            "OverallOfficialAffiliation", "DesignAllocation", "DesignInterventionModel",
            "DesignPrimaryPurpose", "DesignMasking"
        ])
    }

    if page_token:
        params["pageToken"] = page_token

    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def parse_trial(study: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a study from the API response into our database format."""
    protocol = study.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    status_module = protocol.get("statusModule", {})
    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
    design_module = protocol.get("designModule", {})
    eligibility_module = protocol.get("eligibilityModule", {})
    outcomes_module = protocol.get("outcomesModule", {})
    contacts_module = protocol.get("contactsLocationsModule", {})
    arms_module = protocol.get("armsInterventionsModule", {})
    conditions_module = protocol.get("conditionsModule", {})

    # Extract interventions
    interventions = []
    for intervention in arms_module.get("interventions", []):
        name = intervention.get("name", "")
        int_type = intervention.get("type", "")
        if name:
            interventions.append(f"{int_type}: {name}" if int_type else name)

    # Extract locations
    locations = []
    for loc in contacts_module.get("locations", []):
        locations.append({
            "facility": loc.get("facility", ""),
            "city": loc.get("city", ""),
            "state": loc.get("state", ""),
            "country": loc.get("country", ""),
        })

    # Extract outcomes
    primary_outcomes = []
    for outcome in outcomes_module.get("primaryOutcomes", []):
        primary_outcomes.append({
            "measure": outcome.get("measure", ""),
            "timeFrame": outcome.get("timeFrame", ""),
        })

    secondary_outcomes = []
    for outcome in outcomes_module.get("secondaryOutcomes", []):
        secondary_outcomes.append({
            "measure": outcome.get("measure", ""),
            "timeFrame": outcome.get("timeFrame", ""),
        })

    # Get phases (can be a list)
    phases = design_module.get("phases", [])
    phase = phases[0] if phases else None

    # Extract conditions
    conditions = conditions_module.get("conditions", [])

    # Determine therapeutic area from conditions
    therapeutic_area = None
    if conditions:
        # Use first condition as therapeutic area, simplified
        ta = conditions[0].lower()
        if "diabetes" in ta:
            therapeutic_area = "type 2 diabetes" if "type 2" in ta else "diabetes"
        elif "cancer" in ta or "carcinoma" in ta or "tumor" in ta:
            therapeutic_area = ta.replace("carcinoma", "cancer").split(",")[0].strip()
        elif "covid" in ta or "coronavirus" in ta:
            therapeutic_area = "COVID-19"
        else:
            therapeutic_area = ta.split(",")[0].strip()[:100]

    # Get sponsor
    sponsor = None
    sponsor_type = None
    lead_sponsor = sponsor_module.get("leadSponsor", {})
    if lead_sponsor:
        sponsor = lead_sponsor.get("name")
        sponsor_type = lead_sponsor.get("class")  # INDUSTRY, NIH, etc.

    # Parse enrollment
    enrollment_info = design_module.get("enrollmentInfo", {})
    enrollment = enrollment_info.get("count")
    enrollment_type = enrollment_info.get("type")

    return {
        "nct_id": id_module.get("nctId"),
        "title": id_module.get("briefTitle") or id_module.get("officialTitle"),
        "status": status_module.get("overallStatus"),
        "phase": phase,
        "study_type": design_module.get("studyType"),
        "conditions": ", ".join(conditions) if conditions else None,
        "interventions": ", ".join(interventions) if interventions else None,
        "therapeutic_area": therapeutic_area,
        "sponsor": sponsor,
        "sponsor_type": sponsor_type,
        "enrollment": enrollment,
        "enrollment_type": enrollment_type,
        "start_date": status_module.get("startDateStruct", {}).get("date"),
        "completion_date": status_module.get("completionDateStruct", {}).get("date"),
        "primary_completion_date": status_module.get("primaryCompletionDateStruct", {}).get("date"),
        "eligibility_criteria": eligibility_module.get("eligibilityCriteria"),
        "min_age": eligibility_module.get("minimumAge"),
        "max_age": eligibility_module.get("maximumAge"),
        "sex": eligibility_module.get("sex"),
        "primary_outcomes": json.dumps(primary_outcomes) if primary_outcomes else None,
        "secondary_outcomes": json.dumps(secondary_outcomes) if secondary_outcomes else None,
        "locations": json.dumps(locations) if locations else None,
        "num_sites": len(locations) if locations else 0,
        "why_stopped": status_module.get("whyStopped"),
        "has_results": study.get("hasResults", False),
    }


def insert_trials(db, trials: list) -> int:
    """Insert trials into the database."""
    if not trials:
        return 0

    inserted = 0
    with db.session() as session:
        for trial in trials:
            try:
                # Use INSERT OR REPLACE for SQLite
                columns = ", ".join(trial.keys())
                placeholders = ", ".join([f":{k}" for k in trial.keys()])

                session.execute(
                    text(f"INSERT OR REPLACE INTO trials ({columns}) VALUES ({placeholders})"),
                    trial
                )
                inserted += 1
            except Exception as e:
                print(f"Error inserting {trial.get('nct_id')}: {e}")

        session.commit()

    return inserted


def save_checkpoint(page_token: str, total_fetched: int):
    """Save progress checkpoint."""
    checkpoint = {
        "page_token": page_token,
        "total_fetched": total_fetched,
        "timestamp": datetime.now().isoformat()
    }
    with open("data/fetch_checkpoint.json", "w") as f:
        json.dump(checkpoint, f)


def load_checkpoint() -> Optional[Dict[str, Any]]:
    """Load progress checkpoint if exists."""
    try:
        with open("data/fetch_checkpoint.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch all trials from ClinicalTrials.gov")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    from src.database import DatabaseManager
    db = DatabaseManager.get_instance()

    # Get existing NCT IDs for duplicate detection
    existing_ids = get_existing_nct_ids(db)

    # Check for checkpoint
    page_token = None
    total_fetched = 0

    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            page_token = checkpoint.get("page_token")
            total_fetched = checkpoint.get("total_fetched", 0)
            print(f"Resuming from checkpoint: {total_fetched:,} trials fetched")

    # Create session with retries
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("https://", adapter)

    print("Starting to fetch ALL trials from ClinicalTrials.gov...")
    print(f"Target: ~566,000 trials")
    print("-" * 60)

    new_trials = 0
    updated_trials = 0
    batch = []
    batch_size = 500  # Insert in batches for efficiency

    start_time = time.time()

    while True:
        try:
            # Fetch page
            response = fetch_trials_page(session, page_token)
            studies = response.get("studies", [])

            if not studies:
                print("No more studies to fetch")
                break

            # Parse and collect trials
            for study in studies:
                trial = parse_trial(study)
                if trial.get("nct_id"):
                    if trial["nct_id"] in existing_ids:
                        updated_trials += 1
                    else:
                        new_trials += 1
                        existing_ids.add(trial["nct_id"])

                    batch.append(trial)
                    total_fetched += 1

            # Insert batch
            if len(batch) >= batch_size:
                inserted = insert_trials(db, batch)
                batch = []

                # Progress update
                elapsed = time.time() - start_time
                rate = total_fetched / elapsed if elapsed > 0 else 0
                remaining = (566622 - total_fetched) / rate if rate > 0 else 0

                print(f"Progress: {total_fetched:,} trials | New: {new_trials:,} | "
                      f"Updated: {updated_trials:,} | Rate: {rate:.0f}/sec | "
                      f"ETA: {remaining/60:.0f} min")

            # Get next page token
            page_token = response.get("nextPageToken")

            if not page_token:
                print("Reached end of results")
                break

            # Save checkpoint every 5000 trials
            if total_fetched % 5000 == 0:
                save_checkpoint(page_token, total_fetched)

            # Rate limiting
            time.sleep(0.05)  # 50ms between requests

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            print("Saving checkpoint and waiting 30 seconds...")
            save_checkpoint(page_token, total_fetched)
            time.sleep(30)
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            save_checkpoint(page_token, total_fetched)
            raise

    # Insert remaining batch
    if batch:
        insert_trials(db, batch)

    # Final stats
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Total trials fetched: {total_fetched:,}")
    print(f"New trials added: {new_trials:,}")
    print(f"Existing trials updated: {updated_trials:,}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print("=" * 60)

    # Verify final count
    stats = db.get_stats()
    print(f"\nDatabase now contains: {stats['total_trials']:,} trials")

    # Remove checkpoint file
    try:
        os.remove("data/fetch_checkpoint.json")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
