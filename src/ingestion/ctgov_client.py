"""
ClinicalTrials.gov API Client

This module handles all interactions with the ClinicalTrials.gov API (v2).
Documentation: https://clinicaltrials.gov/data-api/api

Key data we extract for TrialIntel:
- Trial metadata (NCT ID, status, phase, dates)
- Eligibility criteria (for optimization analysis)
- Endpoints/outcomes (for benchmarking)
- Sites/investigators (for recommendations)
- Sponsors (for competitive intelligence)
- Historical changes (for amendment prediction)
"""

import requests
import time
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class TrialData:
    """Structured representation of a clinical trial."""
    nct_id: str
    title: str
    status: str
    phase: List[str]
    conditions: List[str]
    interventions: List[str]
    sponsor: str
    collaborators: List[str]
    enrollment: Optional[int]
    start_date: Optional[str]
    completion_date: Optional[str]
    primary_completion_date: Optional[str]
    study_type: str
    eligibility_criteria: Optional[str]
    minimum_age: Optional[str]
    maximum_age: Optional[str]
    sex: Optional[str]
    primary_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    secondary_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    locations: List[Dict[str, Any]] = field(default_factory=list)
    last_update_date: Optional[str] = None
    results_first_posted: Optional[str] = None
    why_stopped: Optional[str] = None


class ClinicalTrialsGovClient:
    """
    Client for ClinicalTrials.gov API v2.
    
    Example usage:
        client = ClinicalTrialsGovClient()
        
        # Search for diabetes trials
        trials = client.search_trials(condition="diabetes", phase=["PHASE3"])
        
        # Get detailed info for a specific trial
        trial = client.get_trial("NCT03689374")
        
        # Stream all trials for a condition (for bulk analysis)
        for trial in client.stream_all_trials(condition="breast cancer"):
            process(trial)
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    # Fields we need for analysis
    ESSENTIAL_FIELDS = [
        "NCTId",
        "BriefTitle",
        "OfficialTitle",
        "OverallStatus",
        "Phase",
        "Condition",
        "InterventionName",
        "InterventionType",
        "LeadSponsorName",
        "CollaboratorName",
        "EnrollmentCount",
        "EnrollmentType",
        "StartDate",
        "CompletionDate",
        "PrimaryCompletionDate",
        "StudyType",
        "EligibilityCriteria",
        "MinimumAge",
        "MaximumAge",
        "Sex",
        "PrimaryOutcomeMeasure",
        "PrimaryOutcomeTimeFrame",
        "SecondaryOutcomeMeasure",
        "SecondaryOutcomeTimeFrame",
        "LocationFacility",
        "LocationCity",
        "LocationState",
        "LocationCountry",
        "LastUpdatePostDate",
        "ResultsFirstPostDate",
        "WhyStopped",
        "ResponsiblePartyInvestigatorFullName",
        "OverallOfficialName",
        "OverallOfficialAffiliation",
    ]
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """
        Initialize the client.
        
        Args:
            rate_limit_delay: Seconds to wait between API calls (be nice to their servers)
        """
        self.session = requests.Session()
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a rate-limited request to the API."""
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def search_trials(
        self,
        condition: Optional[str] = None,
        intervention: Optional[str] = None,
        sponsor: Optional[str] = None,
        status: Optional[List[str]] = None,
        phase: Optional[List[str]] = None,
        location: Optional[str] = None,
        start_date_from: Optional[str] = None,
        start_date_to: Optional[str] = None,
        page_size: int = 100,
        page_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for clinical trials.
        
        Args:
            condition: Disease/condition (e.g., "diabetes", "breast cancer")
            intervention: Drug/treatment name (e.g., "pembrolizumab")
            sponsor: Sponsor organization (e.g., "Pfizer")
            status: Trial statuses (e.g., ["RECRUITING", "COMPLETED"])
            phase: Trial phases (e.g., ["PHASE3"])
            location: Geographic location (e.g., "United States")
            start_date_from: Trials started after this date (YYYY-MM-DD)
            start_date_to: Trials started before this date (YYYY-MM-DD)
            page_size: Results per page (max 1000)
            page_token: Token for pagination
            
        Returns:
            Dict with 'studies' list and 'nextPageToken'
        """
        params = {
            "format": "json",
            "pageSize": min(page_size, 1000),
            "fields": "|".join(self.ESSENTIAL_FIELDS),
        }
        
        # Build query components
        query_parts = []
        
        if condition:
            query_parts.append(f"AREA[Condition]{condition}")
        if intervention:
            query_parts.append(f"AREA[InterventionName]{intervention}")
        if sponsor:
            query_parts.append(f"AREA[LeadSponsorName]{sponsor}")
        if location:
            query_parts.append(f"AREA[LocationCountry]{location}")
        
        if query_parts:
            params["query.term"] = " AND ".join(query_parts)
        
        if status:
            params["filter.overallStatus"] = ",".join(status)
        
        if phase:
            params["filter.phase"] = ",".join(phase)
        
        if start_date_from or start_date_to:
            date_from = start_date_from or "MIN"
            date_to = start_date_to or "MAX"
            params["filter.advanced"] = f"AREA[StartDate]RANGE[{date_from},{date_to}]"
        
        if page_token:
            params["pageToken"] = page_token
        
        return self._make_request("studies", params)
    
    def get_trial(self, nct_id: str) -> Optional[TrialData]:
        """
        Get detailed information for a specific trial.
        
        Args:
            nct_id: NCT identifier (e.g., "NCT03689374")
            
        Returns:
            TrialData object or None if not found
        """
        params = {
            "format": "json",
            "fields": "|".join(self.ESSENTIAL_FIELDS),
        }
        
        try:
            response = self._make_request(f"studies/{nct_id}", params)
            return self._parse_trial(response.get("protocolSection", {}))
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def stream_all_trials(
        self,
        condition: Optional[str] = None,
        intervention: Optional[str] = None,
        sponsor: Optional[str] = None,
        status: Optional[List[str]] = None,
        phase: Optional[List[str]] = None,
        max_trials: Optional[int] = None,
        **kwargs
    ) -> Generator[TrialData, None, None]:
        """
        Stream all trials matching criteria (handles pagination automatically).
        
        Args:
            condition: Disease/condition filter
            intervention: Drug/treatment filter  
            sponsor: Sponsor filter
            status: Status filter
            phase: Phase filter
            max_trials: Maximum number of trials to return
            **kwargs: Additional search parameters
            
        Yields:
            TrialData objects
        """
        count = 0
        page_token = None
        
        while True:
            response = self.search_trials(
                condition=condition,
                intervention=intervention,
                sponsor=sponsor,
                status=status,
                phase=phase,
                page_token=page_token,
                **kwargs
            )
            
            studies = response.get("studies", [])
            if not studies:
                break
            
            for study in studies:
                trial = self._parse_trial(study.get("protocolSection", {}))
                if trial:
                    yield trial
                    count += 1
                    
                    if max_trials and count >= max_trials:
                        return
            
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    
    def _parse_trial(self, protocol: Dict[str, Any]) -> Optional[TrialData]:
        """Parse API response into TrialData object."""
        if not protocol:
            return None
        
        identification = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        design_module = protocol.get("designModule", {})
        eligibility_module = protocol.get("eligibilityModule", {})
        outcomes_module = protocol.get("outcomesModule", {})
        contacts_module = protocol.get("contactsLocationsModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        interventions_module = protocol.get("armsInterventionsModule", {})
        
        # Extract sponsor info
        lead_sponsor = sponsor_module.get("leadSponsor", {})
        collaborators = [
            c.get("name", "") 
            for c in sponsor_module.get("collaborators", [])
        ]
        
        # Extract interventions
        interventions = []
        for interv in interventions_module.get("interventions", []):
            name = interv.get("name", "")
            if name:
                interventions.append(name)
        
        # Extract primary outcomes
        primary_outcomes = []
        for outcome in outcomes_module.get("primaryOutcomes", []):
            primary_outcomes.append({
                "measure": outcome.get("measure", ""),
                "timeFrame": outcome.get("timeFrame", ""),
                "description": outcome.get("description", ""),
            })
        
        # Extract secondary outcomes
        secondary_outcomes = []
        for outcome in outcomes_module.get("secondaryOutcomes", []):
            secondary_outcomes.append({
                "measure": outcome.get("measure", ""),
                "timeFrame": outcome.get("timeFrame", ""),
                "description": outcome.get("description", ""),
            })
        
        # Extract locations
        locations = []
        for loc in contacts_module.get("locations", []):
            locations.append({
                "facility": loc.get("facility", ""),
                "city": loc.get("city", ""),
                "state": loc.get("state", ""),
                "country": loc.get("country", ""),
                "status": loc.get("status", ""),
            })
        
        # Parse dates
        start_date = status_module.get("startDateStruct", {}).get("date")
        completion_date = status_module.get("completionDateStruct", {}).get("date")
        primary_completion = status_module.get("primaryCompletionDateStruct", {}).get("date")
        
        return TrialData(
            nct_id=identification.get("nctId", ""),
            title=identification.get("officialTitle") or identification.get("briefTitle", ""),
            status=status_module.get("overallStatus", ""),
            phase=design_module.get("phases", []),
            conditions=conditions_module.get("conditions", []),
            interventions=interventions,
            sponsor=lead_sponsor.get("name", ""),
            collaborators=collaborators,
            enrollment=design_module.get("enrollmentInfo", {}).get("count"),
            start_date=start_date,
            completion_date=completion_date,
            primary_completion_date=primary_completion,
            study_type=design_module.get("studyType", ""),
            eligibility_criteria=eligibility_module.get("eligibilityCriteria"),
            minimum_age=eligibility_module.get("minimumAge"),
            maximum_age=eligibility_module.get("maximumAge"),
            sex=eligibility_module.get("sex"),
            primary_outcomes=primary_outcomes,
            secondary_outcomes=secondary_outcomes,
            locations=locations,
            last_update_date=status_module.get("lastUpdatePostDateStruct", {}).get("date"),
            results_first_posted=status_module.get("resultsFirstPostDateStruct", {}).get("date"),
            why_stopped=status_module.get("whyStopped"),
        )
    
    def get_trial_history(self, nct_id: str) -> List[Dict[str, Any]]:
        """
        Get the history of changes for a trial (useful for amendment analysis).
        
        Note: This requires the history API endpoint which may have different
        rate limits.
        
        Args:
            nct_id: NCT identifier
            
        Returns:
            List of historical versions with change dates
        """
        params = {
            "format": "json",
        }
        
        try:
            response = self._make_request(f"studies/{nct_id}/history", params)
            return response.get("changes", [])
        except requests.exceptions.HTTPError:
            return []


# Convenience functions for common queries
def get_completed_trials_by_condition(
    condition: str, 
    phase: Optional[List[str]] = None,
    max_trials: int = 1000
) -> List[TrialData]:
    """Get completed trials for analysis."""
    client = ClinicalTrialsGovClient()
    return list(client.stream_all_trials(
        condition=condition,
        status=["COMPLETED"],
        phase=phase,
        max_trials=max_trials
    ))


def get_terminated_trials(
    condition: Optional[str] = None,
    max_trials: int = 500
) -> List[TrialData]:
    """Get terminated trials (useful for failure analysis)."""
    client = ClinicalTrialsGovClient()
    return list(client.stream_all_trials(
        condition=condition,
        status=["TERMINATED", "WITHDRAWN", "SUSPENDED"],
        max_trials=max_trials
    ))


def get_trials_by_sponsor(
    sponsor: str,
    status: Optional[List[str]] = None,
    max_trials: int = 500
) -> List[TrialData]:
    """Get all trials by a specific sponsor."""
    client = ClinicalTrialsGovClient()
    return list(client.stream_all_trials(
        sponsor=sponsor,
        status=status,
        max_trials=max_trials
    ))


if __name__ == "__main__":
    # Quick test
    client = ClinicalTrialsGovClient()
    
    print("Testing ClinicalTrials.gov API client...")
    
    # Test search
    results = client.search_trials(
        condition="diabetes",
        phase=["PHASE3"],
        status=["COMPLETED"],
        page_size=5
    )
    
    print(f"Found {len(results.get('studies', []))} trials")
    
    # Test single trial fetch
    trial = client.get_trial("NCT03689374")
    if trial:
        print(f"\nTrial: {trial.nct_id}")
        print(f"Title: {trial.title[:80]}...")
        print(f"Sponsor: {trial.sponsor}")
        print(f"Phase: {trial.phase}")
        print(f"Enrollment: {trial.enrollment}")
        print(f"Primary Outcomes: {len(trial.primary_outcomes)}")
        print(f"Locations: {len(trial.locations)}")
