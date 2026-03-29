"""
FDA Data Integration Client

This module handles interactions with FDA data sources:
- OpenFDA API (drug approvals, adverse events, recalls)
- FDA Drugs@FDA database (approval history, labels)

Key data we extract for TrialIntel:
- Drug approval timelines and outcomes
- Adverse event reports (FAERS)
- Clinical hold information
- Label changes and safety updates
- Regulatory pathway information

Documentation:
- OpenFDA: https://open.fda.gov/apis/
- Drugs@FDA: https://www.accessdata.fda.gov/scripts/cder/daf/
"""

import requests
import time
import logging
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class DrugApplicationType(Enum):
    """FDA drug application types."""
    NDA = "NDA"  # New Drug Application
    BLA = "BLA"  # Biologics License Application
    ANDA = "ANDA"  # Abbreviated NDA (generic)
    SUPPLEMENT = "SUPPLEMENT"


class ApprovalStatus(Enum):
    """FDA approval status types."""
    APPROVED = "approved"
    TENTATIVE = "tentative_approval"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"


@dataclass
class DrugApproval:
    """Structured representation of an FDA drug approval."""
    application_number: str
    drug_name: str
    brand_name: Optional[str]
    sponsor_name: str
    application_type: str
    approval_date: Optional[str]
    submission_date: Optional[str]
    therapeutic_area: Optional[str]
    route: Optional[str]
    dosage_form: Optional[str]
    active_ingredients: List[str] = field(default_factory=list)
    indications: List[str] = field(default_factory=list)
    approval_pathway: Optional[str] = None  # Priority, Accelerated, Breakthrough, Fast Track
    orphan_drug: bool = False
    pediatric_exclusivity: bool = False
    review_classification: Optional[str] = None  # Standard or Priority
    submission_type: Optional[str] = None


@dataclass
class AdverseEvent:
    """Structured representation of an FDA adverse event report."""
    report_id: str
    receive_date: str
    drug_name: str
    reaction: str
    outcome: Optional[str]
    patient_age: Optional[int]
    patient_sex: Optional[str]
    patient_weight: Optional[float]
    serious: bool
    death: bool
    hospitalization: bool
    life_threatening: bool
    disability: bool
    congenital_anomaly: bool
    reporter_qualification: Optional[str]
    report_source: Optional[str]


@dataclass
class DrugLabel:
    """Structured representation of drug labeling information."""
    application_number: str
    drug_name: str
    brand_name: Optional[str]
    effective_date: Optional[str]
    boxed_warning: Optional[str]
    warnings_precautions: Optional[str]
    adverse_reactions: Optional[str]
    contraindications: Optional[str]
    indications_usage: Optional[str]
    dosage_administration: Optional[str]


class OpenFDAClient:
    """
    Client for OpenFDA API.

    Example usage:
        client = OpenFDAClient()

        # Search for drug approvals
        approvals = client.search_drug_approvals(
            drug_name="pembrolizumab",
            approval_year=2024
        )

        # Get adverse events for a drug
        events = client.search_adverse_events(
            drug_name="pembrolizumab",
            serious=True,
            limit=100
        )

        # Get drug label information
        label = client.get_drug_label("KEYTRUDA")
    """

    BASE_URL = "https://api.fda.gov"

    # Endpoints
    DRUG_APPROVALS_ENDPOINT = "/drug/drugsfda.json"
    ADVERSE_EVENTS_ENDPOINT = "/drug/event.json"
    DRUG_LABELS_ENDPOINT = "/drug/label.json"
    ENFORCEMENT_ENDPOINT = "/drug/enforcement.json"

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.2):
        """
        Initialize the client.

        Args:
            api_key: OpenFDA API key (optional, increases rate limits)
            rate_limit_delay: Seconds to wait between API calls
        """
        self.session = requests.Session()
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0

        # Set default headers
        self.session.headers.update({
            "User-Agent": "TrialIntel/1.0 (Clinical Trial Intelligence Platform)"
        })

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Make a rate-limited request to the API with retries."""
        self._rate_limit()

        url = f"{self.BASE_URL}{endpoint}"

        # Add API key if available
        if self.api_key:
            params["api_key"] = self.api_key

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch from FDA API: {e}")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

        return {}

    # =========================================================================
    # Drug Approvals (Drugs@FDA)
    # =========================================================================

    def search_drug_approvals(
        self,
        drug_name: Optional[str] = None,
        sponsor_name: Optional[str] = None,
        application_type: Optional[DrugApplicationType] = None,
        approval_year: Optional[int] = None,
        approval_date_from: Optional[str] = None,
        approval_date_to: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> Dict[str, Any]:
        """
        Search FDA drug approvals database.

        Args:
            drug_name: Generic or brand drug name
            sponsor_name: Pharmaceutical company
            application_type: NDA, BLA, or ANDA
            approval_year: Year of approval
            approval_date_from: Start date for approval range (YYYYMMDD)
            approval_date_to: End date for approval range (YYYYMMDD)
            limit: Maximum results (max 1000)
            skip: Number of results to skip (pagination)

        Returns:
            Dict with 'results' list and 'meta' pagination info
        """
        search_parts = []

        if drug_name:
            # Search in multiple fields for drug name
            search_parts.append(
                f'(openfda.brand_name:"{drug_name}" OR '
                f'openfda.generic_name:"{drug_name}" OR '
                f'products.brand_name:"{drug_name}")'
            )

        if sponsor_name:
            search_parts.append(f'sponsor_name:"{sponsor_name}"')

        if application_type:
            search_parts.append(f'application_number:"{application_type.value}*"')

        if approval_year:
            search_parts.append(
                f'submissions.submission_status_date:[{approval_year}0101 TO {approval_year}1231]'
            )

        if approval_date_from and approval_date_to:
            search_parts.append(
                f'submissions.submission_status_date:[{approval_date_from} TO {approval_date_to}]'
            )
        elif approval_date_from:
            search_parts.append(
                f'submissions.submission_status_date:[{approval_date_from} TO 99991231]'
            )
        elif approval_date_to:
            search_parts.append(
                f'submissions.submission_status_date:[00000101 TO {approval_date_to}]'
            )

        params = {
            "limit": min(limit, 1000),
            "skip": skip,
        }

        if search_parts:
            params["search"] = " AND ".join(search_parts)

        return self._make_request(self.DRUG_APPROVALS_ENDPOINT, params)

    def get_recent_approvals(
        self,
        days: int = 30,
        application_types: Optional[List[DrugApplicationType]] = None,
        limit: int = 100,
    ) -> List[DrugApproval]:
        """
        Get recent drug approvals.

        Args:
            days: Number of days to look back
            application_types: Filter by application type (NDA, BLA, etc.)
            limit: Maximum results

        Returns:
            List of DrugApproval objects
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        response = self.search_drug_approvals(
            approval_date_from=start_date.strftime("%Y%m%d"),
            approval_date_to=end_date.strftime("%Y%m%d"),
            limit=limit,
        )

        approvals = []
        for result in response.get("results", []):
            approval = self._parse_drug_approval(result)
            if approval:
                if application_types:
                    if any(at.value in approval.application_number for at in application_types):
                        approvals.append(approval)
                else:
                    approvals.append(approval)

        return approvals

    def stream_all_approvals(
        self,
        drug_name: Optional[str] = None,
        sponsor_name: Optional[str] = None,
        max_results: Optional[int] = None,
        **kwargs
    ) -> Generator[DrugApproval, None, None]:
        """
        Stream all matching drug approvals (handles pagination).

        Args:
            drug_name: Filter by drug name
            sponsor_name: Filter by sponsor
            max_results: Maximum results to return
            **kwargs: Additional search parameters

        Yields:
            DrugApproval objects
        """
        count = 0
        skip = 0

        while True:
            response = self.search_drug_approvals(
                drug_name=drug_name,
                sponsor_name=sponsor_name,
                skip=skip,
                **kwargs
            )

            results = response.get("results", [])
            if not results:
                break

            for result in results:
                approval = self._parse_drug_approval(result)
                if approval:
                    yield approval
                    count += 1

                    if max_results and count >= max_results:
                        return

            # Check if more results available
            meta = response.get("meta", {})
            total = meta.get("results", {}).get("total", 0)

            skip += len(results)
            if skip >= total:
                break

    def _parse_drug_approval(self, data: Dict[str, Any]) -> Optional[DrugApproval]:
        """Parse API response into DrugApproval object."""
        if not data:
            return None

        openfda = data.get("openfda", {})
        products = data.get("products", [])
        submissions = data.get("submissions", [])

        # Get the most recent approval submission
        approval_submission = None
        approval_date = None
        for sub in submissions:
            if sub.get("submission_type") == "ORIG" or sub.get("submission_status") == "AP":
                sub_date = sub.get("submission_status_date")
                if sub_date:
                    if not approval_date or sub_date > approval_date:
                        approval_date = sub_date
                        approval_submission = sub

        # Extract active ingredients
        active_ingredients = []
        for product in products:
            for ai in product.get("active_ingredients", []):
                name = ai.get("name")
                if name and name not in active_ingredients:
                    active_ingredients.append(name)

        # Determine approval pathway
        approval_pathway = None
        if approval_submission:
            app_docs = approval_submission.get("application_docs", [])
            for doc in app_docs:
                doc_type = doc.get("type", "").lower()
                if "priority" in doc_type:
                    approval_pathway = "Priority Review"
                elif "accelerated" in doc_type:
                    approval_pathway = "Accelerated Approval"
                elif "breakthrough" in doc_type:
                    approval_pathway = "Breakthrough Therapy"
                elif "fast track" in doc_type:
                    approval_pathway = "Fast Track"

        # Format approval date
        if approval_date and len(approval_date) == 8:
            approval_date = f"{approval_date[:4]}-{approval_date[4:6]}-{approval_date[6:]}"

        return DrugApproval(
            application_number=data.get("application_number", ""),
            drug_name=openfda.get("generic_name", [""])[0] if openfda.get("generic_name") else "",
            brand_name=openfda.get("brand_name", [""])[0] if openfda.get("brand_name") else products[0].get("brand_name") if products else None,
            sponsor_name=data.get("sponsor_name", ""),
            application_type="NDA" if data.get("application_number", "").startswith("NDA") else "BLA" if data.get("application_number", "").startswith("BLA") else "ANDA",
            approval_date=approval_date,
            submission_date=approval_submission.get("submission_status_date") if approval_submission else None,
            therapeutic_area=openfda.get("pharm_class_epc", [""])[0] if openfda.get("pharm_class_epc") else None,
            route=openfda.get("route", [""])[0] if openfda.get("route") else None,
            dosage_form=products[0].get("dosage_form") if products else None,
            active_ingredients=active_ingredients,
            indications=[],  # Would need label data for full indications
            approval_pathway=approval_pathway,
            orphan_drug=any("orphan" in str(sub.get("application_docs", [])).lower() for sub in submissions),
            review_classification=approval_submission.get("review_priority") if approval_submission else None,
            submission_type=approval_submission.get("submission_type") if approval_submission else None,
        )

    # =========================================================================
    # Adverse Events (FAERS)
    # =========================================================================

    def search_adverse_events(
        self,
        drug_name: Optional[str] = None,
        reaction: Optional[str] = None,
        serious: Optional[bool] = None,
        outcome: Optional[str] = None,  # death, hospitalization, etc.
        receive_date_from: Optional[str] = None,
        receive_date_to: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> Dict[str, Any]:
        """
        Search FDA Adverse Event Reporting System (FAERS).

        Args:
            drug_name: Drug name to search
            reaction: Adverse reaction term (MedDRA preferred term)
            serious: Filter to serious events only
            outcome: Filter by outcome (death, hospitalization, disability, etc.)
            receive_date_from: Start date (YYYYMMDD)
            receive_date_to: End date (YYYYMMDD)
            limit: Maximum results
            skip: Pagination offset

        Returns:
            Dict with 'results' list and 'meta' info
        """
        search_parts = []

        if drug_name:
            search_parts.append(
                f'(patient.drug.medicinalproduct:"{drug_name}" OR '
                f'patient.drug.openfda.brand_name:"{drug_name}" OR '
                f'patient.drug.openfda.generic_name:"{drug_name}")'
            )

        if reaction:
            search_parts.append(f'patient.reaction.reactionmeddrapt:"{reaction}"')

        if serious is not None:
            search_parts.append(f'serious:{1 if serious else 2}')

        if outcome:
            outcome_map = {
                "death": "seriousnessdeath:1",
                "hospitalization": "seriousnesshospitalization:1",
                "disability": "seriousnessdisabling:1",
                "life_threatening": "seriousnesslifethreatening:1",
                "congenital_anomaly": "seriousnesscongenitalanomali:1",
            }
            if outcome in outcome_map:
                search_parts.append(outcome_map[outcome])

        if receive_date_from and receive_date_to:
            search_parts.append(f'receivedate:[{receive_date_from} TO {receive_date_to}]')
        elif receive_date_from:
            search_parts.append(f'receivedate:[{receive_date_from} TO 99991231]')
        elif receive_date_to:
            search_parts.append(f'receivedate:[00000101 TO {receive_date_to}]')

        params = {
            "limit": min(limit, 1000),
            "skip": skip,
        }

        if search_parts:
            params["search"] = " AND ".join(search_parts)

        return self._make_request(self.ADVERSE_EVENTS_ENDPOINT, params)

    def get_adverse_event_counts(
        self,
        drug_name: str,
        count_field: str = "patient.reaction.reactionmeddrapt.exact",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get counts of adverse events by reaction type.

        Args:
            drug_name: Drug to analyze
            count_field: Field to count by (reactions, outcomes, etc.)
            limit: Number of top results

        Returns:
            List of dicts with 'term' and 'count'
        """
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "count": count_field,
            "limit": limit,
        }

        response = self._make_request(self.ADVERSE_EVENTS_ENDPOINT, params)
        return response.get("results", [])

    def stream_adverse_events(
        self,
        drug_name: Optional[str] = None,
        serious: Optional[bool] = None,
        max_results: Optional[int] = None,
        **kwargs
    ) -> Generator[AdverseEvent, None, None]:
        """
        Stream adverse events (handles pagination).

        Args:
            drug_name: Filter by drug
            serious: Filter to serious events
            max_results: Maximum events to return
            **kwargs: Additional search parameters

        Yields:
            AdverseEvent objects
        """
        count = 0
        skip = 0

        while True:
            response = self.search_adverse_events(
                drug_name=drug_name,
                serious=serious,
                skip=skip,
                **kwargs
            )

            results = response.get("results", [])
            if not results:
                break

            for result in results:
                events = self._parse_adverse_events(result)
                for event in events:
                    yield event
                    count += 1

                    if max_results and count >= max_results:
                        return

            meta = response.get("meta", {})
            total = meta.get("results", {}).get("total", 0)

            skip += len(results)
            if skip >= total:
                break

    def _parse_adverse_events(self, data: Dict[str, Any]) -> List[AdverseEvent]:
        """Parse API response into AdverseEvent objects."""
        if not data:
            return []

        events = []
        patient = data.get("patient", {})

        # Get drug info (first suspect drug)
        drugs = patient.get("drug", [])
        suspect_drug = None
        for drug in drugs:
            if drug.get("drugcharacterization") == "1":  # 1 = suspect
                suspect_drug = drug
                break
        if not suspect_drug and drugs:
            suspect_drug = drugs[0]

        drug_name = ""
        if suspect_drug:
            drug_name = suspect_drug.get("medicinalproduct", "")
            if not drug_name:
                openfda = suspect_drug.get("openfda", {})
                drug_name = openfda.get("brand_name", [""])[0] if openfda.get("brand_name") else ""

        # Parse each reaction as a separate event
        reactions = patient.get("reaction", [])
        for reaction in reactions:
            events.append(AdverseEvent(
                report_id=data.get("safetyreportid", ""),
                receive_date=data.get("receivedate", ""),
                drug_name=drug_name,
                reaction=reaction.get("reactionmeddrapt", ""),
                outcome=reaction.get("reactionoutcome"),
                patient_age=self._parse_patient_age(patient),
                patient_sex=patient.get("patientsex"),
                patient_weight=patient.get("patientweight"),
                serious=data.get("serious") == "1",
                death=data.get("seriousnessdeath") == "1",
                hospitalization=data.get("seriousnesshospitalization") == "1",
                life_threatening=data.get("seriousnesslifethreatening") == "1",
                disability=data.get("seriousnessdisabling") == "1",
                congenital_anomaly=data.get("seriousnesscongenitalanomali") == "1",
                reporter_qualification=data.get("primarysource", {}).get("qualification"),
                report_source=data.get("primarysource", {}).get("reportercountry"),
            ))

        return events

    def _parse_patient_age(self, patient: Dict[str, Any]) -> Optional[int]:
        """Parse patient age from various formats."""
        age = patient.get("patientonsetage")
        if age:
            try:
                age_unit = patient.get("patientonsetageunit", "801")  # 801 = years
                age_float = float(age)

                # Convert to years if needed
                if age_unit == "800":  # Decades
                    return int(age_float * 10)
                elif age_unit == "801":  # Years
                    return int(age_float)
                elif age_unit == "802":  # Months
                    return int(age_float / 12)
                elif age_unit == "803":  # Weeks
                    return int(age_float / 52)
                elif age_unit == "804":  # Days
                    return int(age_float / 365)
                elif age_unit == "805":  # Hours
                    return 0
            except (ValueError, TypeError):
                pass
        return None

    # =========================================================================
    # Drug Labels
    # =========================================================================

    def get_drug_label(
        self,
        drug_name: Optional[str] = None,
        application_number: Optional[str] = None,
    ) -> Optional[DrugLabel]:
        """
        Get drug labeling information.

        Args:
            drug_name: Brand or generic drug name
            application_number: FDA application number (NDA/BLA)

        Returns:
            DrugLabel object or None
        """
        search_parts = []

        if drug_name:
            search_parts.append(
                f'(openfda.brand_name:"{drug_name}" OR '
                f'openfda.generic_name:"{drug_name}")'
            )

        if application_number:
            search_parts.append(f'openfda.application_number:"{application_number}"')

        if not search_parts:
            return None

        params = {
            "search": " AND ".join(search_parts),
            "limit": 1,
        }

        response = self._make_request(self.DRUG_LABELS_ENDPOINT, params)
        results = response.get("results", [])

        if results:
            return self._parse_drug_label(results[0])
        return None

    def _parse_drug_label(self, data: Dict[str, Any]) -> Optional[DrugLabel]:
        """Parse API response into DrugLabel object."""
        if not data:
            return None

        openfda = data.get("openfda", {})

        # Extract text sections (they come as lists)
        def get_section(key: str) -> Optional[str]:
            value = data.get(key)
            if isinstance(value, list):
                return " ".join(value)
            return value

        return DrugLabel(
            application_number=openfda.get("application_number", [""])[0] if openfda.get("application_number") else "",
            drug_name=openfda.get("generic_name", [""])[0] if openfda.get("generic_name") else "",
            brand_name=openfda.get("brand_name", [""])[0] if openfda.get("brand_name") else None,
            effective_date=data.get("effective_time"),
            boxed_warning=get_section("boxed_warning"),
            warnings_precautions=get_section("warnings_and_cautions") or get_section("warnings"),
            adverse_reactions=get_section("adverse_reactions"),
            contraindications=get_section("contraindications"),
            indications_usage=get_section("indications_and_usage"),
            dosage_administration=get_section("dosage_and_administration"),
        )

    # =========================================================================
    # Drug Recalls / Enforcement
    # =========================================================================

    def search_drug_recalls(
        self,
        drug_name: Optional[str] = None,
        classification: Optional[str] = None,  # Class I, II, or III
        status: Optional[str] = None,  # Ongoing, Completed, Terminated
        recall_date_from: Optional[str] = None,
        recall_date_to: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Search FDA drug recalls and enforcement actions.

        Args:
            drug_name: Drug name to search
            classification: Recall severity (I=most serious, III=least)
            status: Recall status
            recall_date_from: Start date (YYYYMMDD)
            recall_date_to: End date (YYYYMMDD)
            limit: Maximum results

        Returns:
            Dict with 'results' list
        """
        search_parts = []

        if drug_name:
            search_parts.append(f'product_description:"{drug_name}"')

        if classification:
            search_parts.append(f'classification:"{classification}"')

        if status:
            search_parts.append(f'status:"{status}"')

        if recall_date_from and recall_date_to:
            search_parts.append(f'recall_initiation_date:[{recall_date_from} TO {recall_date_to}]')

        params = {
            "limit": min(limit, 1000),
        }

        if search_parts:
            params["search"] = " AND ".join(search_parts)

        return self._make_request(self.ENFORCEMENT_ENDPOINT, params)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_drug_approval_timeline(drug_name: str) -> Dict[str, Any]:
    """
    Get complete approval timeline for a drug.

    Returns:
        Dict with approval dates, pathway info, and milestones
    """
    client = OpenFDAClient()

    response = client.search_drug_approvals(drug_name=drug_name)
    results = response.get("results", [])

    timeline = {
        "drug_name": drug_name,
        "approvals": [],
        "submissions": [],
    }

    for result in results:
        approval = client._parse_drug_approval(result)
        if approval:
            timeline["approvals"].append({
                "application_number": approval.application_number,
                "brand_name": approval.brand_name,
                "approval_date": approval.approval_date,
                "approval_pathway": approval.approval_pathway,
                "sponsor": approval.sponsor_name,
            })

            # Add individual submissions
            for sub in result.get("submissions", []):
                timeline["submissions"].append({
                    "type": sub.get("submission_type"),
                    "status": sub.get("submission_status"),
                    "date": sub.get("submission_status_date"),
                    "class_code": sub.get("submission_class_code"),
                })

    return timeline


def get_drug_safety_profile(drug_name: str) -> Dict[str, Any]:
    """
    Get comprehensive safety profile for a drug.

    Returns:
        Dict with adverse events summary, serious events, and label warnings
    """
    client = OpenFDAClient()

    # Get adverse event counts by reaction
    top_reactions = client.get_adverse_event_counts(
        drug_name=drug_name,
        count_field="patient.reaction.reactionmeddrapt.exact",
        limit=20
    )

    # Get serious event outcomes
    serious_outcomes = client.get_adverse_event_counts(
        drug_name=drug_name,
        count_field="seriousnessdeath",
        limit=5
    )

    # Get label info
    label = client.get_drug_label(drug_name=drug_name)

    return {
        "drug_name": drug_name,
        "top_adverse_reactions": top_reactions,
        "serious_outcome_counts": serious_outcomes,
        "boxed_warning": label.boxed_warning if label else None,
        "warnings": label.warnings_precautions if label else None,
        "contraindications": label.contraindications if label else None,
    }


def compare_drug_safety(drug_names: List[str]) -> Dict[str, Any]:
    """
    Compare safety profiles across multiple drugs.

    Args:
        drug_names: List of drugs to compare

    Returns:
        Dict with comparative safety data
    """
    client = OpenFDAClient()

    comparison = {}

    for drug in drug_names:
        # Get total events
        response = client.search_adverse_events(drug_name=drug, limit=1)
        total_events = response.get("meta", {}).get("results", {}).get("total", 0)

        # Get serious events
        response = client.search_adverse_events(drug_name=drug, serious=True, limit=1)
        serious_events = response.get("meta", {}).get("results", {}).get("total", 0)

        # Get top reactions
        top_reactions = client.get_adverse_event_counts(drug, limit=10)

        comparison[drug] = {
            "total_adverse_events": total_events,
            "serious_events": serious_events,
            "serious_rate": serious_events / total_events if total_events > 0 else 0,
            "top_reactions": [r.get("term") for r in top_reactions[:5]],
        }

    return comparison


if __name__ == "__main__":
    # Quick test
    client = OpenFDAClient()

    print("Testing OpenFDA API client...")

    # Test drug approval search
    print("\n1. Recent drug approvals:")
    approvals = client.get_recent_approvals(days=90, limit=5)
    for approval in approvals[:3]:
        print(f"  - {approval.brand_name or approval.drug_name}: {approval.approval_date} ({approval.sponsor_name})")

    # Test adverse events
    print("\n2. Top adverse reactions for pembrolizumab:")
    reactions = client.get_adverse_event_counts("pembrolizumab", limit=5)
    for r in reactions:
        print(f"  - {r.get('term')}: {r.get('count')} reports")

    # Test drug label
    print("\n3. Drug label lookup:")
    label = client.get_drug_label("KEYTRUDA")
    if label:
        print(f"  Brand: {label.brand_name}")
        print(f"  Boxed warning: {'Yes' if label.boxed_warning else 'No'}")

    print("\nFDA client test complete.")
