"""Ingestion module for TrialIntel."""

from .ctgov_client import ClinicalTrialsGovClient as ClinicalTrialsClient, TrialData
from .data_pipeline import DataPipeline, PipelineStats

__all__ = [
    "ClinicalTrialsClient",
    "TrialData",
    "DataPipeline",
    "PipelineStats",
]
