"""Analysis module for TrialIntel."""

from .protocol_risk_scorer import (
    ProtocolRiskScorer,
    RiskFactor,
    RiskAssessment,
    score_protocol_quick,
    create_scorer_with_db,
)
from .site_intelligence import (
    SiteIntelligenceAnalyzer,
    SitePerformance,
    CountryProfile,
    GeographicDistribution,
    SitePortfolio,
    SiteIntelligenceReport,
)
from .endpoint_benchmarking import (
    EndpointBenchmarker,
    EndpointPattern,
    EndpointAnalysis,
    EndpointRecommendation,
    build_endpoint_benchmarks,
    create_benchmarker_with_db,
)
from .ml_risk_models import (
    MLRiskPredictor,
    ModelMetrics,
    RiskPrediction,
    TrialFeatureExtractor,
    train_risk_models,
    predict_protocol_risk,
)
from .trial_similarity import (
    TrialSimilarityEngine,
    SimilarTrial,
    TrialComparisonResult,
    get_similar_trials,
)
from .enrollment_forecasting import (
    EnrollmentForecaster,
    EnrollmentProjection,
    EnrollmentAlert,
    SiteEnrollmentForecast,
    forecast_trial_enrollment,
)
from .roi_calculator import (
    ROICalculator,
    ROIAnalysis,
    SiteSelectionROI,
    ProtocolOptimizationROI,
    OptimizationSavings,
    calculate_trial_roi,
)
from .site_leaderboard import (
    SiteLeaderboard,
    SiteRanking,
    SiteComparison,
    TrialPerformance,
    get_site_leaderboard,
)
from .alerts_system import (
    AlertsManager,
    Alert,
    AlertSeverity,
    AlertCategory,
    AlertSummary,
    get_alerts_manager,
)
from .scenario_modeler import (
    ScenarioModeler,
    Scenario,
    ScenarioParameter,
    ScenarioImpact,
    ScenarioType,
    ScenarioComparison,
)
from .endpoint_intelligence import (
    EndpointIntelligence,
    EndpointSuccessData,
    EndpointTiming,
    RegulatoryGuidance,
    CompositeEndpointRecommendation,
    EndpointAnalysisResult,
    get_endpoint_intelligence,
)
from .report_generator import (
    ReportGenerator,
    Report,
    ReportSection,
    get_report_generator,
)
from .risk_matrix import (
    RiskMatrixManager,
    Risk,
    RiskMatrix,
    RiskTrend,
    RiskComparison,
    RiskLikelihood,
    RiskImpact,
    RiskStatus,
    get_risk_matrix_manager,
)

__all__ = [
    # Protocol Risk Scorer
    "ProtocolRiskScorer",
    "RiskFactor",
    "RiskAssessment",
    "score_protocol_quick",
    "create_scorer_with_db",
    # Site Intelligence
    "SiteIntelligenceAnalyzer",
    "SitePerformance",
    "CountryProfile",
    "GeographicDistribution",
    "SitePortfolio",
    "SiteIntelligenceReport",
    # Endpoint Benchmarking
    "EndpointBenchmarker",
    "EndpointPattern",
    "EndpointAnalysis",
    "EndpointRecommendation",
    "build_endpoint_benchmarks",
    "create_benchmarker_with_db",
    # ML Risk Models
    "MLRiskPredictor",
    "ModelMetrics",
    "RiskPrediction",
    "TrialFeatureExtractor",
    "train_risk_models",
    "predict_protocol_risk",
    # Trial Similarity
    "TrialSimilarityEngine",
    "SimilarTrial",
    "TrialComparisonResult",
    "get_similar_trials",
    # Enrollment Forecasting
    "EnrollmentForecaster",
    "EnrollmentProjection",
    "EnrollmentAlert",
    "SiteEnrollmentForecast",
    "forecast_trial_enrollment",
    # ROI Calculator
    "ROICalculator",
    "ROIAnalysis",
    "SiteSelectionROI",
    "ProtocolOptimizationROI",
    "OptimizationSavings",
    "calculate_trial_roi",
    # Site Leaderboard
    "SiteLeaderboard",
    "SiteRanking",
    "SiteComparison",
    "TrialPerformance",
    "get_site_leaderboard",
    # Alerts System
    "AlertsManager",
    "Alert",
    "AlertSeverity",
    "AlertCategory",
    "AlertSummary",
    "get_alerts_manager",
    # Scenario Modeler
    "ScenarioModeler",
    "Scenario",
    "ScenarioParameter",
    "ScenarioImpact",
    "ScenarioType",
    "ScenarioComparison",
    # Endpoint Intelligence
    "EndpointIntelligence",
    "EndpointSuccessData",
    "EndpointTiming",
    "RegulatoryGuidance",
    "CompositeEndpointRecommendation",
    "EndpointAnalysisResult",
    "get_endpoint_intelligence",
    # Report Generator
    "ReportGenerator",
    "Report",
    "ReportSection",
    "get_report_generator",
    # Risk Matrix
    "RiskMatrixManager",
    "Risk",
    "RiskMatrix",
    "RiskTrend",
    "RiskComparison",
    "RiskLikelihood",
    "RiskImpact",
    "RiskStatus",
    "get_risk_matrix_manager",
]
