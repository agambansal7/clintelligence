"""
Risk Heat Matrix for TrialIntel.

Provides comprehensive risk visualization:
- Likelihood vs Impact matrix
- Risk trends over time
- Comparative risk analysis
- Mitigation tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import uuid


class RiskLikelihood(Enum):
    RARE = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    ALMOST_CERTAIN = 5


class RiskImpact(Enum):
    NEGLIGIBLE = 1
    MINOR = 2
    MODERATE = 3
    MAJOR = 4
    CATASTROPHIC = 5


class RiskStatus(Enum):
    IDENTIFIED = "identified"
    MITIGATING = "mitigating"
    ACCEPTED = "accepted"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class MitigationStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"


@dataclass
class RiskMitigation:
    """A mitigation action for a risk."""
    id: str
    description: str
    owner: str
    due_date: datetime
    status: MitigationStatus
    effectiveness: Optional[float] = None  # 0-1, how much it reduces risk
    notes: str = ""


@dataclass
class Risk:
    """A single risk item."""
    id: str
    title: str
    description: str
    category: str
    likelihood: RiskLikelihood
    impact: RiskImpact
    risk_score: int  # likelihood * impact
    status: RiskStatus
    created_at: datetime
    updated_at: datetime
    owner: Optional[str] = None
    trial_id: Optional[str] = None
    mitigations: List[RiskMitigation] = field(default_factory=list)
    historical_scores: List[Tuple[datetime, int]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class RiskMatrixCell:
    """A cell in the risk matrix."""
    likelihood: RiskLikelihood
    impact: RiskImpact
    risks: List[Risk]
    cell_score: int
    color: str  # green, yellow, orange, red


@dataclass
class RiskMatrix:
    """Complete risk matrix."""
    cells: List[List[RiskMatrixCell]]
    total_risks: int
    high_risks: int
    medium_risks: int
    low_risks: int
    by_category: Dict[str, int]
    average_score: float


@dataclass
class RiskTrend:
    """Risk trend over time."""
    risk_id: str
    risk_title: str
    data_points: List[Tuple[datetime, int]]
    trend_direction: str  # improving, worsening, stable
    change_percentage: float


@dataclass
class RiskComparison:
    """Comparison of risks across trials."""
    trial_ids: List[str]
    risk_scores: Dict[str, float]
    common_risks: List[str]
    unique_risks: Dict[str, List[str]]
    benchmark_score: float
    relative_position: str  # above, below, at benchmark


class RiskMatrixManager:
    """
    Manages risk identification, assessment, and visualization.
    """

    # Risk score thresholds
    HIGH_RISK_THRESHOLD = 15
    MEDIUM_RISK_THRESHOLD = 8

    # Category definitions
    RISK_CATEGORIES = [
        "enrollment",
        "protocol_complexity",
        "site_performance",
        "regulatory",
        "safety",
        "data_quality",
        "timeline",
        "budget",
        "operational",
        "competitive",
    ]

    def __init__(self, db_manager=None):
        self.db = db_manager
        self.risks: Dict[str, Risk] = {}
        self._counter = 0

    def _generate_id(self) -> str:
        """Generate unique ID."""
        self._counter += 1
        return f"RISK-{datetime.now().strftime('%Y%m%d')}-{self._counter:04d}"

    def _calculate_risk_score(self, likelihood: RiskLikelihood, impact: RiskImpact) -> int:
        """Calculate risk score from likelihood and impact."""
        return likelihood.value * impact.value

    def _get_cell_color(self, score: int) -> str:
        """Get color for risk score."""
        if score >= self.HIGH_RISK_THRESHOLD:
            return "red"
        elif score >= self.MEDIUM_RISK_THRESHOLD:
            return "orange"
        elif score >= 4:
            return "yellow"
        return "green"

    def add_risk(
        self,
        title: str,
        description: str,
        category: str,
        likelihood: RiskLikelihood,
        impact: RiskImpact,
        trial_id: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Risk:
        """Add a new risk."""
        risk_score = self._calculate_risk_score(likelihood, impact)

        risk = Risk(
            id=self._generate_id(),
            title=title,
            description=description,
            category=category,
            likelihood=likelihood,
            impact=impact,
            risk_score=risk_score,
            status=RiskStatus.IDENTIFIED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            trial_id=trial_id,
            owner=owner,
            tags=tags or [],
            historical_scores=[(datetime.now(), risk_score)],
        )

        self.risks[risk.id] = risk
        return risk

    def update_risk_assessment(
        self,
        risk_id: str,
        likelihood: Optional[RiskLikelihood] = None,
        impact: Optional[RiskImpact] = None,
        status: Optional[RiskStatus] = None,
    ) -> Optional[Risk]:
        """Update risk assessment."""
        if risk_id not in self.risks:
            return None

        risk = self.risks[risk_id]

        if likelihood:
            risk.likelihood = likelihood
        if impact:
            risk.impact = impact
        if status:
            risk.status = status

        risk.risk_score = self._calculate_risk_score(risk.likelihood, risk.impact)
        risk.updated_at = datetime.now()
        risk.historical_scores.append((datetime.now(), risk.risk_score))

        return risk

    def add_mitigation(
        self,
        risk_id: str,
        description: str,
        owner: str,
        due_date: datetime,
        effectiveness: Optional[float] = None,
    ) -> Optional[RiskMitigation]:
        """Add mitigation action to a risk."""
        if risk_id not in self.risks:
            return None

        mitigation = RiskMitigation(
            id=f"MIT-{uuid.uuid4().hex[:8].upper()}",
            description=description,
            owner=owner,
            due_date=due_date,
            status=MitigationStatus.NOT_STARTED,
            effectiveness=effectiveness,
        )

        self.risks[risk_id].mitigations.append(mitigation)
        self.risks[risk_id].updated_at = datetime.now()

        return mitigation

    def update_mitigation_status(
        self,
        risk_id: str,
        mitigation_id: str,
        status: MitigationStatus,
        effectiveness: Optional[float] = None,
    ) -> bool:
        """Update mitigation status."""
        if risk_id not in self.risks:
            return False

        for mit in self.risks[risk_id].mitigations:
            if mit.id == mitigation_id:
                mit.status = status
                if effectiveness is not None:
                    mit.effectiveness = effectiveness
                return True

        return False

    def get_risk_matrix(
        self,
        trial_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> RiskMatrix:
        """Generate risk matrix visualization data."""

        # Filter risks
        filtered_risks = list(self.risks.values())
        if trial_id:
            filtered_risks = [r for r in filtered_risks if r.trial_id == trial_id]
        if category:
            filtered_risks = [r for r in filtered_risks if r.category == category]

        # Build matrix (5x5)
        cells = []
        for impact in RiskImpact:
            row = []
            for likelihood in RiskLikelihood:
                cell_risks = [
                    r for r in filtered_risks
                    if r.likelihood == likelihood and r.impact == impact
                ]
                score = self._calculate_risk_score(likelihood, impact)

                row.append(RiskMatrixCell(
                    likelihood=likelihood,
                    impact=impact,
                    risks=cell_risks,
                    cell_score=score,
                    color=self._get_cell_color(score),
                ))
            cells.append(row)

        # Calculate statistics
        by_category = {}
        for risk in filtered_risks:
            by_category[risk.category] = by_category.get(risk.category, 0) + 1

        scores = [r.risk_score for r in filtered_risks]
        avg_score = sum(scores) / len(scores) if scores else 0

        return RiskMatrix(
            cells=cells,
            total_risks=len(filtered_risks),
            high_risks=len([r for r in filtered_risks if r.risk_score >= self.HIGH_RISK_THRESHOLD]),
            medium_risks=len([r for r in filtered_risks if self.MEDIUM_RISK_THRESHOLD <= r.risk_score < self.HIGH_RISK_THRESHOLD]),
            low_risks=len([r for r in filtered_risks if r.risk_score < self.MEDIUM_RISK_THRESHOLD]),
            by_category=by_category,
            average_score=avg_score,
        )

    def get_risk_trends(
        self,
        trial_id: Optional[str] = None,
        days: int = 90,
    ) -> List[RiskTrend]:
        """Get risk trends over time."""
        cutoff = datetime.now() - timedelta(days=days)

        trends = []
        for risk in self.risks.values():
            if trial_id and risk.trial_id != trial_id:
                continue

            # Get historical data within timeframe
            data_points = [
                (dt, score) for dt, score in risk.historical_scores
                if dt >= cutoff
            ]

            if len(data_points) < 2:
                continue

            # Calculate trend
            first_score = data_points[0][1]
            last_score = data_points[-1][1]
            change = last_score - first_score
            change_pct = (change / first_score * 100) if first_score > 0 else 0

            if change > 0:
                direction = "worsening"
            elif change < 0:
                direction = "improving"
            else:
                direction = "stable"

            trends.append(RiskTrend(
                risk_id=risk.id,
                risk_title=risk.title,
                data_points=data_points,
                trend_direction=direction,
                change_percentage=change_pct,
            ))

        return trends

    def compare_to_benchmark(
        self,
        trial_id: str,
        benchmark_trials: Optional[List[str]] = None,
    ) -> RiskComparison:
        """Compare trial risks to benchmark."""
        trial_risks = [r for r in self.risks.values() if r.trial_id == trial_id]
        trial_score = sum(r.risk_score for r in trial_risks) / len(trial_risks) if trial_risks else 0

        # Get benchmark data
        if benchmark_trials:
            benchmark_risks = [
                r for r in self.risks.values()
                if r.trial_id in benchmark_trials
            ]
        else:
            benchmark_risks = [r for r in self.risks.values() if r.trial_id != trial_id]

        benchmark_score = sum(r.risk_score for r in benchmark_risks) / len(benchmark_risks) if benchmark_risks else 0

        # Compare
        trial_categories = set(r.category for r in trial_risks)
        benchmark_categories = set(r.category for r in benchmark_risks)

        common = list(trial_categories & benchmark_categories)
        unique = {
            trial_id: list(trial_categories - benchmark_categories),
            "benchmark": list(benchmark_categories - trial_categories),
        }

        if trial_score > benchmark_score * 1.1:
            position = "above"
        elif trial_score < benchmark_score * 0.9:
            position = "below"
        else:
            position = "at"

        return RiskComparison(
            trial_ids=[trial_id] + (benchmark_trials or []),
            risk_scores={trial_id: trial_score, "benchmark": benchmark_score},
            common_risks=common,
            unique_risks=unique,
            benchmark_score=benchmark_score,
            relative_position=position,
        )

    def get_risks_by_category(self, trial_id: Optional[str] = None) -> Dict[str, List[Risk]]:
        """Get risks grouped by category."""
        result = {cat: [] for cat in self.RISK_CATEGORIES}

        for risk in self.risks.values():
            if trial_id and risk.trial_id != trial_id:
                continue
            if risk.category in result:
                result[risk.category].append(risk)

        return {k: v for k, v in result.items() if v}

    def get_high_priority_risks(
        self,
        trial_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Risk]:
        """Get highest priority risks."""
        risks = list(self.risks.values())
        if trial_id:
            risks = [r for r in risks if r.trial_id == trial_id]

        # Sort by score (descending) and status (escalated first)
        risks.sort(key=lambda r: (
            0 if r.status == RiskStatus.ESCALATED else 1,
            -r.risk_score,
        ))

        return risks[:limit]

    def generate_demo_risks(self, trial_id: str = "NCT05123456") -> List[Risk]:
        """Generate demo risks for display."""
        demo_data = [
            ("Enrollment Rate Below Target", "enrollment", RiskLikelihood.LIKELY, RiskImpact.MAJOR),
            ("Complex Eligibility Criteria", "protocol_complexity", RiskLikelihood.POSSIBLE, RiskImpact.MODERATE),
            ("Site Activation Delays", "site_performance", RiskLikelihood.LIKELY, RiskImpact.MODERATE),
            ("Regulatory Query Response Time", "regulatory", RiskLikelihood.POSSIBLE, RiskImpact.MINOR),
            ("Adverse Event Reporting Burden", "safety", RiskLikelihood.UNLIKELY, RiskImpact.MAJOR),
            ("Data Entry Errors", "data_quality", RiskLikelihood.POSSIBLE, RiskImpact.MINOR),
            ("Competitor Trial Launch", "competitive", RiskLikelihood.LIKELY, RiskImpact.MODERATE),
            ("Budget Overrun Risk", "budget", RiskLikelihood.POSSIBLE, RiskImpact.MAJOR),
            ("Supply Chain Disruption", "operational", RiskLikelihood.UNLIKELY, RiskImpact.CATASTROPHIC),
            ("Protocol Amendment Required", "protocol_complexity", RiskLikelihood.LIKELY, RiskImpact.MAJOR),
        ]

        risks = []
        for title, category, likelihood, impact in demo_data:
            risk = self.add_risk(
                title=title,
                description=f"Risk identified: {title}",
                category=category,
                likelihood=likelihood,
                impact=impact,
                trial_id=trial_id,
            )
            risks.append(risk)

        return risks


def get_risk_matrix_manager(db_manager=None) -> RiskMatrixManager:
    """Get risk matrix manager instance."""
    return RiskMatrixManager(db_manager)
