"""
Smart Alerts and Notifications System for TrialIntel.

Provides configurable alerts for:
- Enrollment delays
- Risk threshold breaches
- Competitive activity
- Milestone reminders
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
import json


class AlertSeverity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertCategory(Enum):
    ENROLLMENT = "enrollment"
    RISK = "risk"
    COMPETITIVE = "competitive"
    MILESTONE = "milestone"
    SITE = "site"
    PROTOCOL = "protocol"


@dataclass
class Alert:
    """Individual alert."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    created_at: datetime
    trial_id: Optional[str] = None
    site_id: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)
    action_required: bool = True
    action_url: Optional[str] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


@dataclass
class AlertRule:
    """Configurable alert rule."""
    id: str
    name: str
    description: str
    category: AlertCategory
    condition: str  # e.g., "enrollment_rate < 0.5"
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    notify_email: bool = False
    notify_slack: bool = False


@dataclass
class AlertSummary:
    """Summary of all active alerts."""
    total_alerts: int
    critical_count: int
    warning_count: int
    info_count: int
    by_category: Dict[str, int]
    recent_alerts: List[Alert]
    action_required_count: int


class AlertsManager:
    """Manages alerts and notifications."""

    def __init__(self, db_manager=None):
        self.db = db_manager
        self.alerts: List[Alert] = []
        self.rules: List[AlertRule] = self._get_default_rules()
        self._alert_counter = 0

    def _get_default_rules(self) -> List[AlertRule]:
        """Get default alert rules."""
        return [
            AlertRule(
                id="enroll_delay_critical",
                name="Critical Enrollment Delay",
                description="Enrollment rate below 50% of target",
                category=AlertCategory.ENROLLMENT,
                condition="enrollment_rate < 0.5 * target_rate",
                threshold=0.5,
                severity=AlertSeverity.CRITICAL,
            ),
            AlertRule(
                id="enroll_delay_warning",
                name="Enrollment Delay Warning",
                description="Enrollment rate below 75% of target",
                category=AlertCategory.ENROLLMENT,
                condition="enrollment_rate < 0.75 * target_rate",
                threshold=0.75,
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                id="risk_score_critical",
                name="High Risk Protocol",
                description="Protocol risk score above 70",
                category=AlertCategory.RISK,
                condition="risk_score > 70",
                threshold=70,
                severity=AlertSeverity.CRITICAL,
            ),
            AlertRule(
                id="risk_score_warning",
                name="Elevated Risk Protocol",
                description="Protocol risk score above 50",
                category=AlertCategory.RISK,
                condition="risk_score > 50",
                threshold=50,
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                id="site_underperform",
                name="Site Underperformance",
                description="Site enrollment below 25% of expected",
                category=AlertCategory.SITE,
                condition="site_enrollment < 0.25 * expected",
                threshold=0.25,
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                id="competitor_new_trial",
                name="New Competitor Trial",
                description="New competing trial started in same indication",
                category=AlertCategory.COMPETITIVE,
                condition="new_competitor_trial",
                threshold=1,
                severity=AlertSeverity.INFO,
            ),
            AlertRule(
                id="milestone_upcoming",
                name="Upcoming Milestone",
                description="Milestone due within 30 days",
                category=AlertCategory.MILESTONE,
                condition="days_to_milestone < 30",
                threshold=30,
                severity=AlertSeverity.INFO,
            ),
        ]

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"ALT-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}"

    def check_enrollment_health(
        self,
        trial_id: str,
        current_enrollment: int,
        target_enrollment: int,
        days_elapsed: int,
        planned_duration_days: int,
    ) -> List[Alert]:
        """Check enrollment and generate alerts if needed."""
        alerts = []

        expected_enrollment = (days_elapsed / planned_duration_days) * target_enrollment
        enrollment_ratio = current_enrollment / expected_enrollment if expected_enrollment > 0 else 0

        if enrollment_ratio < 0.5:
            alerts.append(Alert(
                id=self._generate_alert_id(),
                title="Critical Enrollment Delay",
                description=f"Trial {trial_id} is at {enrollment_ratio:.0%} of expected enrollment ({current_enrollment} vs {expected_enrollment:.0f} expected)",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.ENROLLMENT,
                created_at=datetime.now(),
                trial_id=trial_id,
                data={
                    "current_enrollment": current_enrollment,
                    "expected_enrollment": expected_enrollment,
                    "enrollment_ratio": enrollment_ratio,
                    "days_elapsed": days_elapsed,
                },
                action_required=True,
                action_url=f"/enrollment-war-room?trial={trial_id}",
            ))
        elif enrollment_ratio < 0.75:
            alerts.append(Alert(
                id=self._generate_alert_id(),
                title="Enrollment Behind Schedule",
                description=f"Trial {trial_id} is at {enrollment_ratio:.0%} of expected enrollment",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.ENROLLMENT,
                created_at=datetime.now(),
                trial_id=trial_id,
                data={
                    "current_enrollment": current_enrollment,
                    "expected_enrollment": expected_enrollment,
                    "enrollment_ratio": enrollment_ratio,
                },
                action_required=True,
            ))

        self.alerts.extend(alerts)
        return alerts

    def check_risk_score(
        self,
        trial_id: str,
        risk_score: float,
        risk_factors: List[Dict[str, Any]],
    ) -> List[Alert]:
        """Check risk score and generate alerts."""
        alerts = []

        if risk_score > 70:
            alerts.append(Alert(
                id=self._generate_alert_id(),
                title="High Risk Protocol Detected",
                description=f"Protocol risk score is {risk_score:.0f}/100 - immediate attention required",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.RISK,
                created_at=datetime.now(),
                trial_id=trial_id,
                data={
                    "risk_score": risk_score,
                    "top_risks": risk_factors[:3] if risk_factors else [],
                },
                action_required=True,
                action_url="/protocol-analysis",
            ))
        elif risk_score > 50:
            alerts.append(Alert(
                id=self._generate_alert_id(),
                title="Elevated Protocol Risk",
                description=f"Protocol risk score is {risk_score:.0f}/100 - review recommended",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.RISK,
                created_at=datetime.now(),
                trial_id=trial_id,
                data={"risk_score": risk_score},
                action_required=True,
            ))

        self.alerts.extend(alerts)
        return alerts

    def check_site_performance(
        self,
        site_id: int,
        site_name: str,
        actual_enrollment: int,
        expected_enrollment: int,
        trial_id: Optional[str] = None,
    ) -> List[Alert]:
        """Check site performance and generate alerts."""
        alerts = []

        ratio = actual_enrollment / expected_enrollment if expected_enrollment > 0 else 0

        if ratio < 0.25:
            alerts.append(Alert(
                id=self._generate_alert_id(),
                title="Site Critically Underperforming",
                description=f"{site_name} has enrolled only {actual_enrollment} of {expected_enrollment} expected patients ({ratio:.0%})",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.SITE,
                created_at=datetime.now(),
                trial_id=trial_id,
                site_id=site_id,
                data={
                    "site_name": site_name,
                    "actual": actual_enrollment,
                    "expected": expected_enrollment,
                    "ratio": ratio,
                },
                action_required=True,
            ))
        elif ratio < 0.5:
            alerts.append(Alert(
                id=self._generate_alert_id(),
                title="Site Underperforming",
                description=f"{site_name} is at {ratio:.0%} of expected enrollment",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SITE,
                created_at=datetime.now(),
                trial_id=trial_id,
                site_id=site_id,
                data={"ratio": ratio},
                action_required=True,
            ))

        self.alerts.extend(alerts)
        return alerts

    def add_competitive_alert(
        self,
        competitor_name: str,
        trial_title: str,
        indication: str,
        nct_id: str,
    ) -> Alert:
        """Add alert for new competitor trial."""
        alert = Alert(
            id=self._generate_alert_id(),
            title="New Competitor Trial Detected",
            description=f"{competitor_name} started new {indication} trial: {trial_title}",
            severity=AlertSeverity.INFO,
            category=AlertCategory.COMPETITIVE,
            created_at=datetime.now(),
            trial_id=nct_id,
            data={
                "competitor": competitor_name,
                "indication": indication,
            },
            action_required=False,
            action_url=f"/competitive-intelligence?indication={indication}",
        )
        self.alerts.append(alert)
        return alert

    def add_milestone_reminder(
        self,
        trial_id: str,
        milestone_name: str,
        due_date: datetime,
        days_until: int,
    ) -> Alert:
        """Add milestone reminder alert."""
        severity = (
            AlertSeverity.CRITICAL if days_until <= 7
            else AlertSeverity.WARNING if days_until <= 14
            else AlertSeverity.INFO
        )

        alert = Alert(
            id=self._generate_alert_id(),
            title=f"Milestone Due: {milestone_name}",
            description=f"Due in {days_until} days ({due_date.strftime('%Y-%m-%d')})",
            severity=severity,
            category=AlertCategory.MILESTONE,
            created_at=datetime.now(),
            trial_id=trial_id,
            data={
                "milestone": milestone_name,
                "due_date": due_date.isoformat(),
                "days_until": days_until,
            },
            action_required=days_until <= 14,
        )
        self.alerts.append(alert)
        return alert

    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = user
                return True
        return False

    def get_active_alerts(
        self,
        category: Optional[AlertCategory] = None,
        severity: Optional[AlertSeverity] = None,
        include_acknowledged: bool = False,
    ) -> List[Alert]:
        """Get active alerts with optional filtering."""
        alerts = self.alerts

        if not include_acknowledged:
            alerts = [a for a in alerts if not a.acknowledged]

        if category:
            alerts = [a for a in alerts if a.category == category]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: (
            0 if a.severity == AlertSeverity.CRITICAL else 1 if a.severity == AlertSeverity.WARNING else 2,
            a.created_at
        ), reverse=True)

    def get_alert_summary(self) -> AlertSummary:
        """Get summary of all active alerts."""
        active = [a for a in self.alerts if not a.acknowledged]

        by_category = {}
        for cat in AlertCategory:
            count = len([a for a in active if a.category == cat])
            if count > 0:
                by_category[cat.value] = count

        return AlertSummary(
            total_alerts=len(active),
            critical_count=len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
            warning_count=len([a for a in active if a.severity == AlertSeverity.WARNING]),
            info_count=len([a for a in active if a.severity == AlertSeverity.INFO]),
            by_category=by_category,
            recent_alerts=active[:10],
            action_required_count=len([a for a in active if a.action_required]),
        )

    def generate_demo_alerts(self) -> List[Alert]:
        """Generate demo alerts for display purposes."""
        demo_alerts = [
            Alert(
                id=self._generate_alert_id(),
                title="Critical Enrollment Delay",
                description="DIAMOND-3 trial at 45% of expected enrollment (156 vs 347 expected)",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.ENROLLMENT,
                created_at=datetime.now() - timedelta(hours=2),
                trial_id="NCT05123456",
                data={"enrollment_ratio": 0.45},
                action_required=True,
                action_url="/enrollment-war-room",
            ),
            Alert(
                id=self._generate_alert_id(),
                title="High Risk Protocol Detected",
                description="New T2D protocol has risk score of 73/100 - 5 high-severity factors identified",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.RISK,
                created_at=datetime.now() - timedelta(hours=5),
                data={"risk_score": 73},
                action_required=True,
                action_url="/protocol-analysis",
            ),
            Alert(
                id=self._generate_alert_id(),
                title="Site Underperforming",
                description="Cleveland Clinic Site 3 has enrolled only 12 of 45 expected patients",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SITE,
                created_at=datetime.now() - timedelta(hours=8),
                site_id=1234,
                data={"ratio": 0.27},
                action_required=True,
            ),
            Alert(
                id=self._generate_alert_id(),
                title="New Competitor Trial",
                description="Novo Nordisk started Phase 3 GLP-1 trial in obesity indication",
                severity=AlertSeverity.INFO,
                category=AlertCategory.COMPETITIVE,
                created_at=datetime.now() - timedelta(days=1),
                trial_id="NCT05999888",
                data={"competitor": "Novo Nordisk"},
                action_required=False,
            ),
            Alert(
                id=self._generate_alert_id(),
                title="Milestone Due: Database Lock",
                description="Due in 12 days (2024-02-15)",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.MILESTONE,
                created_at=datetime.now() - timedelta(days=2),
                trial_id="NCT04567890",
                data={"days_until": 12},
                action_required=True,
            ),
            Alert(
                id=self._generate_alert_id(),
                title="Protocol Amendment Risk",
                description="3 eligibility criteria flagged as high amendment risk based on historical data",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.PROTOCOL,
                created_at=datetime.now() - timedelta(hours=24),
                action_required=True,
                action_url="/protocol-analysis",
            ),
        ]

        self.alerts.extend(demo_alerts)
        return demo_alerts


# Singleton instance
_alerts_manager: Optional[AlertsManager] = None


def get_alerts_manager(db_manager=None) -> AlertsManager:
    """Get or create alerts manager singleton."""
    global _alerts_manager
    if _alerts_manager is None:
        _alerts_manager = AlertsManager(db_manager)
    return _alerts_manager
