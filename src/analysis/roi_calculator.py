"""
ROI Calculator for Clinical Trial Optimization

Calculates tangible value from TrialIntel optimizations:
- Time savings from better site selection
- Cost savings from reduced amendments
- Revenue acceleration from faster enrollment
- Risk mitigation value from early warning

Industry benchmarks used:
- Average trial cost: $15-20M (Phase 2), $20-50M (Phase 3)
- Daily delay cost: $37,000-$600,000 depending on drug potential
- Amendment cost: $500K average direct + 3 months delay
- Site activation cost: $50K-100K per site
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CostBenchmarks:
    """Industry cost benchmarks for ROI calculations."""
    # Daily delay costs by therapeutic area ($/day)
    daily_delay_cost: Dict[str, int] = field(default_factory=lambda: {
        "oncology": 300000,
        "cardiovascular": 150000,
        "metabolic": 100000,
        "neurology": 200000,
        "immunology": 175000,
        "rare_disease": 250000,
        "infectious_disease": 125000,
        "respiratory": 100000,
        "default": 150000,
    })

    # Amendment costs
    amendment_direct_cost: int = 500000  # Direct cost per major amendment
    amendment_delay_days: int = 90  # Average delay from amendment

    # Site costs
    site_activation_cost: int = 75000  # Per site
    site_replacement_cost: int = 100000  # Including closeout + new activation
    underperforming_site_loss: int = 50000  # Lost investment in non-enrolling site

    # Patient costs
    cost_per_patient: Dict[str, int] = field(default_factory=lambda: {
        "phase1": 40000,
        "phase2": 35000,
        "phase3": 25000,
        "phase4": 15000,
        "default": 30000,
    })

    # Revenue benchmarks (peak annual sales potential)
    revenue_potential: Dict[str, int] = field(default_factory=lambda: {
        "oncology": 2000000000,  # $2B
        "cardiovascular": 1500000000,
        "metabolic": 1000000000,
        "neurology": 1200000000,
        "immunology": 1500000000,
        "rare_disease": 500000000,
        "default": 1000000000,
    })


@dataclass
class OptimizationSavings:
    """Calculated savings from a specific optimization."""
    optimization_type: str
    description: str
    days_saved: int
    cost_saved: int
    confidence: str  # high, medium, low
    calculation_basis: str


@dataclass
class ROIAnalysis:
    """Complete ROI analysis for trial optimizations."""
    # Summary metrics
    total_days_saved: int
    total_cost_saved: int
    total_revenue_impact: int
    roi_percentage: float  # Return on TrialIntel investment
    confidence_level: str  # high, medium, low

    # Breakdown by optimization type (fields with defaults must come after)
    savings_breakdown: List[OptimizationSavings] = field(default_factory=list)
    key_assumptions: List[str] = field(default_factory=list)

    # Comparison metrics
    baseline_duration_days: int = 0
    optimized_duration_days: int = 0
    baseline_cost: int = 0
    optimized_cost: int = 0


@dataclass
class SiteSelectionROI:
    """ROI from optimized site selection."""
    days_saved: int
    cost_saved: int
    sites_optimized: int
    enrollment_acceleration: float  # Percentage faster
    avoided_site_replacements: int
    details: str


@dataclass
class ProtocolOptimizationROI:
    """ROI from protocol optimization."""
    amendments_avoided: int
    days_saved: int
    cost_saved: int
    risk_reduction: float  # Percentage
    details: str


class ROICalculator:
    """
    Calculates ROI from TrialIntel optimizations.

    Usage:
        calculator = ROICalculator()

        # Full ROI analysis
        roi = calculator.calculate_full_roi(
            therapeutic_area="oncology",
            phase="PHASE3",
            target_enrollment=500,
            num_sites=50,
            baseline_duration_months=24,
            optimizations_applied=["site_selection", "protocol_risk", "enrollment_forecast"]
        )

        # Site selection ROI
        site_roi = calculator.calculate_site_selection_roi(
            num_sites=50,
            therapeutic_area="oncology",
            enrollment_improvement_pct=15
        )

        # Protocol optimization ROI
        protocol_roi = calculator.calculate_protocol_optimization_roi(
            risk_score_before=65,
            risk_score_after=35,
            therapeutic_area="oncology"
        )
    """

    def __init__(self, benchmarks: Optional[CostBenchmarks] = None):
        """Initialize with cost benchmarks."""
        self.benchmarks = benchmarks or CostBenchmarks()

    def calculate_full_roi(
        self,
        therapeutic_area: str,
        phase: str,
        target_enrollment: int,
        num_sites: int,
        baseline_duration_months: float,
        optimizations_applied: List[str],
        protocol_risk_reduction: Optional[float] = None,
        enrollment_improvement_pct: Optional[float] = None,
        similar_trial_insights_used: bool = True,
    ) -> ROIAnalysis:
        """
        Calculate comprehensive ROI from all optimizations.

        Args:
            therapeutic_area: Trial therapeutic area
            phase: Trial phase
            target_enrollment: Target number of patients
            num_sites: Number of sites
            baseline_duration_months: Expected duration without optimization
            optimizations_applied: List of optimization types used
            protocol_risk_reduction: Risk score reduction (0-100)
            enrollment_improvement_pct: Enrollment rate improvement percentage
            similar_trial_insights_used: Whether similar trial matching was used

        Returns:
            ROIAnalysis with complete breakdown
        """
        therapeutic_key = therapeutic_area.lower().replace(" ", "_")
        phase_key = phase.lower().replace("phase", "").strip()

        # Get daily delay cost
        daily_cost = self.benchmarks.daily_delay_cost.get(
            therapeutic_key,
            self.benchmarks.daily_delay_cost["default"]
        )

        savings_list = []
        total_days = 0
        total_cost = 0
        assumptions = []

        # 1. Site Selection Optimization
        if "site_selection" in optimizations_applied:
            site_roi = self.calculate_site_selection_roi(
                num_sites=num_sites,
                therapeutic_area=therapeutic_area,
                enrollment_improvement_pct=enrollment_improvement_pct or 15,
            )
            savings_list.append(OptimizationSavings(
                optimization_type="site_selection",
                description=f"Optimized site selection for {num_sites} sites",
                days_saved=site_roi.days_saved,
                cost_saved=site_roi.cost_saved,
                confidence="high" if similar_trial_insights_used else "medium",
                calculation_basis=site_roi.details,
            ))
            total_days += site_roi.days_saved
            total_cost += site_roi.cost_saved
            assumptions.append(f"Site enrollment improvement of {enrollment_improvement_pct or 15}% based on historical data")

        # 2. Protocol Risk Optimization
        if "protocol_risk" in optimizations_applied and protocol_risk_reduction:
            protocol_roi = self.calculate_protocol_optimization_roi(
                risk_score_before=65,  # Assume average starting risk
                risk_score_after=65 - protocol_risk_reduction,
                therapeutic_area=therapeutic_area,
            )
            savings_list.append(OptimizationSavings(
                optimization_type="protocol_optimization",
                description="Protocol risk scoring and optimization",
                days_saved=protocol_roi.days_saved,
                cost_saved=protocol_roi.cost_saved,
                confidence="medium",
                calculation_basis=protocol_roi.details,
            ))
            total_days += protocol_roi.days_saved
            total_cost += protocol_roi.cost_saved
            assumptions.append(f"Protocol risk reduced by {protocol_risk_reduction} points")

        # 3. Enrollment Forecasting
        if "enrollment_forecast" in optimizations_applied:
            forecast_roi = self._calculate_forecast_roi(
                therapeutic_area=therapeutic_area,
                num_sites=num_sites,
                baseline_duration_months=baseline_duration_months,
            )
            savings_list.append(OptimizationSavings(
                optimization_type="enrollment_forecasting",
                description="Early warning system and enrollment monitoring",
                days_saved=forecast_roi["days_saved"],
                cost_saved=forecast_roi["cost_saved"],
                confidence="medium",
                calculation_basis=forecast_roi["details"],
            ))
            total_days += forecast_roi["days_saved"]
            total_cost += forecast_roi["cost_saved"]
            assumptions.append("Early intervention from forecasting prevents 30-day average delay")

        # 4. Similar Trial Insights
        if similar_trial_insights_used:
            similar_roi = self._calculate_similar_trial_roi(
                therapeutic_area=therapeutic_area,
                phase=phase,
            )
            savings_list.append(OptimizationSavings(
                optimization_type="similar_trial_insights",
                description="Learning from specific similar trials",
                days_saved=similar_roi["days_saved"],
                cost_saved=similar_roi["cost_saved"],
                confidence="medium",
                calculation_basis=similar_roi["details"],
            ))
            total_days += similar_roi["days_saved"]
            total_cost += similar_roi["cost_saved"]
            assumptions.append("Similar trial insights prevent common design pitfalls")

        # Calculate revenue impact
        revenue_potential = self.benchmarks.revenue_potential.get(
            therapeutic_key,
            self.benchmarks.revenue_potential["default"]
        )
        daily_revenue_potential = revenue_potential / 365
        total_revenue_impact = int(total_days * daily_revenue_potential)

        # Calculate baseline and optimized costs
        baseline_duration_days = int(baseline_duration_months * 30.44)
        optimized_duration_days = baseline_duration_days - total_days

        patient_cost = self.benchmarks.cost_per_patient.get(
            f"phase{phase_key}",
            self.benchmarks.cost_per_patient["default"]
        )
        baseline_patient_cost = target_enrollment * patient_cost
        site_cost = num_sites * self.benchmarks.site_activation_cost
        baseline_cost = baseline_patient_cost + site_cost + (baseline_duration_days * daily_cost // 10)  # Simplified

        optimized_cost = baseline_cost - total_cost

        # Calculate ROI percentage (assuming TrialIntel costs ~$100K/year)
        trialintel_cost = 100000
        roi_percentage = ((total_cost - trialintel_cost) / trialintel_cost) * 100 if trialintel_cost > 0 else 0

        # Determine confidence level
        if len([s for s in savings_list if s.confidence == "high"]) >= len(savings_list) // 2:
            confidence = "high"
        elif len([s for s in savings_list if s.confidence == "low"]) >= len(savings_list) // 2:
            confidence = "low"
        else:
            confidence = "medium"

        return ROIAnalysis(
            total_days_saved=total_days,
            total_cost_saved=total_cost,
            total_revenue_impact=total_revenue_impact,
            roi_percentage=round(roi_percentage, 1),
            savings_breakdown=savings_list,
            confidence_level=confidence,
            key_assumptions=assumptions,
            baseline_duration_days=baseline_duration_days,
            optimized_duration_days=optimized_duration_days,
            baseline_cost=baseline_cost,
            optimized_cost=optimized_cost,
        )

    def calculate_site_selection_roi(
        self,
        num_sites: int,
        therapeutic_area: str,
        enrollment_improvement_pct: float = 15,
        avoided_replacements_pct: float = 20,
    ) -> SiteSelectionROI:
        """
        Calculate ROI from optimized site selection.

        Args:
            num_sites: Number of sites selected
            therapeutic_area: Trial therapeutic area
            enrollment_improvement_pct: Expected enrollment rate improvement
            avoided_replacements_pct: Expected reduction in site replacements

        Returns:
            SiteSelectionROI with detailed breakdown
        """
        therapeutic_key = therapeutic_area.lower().replace(" ", "_")

        # Calculate enrollment acceleration
        # Faster enrollment = fewer days to target
        # 15% faster enrollment ≈ 13% fewer days (conservative)
        days_saved_from_enrollment = int(enrollment_improvement_pct * 0.87 * 3)  # ~3 days per % improvement

        # Calculate avoided site replacements
        # Industry average: 20% of sites need replacement
        # With optimization: reduce by avoided_replacements_pct
        baseline_replacements = int(num_sites * 0.20)
        avoided_replacements = int(baseline_replacements * (avoided_replacements_pct / 100))
        replacement_cost_saved = avoided_replacements * self.benchmarks.site_replacement_cost

        # Delay saved from not replacing sites
        # Each replacement ≈ 60 days delay
        days_saved_from_replacements = avoided_replacements * 60

        total_days = days_saved_from_enrollment + days_saved_from_replacements

        # Cost savings from delay reduction
        daily_cost = self.benchmarks.daily_delay_cost.get(
            therapeutic_key,
            self.benchmarks.daily_delay_cost["default"]
        )
        delay_cost_saved = total_days * daily_cost

        total_cost_saved = replacement_cost_saved + delay_cost_saved

        details = (
            f"Enrollment {enrollment_improvement_pct}% faster saves ~{days_saved_from_enrollment} days. "
            f"Avoided {avoided_replacements} site replacements saves {days_saved_from_replacements} days "
            f"and ${replacement_cost_saved:,} in replacement costs."
        )

        return SiteSelectionROI(
            days_saved=total_days,
            cost_saved=total_cost_saved,
            sites_optimized=num_sites,
            enrollment_acceleration=enrollment_improvement_pct,
            avoided_site_replacements=avoided_replacements,
            details=details,
        )

    def calculate_protocol_optimization_roi(
        self,
        risk_score_before: float,
        risk_score_after: float,
        therapeutic_area: str,
    ) -> ProtocolOptimizationROI:
        """
        Calculate ROI from protocol risk optimization.

        Args:
            risk_score_before: Risk score before optimization (0-100)
            risk_score_after: Risk score after optimization (0-100)
            therapeutic_area: Trial therapeutic area

        Returns:
            ProtocolOptimizationROI with detailed breakdown
        """
        therapeutic_key = therapeutic_area.lower().replace(" ", "_")
        risk_reduction = risk_score_before - risk_score_after

        # Amendment probability reduction
        # Higher risk score = higher amendment probability
        # Rule of thumb: 1 point risk reduction ≈ 0.5% amendment probability reduction
        baseline_amendment_prob = risk_score_before / 100 * 0.5  # Max 50% at score 100
        optimized_amendment_prob = risk_score_after / 100 * 0.5
        amendment_prob_reduction = baseline_amendment_prob - optimized_amendment_prob

        # Expected amendments avoided (assuming 2 expected amendments at baseline for high-risk)
        baseline_expected_amendments = baseline_amendment_prob * 2
        optimized_expected_amendments = optimized_amendment_prob * 2
        amendments_avoided = baseline_expected_amendments - optimized_expected_amendments

        # Cost savings
        direct_cost_saved = int(amendments_avoided * self.benchmarks.amendment_direct_cost)
        days_saved = int(amendments_avoided * self.benchmarks.amendment_delay_days)

        daily_cost = self.benchmarks.daily_delay_cost.get(
            therapeutic_key,
            self.benchmarks.daily_delay_cost["default"]
        )
        delay_cost_saved = days_saved * daily_cost

        total_cost_saved = direct_cost_saved + delay_cost_saved

        details = (
            f"Risk score reduced from {risk_score_before:.0f} to {risk_score_after:.0f} "
            f"({risk_reduction:.0f} point reduction). "
            f"Expected to avoid {amendments_avoided:.1f} amendments, "
            f"saving ${direct_cost_saved:,} direct costs and {days_saved} days of delays."
        )

        return ProtocolOptimizationROI(
            amendments_avoided=round(amendments_avoided, 1),
            days_saved=days_saved,
            cost_saved=total_cost_saved,
            risk_reduction=risk_reduction,
            details=details,
        )

    def _calculate_forecast_roi(
        self,
        therapeutic_area: str,
        num_sites: int,
        baseline_duration_months: float,
    ) -> Dict[str, Any]:
        """Calculate ROI from enrollment forecasting early warning system."""
        therapeutic_key = therapeutic_area.lower().replace(" ", "_")

        # Early warning allows intervention 30 days sooner on average
        days_saved = 30

        # Prevents escalation of enrollment issues
        # Average: saves 1 site replacement per 20 sites
        avoided_issues = max(1, num_sites // 20)
        issue_cost_saved = avoided_issues * self.benchmarks.underperforming_site_loss

        daily_cost = self.benchmarks.daily_delay_cost.get(
            therapeutic_key,
            self.benchmarks.daily_delay_cost["default"]
        )
        delay_cost_saved = days_saved * daily_cost

        total_cost_saved = issue_cost_saved + delay_cost_saved

        details = (
            f"Early warning system enables 30-day earlier intervention. "
            f"Prevents escalation of {avoided_issues} site issues, "
            f"saving ${issue_cost_saved:,} in wasted site investment."
        )

        return {
            "days_saved": days_saved,
            "cost_saved": total_cost_saved,
            "details": details,
        }

    def _calculate_similar_trial_roi(
        self,
        therapeutic_area: str,
        phase: str,
    ) -> Dict[str, Any]:
        """Calculate ROI from similar trial insights."""
        therapeutic_key = therapeutic_area.lower().replace(" ", "_")

        # Learning from similar trials prevents common mistakes
        # Conservative estimate: 15 days saved from avoiding known pitfalls
        days_saved = 15

        # Endpoint selection informed by success/failure patterns
        # Reduces risk of choosing endpoints that failed in similar trials
        daily_cost = self.benchmarks.daily_delay_cost.get(
            therapeutic_key,
            self.benchmarks.daily_delay_cost["default"]
        )
        delay_cost_saved = days_saved * daily_cost

        # Additional value from learning why similar trials terminated
        learning_value = 100000  # Conservative estimate

        total_cost_saved = delay_cost_saved + learning_value

        details = (
            f"Individual similar trial analysis prevents known pitfalls. "
            f"Endpoint and eligibility insights from {phase} trials in {therapeutic_area} "
            f"save an estimated 15 days and reduce design risk."
        )

        return {
            "days_saved": days_saved,
            "cost_saved": total_cost_saved,
            "details": details,
        }

    def format_roi_summary(self, roi: ROIAnalysis) -> str:
        """Format ROI analysis as readable summary."""
        lines = [
            "=" * 60,
            "TRIAL OPTIMIZATION ROI SUMMARY",
            "=" * 60,
            "",
            f"TOTAL VALUE:",
            f"  Days Saved: {roi.total_days_saved} days",
            f"  Cost Saved: ${roi.total_cost_saved:,}",
            f"  Revenue Acceleration: ${roi.total_revenue_impact:,}",
            f"  ROI: {roi.roi_percentage:.0f}%",
            "",
            f"TIMELINE IMPACT:",
            f"  Baseline Duration: {roi.baseline_duration_days} days",
            f"  Optimized Duration: {roi.optimized_duration_days} days",
            f"  Improvement: {(roi.baseline_duration_days - roi.optimized_duration_days) / roi.baseline_duration_days * 100:.1f}%",
            "",
            "SAVINGS BREAKDOWN:",
        ]

        for saving in roi.savings_breakdown:
            lines.append(f"  {saving.optimization_type}:")
            lines.append(f"    - Days Saved: {saving.days_saved}")
            lines.append(f"    - Cost Saved: ${saving.cost_saved:,}")
            lines.append(f"    - Confidence: {saving.confidence}")

        lines.extend([
            "",
            f"CONFIDENCE: {roi.confidence_level.upper()}",
            "",
            "KEY ASSUMPTIONS:",
        ])

        for assumption in roi.key_assumptions:
            lines.append(f"  - {assumption}")

        lines.append("=" * 60)

        return "\n".join(lines)


def calculate_trial_roi(
    therapeutic_area: str,
    phase: str,
    target_enrollment: int,
    num_sites: int,
    baseline_duration_months: float,
) -> ROIAnalysis:
    """Convenience function for full ROI calculation."""
    calculator = ROICalculator()
    return calculator.calculate_full_roi(
        therapeutic_area=therapeutic_area,
        phase=phase,
        target_enrollment=target_enrollment,
        num_sites=num_sites,
        baseline_duration_months=baseline_duration_months,
        optimizations_applied=[
            "site_selection",
            "protocol_risk",
            "enrollment_forecast",
        ],
        protocol_risk_reduction=20,
        enrollment_improvement_pct=15,
        similar_trial_insights_used=True,
    )
