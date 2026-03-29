"""
What-If Scenario Modeler for TrialIntel.

Enables interactive scenario planning:
- Protocol parameter adjustments
- Site count changes
- Eligibility criteria modifications
- Timeline and cost impact projections
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import math


class ScenarioType(Enum):
    BASELINE = "baseline"
    OPTIMIZED = "optimized"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    CUSTOM = "custom"


@dataclass
class ScenarioParameter:
    """A single adjustable parameter in a scenario."""
    name: str
    display_name: str
    current_value: Any
    modified_value: Any
    unit: str
    impact_description: str
    category: str  # enrollment, sites, protocol, timeline


@dataclass
class ScenarioImpact:
    """Impact assessment for a scenario."""
    timeline_impact_days: int
    cost_impact: float
    risk_score_change: float
    enrollment_rate_change: float
    success_probability_change: float
    key_tradeoffs: List[str]


@dataclass
class Scenario:
    """A complete scenario with parameters and projected impacts."""
    id: str
    name: str
    description: str
    scenario_type: ScenarioType
    created_at: datetime
    parameters: List[ScenarioParameter]
    impact: ScenarioImpact

    # Projected metrics
    projected_duration_days: int
    projected_cost: float
    projected_enrollment_rate: float
    projected_success_probability: float
    projected_risk_score: float

    # Comparison to baseline
    days_vs_baseline: int
    cost_vs_baseline: float
    risk_vs_baseline: float

    notes: str = ""
    is_recommended: bool = False


@dataclass
class ScenarioComparison:
    """Comparison of multiple scenarios."""
    scenarios: List[Scenario]
    recommended_scenario_id: str
    recommendation_reason: str
    comparison_matrix: Dict[str, Dict[str, Any]]


class ScenarioModeler:
    """
    Interactive scenario modeling for trial planning.

    Allows users to adjust parameters and see projected impacts
    on timeline, cost, and success probability.
    """

    # Cost benchmarks per day of delay by therapeutic area
    DAILY_DELAY_COSTS = {
        "oncology": 250000,
        "cardiovascular": 200000,
        "metabolic": 150000,
        "neurology": 180000,
        "immunology": 175000,
        "rare_disease": 300000,
        "infectious_disease": 120000,
        "respiratory": 140000,
        "default": 150000,
    }

    # Site costs
    SITE_STARTUP_COST = 50000
    SITE_MONTHLY_COST = 15000

    # Impact multipliers
    ENROLLMENT_IMPACT = {
        "add_site": 0.015,  # Each site adds ~1.5% enrollment capacity
        "relax_age": 0.05,  # Relaxing age adds ~5% enrollment
        "relax_hba1c": 0.08,  # Relaxing HbA1c adds ~8% enrollment
        "relax_egfr": 0.06,  # Relaxing eGFR adds ~6% enrollment
        "extend_duration": 0.02,  # Each month adds ~2% enrollment
    }

    def __init__(self, db_manager=None):
        self.db = db_manager
        self.scenarios: Dict[str, Scenario] = {}

    def _generate_scenario_id(self) -> str:
        """Generate unique scenario ID."""
        return f"SCN-{uuid.uuid4().hex[:8].upper()}"

    def create_baseline_scenario(
        self,
        condition: str,
        phase: str,
        target_enrollment: int,
        num_sites: int,
        planned_duration_months: int,
        therapeutic_area: str = "default",
        eligibility_criteria: Optional[Dict[str, Any]] = None,
        risk_score: float = 50.0,
    ) -> Scenario:
        """Create baseline scenario from current protocol parameters."""

        daily_cost = self.DAILY_DELAY_COSTS.get(
            therapeutic_area.lower().replace(" ", "_"),
            self.DAILY_DELAY_COSTS["default"]
        )

        duration_days = planned_duration_months * 30
        total_cost = (
            num_sites * self.SITE_STARTUP_COST +
            num_sites * self.SITE_MONTHLY_COST * planned_duration_months +
            daily_cost * duration_days * 0.1  # Operational costs
        )

        # Estimate enrollment rate (patients per site per month)
        enrollment_rate = target_enrollment / (num_sites * planned_duration_months)

        # Estimate success probability based on phase and risk
        base_success = {"PHASE1": 0.65, "PHASE2": 0.35, "PHASE3": 0.60, "PHASE4": 0.85}.get(phase, 0.5)
        success_probability = base_success * (1 - risk_score / 200)  # Risk reduces success prob

        parameters = [
            ScenarioParameter(
                name="num_sites",
                display_name="Number of Sites",
                current_value=num_sites,
                modified_value=num_sites,
                unit="sites",
                impact_description="More sites = faster enrollment, higher cost",
                category="sites",
            ),
            ScenarioParameter(
                name="target_enrollment",
                display_name="Target Enrollment",
                current_value=target_enrollment,
                modified_value=target_enrollment,
                unit="patients",
                impact_description="Higher target = longer duration, more power",
                category="enrollment",
            ),
            ScenarioParameter(
                name="duration_months",
                display_name="Planned Duration",
                current_value=planned_duration_months,
                modified_value=planned_duration_months,
                unit="months",
                impact_description="Longer duration = better enrollment, higher cost",
                category="timeline",
            ),
        ]

        # Add eligibility parameters if provided
        if eligibility_criteria:
            if "min_age" in eligibility_criteria:
                parameters.append(ScenarioParameter(
                    name="min_age",
                    display_name="Minimum Age",
                    current_value=eligibility_criteria.get("min_age", 18),
                    modified_value=eligibility_criteria.get("min_age", 18),
                    unit="years",
                    impact_description="Lower min age = larger pool, younger population",
                    category="protocol",
                ))
            if "max_age" in eligibility_criteria:
                parameters.append(ScenarioParameter(
                    name="max_age",
                    display_name="Maximum Age",
                    current_value=eligibility_criteria.get("max_age", 65),
                    modified_value=eligibility_criteria.get("max_age", 65),
                    unit="years",
                    impact_description="Higher max age = larger pool, more comorbidities",
                    category="protocol",
                ))

        scenario = Scenario(
            id=self._generate_scenario_id(),
            name="Baseline",
            description="Current protocol parameters",
            scenario_type=ScenarioType.BASELINE,
            created_at=datetime.now(),
            parameters=parameters,
            impact=ScenarioImpact(
                timeline_impact_days=0,
                cost_impact=0,
                risk_score_change=0,
                enrollment_rate_change=0,
                success_probability_change=0,
                key_tradeoffs=[],
            ),
            projected_duration_days=duration_days,
            projected_cost=total_cost,
            projected_enrollment_rate=enrollment_rate,
            projected_success_probability=success_probability,
            projected_risk_score=risk_score,
            days_vs_baseline=0,
            cost_vs_baseline=0,
            risk_vs_baseline=0,
        )

        self.scenarios[scenario.id] = scenario
        return scenario

    def model_site_change(
        self,
        baseline: Scenario,
        new_site_count: int,
    ) -> Scenario:
        """Model impact of changing number of sites."""

        # Get baseline parameters
        baseline_sites = next(
            (p.current_value for p in baseline.parameters if p.name == "num_sites"),
            50
        )
        baseline_duration = baseline.projected_duration_days
        baseline_cost = baseline.projected_cost
        baseline_rate = baseline.projected_enrollment_rate

        # Calculate impact
        site_change = new_site_count - baseline_sites

        # More sites = faster enrollment = shorter duration
        duration_factor = baseline_sites / new_site_count if new_site_count > 0 else 1
        new_duration_days = int(baseline_duration * duration_factor)

        # Cost change from sites
        site_cost_change = site_change * (
            self.SITE_STARTUP_COST +
            self.SITE_MONTHLY_COST * (baseline_duration / 30)
        )

        # But shorter duration saves operational costs
        days_saved = baseline_duration - new_duration_days
        daily_cost = self.DAILY_DELAY_COSTS.get("default", 150000)
        ops_savings = days_saved * daily_cost * 0.1

        new_cost = baseline_cost + site_cost_change - ops_savings

        # Enrollment rate stays similar per site
        new_rate = baseline_rate

        parameters = []
        for p in baseline.parameters:
            new_param = ScenarioParameter(
                name=p.name,
                display_name=p.display_name,
                current_value=p.current_value,
                modified_value=new_site_count if p.name == "num_sites" else p.modified_value,
                unit=p.unit,
                impact_description=p.impact_description,
                category=p.category,
            )
            parameters.append(new_param)

        tradeoffs = []
        if site_change > 0:
            tradeoffs.append(f"Adding {site_change} sites increases startup costs by ${site_change * self.SITE_STARTUP_COST:,.0f}")
            tradeoffs.append(f"Projected to save {days_saved:.0f} days in enrollment timeline")
        else:
            tradeoffs.append(f"Removing {abs(site_change)} sites saves ${abs(site_change) * self.SITE_STARTUP_COST:,.0f} in startup costs")
            tradeoffs.append(f"May extend timeline by {abs(days_saved):.0f} days")

        scenario = Scenario(
            id=self._generate_scenario_id(),
            name=f"{new_site_count} Sites",
            description=f"Modified site count from {baseline_sites} to {new_site_count}",
            scenario_type=ScenarioType.CUSTOM,
            created_at=datetime.now(),
            parameters=parameters,
            impact=ScenarioImpact(
                timeline_impact_days=new_duration_days - baseline_duration,
                cost_impact=new_cost - baseline_cost,
                risk_score_change=0,
                enrollment_rate_change=0,
                success_probability_change=0,
                key_tradeoffs=tradeoffs,
            ),
            projected_duration_days=new_duration_days,
            projected_cost=new_cost,
            projected_enrollment_rate=new_rate,
            projected_success_probability=baseline.projected_success_probability,
            projected_risk_score=baseline.projected_risk_score,
            days_vs_baseline=new_duration_days - baseline_duration,
            cost_vs_baseline=new_cost - baseline_cost,
            risk_vs_baseline=0,
        )

        self.scenarios[scenario.id] = scenario
        return scenario

    def model_eligibility_change(
        self,
        baseline: Scenario,
        changes: Dict[str, Any],
        therapeutic_area: str = "default",
    ) -> Scenario:
        """Model impact of eligibility criteria changes."""

        baseline_duration = baseline.projected_duration_days
        baseline_cost = baseline.projected_cost
        baseline_rate = baseline.projected_enrollment_rate
        baseline_risk = baseline.projected_risk_score

        # Calculate enrollment pool expansion
        pool_expansion = 1.0
        risk_change = 0
        tradeoffs = []

        if "relax_age" in changes:
            pool_expansion *= 1 + self.ENROLLMENT_IMPACT["relax_age"]
            risk_change += 3  # Slightly higher risk with broader age
            tradeoffs.append("Broader age range increases patient pool by ~5%")
            tradeoffs.append("May increase protocol complexity for older/younger patients")

        if "relax_hba1c" in changes:
            pool_expansion *= 1 + self.ENROLLMENT_IMPACT["relax_hba1c"]
            risk_change += 5  # Higher risk with less controlled patients
            tradeoffs.append("Relaxed HbA1c criteria increases pool by ~8%")
            tradeoffs.append("May need to account for higher baseline variability")

        if "relax_egfr" in changes:
            pool_expansion *= 1 + self.ENROLLMENT_IMPACT["relax_egfr"]
            risk_change += 4
            tradeoffs.append("Relaxed eGFR criteria increases pool by ~6%")
            tradeoffs.append("May require additional renal safety monitoring")

        # Duration decreases with larger pool
        new_duration_days = int(baseline_duration / pool_expansion)
        days_saved = baseline_duration - new_duration_days

        # Cost savings from shorter duration
        daily_cost = self.DAILY_DELAY_COSTS.get(
            therapeutic_area.lower().replace(" ", "_"),
            self.DAILY_DELAY_COSTS["default"]
        )
        cost_savings = days_saved * daily_cost * 0.1
        new_cost = baseline_cost - cost_savings

        # New enrollment rate
        new_rate = baseline_rate * pool_expansion

        # New risk score
        new_risk = min(100, baseline_risk + risk_change)

        # Copy parameters with modifications
        parameters = [p for p in baseline.parameters]

        scenario = Scenario(
            id=self._generate_scenario_id(),
            name="Relaxed Eligibility",
            description="Modified eligibility criteria to increase patient pool",
            scenario_type=ScenarioType.OPTIMIZED,
            created_at=datetime.now(),
            parameters=parameters,
            impact=ScenarioImpact(
                timeline_impact_days=-days_saved,
                cost_impact=-cost_savings,
                risk_score_change=risk_change,
                enrollment_rate_change=(pool_expansion - 1) * 100,
                success_probability_change=-risk_change * 0.5,  # Risk increases, success prob decreases
                key_tradeoffs=tradeoffs,
            ),
            projected_duration_days=new_duration_days,
            projected_cost=new_cost,
            projected_enrollment_rate=new_rate,
            projected_success_probability=baseline.projected_success_probability * (1 - risk_change / 200),
            projected_risk_score=new_risk,
            days_vs_baseline=-days_saved,
            cost_vs_baseline=-cost_savings,
            risk_vs_baseline=risk_change,
        )

        self.scenarios[scenario.id] = scenario
        return scenario

    def model_duration_change(
        self,
        baseline: Scenario,
        new_duration_months: int,
        therapeutic_area: str = "default",
    ) -> Scenario:
        """Model impact of changing planned duration."""

        baseline_duration = baseline.projected_duration_days
        new_duration_days = new_duration_months * 30
        duration_change = new_duration_days - baseline_duration

        # Cost change
        daily_cost = self.DAILY_DELAY_COSTS.get(
            therapeutic_area.lower().replace(" ", "_"),
            self.DAILY_DELAY_COSTS["default"]
        )

        baseline_sites = next(
            (p.current_value for p in baseline.parameters if p.name == "num_sites"),
            50
        )

        # Site costs change with duration
        site_cost_change = baseline_sites * self.SITE_MONTHLY_COST * (duration_change / 30)
        ops_cost_change = duration_change * daily_cost * 0.1

        new_cost = baseline.projected_cost + site_cost_change + ops_cost_change

        # Longer duration = higher enrollment probability
        enrollment_factor = 1 + (duration_change / baseline_duration) * 0.5
        new_rate = baseline.projected_enrollment_rate

        # Risk decreases slightly with more time
        risk_change = -5 if duration_change > 0 else 5

        tradeoffs = []
        if duration_change > 0:
            tradeoffs.append(f"Extended duration adds ${ops_cost_change:,.0f} in operational costs")
            tradeoffs.append("More time reduces enrollment pressure and risk")
            tradeoffs.append("Delays time-to-market and revenue")
        else:
            tradeoffs.append(f"Shortened duration saves ${abs(ops_cost_change):,.0f}")
            tradeoffs.append("Accelerates time-to-market")
            tradeoffs.append("Increases enrollment pressure and risk")

        parameters = []
        for p in baseline.parameters:
            new_param = ScenarioParameter(
                name=p.name,
                display_name=p.display_name,
                current_value=p.current_value,
                modified_value=new_duration_months if p.name == "duration_months" else p.modified_value,
                unit=p.unit,
                impact_description=p.impact_description,
                category=p.category,
            )
            parameters.append(new_param)

        scenario = Scenario(
            id=self._generate_scenario_id(),
            name=f"{new_duration_months} Months",
            description=f"Modified duration from {baseline_duration // 30} to {new_duration_months} months",
            scenario_type=ScenarioType.CUSTOM,
            created_at=datetime.now(),
            parameters=parameters,
            impact=ScenarioImpact(
                timeline_impact_days=duration_change,
                cost_impact=site_cost_change + ops_cost_change,
                risk_score_change=risk_change,
                enrollment_rate_change=0,
                success_probability_change=abs(risk_change) * 0.3 * (1 if duration_change > 0 else -1),
                key_tradeoffs=tradeoffs,
            ),
            projected_duration_days=new_duration_days,
            projected_cost=new_cost,
            projected_enrollment_rate=new_rate,
            projected_success_probability=baseline.projected_success_probability + risk_change * 0.003,
            projected_risk_score=max(0, min(100, baseline.projected_risk_score + risk_change)),
            days_vs_baseline=duration_change,
            cost_vs_baseline=site_cost_change + ops_cost_change,
            risk_vs_baseline=risk_change,
        )

        self.scenarios[scenario.id] = scenario
        return scenario

    def generate_optimized_scenarios(
        self,
        baseline: Scenario,
        therapeutic_area: str = "default",
    ) -> List[Scenario]:
        """Generate a set of optimized scenarios automatically."""
        scenarios = []

        baseline_sites = next(
            (p.current_value for p in baseline.parameters if p.name == "num_sites"),
            50
        )

        # Aggressive: More sites, shorter timeline
        aggressive = self.model_site_change(baseline, int(baseline_sites * 1.5))
        aggressive.name = "Aggressive"
        aggressive.description = "50% more sites for faster enrollment"
        aggressive.scenario_type = ScenarioType.AGGRESSIVE
        aggressive.is_recommended = False
        scenarios.append(aggressive)

        # Conservative: Fewer sites, lower risk
        conservative = self.model_site_change(baseline, int(baseline_sites * 0.75))
        conservative.name = "Conservative"
        conservative.description = "25% fewer sites, lower cost"
        conservative.scenario_type = ScenarioType.CONSERVATIVE
        scenarios.append(conservative)

        # Optimized: Balance of improvements
        optimized = self.model_eligibility_change(
            baseline,
            {"relax_age": True},
            therapeutic_area
        )
        optimized.name = "Optimized"
        optimized.description = "Relaxed eligibility for faster enrollment"
        optimized.scenario_type = ScenarioType.OPTIMIZED
        optimized.is_recommended = True
        scenarios.append(optimized)

        return scenarios

    def compare_scenarios(self, scenario_ids: List[str]) -> ScenarioComparison:
        """Compare multiple scenarios."""
        scenarios = [self.scenarios[sid] for sid in scenario_ids if sid in self.scenarios]

        if not scenarios:
            raise ValueError("No valid scenarios found")

        # Build comparison matrix
        matrix = {}
        for s in scenarios:
            matrix[s.id] = {
                "name": s.name,
                "duration_days": s.projected_duration_days,
                "cost": s.projected_cost,
                "risk_score": s.projected_risk_score,
                "success_prob": s.projected_success_probability,
                "enrollment_rate": s.projected_enrollment_rate,
            }

        # Find recommended scenario (lowest risk-adjusted cost)
        def score_scenario(s: Scenario) -> float:
            # Balance cost, time, and risk
            time_score = s.projected_duration_days / 365  # Normalize to years
            cost_score = s.projected_cost / 10_000_000  # Normalize to $10M
            risk_score = s.projected_risk_score / 100
            return time_score + cost_score + risk_score

        recommended = min(scenarios, key=score_scenario)

        return ScenarioComparison(
            scenarios=scenarios,
            recommended_scenario_id=recommended.id,
            recommendation_reason=f"Best balance of timeline ({recommended.projected_duration_days} days), "
                                 f"cost (${recommended.projected_cost:,.0f}), and risk ({recommended.projected_risk_score:.0f})",
            comparison_matrix=matrix,
        )

    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Get scenario by ID."""
        return self.scenarios.get(scenario_id)

    def get_all_scenarios(self) -> List[Scenario]:
        """Get all scenarios."""
        return list(self.scenarios.values())

    def delete_scenario(self, scenario_id: str) -> bool:
        """Delete a scenario."""
        if scenario_id in self.scenarios:
            del self.scenarios[scenario_id]
            return True
        return False
