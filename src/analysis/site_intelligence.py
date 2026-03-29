"""
Enhanced Site & Geographic Intelligence

Provides comprehensive site selection and geographic analysis including:
1. Site performance scoring (enrollment velocity, completion rate, experience)
2. Geographic distribution and coverage optimization
3. Regulatory context by country/region
4. Site capacity and saturation assessment
5. Competitive site analysis
6. Recommended site portfolio optimization
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SitePerformance:
    """Performance metrics for a clinical trial site."""
    facility_name: str
    city: str
    country: str
    region: str

    # Performance metrics
    total_trials: int
    completed_trials: int
    terminated_trials: int
    active_trials: int
    completion_rate: float  # 0-100

    # Enrollment metrics
    avg_enrollment: float
    total_patients_enrolled: int
    enrollment_velocity: float  # patients/month estimated

    # Experience
    therapeutic_area_experience: int  # trials in this area
    phase_experience: int  # trials in this phase
    years_active: int

    # Capacity assessment
    capacity_score: str  # high, medium, low
    saturation_risk: str  # high, medium, low

    # Overall score
    overall_score: float  # 0-100
    recommendation: str  # highly_recommended, recommended, consider, caution


@dataclass
class CountryProfile:
    """Profile for a country's clinical trial landscape."""
    country: str
    region: str

    # Trial metrics
    total_trials: int
    completed_trials: int
    recruiting_trials: int
    completion_rate: float

    # Site metrics
    num_sites: int
    avg_sites_per_trial: float

    # Enrollment
    total_enrollment: int
    avg_enrollment_per_trial: float

    # Regulatory context
    regulatory_complexity: str  # low, medium, high
    regulatory_notes: str
    avg_approval_time: str  # estimated

    # Competition
    competitor_trials: int
    market_saturation: str  # low, medium, high

    # Recommendation
    recommendation: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)


@dataclass
class GeographicDistribution:
    """Geographic distribution analysis."""
    total_countries: int
    total_sites: int
    total_enrollment_capacity: int

    # By region
    by_region: Dict[str, Dict[str, Any]]

    # Top countries
    top_countries: List[CountryProfile]

    # Coverage analysis
    coverage_score: float  # 0-100
    diversity_score: float  # 0-100
    recommendations: List[str]


@dataclass
class SitePortfolio:
    """Recommended site portfolio for a trial."""
    total_recommended_sites: int
    total_recommended_countries: int
    estimated_enrollment_capacity: int
    estimated_enrollment_months: float

    # By tier
    tier1_sites: List[SitePerformance]  # Top performers
    tier2_sites: List[SitePerformance]  # Good performers
    tier3_sites: List[SitePerformance]  # Backup sites

    # Geographic balance
    geographic_distribution: Dict[str, int]  # region -> site count
    country_distribution: Dict[str, int]  # country -> site count

    # Risk assessment
    portfolio_risk: str  # low, medium, high
    risk_factors: List[str]

    # Optimization suggestions
    optimization_suggestions: List[str]


@dataclass
class SiteIntelligenceReport:
    """Complete site intelligence report."""
    condition: str
    phase: str
    target_enrollment: int
    target_sites: int

    # Site recommendations
    top_sites: List[SitePerformance]
    sites_by_country: Dict[str, List[SitePerformance]]

    # Geographic analysis
    geographic_distribution: GeographicDistribution
    country_profiles: List[CountryProfile]

    # Portfolio recommendation
    recommended_portfolio: SitePortfolio

    # Competitive intelligence
    competitor_site_overlap: int
    competitor_heavy_regions: List[str]

    # Key insights
    key_insights: List[str]
    recommendations: List[str]


class SiteIntelligenceAnalyzer:
    """
    Comprehensive site and geographic intelligence analysis.
    """

    # Regional mappings
    COUNTRY_TO_REGION = {
        "United States": "North America", "Canada": "North America", "Mexico": "North America",
        "Germany": "Western Europe", "France": "Western Europe", "United Kingdom": "Western Europe",
        "Italy": "Western Europe", "Spain": "Western Europe", "Netherlands": "Western Europe",
        "Belgium": "Western Europe", "Switzerland": "Western Europe", "Austria": "Western Europe",
        "Poland": "Eastern Europe", "Czech Republic": "Eastern Europe", "Hungary": "Eastern Europe",
        "Romania": "Eastern Europe", "Bulgaria": "Eastern Europe", "Russia": "Eastern Europe",
        "Ukraine": "Eastern Europe",
        "Japan": "Asia Pacific", "China": "Asia Pacific", "South Korea": "Asia Pacific",
        "Taiwan": "Asia Pacific", "Australia": "Asia Pacific", "New Zealand": "Asia Pacific",
        "India": "Asia Pacific", "Singapore": "Asia Pacific", "Hong Kong": "Asia Pacific",
        "Brazil": "Latin America", "Argentina": "Latin America", "Chile": "Latin America",
        "Colombia": "Latin America", "Peru": "Latin America",
        "Israel": "Middle East", "South Africa": "Africa", "Turkey": "Middle East",
    }

    # Regulatory complexity by country
    REGULATORY_COMPLEXITY = {
        "United States": ("medium", "FDA requirements, IND needed, IRB approval per site"),
        "Germany": ("medium", "BfArM approval, ethics committee"),
        "France": ("medium", "ANSM approval, CPP ethics"),
        "United Kingdom": ("medium", "MHRA approval, REC ethics"),
        "Japan": ("high", "PMDA approval, strict requirements, language barriers"),
        "China": ("high", "NMPA approval, local partner often required"),
        "India": ("medium", "CDSCO approval, ethics committee, growing infrastructure"),
        "Brazil": ("medium", "ANVISA approval, ethics committee"),
        "Australia": ("low", "TGA notification, HREC ethics, efficient process"),
        "Canada": ("low", "Health Canada, REB approval, efficient"),
        "Poland": ("low", "Fast approval, experienced sites, cost-effective"),
        "Czech Republic": ("low", "Efficient regulatory, experienced sites"),
        "South Korea": ("medium", "MFDS approval, good infrastructure"),
        "Israel": ("low", "Fast approval, high quality sites"),
        "Netherlands": ("low", "Efficient process, good infrastructure"),
        "Spain": ("medium", "AEMPS approval, ethics committee"),
        "Italy": ("medium", "AIFA approval, ethics committee"),
    }

    def __init__(self, db_manager, api_key: Optional[str] = None):
        self.db = db_manager
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def _get_region(self, country: str) -> str:
        return self.COUNTRY_TO_REGION.get(country, "Other")

    def _get_regulatory_info(self, country: str) -> Tuple[str, str]:
        return self.REGULATORY_COMPLEXITY.get(country, ("medium", "Standard regulatory process"))

    def _calculate_site_score(self, site: Dict[str, Any], condition: str, phase: str) -> float:
        score = 0
        completion_rate = site.get("completion_rate", 0)
        score += min(40, completion_rate * 0.4)

        total_trials = site.get("total_trials", 0)
        if total_trials >= 20: score += 25
        elif total_trials >= 10: score += 20
        elif total_trials >= 5: score += 15
        elif total_trials >= 3: score += 10
        else: score += 5

        ta_experience = site.get("therapeutic_area_trials", 0)
        if ta_experience >= 10: score += 20
        elif ta_experience >= 5: score += 15
        elif ta_experience >= 2: score += 10
        else: score += 5

        avg_enrollment = site.get("avg_enrollment", 0)
        if avg_enrollment >= 100: score += 15
        elif avg_enrollment >= 50: score += 12
        elif avg_enrollment >= 20: score += 8
        else: score += 5

        return min(100, score)

    def _assess_capacity(self, active_trials: int, total_trials: int) -> Tuple[str, str]:
        if total_trials == 0: return "unknown", "unknown"
        active_ratio = active_trials / total_trials if total_trials > 0 else 0
        if active_trials >= 10 or active_ratio > 0.5: return "low", "high"
        elif active_trials >= 5 or active_ratio > 0.3: return "medium", "medium"
        else: return "high", "low"

    def _get_site_data(self, condition: str, phase: str, country_filter: Optional[str] = None, limit: int = 500) -> List[Dict]:
        from sqlalchemy import text
        country_clause = ""
        if country_filter and country_filter != "Any":
            country_clause = f"AND LOWER(json_extract(value, '$.country')) LIKE '%{country_filter.lower()}%'"

        query = text(f"""
            WITH site_trials AS (
                SELECT json_extract(value, '$.facility') as facility_name,
                    json_extract(value, '$.city') as city,
                    json_extract(value, '$.country') as country,
                    t.nct_id, t.status, t.enrollment, t.phase
                FROM trials t, json_each(t.locations)
                WHERE t.locations IS NOT NULL
                AND json_extract(value, '$.facility') IS NOT NULL
                AND (LOWER(t.conditions) LIKE :condition OR LOWER(t.therapeutic_area) LIKE :condition)
                {country_clause}
            )
            SELECT facility_name, city, country,
                COUNT(DISTINCT nct_id) as total_trials,
                SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status IN ('TERMINATED', 'WITHDRAWN') THEN 1 ELSE 0 END) as terminated,
                SUM(CASE WHEN status IN ('RECRUITING', 'ACTIVE_NOT_RECRUITING') THEN 1 ELSE 0 END) as active,
                AVG(CASE WHEN enrollment > 0 THEN enrollment ELSE NULL END) as avg_enrollment,
                SUM(CASE WHEN enrollment > 0 THEN enrollment ELSE 0 END) as total_enrollment,
                COUNT(DISTINCT CASE WHEN phase = :phase THEN nct_id END) as phase_trials
            FROM site_trials
            GROUP BY facility_name, city, country
            HAVING total_trials >= 2
            ORDER BY completed DESC, total_trials DESC
            LIMIT :limit
        """)

        results = self.db.execute_raw(query.text, {"condition": f"%{condition.lower()}%", "phase": phase, "limit": limit})
        sites = []
        for r in results:
            total, completed, terminated, active = r[3] or 0, r[4] or 0, r[5] or 0, r[6] or 0
            completion_rate = (completed / total * 100) if total > 0 else 0
            capacity, saturation = self._assess_capacity(active, total)
            sites.append({
                "facility_name": r[0], "city": r[1], "country": r[2],
                "total_trials": total, "completed": completed, "terminated": terminated, "active": active,
                "completion_rate": completion_rate, "avg_enrollment": r[7] or 0,
                "total_enrollment": r[8] or 0, "phase_trials": r[9] or 0,
                "capacity": capacity, "saturation": saturation, "therapeutic_area_trials": total
            })
        return sites

    def _get_country_data(self, condition: str, phase: str) -> List[Dict]:
        from sqlalchemy import text
        query = text("""
            WITH country_trials AS (
                SELECT DISTINCT json_extract(value, '$.country') as country,
                    t.nct_id, t.status, t.enrollment, t.sponsor
                FROM trials t, json_each(t.locations)
                WHERE t.locations IS NOT NULL
                AND (LOWER(t.conditions) LIKE :condition OR LOWER(t.therapeutic_area) LIKE :condition)
                AND json_extract(value, '$.country') IS NOT NULL
            )
            SELECT country, COUNT(DISTINCT nct_id) as total_trials,
                SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'RECRUITING' THEN 1 ELSE 0 END) as recruiting,
                SUM(CASE WHEN status IN ('TERMINATED', 'WITHDRAWN') THEN 1 ELSE 0 END) as terminated,
                AVG(CASE WHEN enrollment > 0 THEN enrollment ELSE NULL END) as avg_enrollment,
                SUM(CASE WHEN enrollment > 0 THEN enrollment ELSE 0 END) as total_enrollment,
                COUNT(DISTINCT sponsor) as num_sponsors
            FROM country_trials GROUP BY country HAVING total_trials >= 2
            ORDER BY total_trials DESC LIMIT 50
        """)
        results = self.db.execute_raw(query.text, {"condition": f"%{condition.lower()}%"})
        countries = []
        for r in results:
            total, completed, recruiting, terminated = r[1] or 0, r[2] or 0, r[3] or 0, r[4] or 0
            completion_rate = (completed / (completed + terminated) * 100) if (completed + terminated) > 0 else 0
            reg_complexity, reg_notes = self._get_regulatory_info(r[0])
            saturation = "high" if recruiting > 10 else "medium" if recruiting > 5 else "low"
            countries.append({
                "country": r[0], "region": self._get_region(r[0]), "total_trials": total,
                "completed": completed, "recruiting": recruiting, "terminated": terminated,
                "completion_rate": completion_rate, "avg_enrollment": r[5] or 0,
                "total_enrollment": r[6] or 0, "num_sponsors": r[7] or 0,
                "regulatory_complexity": reg_complexity, "regulatory_notes": reg_notes,
                "market_saturation": saturation
            })
        return countries

    def _build_site_performance(self, site: Dict, condition: str, phase: str) -> SitePerformance:
        score = self._calculate_site_score(site, condition, phase)
        if score >= 80 and site["capacity"] != "low": recommendation = "highly_recommended"
        elif score >= 60: recommendation = "recommended"
        elif score >= 40: recommendation = "consider"
        else: recommendation = "caution"
        velocity = site["avg_enrollment"] / 24 if site["avg_enrollment"] > 0 else 0

        return SitePerformance(
            facility_name=site["facility_name"] or "Unknown", city=site["city"] or "Unknown",
            country=site["country"] or "Unknown", region=self._get_region(site["country"] or ""),
            total_trials=site["total_trials"], completed_trials=site["completed"],
            terminated_trials=site["terminated"], active_trials=site["active"],
            completion_rate=site["completion_rate"], avg_enrollment=site["avg_enrollment"],
            total_patients_enrolled=site["total_enrollment"], enrollment_velocity=velocity,
            therapeutic_area_experience=site["therapeutic_area_trials"],
            phase_experience=site["phase_trials"], years_active=5,
            capacity_score=site["capacity"], saturation_risk=site["saturation"],
            overall_score=score, recommendation=recommendation
        )

    def _build_country_profile(self, country: Dict) -> CountryProfile:
        pros, cons = [], []
        if country["completion_rate"] >= 70: pros.append("High completion rate")
        elif country["completion_rate"] < 50: cons.append("Low completion rate")
        if country["regulatory_complexity"] == "low": pros.append("Efficient regulatory process")
        elif country["regulatory_complexity"] == "high": cons.append("Complex regulatory requirements")
        if country["market_saturation"] == "low": pros.append("Low competition")
        elif country["market_saturation"] == "high": cons.append("High competition for patients")
        if country["total_trials"] >= 20: pros.append("Experienced site network")
        if country["avg_enrollment"] >= 100: pros.append("Strong enrollment capacity")

        if len(pros) >= 3 and len(cons) <= 1: recommendation = "highly_recommended"
        elif len(pros) >= 2: recommendation = "recommended"
        elif len(cons) >= 2: recommendation = "consider_carefully"
        else: recommendation = "neutral"

        return CountryProfile(
            country=country["country"], region=country["region"],
            total_trials=country["total_trials"], completed_trials=country["completed"],
            recruiting_trials=country["recruiting"], completion_rate=country["completion_rate"],
            num_sites=0, avg_sites_per_trial=0, total_enrollment=country["total_enrollment"],
            avg_enrollment_per_trial=country["avg_enrollment"],
            regulatory_complexity=country["regulatory_complexity"],
            regulatory_notes=country["regulatory_notes"], avg_approval_time="2-4 months",
            competitor_trials=country["recruiting"], market_saturation=country["market_saturation"],
            recommendation=recommendation, pros=pros, cons=cons
        )

    def _build_portfolio(self, sites: List[SitePerformance], target_enrollment: int, target_sites: int) -> SitePortfolio:
        sorted_sites = sorted(sites, key=lambda x: x.overall_score, reverse=True)
        tier1 = [s for s in sorted_sites if s.recommendation == "highly_recommended"][:target_sites // 3 + 1]
        tier2 = [s for s in sorted_sites if s.recommendation == "recommended"][:target_sites // 3 + 1]
        tier3 = [s for s in sorted_sites if s.recommendation == "consider"][:target_sites // 3 + 1]
        all_selected = (tier1 + tier2 + tier3)[:target_sites]

        geo_dist, country_dist = defaultdict(int), defaultdict(int)
        for site in all_selected:
            geo_dist[site.region] += 1
            country_dist[site.country] += 1

        total_capacity = sum(s.avg_enrollment for s in all_selected)
        total_velocity = sum(s.enrollment_velocity for s in all_selected if s.enrollment_velocity > 0)
        estimated_months = target_enrollment / total_velocity if total_velocity > 0 else 24

        risk_factors = []
        if len(set(s.country for s in all_selected)) < 3: risk_factors.append("Limited geographic diversity")
        if sum(1 for s in all_selected if s.saturation_risk == "high") > len(all_selected) * 0.3:
            risk_factors.append("High site saturation in portfolio")
        portfolio_risk = "high" if len(risk_factors) >= 2 else "medium" if len(risk_factors) == 1 else "low"

        suggestions = []
        if len(geo_dist) < 3: suggestions.append("Consider adding sites in additional regions")
        if estimated_months > 24: suggestions.append("Consider adding more high-performing sites")

        return SitePortfolio(
            total_recommended_sites=len(all_selected), total_recommended_countries=len(country_dist),
            estimated_enrollment_capacity=int(total_capacity), estimated_enrollment_months=estimated_months,
            tier1_sites=tier1, tier2_sites=tier2, tier3_sites=tier3,
            geographic_distribution=dict(geo_dist), country_distribution=dict(country_dist),
            portfolio_risk=portfolio_risk, risk_factors=risk_factors, optimization_suggestions=suggestions
        )

    def analyze(self, condition: str, phase: str, target_enrollment: int = 200,
                target_sites: int = 30, country_filter: Optional[str] = None) -> SiteIntelligenceReport:
        logger.info(f"Analyzing sites for {condition} {phase}")
        site_data = self._get_site_data(condition, phase, country_filter)
        country_data = self._get_country_data(condition, phase)

        site_performances = [self._build_site_performance(s, condition, phase) for s in site_data]
        site_performances.sort(key=lambda x: x.overall_score, reverse=True)

        sites_by_country = defaultdict(list)
        for site in site_performances:
            sites_by_country[site.country].append(site)

        country_profiles = [self._build_country_profile(c) for c in country_data]
        country_profiles.sort(key=lambda x: x.total_trials, reverse=True)

        by_region = defaultdict(lambda: {"trials": 0, "sites": 0, "countries": set()})
        for profile in country_profiles:
            by_region[profile.region]["trials"] += profile.total_trials
            by_region[profile.region]["countries"].add(profile.country)
        for region in by_region:
            by_region[region]["countries"] = len(by_region[region]["countries"])

        coverage_score = min(100, len(by_region) * 15)
        diversity_score = min(100, len(country_profiles) * 5)

        geo_recommendations = []
        if len(by_region) < 3: geo_recommendations.append("Consider expanding to additional regions")

        geo_distribution = GeographicDistribution(
            total_countries=len(country_profiles), total_sites=len(site_performances),
            total_enrollment_capacity=sum(s.total_patients_enrolled for s in site_performances),
            by_region=dict(by_region), top_countries=country_profiles[:10],
            coverage_score=coverage_score, diversity_score=diversity_score,
            recommendations=geo_recommendations
        )

        portfolio = self._build_portfolio(site_performances, target_enrollment, target_sites)

        competitor_regions = [p.region for p in country_profiles if p.market_saturation == "high"]

        insights = []
        if site_performances:
            top_country = max(sites_by_country.items(), key=lambda x: len(x[1]))[0]
            insights.append(f"Strongest site network in {top_country}")
        if country_profiles:
            best_completion = max(country_profiles, key=lambda x: x.completion_rate)
            insights.append(f"Highest completion rate: {best_completion.country} ({best_completion.completion_rate:.0f}%)")
        high_capacity = [s for s in site_performances if s.capacity_score == "high"]
        if high_capacity: insights.append(f"{len(high_capacity)} sites with high capacity available")

        recommendations = portfolio.optimization_suggestions.copy()
        if competitor_regions:
            recommendations.append(f"High competition in: {', '.join(set(competitor_regions)[:3])}")

        return SiteIntelligenceReport(
            condition=condition, phase=phase,
            target_enrollment=target_enrollment, target_sites=target_sites,
            top_sites=site_performances[:50], sites_by_country=dict(sites_by_country),
            geographic_distribution=geo_distribution, country_profiles=country_profiles,
            recommended_portfolio=portfolio, competitor_site_overlap=0,
            competitor_heavy_regions=list(set(competitor_regions)),
            key_insights=insights, recommendations=recommendations
        )
