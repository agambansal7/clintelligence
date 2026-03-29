"""
Reporting Suite for TrialIntel.

Generates comprehensive reports:
- Executive summaries
- Protocol analysis reports
- Site selection reports
- ROI analysis reports
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import io


@dataclass
class ReportSection:
    """A section of a report."""
    title: str
    content: str
    data: Optional[Dict[str, Any]] = None
    charts: Optional[List[Dict[str, Any]]] = None
    tables: Optional[List[Dict[str, Any]]] = None


@dataclass
class Report:
    """A complete report."""
    id: str
    title: str
    report_type: str
    created_at: datetime
    sections: List[ReportSection]
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """
    Generates comprehensive reports from TrialIntel analyses.
    """

    def __init__(self, db_manager=None):
        self.db = db_manager
        self._report_counter = 0

    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        self._report_counter += 1
        return f"RPT-{datetime.now().strftime('%Y%m%d')}-{self._report_counter:04d}"

    def generate_executive_summary(
        self,
        trial_name: str,
        condition: str,
        phase: str,
        risk_assessment: Optional[Dict[str, Any]] = None,
        enrollment_forecast: Optional[Dict[str, Any]] = None,
        site_recommendations: Optional[List[Dict[str, Any]]] = None,
        roi_analysis: Optional[Dict[str, Any]] = None,
    ) -> Report:
        """Generate executive summary report."""

        sections = []

        # Overview section
        overview_content = f"""
## Trial Overview

**Trial Name:** {trial_name}
**Indication:** {condition}
**Phase:** {phase}
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

This executive summary provides a comprehensive analysis of the trial protocol,
enrollment projections, site recommendations, and expected ROI from TrialIntel optimizations.
"""
        sections.append(ReportSection(
            title="Overview",
            content=overview_content,
        ))

        key_findings = []
        recommendations = []

        # Risk Assessment section
        if risk_assessment:
            risk_score = risk_assessment.get("overall_risk_score", 50)
            risk_level = "HIGH" if risk_score > 70 else "MEDIUM" if risk_score > 40 else "LOW"

            risk_content = f"""
## Protocol Risk Assessment

**Overall Risk Score:** {risk_score}/100 ({risk_level})
**Amendment Probability:** {risk_assessment.get('amendment_probability', 0.3):.0%}
**Enrollment Delay Risk:** {risk_assessment.get('enrollment_delay_probability', 0.4):.0%}
**Termination Risk:** {risk_assessment.get('termination_probability', 0.1):.0%}

### Key Risk Factors
"""
            for rf in risk_assessment.get("risk_factors", [])[:5]:
                risk_content += f"\n- **{rf.get('category', 'General')}**: {rf.get('description', 'N/A')}"
                risk_content += f"\n  - Recommendation: {rf.get('recommendation', 'N/A')}"

            sections.append(ReportSection(
                title="Risk Assessment",
                content=risk_content,
                data=risk_assessment,
            ))

            key_findings.append(f"Protocol risk score is {risk_score}/100 ({risk_level})")
            if risk_score > 50:
                recommendations.append("Consider protocol amendments to reduce risk factors")

        # Enrollment Forecast section
        if enrollment_forecast:
            target = enrollment_forecast.get("target_enrollment", 0)
            days = enrollment_forecast.get("projected_days_to_target", 0)
            completion_date = enrollment_forecast.get("projected_completion_date", "TBD")

            forecast_content = f"""
## Enrollment Forecast

**Target Enrollment:** {target:,} patients
**Projected Days to Target:** {days:,.0f} days
**Projected Completion Date:** {completion_date}
**Confidence Level:** {enrollment_forecast.get('confidence_level', 'medium').upper()}

### Timeline Scenarios
- **Optimistic:** {enrollment_forecast.get('optimistic_days', days * 0.8):.0f} days
- **Expected:** {days:.0f} days
- **Pessimistic:** {enrollment_forecast.get('pessimistic_days', days * 1.3):.0f} days

### Enrollment Risk Factors
"""
            for risk in enrollment_forecast.get("risk_factors", [])[:3]:
                forecast_content += f"\n- {risk}"

            sections.append(ReportSection(
                title="Enrollment Forecast",
                content=forecast_content,
                data=enrollment_forecast,
            ))

            key_findings.append(f"Projected enrollment completion in {days:.0f} days")
            if enrollment_forecast.get("enrollment_risk_score", 0) > 50:
                recommendations.append("Implement enrollment acceleration strategies")

        # Site Recommendations section
        if site_recommendations:
            site_content = f"""
## Site Selection Recommendations

**Total Sites Analyzed:** {len(site_recommendations)}
**Top Recommendation:** {site_recommendations[0].get('facility_name', 'N/A') if site_recommendations else 'N/A'}

### Top 5 Recommended Sites

| Rank | Facility | Location | Score | Trials |
|------|----------|----------|-------|--------|
"""
            for i, site in enumerate(site_recommendations[:5], 1):
                site_content += f"| {i} | {site.get('facility_name', 'N/A')[:30]} | {site.get('city', 'N/A')}, {site.get('country', 'N/A')} | {site.get('overall_score', 0):.0f} | {site.get('total_trials', 0)} |\n"

            sections.append(ReportSection(
                title="Site Recommendations",
                content=site_content,
                data={"sites": site_recommendations[:10]},
            ))

            key_findings.append(f"Identified {len(site_recommendations)} potential sites")
            recommendations.append("Prioritize top-ranked sites for feasibility assessment")

        # ROI Analysis section
        if roi_analysis:
            days_saved = roi_analysis.get("total_days_saved", 0)
            cost_saved = roi_analysis.get("total_cost_saved", 0)
            roi_pct = roi_analysis.get("roi_percentage", 0)

            roi_content = f"""
## ROI Analysis

**Total Days Saved:** {days_saved:,}
**Total Cost Saved:** ${cost_saved:,.0f}
**Revenue Impact:** ${roi_analysis.get('total_revenue_impact', 0):,.0f}
**ROI Percentage:** {roi_pct:,.0f}%

### Savings Breakdown
"""
            for saving in roi_analysis.get("savings_breakdown", []):
                roi_content += f"\n- **{saving.get('optimization_type', 'N/A').replace('_', ' ').title()}**: ${saving.get('cost_saved', 0):,.0f} ({saving.get('days_saved', 0)} days)"

            sections.append(ReportSection(
                title="ROI Analysis",
                content=roi_content,
                data=roi_analysis,
            ))

            key_findings.append(f"Projected ${cost_saved:,.0f} in savings ({days_saved} days faster)")
            recommendations.append("Implement recommended optimizations to realize projected ROI")

        # Recommendations section
        recs_content = """
## Key Recommendations

Based on the comprehensive analysis, we recommend the following actions:

"""
        for i, rec in enumerate(recommendations, 1):
            recs_content += f"{i}. {rec}\n"

        sections.append(ReportSection(
            title="Recommendations",
            content=recs_content,
        ))

        # Summary
        summary = f"""
Executive summary for {trial_name} ({phase} in {condition}).
Risk assessment, enrollment forecasting, site selection, and ROI analysis indicate
{'favorable conditions for trial execution' if (risk_assessment or {}).get('overall_risk_score', 50) < 50 else 'areas requiring attention to optimize trial success'}.
"""

        return Report(
            id=self._generate_report_id(),
            title=f"Executive Summary: {trial_name}",
            report_type="executive_summary",
            created_at=datetime.now(),
            sections=sections,
            summary=summary.strip(),
            key_findings=key_findings,
            recommendations=recommendations,
            metadata={
                "trial_name": trial_name,
                "condition": condition,
                "phase": phase,
            },
        )

    def generate_protocol_analysis_report(
        self,
        protocol_name: str,
        condition: str,
        phase: str,
        eligibility_criteria: str,
        risk_assessment: Dict[str, Any],
        similar_trials: Optional[List[Dict[str, Any]]] = None,
        endpoint_analysis: Optional[Dict[str, Any]] = None,
    ) -> Report:
        """Generate detailed protocol analysis report."""

        sections = []

        # Protocol Overview
        overview_content = f"""
## Protocol Overview

**Protocol Name:** {protocol_name}
**Indication:** {condition}
**Phase:** {phase}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}

### Eligibility Criteria Summary
```
{eligibility_criteria[:500]}{'...' if len(eligibility_criteria) > 500 else ''}
```
"""
        sections.append(ReportSection(
            title="Protocol Overview",
            content=overview_content,
        ))

        # Risk Analysis
        risk_score = risk_assessment.get("overall_risk_score", 50)
        risk_content = f"""
## Risk Analysis

### Overall Assessment
- **Risk Score:** {risk_score}/100
- **Risk Category:** {'High Risk' if risk_score > 70 else 'Medium Risk' if risk_score > 40 else 'Low Risk'}

### Risk Factor Breakdown
"""
        for rf in risk_assessment.get("risk_factors", []):
            risk_content += f"""
#### {rf.get('category', 'Unknown')} - {rf.get('severity', 'unknown').upper()}
- **Description:** {rf.get('description', 'N/A')}
- **Historical Evidence:** {rf.get('historical_evidence', 'N/A')}
- **Recommendation:** {rf.get('recommendation', 'N/A')}
"""

        sections.append(ReportSection(
            title="Risk Analysis",
            content=risk_content,
            data=risk_assessment,
        ))

        # Similar Trials Comparison
        if similar_trials:
            similar_content = """
## Similar Historical Trials

Analysis of completed trials with similar characteristics:

| NCT ID | Status | Enrollment | Duration | Similarity |
|--------|--------|------------|----------|------------|
"""
            for trial in similar_trials[:10]:
                similar_content += f"| {trial.get('nct_id', 'N/A')} | {trial.get('status', 'N/A')} | {trial.get('enrollment', 'N/A')} | {trial.get('duration_months', 'N/A')}mo | {trial.get('similarity_score', 0):.0f}% |\n"

            similar_content += """
### Lessons from Similar Trials
- Trials with similar eligibility criteria achieved average enrollment of X patients
- Y% of similar trials required protocol amendments
- Most common reasons for termination: [reasons]
"""

            sections.append(ReportSection(
                title="Similar Trials",
                content=similar_content,
                data={"similar_trials": similar_trials[:10]},
            ))

        # Endpoint Analysis
        if endpoint_analysis:
            endpoint_content = f"""
## Endpoint Analysis

### Proposed Endpoints
"""
            for ep, score in endpoint_analysis.get("endpoint_scores", {}).items():
                endpoint_content += f"- **{ep}**: Score {score}/100\n"

            endpoint_content += f"""
### Regulatory Alignment
- **Alignment Score:** {endpoint_analysis.get('regulatory_alignment_score', 0):.0f}%
- **Historical Success Rate:** {endpoint_analysis.get('historical_success_rate', 0):.0%}
- **Est. Time to Significance:** {endpoint_analysis.get('estimated_time_to_significance', 0):.0f} months

### Recommendations
"""
            for rec in endpoint_analysis.get("recommendations", []):
                endpoint_content += f"- {rec}\n"

            sections.append(ReportSection(
                title="Endpoint Analysis",
                content=endpoint_content,
                data=endpoint_analysis,
            ))

        key_findings = [
            f"Overall protocol risk score: {risk_score}/100",
            f"{len(risk_assessment.get('risk_factors', []))} risk factors identified",
        ]

        recommendations = risk_assessment.get("recommendations", [])

        return Report(
            id=self._generate_report_id(),
            title=f"Protocol Analysis: {protocol_name}",
            report_type="protocol_analysis",
            created_at=datetime.now(),
            sections=sections,
            summary=f"Comprehensive analysis of {protocol_name} for {condition} {phase} trial.",
            key_findings=key_findings,
            recommendations=recommendations,
            metadata={
                "protocol_name": protocol_name,
                "condition": condition,
                "phase": phase,
            },
        )

    def export_to_markdown(self, report: Report) -> str:
        """Export report to Markdown format."""

        md = f"""# {report.title}

**Report ID:** {report.id}
**Generated:** {report.created_at.strftime('%Y-%m-%d %H:%M')}
**Type:** {report.report_type.replace('_', ' ').title()}

---

## Executive Summary

{report.summary}

### Key Findings
"""
        for i, finding in enumerate(report.key_findings, 1):
            md += f"{i}. {finding}\n"

        md += "\n### Recommendations\n"
        for i, rec in enumerate(report.recommendations, 1):
            md += f"{i}. {rec}\n"

        md += "\n---\n"

        for section in report.sections:
            md += f"\n{section.content}\n"

        md += f"""
---

*Report generated by TrialIntel - Clinical Trial Intelligence Platform*
*© {datetime.now().year} TrialIntel*
"""

        return md

    def export_to_json(self, report: Report) -> str:
        """Export report to JSON format."""

        report_dict = {
            "id": report.id,
            "title": report.title,
            "report_type": report.report_type,
            "created_at": report.created_at.isoformat(),
            "summary": report.summary,
            "key_findings": report.key_findings,
            "recommendations": report.recommendations,
            "metadata": report.metadata,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "data": s.data,
                }
                for s in report.sections
            ],
        }

        return json.dumps(report_dict, indent=2)

    def export_to_html(self, report: Report) -> str:
        """Export report to HTML format."""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #1E3A5F; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        h2 {{ color: #4B5563; margin-top: 30px; }}
        .meta {{ color: #6B7280; font-size: 0.9em; }}
        .summary {{ background: #F3F4F6; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .findings {{ background: #DBEAFE; padding: 15px; border-radius: 8px; }}
        .recommendations {{ background: #D1FAE5; padding: 15px; border-radius: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #D1D5DB; padding: 10px; text-align: left; }}
        th {{ background: #F3F4F6; }}
        pre {{ background: #F9FAFB; padding: 15px; border-radius: 4px; overflow-x: auto; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #E5E7EB; color: #9CA3AF; font-size: 0.8em; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <p class="meta">
        Report ID: {report.id} |
        Generated: {report.created_at.strftime('%Y-%m-%d %H:%M')} |
        Type: {report.report_type.replace('_', ' ').title()}
    </p>

    <div class="summary">
        <h2>Executive Summary</h2>
        <p>{report.summary}</p>
    </div>

    <div class="findings">
        <h3>Key Findings</h3>
        <ul>
"""
        for finding in report.key_findings:
            html += f"            <li>{finding}</li>\n"

        html += """        </ul>
    </div>

    <div class="recommendations">
        <h3>Recommendations</h3>
        <ol>
"""
        for rec in report.recommendations:
            html += f"            <li>{rec}</li>\n"

        html += """        </ol>
    </div>
"""

        for section in report.sections:
            # Convert markdown to basic HTML
            content = section.content
            content = content.replace("## ", "<h2>").replace("\n##", "</h2>\n<h2>")
            content = content.replace("### ", "<h3>").replace("\n###", "</h3>\n<h3>")
            content = content.replace("**", "<strong>").replace("**", "</strong>")
            content = content.replace("```", "<pre>").replace("```", "</pre>")
            content = content.replace("\n- ", "\n<li>").replace("<li>", "</li><li>")
            content = content.replace("\n", "<br>\n")

            html += f"""
    <section>
        {content}
    </section>
"""

        html += f"""
    <div class="footer">
        <p>Report generated by TrialIntel - Clinical Trial Intelligence Platform</p>
        <p>© {datetime.now().year} TrialIntel</p>
    </div>
</body>
</html>
"""

        return html


def get_report_generator(db_manager=None) -> ReportGenerator:
    """Get report generator instance."""
    return ReportGenerator(db_manager)
