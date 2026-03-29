"""
TrialIntel Dashboard

Interactive dashboard for exploring clinical trial intelligence.
This is a proof-of-concept UI that could be white-labeled for Jeeva.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.protocol_risk_scorer import ProtocolRiskScorer
from analysis.endpoint_benchmarking import EndpointBenchmarker


# Page config
st.set_page_config(
    page_title="TrialIntel - Clinical Trial Intelligence",
    page_icon="🔬",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa500; font-weight: bold; }
    .risk-low { color: #00c853; font-weight: bold; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🔬 TrialIntel")
    st.markdown("*Clinical Trial Intelligence powered by ClinicalTrials.gov data*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Tool",
        [
            "Protocol Risk Scorer",
            "Site Intelligence",
            "Endpoint Benchmarking",
            "Competitive Radar",
        ]
    )
    
    if page == "Protocol Risk Scorer":
        protocol_risk_page()
    elif page == "Site Intelligence":
        site_intelligence_page()
    elif page == "Endpoint Benchmarking":
        endpoint_benchmarking_page()
    elif page == "Competitive Radar":
        competitive_radar_page()


def protocol_risk_page():
    """Protocol Risk Scorer page."""
    st.header("📊 Protocol Risk Scorer")
    st.markdown("""
    Analyze your protocol against historical trial data to identify risks 
    before they become problems.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Protocol Details")
        
        condition = st.text_input(
            "Primary Indication",
            value="Type 2 Diabetes",
            help="The primary disease/condition being studied"
        )
        
        phase = st.selectbox(
            "Trial Phase",
            ["PHASE1", "PHASE2", "PHASE3", "PHASE4"],
            index=2
        )
        
        target_enrollment = st.number_input(
            "Target Enrollment",
            min_value=10,
            max_value=50000,
            value=500,
            step=50
        )
        
        planned_sites = st.number_input(
            "Planned Sites",
            min_value=1,
            max_value=500,
            value=50,
            step=5
        )
        
        planned_duration = st.number_input(
            "Planned Duration (months)",
            min_value=3,
            max_value=120,
            value=24,
            step=3
        )
    
    with col2:
        st.subheader("Eligibility Criteria")
        
        eligibility_criteria = st.text_area(
            "Inclusion/Exclusion Criteria",
            value="""Inclusion Criteria:
- Male or female, age 18-65 years
- Diagnosed with type 2 diabetes ≥180 days prior
- HbA1c between 7.5% and 10.0%
- On stable metformin dose ≥1500mg for 90 days
- BMI between 25-40 kg/m²

Exclusion Criteria:
- History of pancreatitis
- eGFR < 60 mL/min/1.73m²
- Prior use of GLP-1 receptor agonists
- Myocardial infarction within 180 days
- NYHA Class III or IV heart failure
- ALT or AST > 3x ULN
- History of bariatric surgery
- Active malignancy
- Pregnant or nursing
- Current use of systemic corticosteroids
- History of diabetic ketoacidosis
- Uncontrolled hypertension (>160/100)""",
            height=300
        )
        
        primary_endpoints = st.text_input(
            "Primary Endpoint(s)",
            value="Change in HbA1c from baseline at week 52",
            help="Comma-separated list of primary endpoints"
        )
    
    if st.button("🔍 Analyze Protocol", type="primary"):
        with st.spinner("Analyzing protocol against historical data..."):
            # Run analysis
            scorer = ProtocolRiskScorer()
            assessment = scorer.score_protocol(
                condition=condition,
                phase=phase,
                eligibility_criteria=eligibility_criteria,
                primary_endpoints=[ep.strip() for ep in primary_endpoints.split(",")],
                target_enrollment=target_enrollment,
                planned_sites=planned_sites,
                planned_duration_months=planned_duration,
            )
            
            # Display results
            st.divider()
            st.subheader("Risk Assessment Results")
            
            # Overall score gauge
            risk_level = "low" if assessment.overall_risk_score < 30 else "medium" if assessment.overall_risk_score < 60 else "high"
            risk_color = "#00c853" if risk_level == "low" else "#ffa500" if risk_level == "medium" else "#ff4b4b"
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Overall Risk Score",
                    f"{assessment.overall_risk_score:.0f}/100",
                    delta=f"{risk_level.upper()} RISK",
                    delta_color="off"
                )
            
            with col2:
                st.metric(
                    "Amendment Probability",
                    f"{assessment.amendment_probability:.0%}",
                )
            
            with col3:
                st.metric(
                    "Enrollment Delay Risk",
                    f"{assessment.enrollment_delay_probability:.0%}",
                )
            
            with col4:
                st.metric(
                    "Termination Risk",
                    f"{assessment.termination_probability:.0%}",
                )
            
            # Risk factors
            st.subheader("Risk Factors Identified")
            
            for rf in assessment.risk_factors:
                severity_icon = "🔴" if rf.severity == "high" else "🟡" if rf.severity == "medium" else "🟢"
                with st.expander(f"{severity_icon} {rf.description}"):
                    st.markdown(f"**Category:** {rf.category.title()}")
                    st.markdown(f"**Historical Evidence:** {rf.historical_evidence}")
                    st.markdown(f"**Recommendation:** {rf.recommendation}")
            
            # Recommendations
            st.subheader("Recommendations")
            for rec in assessment.recommendations:
                st.markdown(f"• {rec}")
            
            # Benchmark trials
            if assessment.benchmark_trials:
                st.subheader("Similar Trials for Reference")
                st.markdown(
                    "These completed trials have similar characteristics: " +
                    ", ".join([f"[{nct}](https://clinicaltrials.gov/study/{nct})" 
                              for nct in assessment.benchmark_trials])
                )


def site_intelligence_page():
    """Site Intelligence page."""
    st.header("🏥 Site Intelligence")
    st.markdown("""
    Find optimal sites for your trial based on historical performance data.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        therapeutic_area = st.selectbox(
            "Therapeutic Area",
            ["diabetes", "oncology", "cardiovascular", "neurology", "immunology", "respiratory"]
        )
        
        target_enrollment = st.number_input(
            "Target Enrollment",
            min_value=10,
            max_value=10000,
            value=500
        )
        
        num_sites = st.slider(
            "Number of Sites to Recommend",
            min_value=5,
            max_value=50,
            value=20
        )
        
        country = st.selectbox(
            "Country Filter",
            [None, "United States", "Germany", "United Kingdom", "France", "Japan"],
            format_func=lambda x: "All Countries" if x is None else x
        )
        
        prioritize_diversity = st.checkbox("Prioritize Diverse Sites", value=True)
        
        analyze_btn = st.button("🔍 Find Sites", type="primary")
    
    with col2:
        if analyze_btn:
            st.info("In production, this would query the TrialIntel database of 100,000+ sites.")
            
            # Mock data for demonstration
            mock_sites = [
                {"facility": "Mayo Clinic", "city": "Rochester", "state": "Minnesota", "country": "United States", "score": 92, "trials": 45, "diversity": 65},
                {"facility": "Cleveland Clinic", "city": "Cleveland", "state": "Ohio", "country": "United States", "score": 88, "trials": 38, "diversity": 70},
                {"facility": "Johns Hopkins", "city": "Baltimore", "state": "Maryland", "country": "United States", "score": 85, "trials": 42, "diversity": 75},
                {"facility": "MD Anderson", "city": "Houston", "state": "Texas", "country": "United States", "score": 82, "trials": 35, "diversity": 85},
                {"facility": "Duke University", "city": "Durham", "state": "North Carolina", "country": "United States", "score": 78, "trials": 28, "diversity": 80},
            ]
            
            df = pd.DataFrame(mock_sites)
            
            st.subheader("Recommended Sites")
            
            fig = px.bar(
                df,
                x="facility",
                y="score",
                color="diversity",
                color_continuous_scale="Viridis",
                title="Site Match Scores",
                labels={"score": "Match Score", "facility": "Facility", "diversity": "Diversity Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                df,
                column_config={
                    "facility": "Facility Name",
                    "city": "City",
                    "state": "State",
                    "country": "Country",
                    "score": st.column_config.ProgressColumn(
                        "Match Score",
                        min_value=0,
                        max_value=100,
                    ),
                    "trials": "Historical Trials",
                    "diversity": st.column_config.ProgressColumn(
                        "Diversity Score",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True,
            )


def endpoint_benchmarking_page():
    """Endpoint Benchmarking page."""
    st.header("🎯 Endpoint Benchmarking")
    st.markdown("""
    Analyze endpoints used in historical trials to make informed endpoint selection.
    """)
    
    condition = st.selectbox(
        "Select Indication",
        ["diabetes", "breast_cancer", "lung_cancer", "alzheimer", "heart_failure", "rheumatoid_arthritis"]
    )
    
    phase_filter = st.multiselect(
        "Phase Filter (optional)",
        ["PHASE1", "PHASE2", "PHASE3", "PHASE4"],
        default=["PHASE3"]
    )
    
    if st.button("🔍 Analyze Endpoints", type="primary"):
        st.info("In production, this would analyze real endpoint data from ClinicalTrials.gov.")
        
        # Mock endpoint data
        if condition == "diabetes":
            primary_endpoints = [
                {"endpoint": "HbA1c Change", "frequency": 523, "success_rate": 0.72, "typical_timeframe": "24-52 weeks"},
                {"endpoint": "Fasting Glucose", "frequency": 234, "success_rate": 0.68, "typical_timeframe": "12-24 weeks"},
                {"endpoint": "Body Weight Change", "frequency": 189, "success_rate": 0.65, "typical_timeframe": "26-52 weeks"},
                {"endpoint": "MACE Events", "frequency": 45, "success_rate": 0.55, "typical_timeframe": "2-4 years"},
            ]
            regulatory_insights = [
                "HbA1c is the gold standard primary endpoint for diabetes trials",
                "FDA requires cardiovascular safety outcomes for diabetes drugs",
                "Body weight change increasingly important as secondary endpoint",
            ]
        elif condition == "breast_cancer":
            primary_endpoints = [
                {"endpoint": "Progression-Free Survival", "frequency": 412, "success_rate": 0.58, "typical_timeframe": "12-36 months"},
                {"endpoint": "Overall Survival", "frequency": 287, "success_rate": 0.45, "typical_timeframe": "24-60 months"},
                {"endpoint": "Pathologic Complete Response", "frequency": 156, "success_rate": 0.62, "typical_timeframe": "At surgery"},
                {"endpoint": "Objective Response Rate", "frequency": 134, "success_rate": 0.55, "typical_timeframe": "12-16 weeks"},
            ]
            regulatory_insights = [
                "pCR accepted for accelerated approval in neoadjuvant setting",
                "Overall survival remains preferred for full approval",
                "PFS acceptable in metastatic setting with adequate effect size",
            ]
        else:
            primary_endpoints = [
                {"endpoint": "Primary Efficacy", "frequency": 200, "success_rate": 0.60, "typical_timeframe": "12-24 weeks"},
            ]
            regulatory_insights = [
                "Contact TrialIntel for detailed analysis of this indication",
            ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Primary Endpoints by Usage")
            
            df = pd.DataFrame(primary_endpoints)
            
            fig = px.bar(
                df,
                x="endpoint",
                y="frequency",
                color="success_rate",
                color_continuous_scale="RdYlGn",
                title="Endpoint Usage in Historical Trials",
                labels={"frequency": "# Trials", "endpoint": "Endpoint", "success_rate": "Success Rate"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Endpoint Details")
            st.dataframe(
                df,
                column_config={
                    "endpoint": "Endpoint",
                    "frequency": "# Trials",
                    "success_rate": st.column_config.ProgressColumn(
                        "Success Rate",
                        min_value=0,
                        max_value=1,
                    ),
                    "typical_timeframe": "Typical Timeframe",
                },
                hide_index=True,
            )
        
        st.subheader("Regulatory Insights")
        for insight in regulatory_insights:
            st.markdown(f"• {insight}")


def competitive_radar_page():
    """Competitive Radar page."""
    st.header("📡 Competitive Radar")
    st.markdown("""
    Monitor competitor trials in your therapeutic area.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        indication = st.text_input(
            "Indication",
            value="Type 2 Diabetes"
        )
        
        phase = st.multiselect(
            "Phase Filter",
            ["PHASE1", "PHASE2", "PHASE3", "PHASE4"],
            default=["PHASE2", "PHASE3"]
        )
        
        status = st.multiselect(
            "Status Filter",
            ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"],
            default=["RECRUITING", "NOT_YET_RECRUITING"]
        )
        
        competitors = st.text_input(
            "Competitors to Watch (comma-separated)",
            value="Novo Nordisk, Eli Lilly, Sanofi"
        )
        
        analyze_btn = st.button("🔍 Scan Competitive Landscape", type="primary")
    
    with col2:
        if analyze_btn:
            st.info("In production, this would query live ClinicalTrials.gov data.")
            
            # Mock competitive data
            mock_trials = [
                {
                    "nct_id": "NCT05012345",
                    "sponsor": "Novo Nordisk",
                    "title": "Semaglutide Phase 3 Study",
                    "phase": "PHASE3",
                    "status": "RECRUITING",
                    "enrollment": 3500,
                    "start_date": "2024-01",
                    "endpoints": ["HbA1c", "Body Weight"]
                },
                {
                    "nct_id": "NCT05023456",
                    "sponsor": "Eli Lilly",
                    "title": "Tirzepatide Extension Study",
                    "phase": "PHASE3",
                    "status": "RECRUITING",
                    "enrollment": 2800,
                    "start_date": "2024-03",
                    "endpoints": ["HbA1c", "MACE"]
                },
                {
                    "nct_id": "NCT05034567",
                    "sponsor": "Sanofi",
                    "title": "Novel GLP-1 Agonist Phase 2",
                    "phase": "PHASE2",
                    "status": "NOT_YET_RECRUITING",
                    "enrollment": 600,
                    "start_date": "2024-06",
                    "endpoints": ["HbA1c", "Safety"]
                },
            ]
            
            st.subheader("Competitor Trial Activity")
            
            df = pd.DataFrame(mock_trials)
            
            # Timeline visualization
            fig = px.timeline(
                df,
                x_start="start_date",
                x_end="start_date",  # Would be end_date in real data
                y="sponsor",
                color="phase",
                title="Competitor Trial Timeline",
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trial details
            for trial in mock_trials:
                with st.expander(f"📋 {trial['nct_id']} - {trial['sponsor']}"):
                    st.markdown(f"**Title:** {trial['title']}")
                    st.markdown(f"**Phase:** {trial['phase']}")
                    st.markdown(f"**Status:** {trial['status']}")
                    st.markdown(f"**Enrollment:** {trial['enrollment']}")
                    st.markdown(f"**Endpoints:** {', '.join(trial['endpoints'])}")
                    st.markdown(f"[View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{trial['nct_id']})")


if __name__ == "__main__":
    main()
