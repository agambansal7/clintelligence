"""
Clintelligence - Clinical Trial Protocol Intelligence

By Jeeva Clinical Trials
AI-powered protocol analysis platform.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Clintelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clintelligence Theme - Modern Single Page
st.markdown("""
<style>
    /* ===== FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    /* ===== ROOT VARIABLES ===== */
    :root {
        --primary: #F45900;
        --primary-light: #FF6B1A;
        --primary-dark: #D94E00;
        --secondary: #314DD8;
        --secondary-light: #4A63E8;
        --navy: #0F172A;
        --text-dark: #1E293B;
        --text-medium: #475569;
        --text-light: #64748B;
        --bg-white: #FFFFFF;
        --bg-light: #F8FAFC;
        --bg-blue-light: #EEF2FF;
        --bg-orange-light: #FFF7ED;
        --border: #E2E8F0;
        --success: #10B981;
        --warning: #F59E0B;
        --error: #EF4444;
    }

    /* ===== GLOBAL STYLES ===== */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
        font-size: 16px;
    }

    .block-container {
        padding: 1.5rem 2.5rem !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    /* Hide Streamlit branding and sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}

    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--navy) !important;
    }

    h1 {
        font-weight: 800 !important;
        font-size: 2.75rem !important;
        letter-spacing: -0.03em;
        line-height: 1.2 !important;
    }

    h2 {
        font-weight: 700 !important;
        font-size: 1.75rem !important;
        letter-spacing: -0.02em;
    }

    h3 {
        font-weight: 600 !important;
        font-size: 1.35rem !important;
    }

    p, li, span, label, div {
        color: var(--text-medium);
        line-height: 1.7;
        font-size: 1.05rem;
    }

    /* Larger text for labels */
    .stTextArea label, .stTextInput label, .stSelectbox label, .stSlider label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* ===== TEXT INPUTS - FIXED VISIBILITY ===== */
    .stTextArea textarea {
        background: var(--bg-white) !important;
        border: 2px solid var(--border) !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        padding: 1.25rem 1.5rem !important;
        color: var(--text-dark) !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.7 !important;
        transition: all 0.2s ease;
    }

    .stTextArea textarea::placeholder {
        color: #94A3B8 !important;
        opacity: 1 !important;
        font-size: 1.05rem !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(244, 89, 0, 0.1) !important;
        outline: none !important;
    }

    .stTextInput input {
        background: var(--bg-white) !important;
        border: 2px solid var(--border) !important;
        border-radius: 10px !important;
        font-size: 1.1rem !important;
        padding: 0.875rem 1.25rem !important;
        color: var(--text-dark) !important;
    }

    .stTextInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(244, 89, 0, 0.1) !important;
    }

    /* ===== HIDE SIDEBAR ELEMENTS ===== */
    [data-testid="stSidebar"] .stRadio > div > label > div:first-child {
        display: none;
    }

    /* ===== STAT CARDS ===== */
    .stat-card {
        background: var(--bg-white);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .stat-card:hover {
        border-color: var(--primary);
        box-shadow: 0 8px 25px rgba(244, 89, 0, 0.12);
        transform: translateY(-3px);
    }

    .stat-value {
        font-family: 'Poppins', sans-serif;
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--navy);
        line-height: 1;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-light);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ===== METRICS ===== */
    [data-testid="metric-container"] {
        background: var(--bg-white);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        transition: all 0.3s ease;
    }

    [data-testid="metric-container"]:hover {
        border-color: var(--secondary);
        box-shadow: 0 4px 15px rgba(49, 77, 216, 0.1);
    }

    [data-testid="metric-container"] label {
        font-weight: 600 !important;
        color: var(--text-light) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        color: var(--navy) !important;
        font-size: 1.75rem !important;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px;
        padding: 1rem 2.5rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(244, 89, 0, 0.3);
        text-transform: none;
        letter-spacing: -0.01em;
    }

    .stButton > button:hover {
        background: var(--primary-dark) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(244, 89, 0, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    .stButton > button[kind="secondary"] {
        background: var(--bg-white) !important;
        color: var(--secondary) !important;
        border: 2px solid var(--secondary) !important;
        box-shadow: none;
    }

    .stButton > button[kind="secondary"]:hover {
        background: var(--bg-blue-light) !important;
        box-shadow: 0 4px 15px rgba(49, 77, 216, 0.15);
    }

    /* ===== SELECTBOX ===== */
    .stSelectbox > div > div {
        border: 2px solid var(--border) !important;
        border-radius: 10px !important;
        background: var(--bg-white) !important;
        font-size: 1.1rem !important;
    }

    .stSelectbox > div > div:hover {
        border-color: var(--secondary) !important;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-light);
        border-radius: 14px;
        padding: 8px;
        border: none;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 1rem 1.75rem;
        font-weight: 600;
        font-size: 1.1rem !important;
        color: var(--text-medium);
        background: transparent;
        transition: all 0.2s ease;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary);
        background: var(--bg-orange-light);
    }

    .stTabs [aria-selected="true"] {
        background: var(--bg-white) !important;
        color: var(--primary) !important;
        font-weight: 700;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background: var(--bg-light);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        color: var(--text-dark);
        padding: 1rem 1.25rem !important;
        transition: all 0.2s ease;
    }

    .streamlit-expanderHeader:hover {
        background: var(--bg-blue-light);
        color: var(--secondary);
    }

    /* ===== ALERTS ===== */
    .stAlert {
        border-radius: 10px;
        border: none;
        padding: 1rem 1.25rem;
    }

    .stSuccess, [data-baseweb="notification"][kind="positive"] {
        background: #ECFDF5 !important;
        border-left: 4px solid var(--success);
    }

    .stWarning, [data-baseweb="notification"][kind="warning"] {
        background: #FFFBEB !important;
        border-left: 4px solid var(--warning);
    }

    .stError, [data-baseweb="notification"][kind="negative"] {
        background: #FEF2F2 !important;
        border-left: 4px solid var(--error);
    }

    .stInfo, [data-baseweb="notification"][kind="info"] {
        background: var(--bg-blue-light) !important;
        border-left: 4px solid var(--secondary);
    }

    /* ===== FEATURE CARDS ===== */
    .feature-card {
        background: var(--bg-white);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        border-color: var(--primary);
        box-shadow: 0 8px 30px rgba(244, 89, 0, 0.1);
        transform: translateY(-2px);
    }

    /* ===== HERO SECTION ===== */
    .hero-section {
        padding: 0 0 1.5rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid var(--bg-light);
    }

    /* ===== DIVIDERS ===== */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.5rem 0;
    }

    /* ===== SLIDERS ===== */
    .stSlider > div > div > div > div {
        background: var(--primary) !important;
    }

    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        background: var(--primary) !important;
    }

    /* ===== NUMBER INPUT ===== */
    .stNumberInput > div > div > input {
        border: 2px solid var(--border) !important;
        border-radius: 10px !important;
    }

    .stNumberInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(244, 89, 0, 0.1) !important;
    }

    /* ===== DATAFRAMES ===== */
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    /* ===== SPINNER ===== */
    .stSpinner > div {
        border-top-color: var(--primary) !important;
    }

    /* ===== PROGRESS ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%) !important;
    }

    /* ===== PLOTLY ===== */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-light);
    }

    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-light);
    }

    /* ===== ANIMATIONS ===== */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .element-container {
        animation: fadeInUp 0.4s ease-out;
    }

    /* ===== SCORE INDICATORS ===== */
    .score-excellent { background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); color: #065F46; }
    .score-good { background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); color: #92400E; }
    .score-poor { background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); color: #991B1B; }
</style>
""", unsafe_allow_html=True)

# Initialize session state for protocol data
if "protocol_analyzed" not in st.session_state:
    st.session_state.protocol_analyzed = False
if "extracted_protocol" not in st.session_state:
    st.session_state.extracted_protocol = None
if "similar_trials" not in st.session_state:
    st.session_state.similar_trials = []
if "protocol_metrics" not in st.session_state:
    st.session_state.protocol_metrics = {}
if "risk_assessment" not in st.session_state:
    st.session_state.risk_assessment = {}
if "matching_context" not in st.session_state:
    st.session_state.matching_context = None
if "optimization_report" not in st.session_state:
    st.session_state.optimization_report = None
if "enhanced_analysis" not in st.session_state:
    st.session_state.enhanced_analysis = None
if "enrollment_forecast" not in st.session_state:
    st.session_state.enrollment_forecast = None
if "risk_assessment_detailed" not in st.session_state:
    st.session_state.risk_assessment_detailed = None
if "site_intelligence" not in st.session_state:
    st.session_state.site_intelligence = None
if "eligibility_optimization" not in st.session_state:
    st.session_state.eligibility_optimization = None
if "endpoint_benchmark" not in st.session_state:
    st.session_state.endpoint_benchmark = None


# ============== DATABASE CONNECTION ==============
@st.cache_resource
def get_database():
    """Get database connection."""
    try:
        from src.database import DatabaseManager
        return DatabaseManager.get_instance()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


@st.cache_data(ttl=300)
def get_db_stats():
    """Get database statistics."""
    db = get_database()
    if db:
        return db.get_stats()
    return None


# ============== PROTOCOL DISPLAY HELPER ==============
def show_protocol_summary():
    """Show current protocol summary in sidebar or main area."""
    if st.session_state.protocol_analyzed and st.session_state.extracted_protocol:
        p = st.session_state.extracted_protocol
        st.success("**Protocol Loaded**")
        st.write(f"**Condition:** {p.condition}")
        st.write(f"**Phase:** {p.phase}")
        st.write(f"**Enrollment:** {p.target_enrollment:,}")
        if p.primary_endpoint:
            st.write(f"**Endpoint:** {p.primary_endpoint[:50]}...")
        return True
    else:
        st.warning("**No Protocol Loaded**")
        st.caption("Go to 'Enter Protocol' to analyze your trial protocol first.")
        return False


def require_protocol():
    """Check if protocol is loaded, show warning if not."""
    if not st.session_state.protocol_analyzed:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FFF4ED, #FFEDD5); border: 2px solid #F45900;
                    border-radius: 16px; padding: 2.5rem; text-align: center; margin: 2rem 0;">
            <div style="width: 56px; height: 56px; background: linear-gradient(135deg, #F45900, #FF6B1A); border-radius: 50%;
                        display: flex; align-items: center; justify-content: center; margin: 0 auto 1.25rem;
                        box-shadow: 0 4px 15px rgba(244, 89, 0, 0.3);">
                <span style="color: white; font-size: 1.75rem;">📝</span>
            </div>
            <h3 style="font-family: 'Poppins', sans-serif; font-size: 1.35rem; font-weight: 700; color: #182A81; margin-bottom: 0.75rem;">
                Protocol Required
            </h3>
            <p style="color: #4A4A4A; font-size: 1rem; max-width: 450px; margin: 0 auto; line-height: 1.6;">
                Please enter and analyze your protocol first using <strong style="color: #F45900;">"Enter Protocol"</strong> in the sidebar.
                All features use your protocol information to provide tailored intelligence.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return False
    return True


# ============== SIDEBAR ==============
def render_sidebar():
    """Render sidebar with Jeeva-themed navigation."""
    with st.sidebar:
        # Logo and brand - Jeeva style
        st.markdown("""
        <div style="padding: 0.5rem 0 1.5rem 0;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 42px; height: 42px; background: linear-gradient(135deg, #F45900, #FF6B1A);
                            border-radius: 10px; display: flex; align-items: center; justify-content: center;
                            box-shadow: 0 4px 12px rgba(244, 89, 0, 0.3);">
                    <span style="color: white; font-size: 1.25rem; font-weight: bold;">T</span>
                </div>
                <div>
                    <div style="font-family: 'Poppins', sans-serif; font-size: 1.35rem; font-weight: 700; color: #182A81; letter-spacing: -0.02em;">TrialIntel</div>
                    <div style="font-size: 0.7rem; color: #6B7280; font-weight: 500;">by Jeeva Clinical Trials</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Protocol status indicator
        if st.session_state.protocol_analyzed and st.session_state.extracted_protocol:
            p = st.session_state.extracted_protocol
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FFF4ED, #FFEDD5); border: 2px solid #F45900;
                        border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <div style="width: 10px; height: 10px; background: #F45900; border-radius: 50%;
                                box-shadow: 0 0 0 3px rgba(244, 89, 0, 0.2);"></div>
                    <span style="font-size: 0.75rem; font-weight: 700; color: #F45900; text-transform: uppercase; letter-spacing: 0.05em;">Protocol Active</span>
                </div>
                <div style="font-family: 'Poppins', sans-serif; font-size: 0.95rem; font-weight: 600; color: #182A81; margin-bottom: 0.25rem;">{p.condition[:28]}{'...' if len(p.condition) > 28 else ''}</div>
                <div style="font-size: 0.85rem; color: #4A4A4A;">{p.phase} · {p.target_enrollment:,} patients</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Clear Protocol", use_container_width=True, type="secondary"):
                st.session_state.protocol_analyzed = False
                st.session_state.extracted_protocol = None
                st.session_state.similar_trials = []
                st.session_state.protocol_metrics = {}
                st.session_state.risk_assessment = {}
                st.session_state.matching_context = None
                st.session_state.optimization_report = None
                st.session_state.enhanced_analysis = None
                st.session_state.enrollment_forecast = None
                st.session_state.risk_assessment_detailed = None
                st.session_state.site_intelligence = None
                st.session_state.eligibility_optimization = None
                st.session_state.endpoint_benchmark = None
                st.rerun()

            st.divider()

        # Navigation label
        st.markdown('<p style="font-size: 0.7rem; font-weight: 700; color: #6B7280; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;">Features</p>', unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            [
                "📝 Enter Protocol",
                "⭐ Protocol Optimization",
                "📊 Risk Analysis",
                "🌍 Site Intelligence",
                "📈 Enrollment Forecast",
                "🔍 Similar Trials",
                "✅ Eligibility",
                "🎯 Endpoints",
            ],
            label_visibility="collapsed"
        )

        # Footer
        st.markdown("""
        <div style="position: absolute; bottom: 1.5rem; left: 1rem; right: 1rem;">
            <div style="padding-top: 1rem; border-top: 1px solid #E5E5E5;">
                <div style="font-size: 0.7rem; color: #6B7280; margin-bottom: 0.25rem;">Powered by</div>
                <div style="font-size: 0.85rem; color: #314DD8; font-weight: 600;">Claude AI · 566K trials</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        return page


# ============== PROTOCOL ENTRY (Main Entry Point) ==============
def render_protocol_entry():
    """Main protocol entry - extracts all information for other features."""
    # Hero section - Jeeva style
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 2.25rem; font-weight: 700; margin-bottom: 0.75rem; color: #182A81;">
            Enter Your Trial Protocol
        </h1>
        <p style="font-size: 1.05rem; color: #4A4A4A; max-width: 650px; line-height: 1.7;">
            Paste your protocol synopsis and our AI will extract all key information
            to power intelligent analysis across all features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Protocol input section
    st.markdown("""
    <div style="margin-bottom: 0.5rem;">
        <span style="font-size: 0.85rem; font-weight: 600; color: #314DD8; text-transform: uppercase;
                   letter-spacing: 0.05em;">Protocol Synopsis</span>
    </div>
    """, unsafe_allow_html=True)

    protocol_text = st.text_area(
        "Protocol Details",
        placeholder="""Paste your complete protocol synopsis here. Include:

• Study Title and Condition/Indication
• Phase (1, 2, 3, or 4)
• Target enrollment number
• Primary endpoint(s)
• Key inclusion/exclusion criteria
• Study duration
• Intervention details (drug, device, procedure)
• Comparator (if any)
• Number of planned sites

The more detail you provide, the better the analysis.""",
        height=280,
        label_visibility="collapsed"
    )

    # Settings section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #F5F5F5, #EAECFB); border-radius: 12px; padding: 1.25rem; margin: 1.5rem 0;">
        <h4 style="font-size: 0.8rem; font-weight: 700; color: #314DD8; text-transform: uppercase;
                   letter-spacing: 0.08em; margin-bottom: 0.5rem;">Search Settings</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        max_similar = st.slider("Max candidates", 200, 1000, 500, help="Maximum trials to search")
    with col2:
        min_similarity = st.slider("Min similarity %", 20, 60, 40, help="Minimum match threshold")
    with col3:
        rank_top_n = st.slider("Top N for AI", 50, 200, 100, help="Trials for AI to rank")

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    if st.button("🚀 Analyze Protocol", type="primary", use_container_width=True):
        if not protocol_text or len(protocol_text.strip()) < 100:
            st.error("Please enter more protocol details (at least 100 characters)")
            return

        if not os.getenv("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY not configured in .env file")
            return

        db = get_database()
        if not db:
            return

        try:
            with st.spinner("🤖 Claude is analyzing your protocol... This may take a moment."):
                from src.analysis.protocol_analyzer import ProtocolAnalyzer

                analyzer = ProtocolAnalyzer()
                results = analyzer.analyze_and_match(
                    protocol_text=protocol_text,
                    db_manager=db,
                    include_site_recommendations=True,
                    min_similarity=min_similarity,
                    max_candidates=max_similar,
                    semantic_rank_top_n=rank_top_n
                )

            # Store in session state for all features
            st.session_state.extracted_protocol = results["extracted_protocol"]
            st.session_state.protocol_metrics = results["metrics"]
            st.session_state.risk_assessment = results["risk_assessment"]
            st.session_state.similar_trials = results.get("similar_trials", [])
            st.session_state.site_recommendations = results.get("site_recommendations", [])
            st.session_state.matching_context = results.get("matching_context")
            st.session_state.protocol_analyzed = True
            st.session_state.protocol_text = protocol_text
            st.session_state.optimization_report = None  # Reset optimization report

            # Success state
            p = st.session_state.extracted_protocol

            st.markdown("""
            <div style="background: linear-gradient(135deg, #FFF4ED, #FFEDD5); border: 2px solid #F45900;
                        border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <div style="width: 28px; height: 28px; background: #F45900; border-radius: 50%;
                                display: flex; align-items: center; justify-content: center;
                                box-shadow: 0 2px 8px rgba(244, 89, 0, 0.3);">
                        <span style="color: white; font-size: 16px;">✓</span>
                    </div>
                    <span style="font-family: 'Poppins', sans-serif; font-size: 1.15rem; font-weight: 600; color: #182A81;">Protocol Analyzed Successfully</span>
                </div>
                <p style="color: #4A4A4A; margin: 0; font-size: 0.95rem;">All features are now ready. Select any option from the sidebar to continue.</p>
            </div>
            """, unsafe_allow_html=True)

            # Extracted info cards
            st.markdown("""
            <h3 style="font-family: 'Poppins', sans-serif; font-size: 1.1rem; font-weight: 600; color: #182A81; margin: 2rem 0 1rem 0;">Extracted Information</h3>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Condition</div>
                    <div class="stat-value" style="font-size: 1.25rem;">{p.condition[:25]}{'...' if len(p.condition) > 25 else ''}</div>
                    <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e5e7eb;">
                        <div style="font-size: 0.8rem; color: #6b7280;">Phase</div>
                        <div style="font-weight: 600; color: #374151;">{p.phase}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Target Enrollment</div>
                    <div class="stat-value">{p.target_enrollment:,}</div>
                    <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e5e7eb;">
                        <div style="font-size: 0.8rem; color: #6b7280;">Study Type</div>
                        <div style="font-weight: 600; color: #374151;">{p.study_type}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Intervention</div>
                    <div class="stat-value" style="font-size: 1.25rem;">{p.intervention_type}</div>
                    <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e5e7eb;">
                        <div style="font-size: 0.8rem; color: #6b7280;">Comparator</div>
                        <div style="font-weight: 600; color: #374151;">{p.comparator or 'None'}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Primary endpoint card
            if p.primary_endpoint:
                st.markdown(f"""
                <div class="feature-card" style="margin-top: 1rem;">
                    <div style="font-size: 0.75rem; font-weight: 600; color: #6b7280; text-transform: uppercase;
                               letter-spacing: 0.05em; margin-bottom: 0.5rem;">Primary Endpoint</div>
                    <div style="font-size: 0.95rem; color: #374151;">{p.primary_endpoint}</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# ============== PROTOCOL OPTIMIZATION (STAR FEATURE) ==============
def render_protocol_optimization():
    """Star Feature: Comprehensive AI-powered protocol optimization."""
    # Hero with star badge - Jeeva style
    st.markdown("""
    <div class="hero-section">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
            <div style="background: linear-gradient(135deg, #F45900, #FF6B1A); padding: 0.35rem 1rem;
                        border-radius: 20px; font-size: 0.7rem; font-weight: 700; color: white;
                        text-transform: uppercase; letter-spacing: 0.08em; box-shadow: 0 2px 8px rgba(244,89,0,0.3);">⭐ Featured</div>
        </div>
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 2.25rem; font-weight: 700; margin-bottom: 0.75rem; color: #182A81;">
            Protocol Optimization
        </h1>
        <p style="font-size: 1.05rem; color: #4A4A4A; max-width: 650px; line-height: 1.7;">
            AI-powered comprehensive analysis including design evaluation, regulatory checks,
            power assessment, and actionable recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not require_protocol():
        return

    p = st.session_state.extracted_protocol
    metrics = st.session_state.protocol_metrics
    similar_trials = st.session_state.similar_trials

    # Protocol context badge - Jeeva style
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #EAECFB, #F5F5F5); border-radius: 10px; padding: 0.875rem 1.25rem; margin-bottom: 1.5rem;
                display: inline-flex; align-items: center; gap: 0.75rem; border: 1px solid #E5E5E5;">
        <span style="font-size: 0.85rem; color: #6B7280; font-weight: 500;">Analyzing:</span>
        <span style="font-weight: 700; color: #182A81;">{p.condition}</span>
        <span style="color: #CBD5E1; font-weight: 300;">|</span>
        <span style="color: #F45900; font-weight: 600;">{p.phase}</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        analyze_btn = st.button("🚀 Generate Optimization Report", type="primary", use_container_width=True)
    with col2:
        if st.session_state.optimization_report:
            if st.button("↻ Refresh", use_container_width=True, type="secondary"):
                st.session_state.optimization_report = None
                st.rerun()

    if analyze_btn or st.session_state.optimization_report:
        if analyze_btn:
            with st.spinner("🤖 Analyzing protocol design, benchmarks, and generating AI recommendations..."):
                try:
                    from src.analysis.enhanced_protocol_optimizer import EnhancedProtocolOptimizer

                    db = get_database()
                    optimizer = EnhancedProtocolOptimizer(db=db)

                    matching_context = st.session_state.get("matching_context")

                    report = optimizer.optimize(
                        extracted_protocol=p,
                        similar_trials=similar_trials,
                        metrics=metrics,
                        matching_context=matching_context
                    )

                    st.session_state.optimization_report = report

                except Exception as e:
                    st.error(f"Failed to generate optimization report: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return

        report = st.session_state.optimization_report

        if not report:
            st.warning("Unable to generate report. Please try again.")
            return

        # Display enhanced report
        _render_enhanced_optimization_report(report)


def _render_enhanced_optimization_report(report):
    """Render enhanced optimization report with multiple sections."""
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Score card with visual indicator
    score = report.overall_score
    score_color = "#10b981" if score >= 75 else "#f59e0b" if score >= 55 else "#ef4444"
    readiness_text = report.readiness_level.replace("_", " ").title()

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #ffffff, #f8fafc); border: 1px solid #e2e8f0;
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <div style="position: relative; width: 80px; height: 80px;">
                    <svg viewBox="0 0 36 36" style="width: 80px; height: 80px; transform: rotate(-90deg);">
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                              fill="none" stroke="#e5e7eb" stroke-width="3"/>
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                              fill="none" stroke="{score_color}" stroke-width="3"
                              stroke-dasharray="{score}, 100"/>
                    </svg>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                                font-size: 1.25rem; font-weight: 700; color: {score_color};">{score}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;
                               letter-spacing: 0.05em;">Protocol Score</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #1e293b; margin-top: 0.25rem;">{readiness_text}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"""
    <div class="stat-card" style="text-align: center;">
        <div class="stat-value">{report.total_recommendations}</div>
        <div class="stat-label">Recommendations</div>
    </div>
    """, unsafe_allow_html=True)
    col2.markdown(f"""
    <div class="stat-card" style="text-align: center;">
        <div class="stat-value" style="color: #ef4444;">{report.high_priority_count}</div>
        <div class="stat-label">High Priority</div>
    </div>
    """, unsafe_allow_html=True)
    col3.markdown("", unsafe_allow_html=True)
    col4.markdown("", unsafe_allow_html=True)

    # Summary
    st.markdown(f"""
    <div class="feature-card" style="margin: 1rem 0;">
        <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;
                   letter-spacing: 0.05em; margin-bottom: 0.5rem;">Summary</div>
        <p style="color: #374151; margin: 0; line-height: 1.6;">{report.summary}</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Strengths and Gaps
    col1, col2 = st.columns(2)
    with col1:
        if report.key_strengths:
            st.markdown("""
            <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 12px; padding: 1rem;">
                <div style="font-size: 0.75rem; font-weight: 600; color: #059669; text-transform: uppercase;
                           letter-spacing: 0.05em; margin-bottom: 0.75rem;">Key Strengths</div>
            """, unsafe_allow_html=True)
            for strength in report.key_strengths:
                st.markdown(f"""<div style="color: #065f46; font-size: 0.9rem; margin-bottom: 0.5rem;">• {strength}</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        if report.critical_gaps:
            st.markdown("""
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 1rem;">
                <div style="font-size: 0.75rem; font-weight: 600; color: #dc2626; text-transform: uppercase;
                           letter-spacing: 0.05em; margin-bottom: 0.75rem;">Critical Gaps</div>
            """, unsafe_allow_html=True)
            for gap in report.critical_gaps:
                st.markdown(f"""<div style="color: #991b1b; font-size: 0.9rem; margin-bottom: 0.5rem;">• {gap}</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Estimated Improvements
    if report.estimated_improvement:
        st.markdown("""
        <h3 style="font-size: 1rem; font-weight: 600; color: #374151; margin: 1.5rem 0 1rem 0;">Estimated Improvements</h3>
        """, unsafe_allow_html=True)
        cols = st.columns(len(report.estimated_improvement))
        for i, (key, value) in enumerate(report.estimated_improvement.items()):
            cols[i].markdown(f"""
            <div class="stat-card" style="text-align: center;">
                <div class="stat-value" style="color: #3b82f6;">{value}</div>
                <div class="stat-label">{key.replace("_", " ").title()}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Recommendations",
        "Design Analysis",
        "Regulatory",
        "Statistics",
        "Timeline",
        "Competition"
    ])

    with tab1:
        _render_recommendations_tab(report)

    with tab2:
        _render_design_analysis_tab(report)

    with tab3:
        _render_regulatory_tab(report)

    with tab4:
        _render_statistics_tab(report)

    with tab5:
        _render_timeline_tab(report)

    with tab6:
        _render_competition_tab(report)


def _render_recommendations_tab(report):
    """Render recommendations tab."""
    st.subheader("Optimization Recommendations")

    if not report.recommendations:
        st.success("No major recommendations - your protocol is well-optimized!")
        return

    # High Priority
    high_priority = [r for r in report.recommendations if r.priority == "high"]
    if high_priority:
        st.markdown("### 🔴 High Priority")
        for rec in high_priority:
            with st.expander(f"**{rec.title}** ({rec.category.upper()})", expanded=True):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Current State:**")
                    st.write(rec.current_state)
                with col2:
                    st.markdown("**Expected Impact:**")
                    st.success(rec.expected_impact)

                st.markdown("**Recommendation:**")
                st.info(rec.recommendation)

                if rec.evidence:
                    st.markdown("**Evidence:**")
                    for ev in rec.evidence:
                        st.caption(f"• {ev}")

                st.caption(f"Complexity: {rec.implementation_complexity.title()} | Confidence: {rec.confidence*100:.0f}%")

    # Medium Priority
    medium_priority = [r for r in report.recommendations if r.priority == "medium"]
    if medium_priority:
        st.markdown("### 🟡 Medium Priority")
        for rec in medium_priority:
            with st.expander(f"**{rec.title}** ({rec.category.upper()})"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Current State:**")
                    st.write(rec.current_state)
                with col2:
                    st.markdown("**Expected Impact:**")
                    st.warning(rec.expected_impact)

                st.markdown("**Recommendation:**")
                st.info(rec.recommendation)

    # Low Priority
    low_priority = [r for r in report.recommendations if r.priority == "low"]
    if low_priority:
        st.markdown("### 🟢 Low Priority")
        for rec in low_priority:
            with st.expander(f"**{rec.title}** ({rec.category.upper()})"):
                st.write(f"**Current:** {rec.current_state}")
                st.write(f"**Recommendation:** {rec.recommendation}")


def _render_design_analysis_tab(report):
    """Render design analysis tab."""
    st.subheader("Design Element Analysis")

    if not report.design_elements:
        st.info("No design elements analyzed")
        return

    # Score by category chart
    if report.design_score_by_category:
        st.markdown("### Category Scores")
        categories = list(report.design_score_by_category.keys())
        scores = list(report.design_score_by_category.values())

        fig = go.Figure(go.Bar(
            x=scores,
            y=categories,
            orientation='h',
            marker_color=['#4caf50' if s >= 75 else '#ff9800' if s >= 55 else '#f44336' for s in scores]
        ))
        fig.update_layout(
            title="Design Scores by Category",
            xaxis_title="Score (0-100)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    # Element details
    st.markdown("### Element Details")

    element_data = []
    for element in report.design_elements:
        assessment_emoji = {
            "optimal": "✅",
            "acceptable": "🟡",
            "needs_improvement": "⚠️",
            "concerning": "🔴"
        }.get(element.assessment, "⚪")

        element_data.append({
            "Element": element.name,
            "Category": element.category.title(),
            "Current": element.current_value[:40] + "..." if len(element.current_value) > 40 else element.current_value,
            "Benchmark": element.benchmark_value[:40] + "..." if len(element.benchmark_value) > 40 else element.benchmark_value,
            "Score": f"{element.score:.0f}",
            "Status": assessment_emoji
        })

    st.dataframe(pd.DataFrame(element_data), use_container_width=True, hide_index=True)

    # Phase guidance
    if report.phase_guidance:
        st.markdown("### Phase-Specific Guidance")
        guidance = report.phase_guidance

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Typical Enrollment:** {guidance.typical_enrollment[0]:,} - {guidance.typical_enrollment[1]:,}")
            st.markdown(f"**Typical Duration:** {guidance.typical_duration_months[0]} - {guidance.typical_duration_months[1]} months")
            st.markdown(f"**Success Rate Benchmark:** {guidance.success_rate_benchmark*100:.0f}%")

        with col2:
            st.markdown("**Key Focus Areas:**")
            for area in guidance.key_focus_areas:
                st.markdown(f"- {area}")

        st.markdown("**Common Pitfalls to Avoid:**")
        for pitfall in guidance.common_pitfalls:
            st.warning(pitfall)


def _render_regulatory_tab(report):
    """Render regulatory alignment tab."""
    st.subheader("Regulatory Alignment")

    col1, col2 = st.columns([1, 3])
    with col1:
        score = report.regulatory_score
        if score >= 80:
            st.success(f"### {score:.0f}%\n**Regulatory Score**")
        elif score >= 60:
            st.warning(f"### {score:.0f}%\n**Regulatory Score**")
        else:
            st.error(f"### {score:.0f}%\n**Regulatory Score**")

    if not report.regulatory_checks:
        st.info("No regulatory checks performed")
        return

    # Summary counts
    aligned = sum(1 for c in report.regulatory_checks if c.status == "aligned")
    needs_attention = sum(1 for c in report.regulatory_checks if c.status == "needs_attention")
    missing = sum(1 for c in report.regulatory_checks if c.status == "missing")

    with col2:
        cols = st.columns(3)
        cols[0].metric("✅ Aligned", aligned)
        cols[1].metric("⚠️ Needs Attention", needs_attention)
        cols[2].metric("❌ Missing", missing)

    st.divider()

    # Detailed checks
    for check in report.regulatory_checks:
        status_emoji = {"aligned": "✅", "needs_attention": "⚠️", "missing": "❌"}.get(check.status, "⚪")

        with st.expander(f"{status_emoji} {check.requirement}"):
            st.markdown(f"**Category:** {check.category.title()}")
            st.markdown(f"**Status:** {check.status.replace('_', ' ').title()}")
            st.markdown(f"**Details:** {check.details}")
            if check.recommendation:
                st.info(f"**Recommendation:** {check.recommendation}")


def _render_statistics_tab(report):
    """Render statistical analysis tab."""
    st.subheader("Statistical Power Assessment")

    power = report.power_analysis

    col1, col2, col3 = st.columns(3)

    power_pct = power.estimated_power * 100
    if power_pct >= 80:
        col1.success(f"### {power_pct:.0f}%\n**Estimated Power**")
    elif power_pct >= 70:
        col1.warning(f"### {power_pct:.0f}%\n**Estimated Power**")
    else:
        col1.error(f"### {power_pct:.0f}%\n**Estimated Power**")

    col2.metric("Sample Size Adequate", "Yes" if power.sample_size_adequate else "No")
    col3.metric("Recommended Size", f"{power.recommended_sample_size:,}")

    st.markdown(f"**Effect Size Assumption:** {power.effect_size_assumption}")

    if power.notes:
        st.markdown("**Notes:**")
        for note in power.notes:
            st.info(note)

    # Power visualization
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=power_pct,
        title={'text': "Statistical Power"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4caf50" if power_pct >= 80 else "#ff9800"},
            'steps': [
                {'range': [0, 70], 'color': "#ffebee"},
                {'range': [70, 80], 'color': "#fff3e0"},
                {'range': [80, 100], 'color': "#e8f5e9"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)


def _render_timeline_tab(report):
    """Render timeline assessment tab."""
    st.subheader("Timeline Feasibility")

    timeline = report.timeline_assessment

    feasibility_colors = {"realistic": "🟢", "conservative": "🟡", "aggressive": "🔴"}
    st.markdown(f"### {feasibility_colors.get(timeline.feasibility, '⚪')} Timeline: {timeline.feasibility.title()}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated Duration", f"{timeline.proposed_duration_months:.0f} months")
    with col2:
        st.metric("Benchmark Duration", f"{timeline.benchmark_duration_months:.0f} months")

    # Risk factors
    if timeline.risk_factors:
        st.markdown("**Risk Factors:**")
        for risk in timeline.risk_factors:
            st.warning(risk)

    # Recommendations
    if timeline.recommendations:
        st.markdown("**Recommendations:**")
        for rec in timeline.recommendations:
            st.info(rec)

    # Milestone timeline
    if timeline.milestone_estimates:
        st.markdown("### Milestone Estimates")

        milestones = list(timeline.milestone_estimates.keys())
        months = list(timeline.milestone_estimates.values())

        fig = go.Figure(go.Bar(
            x=months,
            y=milestones,
            orientation='h',
            marker_color='#2196f3'
        ))
        fig.update_layout(
            title="Estimated Milestone Timeline",
            xaxis_title="Months from Start",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_competition_tab(report):
    """Render competitive analysis tab."""
    st.subheader("Competitive Landscape")

    competitive = report.competitive_position

    competition_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    st.markdown(f"### {competition_colors.get(competitive.enrollment_competition_level, '⚪')} Competition Level: {competitive.enrollment_competition_level.title()}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active Competing Trials", competitive.active_competitors)
    with col2:
        st.metric("Recently Completed", competitive.recently_completed)

    # Advantages
    if competitive.competitive_advantages:
        st.markdown("**✅ Competitive Advantages:**")
        for adv in competitive.competitive_advantages:
            st.success(adv)

    # Risks
    if competitive.competitive_risks:
        st.markdown("**⚠️ Competitive Risks:**")
        for risk in competitive.competitive_risks:
            st.warning(risk)


# ============== FEATURE 1: RISK ASSESSMENT ==============
def render_risk_assessment():
    """Feature 1: Comprehensive Risk & Termination Analysis."""
    # Hero section - Jeeva style
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 2.25rem; font-weight: 700; margin-bottom: 0.75rem; color: #182A81;">
            Risk & Termination Analysis
        </h1>
        <p style="font-size: 1.05rem; color: #4A4A4A; max-width: 650px; line-height: 1.7;">
            AI-powered risk assessment with termination pattern analysis,
            protective factors identification, and actionable mitigation strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not require_protocol():
        return

    p = st.session_state.extracted_protocol
    basic_risk = st.session_state.risk_assessment
    metrics = st.session_state.protocol_metrics

    # Protocol context - Jeeva style
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #EAECFB, #F5F5F5); border-radius: 10px; padding: 0.875rem 1.25rem; margin-bottom: 1.5rem;
                display: inline-flex; align-items: center; gap: 0.75rem; border: 1px solid #E5E5E5;">
        <span style="font-size: 0.85rem; color: #6B7280; font-weight: 500;">Analyzing:</span>
        <span style="font-weight: 700; color: #182A81;">{p.condition}</span>
        <span style="color: #CBD5E1; font-weight: 300;">|</span>
        <span style="color: #F45900; font-weight: 600;">{p.phase}</span>
    </div>
    """, unsafe_allow_html=True)

    # Historical benchmark cards
    st.markdown("""
    <h3 style="font-size: 0.85rem; font-weight: 700; color: #314DD8; text-transform: uppercase;
               letter-spacing: 0.08em; margin-bottom: 1rem;">Historical Benchmark</h3>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f"""<div class="stat-card"><div class="stat-value">{metrics.get('total_similar', 0)}</div><div class="stat-label">Similar Trials</div></div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="stat-card"><div class="stat-value" style="color: #10B981;">{metrics.get('completion_rate', 0):.0f}%</div><div class="stat-label">Success Rate</div></div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div class="stat-card"><div class="stat-value" style="color: #EF4444;">{metrics.get('termination_rate', 0):.0f}%</div><div class="stat-label">Termination Rate</div></div>""", unsafe_allow_html=True)
    col4.markdown(f"""<div class="stat-card"><div class="stat-value">{metrics.get('avg_enrollment', 0):,.0f}</div><div class="stat-label">Avg Enrollment</div></div>""", unsafe_allow_html=True)
    col5.markdown(f"""<div class="stat-card"><div class="stat-value">{metrics.get('avg_duration_months', 0):.0f} mo</div><div class="stat-label">Avg Duration</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Deep analysis button
    if st.button("Run Deep Risk Analysis", type="primary", use_container_width=True):
        db = get_database()
        if not db:
            return

        with st.spinner("Analyzing risks and termination patterns... This may take a minute."):
            try:
                from src.analysis.risk_analyzer import RiskAnalyzer

                analyzer = RiskAnalyzer(db)
                assessment = analyzer.analyze(
                    condition=p.condition,
                    phase=p.phase,
                    target_enrollment=p.target_enrollment,
                    num_sites=30,  # Default, could be from protocol
                    intervention=f"{p.intervention_type} - {getattr(p, 'intervention_name', '')}",
                    endpoint=p.primary_endpoint,
                    eligibility_criteria=p.eligibility_criteria or "",
                    use_ai_analysis=True
                )

                st.session_state.risk_assessment_detailed = assessment
                st.success("Analysis complete!")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display detailed analysis if available
    if "risk_assessment_detailed" in st.session_state and st.session_state.risk_assessment_detailed:
        _render_detailed_risk_assessment(st.session_state.risk_assessment_detailed)
    else:
        _render_basic_risk_assessment(basic_risk, metrics)


def _render_basic_risk_assessment(risk, metrics):
    """Render basic risk assessment (before deep analysis)."""
    st.info("👆 Click **Run Deep Risk Analysis** for AI-powered insights, termination patterns, and mitigation strategies.")

    col1, col2 = st.columns([1, 3])

    with col1:
        risk_level = risk.get("risk_level", "medium")
        risk_score = risk.get("overall_risk_score", 50)

        if risk_level == "high":
            st.error(f"### HIGH RISK\n\nScore: **{risk_score}/100**")
        elif risk_level == "medium":
            st.warning(f"### MEDIUM RISK\n\nScore: **{risk_score}/100**")
        else:
            st.success(f"### LOW RISK\n\nScore: **{risk_score}/100**")

    with col2:
        st.subheader("Initial Risk Factors")
        risks = risk.get("risks", [])
        if risks:
            for r in risks:
                icon = "🔴" if r["severity"] == "high" else "🟡" if r["severity"] == "medium" else "🟢"
                st.markdown(f"{icon} **{r['category']}**: {r['risk']}")
        else:
            st.info("Run deep analysis for detailed risk factors.")


def _render_detailed_risk_assessment(assessment):
    """Render full detailed risk assessment."""

    # ===== TABS =====
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Risk Overview",
        "⚠️ Risk Factors",
        "💀 Termination Analysis",
        "🛡️ Protective Factors",
        "📅 Timeline Risks"
    ])

    # ===== TAB 1: OVERVIEW =====
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            # Overall risk score
            rs = assessment.risk_score
            if rs.risk_level == "high":
                st.error(f"### ⚠️ HIGH RISK\n**Score: {rs.overall_score:.0f}/100**")
            elif rs.risk_level == "medium":
                st.warning(f"### 🟡 MEDIUM RISK\n**Score: {rs.overall_score:.0f}/100**")
            else:
                st.success(f"### ✅ LOW RISK\n**Score: {rs.overall_score:.0f}/100**")

            st.metric("Success Probability", f"{assessment.success_probability:.0f}%")
            st.caption(f"vs Historical: {assessment.historical_success_rate:.0f}%")

        with col2:
            # Risk component breakdown
            st.subheader("Risk Component Breakdown")

            risk_components = {
                "Enrollment": rs.enrollment_risk,
                "Scientific": rs.scientific_risk,
                "Design": rs.design_risk,
                "Operational": rs.operational_risk,
                "Regulatory": rs.regulatory_risk,
                "Market": rs.market_risk
            }

            fig = go.Figure(go.Bar(
                x=list(risk_components.values()),
                y=list(risk_components.keys()),
                orientation='h',
                marker_color=['#FF6B6B' if v > 60 else '#FFD93D' if v > 40 else '#6BCB77' for v in risk_components.values()],
                text=[f"{v:.0f}" for v in risk_components.values()],
                textposition='auto'
            ))
            fig.update_layout(
                xaxis_title="Risk Score (0-100)",
                height=280,
                xaxis=dict(range=[0, 100]),
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Interaction effects
        if rs.interaction_effects:
            st.divider()
            st.subheader("⚡ Risk Interactions")
            for effect in rs.interaction_effects:
                st.warning(f"• {effect}")

        # Top risks and priorities
        col1, col2 = st.columns(2)
        with col1:
            if assessment.top_risks:
                st.subheader("🔴 Top Risks")
                for i, risk in enumerate(assessment.top_risks[:5], 1):
                    st.markdown(f"**{i}.** {risk}")

        with col2:
            if assessment.mitigation_priorities:
                st.subheader("🎯 Mitigation Priorities")
                for i, priority in enumerate(assessment.mitigation_priorities[:5], 1):
                    st.markdown(f"**{i}.** {priority}")

        # Key insights
        if assessment.key_insights:
            st.divider()
            st.subheader("💡 Key Insights")
            for insight in assessment.key_insights:
                st.info(insight)

    # ===== TAB 2: RISK FACTORS =====
    with tab2:
        st.subheader("Detailed Risk Factors")

        if assessment.risk_factors:
            # Group by category
            by_category = {}
            for rf in assessment.risk_factors:
                if rf.category not in by_category:
                    by_category[rf.category] = []
                by_category[rf.category].append(rf)

            for category, factors in by_category.items():
                st.markdown(f"### {category.title()} Risks")
                for rf in factors:
                    severity_icon = "🔴" if rf.severity == "high" else "🟡" if rf.severity == "medium" else "🟢"

                    with st.expander(f"{severity_icon} {rf.name} (Score: {rf.score:.0f})"):
                        st.markdown(f"**Description:** {rf.description}")
                        if rf.evidence:
                            st.markdown(f"**Evidence:** {rf.evidence}")
                        st.success(f"**Mitigation:** {rf.mitigation}")
                        st.caption(f"Confidence: {rf.confidence}")
        else:
            st.info("No detailed risk factors available. Run deep analysis for more insights.")

    # ===== TAB 3: TERMINATION ANALYSIS =====
    with tab3:
        st.subheader("Why Similar Trials Failed")
        st.caption(f"Analysis of {assessment.terminated_count} terminated trials")

        if assessment.termination_categories:
            # Pie chart
            categories = assessment.termination_categories
            non_zero = [c for c in categories if c.count > 0]

            if non_zero:
                col1, col2 = st.columns([1, 2])

                with col1:
                    fig = px.pie(
                        values=[c.count for c in non_zero],
                        names=[c.name for c in non_zero],
                        title="Termination Reasons",
                        hole=0.4
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Risk by Category")
                    for cat in sorted(non_zero, key=lambda x: -x.count):
                        if cat.risk_level == "high":
                            st.error(f"🔴 **{cat.name}**: {cat.count} ({cat.percentage:.1f}%) - HIGH RISK")
                        elif cat.risk_level == "medium":
                            st.warning(f"🟡 **{cat.name}**: {cat.count} ({cat.percentage:.1f}%) - MEDIUM RISK")
                        else:
                            st.success(f"🟢 **{cat.name}**: {cat.count} ({cat.percentage:.1f}%) - LOW RISK")

                # Detailed category analysis
                st.divider()
                st.subheader("Mitigation Strategies by Category")

                for cat in sorted(non_zero, key=lambda x: -x.count)[:5]:
                    with st.expander(f"📋 {cat.name} - Mitigation Strategies"):
                        if cat.mitigation_strategies:
                            for strategy in cat.mitigation_strategies:
                                st.markdown(f"• {strategy}")

                        if cat.examples:
                            st.markdown("**Example termination reasons:**")
                            for ex in cat.examples[:3]:
                                st.caption(f"• {ex[:150]}...")

        # Root cause analyses
        if assessment.root_cause_analyses:
            st.divider()
            st.subheader("🔍 Root Cause Analysis")

            for rca in assessment.root_cause_analyses:
                with st.expander(f"Root Causes: {rca.category}"):
                    if rca.root_causes:
                        st.markdown("**Root Causes:**")
                        for cause in rca.root_causes:
                            st.markdown(f"• {cause}")

                    if rca.early_warning_signs:
                        st.markdown("**Early Warning Signs:**")
                        for sign in rca.early_warning_signs:
                            st.warning(f"⚠️ {sign}")

                    if rca.prevention_strategies:
                        st.markdown("**Prevention Strategies:**")
                        for strategy in rca.prevention_strategies:
                            st.success(f"✓ {strategy}")

    # ===== TAB 4: PROTECTIVE FACTORS =====
    with tab4:
        st.subheader("🛡️ Protective Factors")
        st.markdown("Factors that correlate with trial success. Adopt these to improve your chances.")

        if assessment.protective_factors:
            for pf in assessment.protective_factors:
                impact_icon = "💪" if pf.impact == "strong" else "👍" if pf.impact == "moderate" else "👌"
                status_color = "success" if pf.your_protocol_status == "present" else "warning" if pf.your_protocol_status == "partial" else "error"

                with st.expander(f"{impact_icon} {pf.name} ({pf.impact.title()} Impact)"):
                    st.markdown(f"**Why it helps:** {pf.description}")

                    if status_color == "success":
                        st.success(f"✅ Your protocol: {pf.your_protocol_status.upper()}")
                    elif status_color == "warning":
                        st.warning(f"⚠️ Your protocol: {pf.your_protocol_status.upper()}")
                    else:
                        st.error(f"❌ Your protocol: {pf.your_protocol_status.upper()}")

                    if pf.recommendation:
                        st.info(f"**Recommendation:** {pf.recommendation}")
        else:
            st.info("Run deep analysis to identify protective factors from successful trials.")

    # ===== TAB 5: TIMELINE RISKS =====
    with tab5:
        st.subheader("📅 Risk Throughout Trial Lifecycle")
        st.markdown("Different risks emerge at different stages. Plan accordingly.")

        if assessment.temporal_risks:
            for tr in assessment.temporal_risks:
                risk_color = "error" if tr.risk_level == "high" else "warning" if tr.risk_level == "medium" else "success"

                with st.expander(f"{'🔴' if tr.risk_level == 'high' else '🟡' if tr.risk_level == 'medium' else '🟢'} {tr.phase}"):
                    st.markdown(f"**Risk Level:** {tr.risk_level.upper()}")

                    if tr.common_issues:
                        st.markdown("**Common Issues:**")
                        for issue in tr.common_issues:
                            st.markdown(f"• {issue}")

                    if tr.watchpoints:
                        st.markdown("**Key Watchpoints:**")
                        for wp in tr.watchpoints:
                            st.warning(f"👁️ {wp}")

            # Visual timeline
            st.divider()
            st.subheader("Risk Timeline Visualization")

            phases = [tr.phase.split("(")[0].strip() for tr in assessment.temporal_risks]
            risk_levels = [3 if tr.risk_level == "high" else 2 if tr.risk_level == "medium" else 1 for tr in assessment.temporal_risks]
            colors = ["#FF6B6B" if rl == 3 else "#FFD93D" if rl == 2 else "#6BCB77" for rl in risk_levels]

            fig = go.Figure(go.Bar(
                x=phases,
                y=risk_levels,
                marker_color=colors,
                text=["High" if rl == 3 else "Medium" if rl == 2 else "Low" for rl in risk_levels],
                textposition='auto'
            ))
            fig.update_layout(
                yaxis_title="Risk Level",
                height=300,
                yaxis=dict(tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"])
            )
            st.plotly_chart(fig, use_container_width=True)


# ============== FEATURE 2: SITE RECOMMENDATIONS ==============
def render_site_intelligence():
    """Feature 2: Comprehensive Site & Geographic Intelligence."""
    # Hero section - Jeeva style
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 2.25rem; font-weight: 700; margin-bottom: 0.75rem; color: #182A81;">
            Site & Geographic Intelligence
        </h1>
        <p style="font-size: 1.05rem; color: #4A4A4A; max-width: 650px; line-height: 1.7;">
            Comprehensive site selection analysis with performance scoring,
            regulatory context, and portfolio optimization strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not require_protocol():
        return

    p = st.session_state.extracted_protocol

    # Protocol context - Jeeva style
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #EAECFB, #F5F5F5); border-radius: 10px; padding: 0.875rem 1.25rem; margin-bottom: 1.5rem;
                display: inline-flex; align-items: center; gap: 0.75rem; border: 1px solid #E5E5E5;">
        <span style="font-size: 0.85rem; color: #6B7280; font-weight: 500;">Analyzing:</span>
        <span style="font-weight: 700; color: #182A81;">{p.condition}</span>
        <span style="color: #CBD5E1; font-weight: 300;">|</span>
        <span style="color: #F45900; font-weight: 600;">{p.phase}</span>
    </div>
    """, unsafe_allow_html=True)

    # Settings card - Jeeva style
    st.markdown("""
    <div style="background: linear-gradient(135deg, #F5F5F5, #EAECFB); border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem;">
        <h4 style="font-size: 0.8rem; font-weight: 700; color: #314DD8; text-transform: uppercase;
                   letter-spacing: 0.08em; margin-bottom: 0.5rem;">Analysis Settings</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        target_sites = st.number_input("Target Sites", min_value=5, value=30, step=5)
    with col2:
        target_enrollment = st.number_input("Target Enrollment", min_value=10, value=p.target_enrollment or 200)
    with col3:
        country_filter = st.selectbox("Country Focus", ["All Countries", "United States", "Germany", "United Kingdom", "France", "Canada", "Japan", "China", "Australia"])

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    if st.button("🌍 Generate Site Intelligence", type="primary", use_container_width=True):
        db = get_database()
        if not db:
            return

        with st.spinner("Analyzing sites, geography, and building optimal portfolio..."):
            try:
                from src.analysis.site_intelligence import SiteIntelligenceAnalyzer

                analyzer = SiteIntelligenceAnalyzer(db)
                report = analyzer.analyze(
                    condition=p.condition,
                    phase=p.phase,
                    target_enrollment=target_enrollment,
                    target_sites=target_sites,
                    country_filter=country_filter if country_filter != "All Countries" else None
                )

                st.session_state.site_intelligence = report
                st.success("Site intelligence analysis complete!")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display results if available
    if st.session_state.site_intelligence:
        _render_site_intelligence_results()


def _render_site_intelligence_results():
    """Render site intelligence results with tabs."""
    report = st.session_state.site_intelligence

    st.divider()

    # Key insights
    if report.key_insights:
        st.subheader("🔑 Key Insights")
        for insight in report.key_insights:
            st.info(f"💡 {insight}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Top Sites Found", len(report.top_sites))
    col2.metric("Countries Analyzed", len(report.country_profiles))
    col3.metric("Portfolio Sites", report.recommended_portfolio.total_recommended_sites)
    col4.metric("Est. Capacity", f"{report.recommended_portfolio.estimated_enrollment_capacity:,}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 Site Recommendations",
        "🌍 Geographic Distribution",
        "🗺️ Country Profiles",
        "📊 Portfolio Optimization"
    ])

    with tab1:
        _render_site_recommendations_tab(report)

    with tab2:
        _render_geographic_tab(report)

    with tab3:
        _render_country_profiles_tab(report)

    with tab4:
        _render_portfolio_tab(report)


def _render_site_recommendations_tab(report):
    """Render site recommendations tab."""
    st.subheader("Top Recommended Sites")

    if not report.top_sites:
        st.warning("No sites found matching criteria")
        return

    # Summary by recommendation level
    highly_rec = [s for s in report.top_sites if s.recommendation == "highly_recommended"]
    recommended = [s for s in report.top_sites if s.recommendation == "recommended"]
    consider = [s for s in report.top_sites if s.recommendation == "consider"]

    col1, col2, col3 = st.columns(3)
    col1.metric("⭐ Highly Recommended", len(highly_rec))
    col2.metric("✓ Recommended", len(recommended))
    col3.metric("○ Consider", len(consider))

    # Site table
    site_data = []
    for i, site in enumerate(report.top_sites[:30], 1):
        rec_emoji = "⭐" if site.recommendation == "highly_recommended" else ("✓" if site.recommendation == "recommended" else "○")
        site_data.append({
            "Rank": i,
            "Rec": rec_emoji,
            "Facility": site.facility_name[:35] + "..." if len(site.facility_name) > 35 else site.facility_name,
            "City": site.city or "N/A",
            "Country": site.country,
            "Score": f"{site.overall_score:.0f}",
            "Trials": site.total_trials,
            "Completion": f"{site.completion_rate:.0f}%",
            "TA Exp": site.therapeutic_area_experience,
            "Capacity": site.capacity_score.title(),
        })

    st.dataframe(pd.DataFrame(site_data), use_container_width=True, hide_index=True)

    # Top sites by country visualization
    if report.sites_by_country:
        country_counts = {k: len(v) for k, v in report.sites_by_country.items()}
        top_countries = dict(sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        fig = px.bar(
            x=list(top_countries.values()),
            y=list(top_countries.keys()),
            orientation='h',
            title="Top Sites by Country",
            labels={"x": "Number of Sites", "y": "Country"}
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


def _render_geographic_tab(report):
    """Render geographic distribution tab."""
    geo = report.geographic_distribution

    st.subheader("Geographic Coverage Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Countries", geo.total_countries)
    col2.metric("Coverage Score", f"{geo.coverage_score:.0f}/100")
    col3.metric("Diversity Score", f"{geo.diversity_score:.0f}/100")

    # Regional distribution
    if geo.by_region:
        st.subheader("Regional Distribution")

        region_data = []
        for region, data in geo.by_region.items():
            region_data.append({
                "Region": region,
                "Trials": data.get("trials", 0),
                "Sites": data.get("sites", 0),
                "Countries": data.get("countries", 0),
            })

        if region_data:
            df = pd.DataFrame(region_data)
            df = df.sort_values("Trials", ascending=False)

            fig = px.bar(
                df, x="Trials", y="Region", orientation='h',
                title="Trial Activity by Region",
                color="Trials", color_continuous_scale="Blues"
            )
            fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df, use_container_width=True, hide_index=True)

    # Recommendations
    if geo.recommendations:
        st.subheader("Geographic Recommendations")
        for rec in geo.recommendations:
            st.markdown(f"- {rec}")


def _render_country_profiles_tab(report):
    """Render country profiles tab."""
    st.subheader("Country-Level Analysis")

    if not report.country_profiles:
        st.warning("No country data available")
        return

    # Country selector
    countries = [p.country for p in report.country_profiles]
    selected = st.selectbox("Select Country for Details", countries)

    # Summary table
    country_data = []
    for cp in report.country_profiles[:20]:
        country_data.append({
            "Country": cp.country,
            "Region": cp.region,
            "Trials": cp.total_trials,
            "Completed": cp.completed_trials,
            "Completion %": f"{cp.completion_rate:.0f}%",
            "Sites": cp.num_sites,
            "Regulatory": cp.regulatory_complexity.title(),
            "Saturation": cp.market_saturation.title(),
            "Rec": cp.recommendation.title(),
        })

    st.dataframe(pd.DataFrame(country_data), use_container_width=True, hide_index=True)

    # Selected country details
    selected_profile = next((p for p in report.country_profiles if p.country == selected), None)
    if selected_profile:
        st.divider()
        st.subheader(f"📍 {selected_profile.country} Details")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Trial Metrics**")
            st.write(f"- Total Trials: {selected_profile.total_trials:,}")
            st.write(f"- Completed: {selected_profile.completed_trials:,}")
            st.write(f"- Recruiting: {selected_profile.recruiting_trials}")
            st.write(f"- Completion Rate: {selected_profile.completion_rate:.1f}%")

        with col2:
            st.markdown("**Regulatory Context**")
            st.write(f"- Complexity: {selected_profile.regulatory_complexity.title()}")
            st.write(f"- Approval Time: {selected_profile.avg_approval_time}")
            st.write(f"- Market Saturation: {selected_profile.market_saturation.title()}")
            if selected_profile.regulatory_notes:
                st.write(f"- Notes: {selected_profile.regulatory_notes}")

        # Pros/Cons
        col1, col2 = st.columns(2)
        with col1:
            if selected_profile.pros:
                st.markdown("**✅ Advantages**")
                for pro in selected_profile.pros:
                    st.markdown(f"- {pro}")
        with col2:
            if selected_profile.cons:
                st.markdown("**⚠️ Considerations**")
                for con in selected_profile.cons:
                    st.markdown(f"- {con}")


def _render_portfolio_tab(report):
    """Render portfolio optimization tab."""
    portfolio = report.recommended_portfolio

    st.subheader("Optimized Site Portfolio")

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recommended Sites", portfolio.total_recommended_sites)
    col2.metric("Countries", portfolio.total_recommended_countries)
    col3.metric("Est. Capacity", f"{portfolio.estimated_enrollment_capacity:,}")
    col4.metric("Est. Duration", f"{portfolio.estimated_enrollment_months:.1f} mo")

    # Risk assessment
    risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(portfolio.portfolio_risk, "⚪")
    st.markdown(f"**Portfolio Risk:** {risk_color} {portfolio.portfolio_risk.title()}")

    if portfolio.risk_factors:
        with st.expander("Risk Factors"):
            for rf in portfolio.risk_factors:
                st.markdown(f"- {rf}")

    st.divider()

    # Tiered sites
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**⭐ Tier 1 Sites** (Top Performers)")
        if portfolio.tier1_sites:
            for site in portfolio.tier1_sites[:8]:
                st.write(f"- {site.facility_name[:30]}... ({site.country})")
        else:
            st.caption("No Tier 1 sites")

    with col2:
        st.markdown("**✓ Tier 2 Sites** (Good Performers)")
        if portfolio.tier2_sites:
            for site in portfolio.tier2_sites[:8]:
                st.write(f"- {site.facility_name[:30]}... ({site.country})")
        else:
            st.caption("No Tier 2 sites")

    with col3:
        st.markdown("**○ Tier 3 Sites** (Backup)")
        if portfolio.tier3_sites:
            for site in portfolio.tier3_sites[:8]:
                st.write(f"- {site.facility_name[:30]}... ({site.country})")
        else:
            st.caption("No Tier 3 sites")

    # Geographic distribution chart
    if portfolio.country_distribution:
        st.divider()
        st.subheader("Portfolio Geographic Distribution")

        top_countries = dict(sorted(portfolio.country_distribution.items(), key=lambda x: x[1], reverse=True)[:10])
        fig = px.pie(
            values=list(top_countries.values()),
            names=list(top_countries.keys()),
            title="Sites by Country",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    # Optimization suggestions
    if portfolio.optimization_suggestions:
        st.divider()
        st.subheader("💡 Optimization Suggestions")
        for suggestion in portfolio.optimization_suggestions:
            st.info(suggestion)

    # Overall recommendations
    if report.recommendations:
        st.divider()
        st.subheader("📋 Recommendations")
        for rec in report.recommendations:
            st.markdown(f"- {rec}")


# ============== FEATURE 3: ENROLLMENT FORECAST ==============
def render_enrollment_forecast():
    """Feature 3: Enhanced enrollment prediction with S-curve modeling."""
    # Hero section - Jeeva style
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 2.25rem; font-weight: 700; margin-bottom: 0.75rem; color: #182A81;">
            Enrollment Timeline Forecast
        </h1>
        <p style="font-size: 1.05rem; color: #4A4A4A; max-width: 650px; line-height: 1.7;">
            Advanced enrollment predictions using S-curve modeling, site activation timing,
            and risk-adjusted projections based on historical data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not require_protocol():
        return

    p = st.session_state.extracted_protocol

    # Protocol context - Jeeva style
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #EAECFB, #F5F5F5); border-radius: 10px; padding: 0.875rem 1.25rem; margin-bottom: 1.5rem;
                display: inline-flex; align-items: center; gap: 0.75rem; border: 1px solid #E5E5E5;">
        <span style="font-size: 0.85rem; color: #6B7280; font-weight: 500;">Analyzing:</span>
        <span style="font-weight: 700; color: #182A81;">{p.condition}</span>
        <span style="color: #CBD5E1; font-weight: 300;">|</span>
        <span style="color: #F45900; font-weight: 600;">{p.phase}</span>
    </div>
    """, unsafe_allow_html=True)

    # Settings card - Jeeva style
    st.markdown("""
    <div style="background: linear-gradient(135deg, #F5F5F5, #EAECFB); border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem;">
        <h4 style="font-size: 0.8rem; font-weight: 700; color: #314DD8; text-transform: uppercase;
                   letter-spacing: 0.08em; margin-bottom: 0.5rem;">Trial Parameters</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        target = st.number_input("Target Enrollment", min_value=10, value=p.target_enrollment or 200)
    with col2:
        num_sites = st.number_input("Planned Sites", min_value=1, value=30)
    with col3:
        screen_failure = st.slider("Screen Failure Rate", 0.1, 0.5, 0.25, 0.05,
                                   help="% of screened patients who fail screening")

    # Risk factors
    with st.expander("Advanced Risk Factors", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            competition = st.slider("Competition Level", 0.0, 1.0, 0.3, 0.1,
                                   help="0=No competition, 1=Heavy competition")
            site_experience = st.slider("Site Experience", 0.0, 1.0, 0.5, 0.1,
                                       help="0=Inexperienced, 1=Very experienced")
        with col2:
            geographic = st.slider("Geographic Spread", 0.0, 1.0, 0.3, 0.1,
                                  help="0=Single country, 1=Global")

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    if st.button("Generate Advanced Forecast", type="primary", use_container_width=True):
        db = get_database()
        if not db:
            return

        with st.spinner("Generating S-curve enrollment forecast..."):
            try:
                from src.analysis.enrollment_forecaster import EnrollmentForecaster

                forecaster = EnrollmentForecaster(db)
                forecast = forecaster.forecast(
                    target_enrollment=target,
                    num_sites=num_sites,
                    condition=p.condition,
                    phase=p.phase,
                    eligibility_criteria=p.eligibility_criteria or "",
                    competition_level=competition,
                    site_experience=site_experience,
                    geographic_spread=geographic,
                    screen_failure_rate=screen_failure
                )

                # Store in session state for reference
                st.session_state.enrollment_forecast = forecast

                st.divider()

                # ===== TABS FOR DIFFERENT VIEWS =====
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Timeline Overview",
                    "📈 Enrollment Curves",
                    "🎯 Milestones & Targets",
                    "⚠️ Risk Analysis"
                ])

                # ===== TAB 1: OVERVIEW =====
                with tab1:
                    st.subheader("Enrollment Timeline Scenarios")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.success(f"### 🟢 Optimistic\n**{forecast.optimistic.total_months:.0f} months**")
                        st.caption(f"{forecast.optimistic.monthly_rate:.1f} pts/month")
                    with col2:
                        st.warning(f"### 🟡 Expected\n**{forecast.expected.total_months:.0f} months**")
                        st.caption(f"{forecast.expected.monthly_rate:.1f} pts/month")
                    with col3:
                        st.error(f"### 🔴 Pessimistic\n**{forecast.pessimistic.total_months:.0f} months**")
                        st.caption(f"{forecast.pessimistic.monthly_rate:.1f} pts/month")
                    with col4:
                        st.info(f"### 📊 Risk-Adjusted\n**{forecast.risk_adjusted.total_months:.0f} months**")
                        st.caption(f"Multiplier: {forecast.risk_multiplier:.2f}x")

                    st.divider()

                    # Historical benchmark
                    st.subheader("📚 Historical Benchmark")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Similar Trials Analyzed", forecast.similar_trials_count)
                    col2.metric("Median Rate", f"{forecast.historical_rate_median:.2f}/site/mo")
                    col3.metric("25th Percentile", f"{forecast.historical_rate_p25:.2f}/site/mo")
                    col4.metric("75th Percentile", f"{forecast.historical_rate_p75:.2f}/site/mo")

                    # Screening requirements
                    st.divider()
                    st.subheader("🔬 Screening Requirements")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Patients to Screen", f"{forecast.patients_to_screen:,}")
                    col2.metric("Screen Failure Rate", f"{forecast.screening.screen_failure_rate*100:.0f}%")
                    col3.metric("Randomization Rate", f"{forecast.screening.randomization_rate*100:.0f}%")

                # ===== TAB 2: ENROLLMENT CURVES =====
                with tab2:
                    st.subheader("S-Curve Enrollment Projections")
                    st.caption("Realistic enrollment curves with ramp-up and plateau phases")

                    # Get max months for x-axis
                    max_months = int(forecast.pessimistic.total_months) + 6

                    fig = go.Figure()

                    # Add all scenarios
                    scenarios = [
                        (forecast.optimistic, "Optimistic", "green", "dash"),
                        (forecast.expected, "Expected", "orange", "solid"),
                        (forecast.pessimistic, "Pessimistic", "red", "dash"),
                        (forecast.risk_adjusted, "Risk-Adjusted", "purple", "dot"),
                    ]

                    for scenario, name, color, dash in scenarios:
                        months = list(range(len(scenario.cumulative_enrollment)))
                        fig.add_trace(go.Scatter(
                            x=months,
                            y=scenario.cumulative_enrollment,
                            name=name,
                            line=dict(color=color, dash=dash, width=2 if dash == "solid" else 1.5),
                            hovertemplate=f"{name}<br>Month: %{{x}}<br>Enrolled: %{{y:.0f}}<extra></extra>"
                        ))

                    # Target line
                    fig.add_hline(y=target, line_dash="dot", line_color="gray",
                                 annotation_text=f"Target: {target:,}")

                    fig.update_layout(
                        xaxis_title="Months from First Patient In",
                        yaxis_title="Cumulative Patients Enrolled",
                        height=450,
                        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Monthly enrollment rate chart
                    st.subheader("Monthly Enrollment Rate")

                    fig2 = go.Figure()
                    months = list(range(len(forecast.expected.monthly_enrollment)))
                    fig2.add_trace(go.Bar(
                        x=months,
                        y=forecast.expected.monthly_enrollment,
                        name="Expected Monthly",
                        marker_color="orange"
                    ))
                    fig2.update_layout(
                        xaxis_title="Month",
                        yaxis_title="Patients Enrolled",
                        height=300
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    # Site activation
                    st.subheader("🏥 Site Activation Schedule")
                    site_plan = forecast.site_activation
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Sites", site_plan.total_sites)
                        st.metric("Activation Period", f"{site_plan.activation_months} months")
                    with col2:
                        site_data = []
                        for m, (new, cum) in enumerate(zip(site_plan.sites_per_month, site_plan.cumulative_active)):
                            site_data.append({
                                "Month": m + 1,
                                "New Sites": new,
                                "Cumulative Active": cum
                            })
                        st.dataframe(pd.DataFrame(site_data), hide_index=True)

                # ===== TAB 3: MILESTONES =====
                with tab3:
                    st.subheader("🎯 Enrollment Milestones")

                    # Show milestones for expected scenario
                    milestones = forecast.expected.milestones

                    if milestones:
                        milestone_data = []
                        for m in milestones:
                            milestone_data.append({
                                "Target": f"{m.cumulative_percent:.0f}%",
                                "Patients": m.target_patients,
                                "Month": m.month,
                                "Notes": m.notes
                            })

                        st.dataframe(pd.DataFrame(milestone_data), use_container_width=True, hide_index=True)

                        # Visual timeline
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=[m.month for m in milestones],
                            y=[m.target_patients for m in milestones],
                            mode='markers+text',
                            marker=dict(size=20, color=['#90EE90', '#FFD700', '#FFA500', '#32CD32']),
                            text=[f"{m.cumulative_percent:.0f}%" for m in milestones],
                            textposition="top center",
                            name="Milestones"
                        ))
                        fig.update_layout(
                            xaxis_title="Month",
                            yaxis_title="Patients Enrolled",
                            height=300,
                            title="Milestone Timeline"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Monthly targets table
                    st.subheader("📅 Monthly Enrollment Targets")
                    monthly_targets = forecaster.get_monthly_targets(forecast, "expected")

                    # Show first 12 months or until target reached
                    display_targets = [t for t in monthly_targets if t["month"] <= 12 or t["cumulative_target"] < target][:18]

                    if display_targets:
                        targets_df = pd.DataFrame(display_targets)
                        targets_df.columns = ["Month", "Monthly Target", "Cumulative Target", "% Complete"]
                        st.dataframe(targets_df, use_container_width=True, hide_index=True)

                # ===== TAB 4: RISK ANALYSIS =====
                with tab4:
                    st.subheader("⚠️ Risk Factor Analysis")

                    # Risk factors visualization
                    rf = forecast.risk_factors

                    risk_data = {
                        "Factor": ["Eligibility Complexity", "Competition Level", "Site Experience", "Geographic Spread"],
                        "Value": [rf.eligibility_complexity, rf.competition_level, rf.site_experience, rf.geographic_spread],
                        "Impact": ["Higher = Slower", "Higher = Slower", "Higher = Faster", "Higher = Faster"]
                    }

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = go.Figure(go.Bar(
                            x=risk_data["Value"],
                            y=risk_data["Factor"],
                            orientation='h',
                            marker_color=['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4'],
                            text=[f"{v:.0%}" for v in risk_data["Value"]],
                            textposition='auto'
                        ))
                        fig.update_layout(
                            xaxis_title="Factor Level (0-1)",
                            height=250,
                            xaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.metric("Condition Prevalence", rf.condition_prevalence.title())
                        st.metric("Risk Multiplier", f"{forecast.risk_multiplier:.2f}x")

                        if forecast.risk_multiplier < 0.7:
                            st.error("High risk - enrollment may be significantly slower")
                        elif forecast.risk_multiplier < 0.9:
                            st.warning("Moderate risk - some enrollment challenges expected")
                        else:
                            st.success("Low risk - enrollment conditions favorable")

                    # Key insights
                    st.divider()
                    st.subheader("💡 Key Insights")
                    for insight in forecast.key_insights:
                        st.info(insight)

                    # Recommendations
                    if forecast.recommendations:
                        st.subheader("📋 Recommendations")
                        for rec in forecast.recommendations:
                            st.markdown(f"• {rec}")

            except Exception as e:
                st.error(f"Forecast failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ============== FEATURE 4: SIMILAR TRIALS & COMPETITIVE INTELLIGENCE ==============
def render_similar_trials_and_competition():
    """Feature 4: Enhanced similar trials analysis with competitive intelligence."""
    # Hero section - Jeeva style
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 2.25rem; font-weight: 700; margin-bottom: 0.75rem; color: #182A81;">
            Similar Trials & Competition
        </h1>
        <p style="font-size: 1.05rem; color: #4A4A4A; max-width: 650px; line-height: 1.7;">
            Deep analysis of comparable trials with similarity breakdowns,
            lessons learned, and competitive threat assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not require_protocol():
        return

    p = st.session_state.extracted_protocol
    similar = st.session_state.similar_trials
    matching_context = st.session_state.matching_context

    # Protocol context - Jeeva style
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #EAECFB, #F5F5F5); border-radius: 10px; padding: 0.875rem 1.25rem; margin-bottom: 1.5rem;
                display: inline-flex; align-items: center; gap: 0.75rem; border: 1px solid #E5E5E5;">
        <span style="font-size: 0.85rem; color: #6B7280; font-weight: 500;">Analyzing:</span>
        <span style="font-weight: 700; color: #182A81;">{p.condition}</span>
        <span style="color: #CBD5E1; font-weight: 300;">|</span>
        <span style="color: #F45900; font-weight: 600;">{p.phase}</span>
        <span style="color: #CBD5E1; font-weight: 300;">|</span>
        <span style="color: #314DD8; font-weight: 600;">{len(similar) if similar else 0} similar trials</span>
    </div>
    """, unsafe_allow_html=True)

    if not similar:
        st.warning("No similar trials found. Please analyze your protocol first.")
        return

    # Check if we need to run enhanced analysis
    run_analysis = st.button("🔍 Run Deep Analysis", type="primary", use_container_width=True)

    # Use cached analysis if available
    analysis = st.session_state.enhanced_analysis

    if run_analysis:
        with st.spinner("Running comprehensive analysis... This may take a minute."):
            try:
                from src.analysis.similar_trials_analyzer import SimilarTrialsAnalyzer

                db = get_database()
                analyzer = SimilarTrialsAnalyzer(db)

                # Prepare user protocol dict
                user_protocol = {
                    "condition": p.condition,
                    "phase": p.phase,
                    "intervention": f"{p.intervention_type} - {getattr(p, 'intervention_name', '')}",
                    "primary_endpoint": p.primary_endpoint,
                    "target_enrollment": p.target_enrollment,
                    "eligibility": p.eligibility_criteria[:500] if p.eligibility_criteria else ""
                }

                # Convert similar trials to dicts
                similar_dicts = []
                for t in similar:
                    similar_dicts.append({
                        "nct_id": t.nct_id,
                        "title": t.title,
                        "status": t.status,
                        "phase": t.phase,
                        "sponsor": t.sponsor,
                        "conditions": t.conditions,
                        "interventions": t.interventions,
                        "enrollment": t.enrollment,
                        "enrollment_type": getattr(t, 'enrollment_type', ''),
                        "num_sites": t.num_sites,
                        "start_date": getattr(t, 'start_date', None),
                        "completion_date": getattr(t, 'completion_date', None),
                        "primary_outcomes": t.primary_outcomes,
                        "eligibility_criteria": t.eligibility_criteria,
                        "why_stopped": t.why_stopped,
                        "duration_months": t.duration_months
                    })

                # Run comprehensive analysis
                analysis = analyzer.analyze(
                    user_protocol=user_protocol,
                    similar_trials=similar_dicts,
                    analyze_top_n_similarity=10,
                    extract_lessons=True,
                    analyze_competition=True
                )

                st.session_state.enhanced_analysis = analysis
                st.success("Analysis complete!")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display results
    if analysis:
        _render_enhanced_analysis(analysis, p)
    else:
        _render_basic_similar_trials(similar, p)


def _render_basic_similar_trials(similar, p):
    """Render basic similar trials view (before deep analysis)."""
    # Summary metrics
    completed = [t for t in similar if t.status == 'COMPLETED']
    terminated = [t for t in similar if t.status in ['TERMINATED', 'WITHDRAWN']]
    active = [t for t in similar if t.status in ['RECRUITING', 'ACTIVE_NOT_RECRUITING']]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Similar Trials", len(similar))
    col2.metric("Completed", len(completed))
    col3.metric("Terminated", len(terminated))
    col4.metric("Active/Recruiting", len(active))

    st.info("👆 Click **Run Deep Analysis** above for AI-powered insights, lessons learned, and competitive analysis.")

    st.divider()

    # Basic table
    show_only = st.selectbox("Filter", ["All", "Completed", "Terminated", "Active/Recruiting"])

    if show_only == "Completed":
        display_trials = completed
    elif show_only == "Terminated":
        display_trials = terminated
    elif show_only == "Active/Recruiting":
        display_trials = active
    else:
        display_trials = similar

    trial_data = []
    for t in display_trials[:50]:
        status_icon = "✅" if t.status == "COMPLETED" else "❌" if t.status in ["TERMINATED", "WITHDRAWN"] else "🔄"
        trial_data.append({
            "Status": status_icon,
            "NCT ID": t.nct_id,
            "Similarity": f"{t.overall_similarity:.0f}%",
            "Sponsor": (t.sponsor[:25] + "...") if t.sponsor and len(t.sponsor) > 25 else (t.sponsor or "N/A"),
            "Enrollment": str(t.enrollment) if t.enrollment else "N/A",
            "Duration": f"{t.duration_months:.0f}mo" if t.duration_months else "N/A",
        })

    st.dataframe(pd.DataFrame(trial_data), use_container_width=True, hide_index=True)


def _render_enhanced_analysis(analysis, p):
    """Render full enhanced analysis results."""
    insights = analysis.get("aggregate_insights")
    completed = analysis.get("completed_trials", [])
    terminated = analysis.get("terminated_trials", [])
    active = analysis.get("active_trials", [])
    competitive_intel = analysis.get("competitive_intel")
    top_recommendations = analysis.get("top_recommendations", [])
    enhanced_matches = analysis.get("enhanced_matches", [])

    # ===== TAB NAVIGATION =====
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview & Insights",
        "🔍 Similar Trials Detail",
        "⚔️ Competitive Analysis",
        "📚 Lessons Learned"
    ])

    # ===== TAB 1: OVERVIEW =====
    with tab1:
        st.subheader("Aggregate Insights")

        if insights:
            # Key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Similar Trials", insights.total_similar_trials)
            col2.metric("Success Rate", f"{insights.success_rate:.0f}%")
            col3.metric("Avg Enrollment", f"{insights.avg_enrollment:.0f}")
            col4.metric("Avg Duration", f"{insights.avg_duration_months:.1f}mo" if insights.avg_duration_months else "N/A")
            col5.metric("Avg Sites", f"{insights.avg_sites:.0f}" if insights.avg_sites else "N/A")

            st.divider()

            # Benchmarks
            st.subheader("📈 Your Protocol vs Benchmarks")
            if insights.enrollment_benchmark:
                st.markdown(f"**Enrollment:** {insights.enrollment_benchmark}")
            if insights.duration_benchmark:
                st.markdown(f"**Duration:** {insights.duration_benchmark}")
            if insights.sites_benchmark:
                st.markdown(f"**Sites:** {insights.sites_benchmark}")

            st.divider()

            # Key insights
            st.subheader("💡 Key Insights")
            for insight in insights.key_insights:
                st.markdown(f"• {insight}")

            # Failure reasons
            if insights.common_failure_reasons:
                st.divider()
                st.subheader("⚠️ Common Failure Reasons")
                failure_df = pd.DataFrame(insights.common_failure_reasons)
                if not failure_df.empty:
                    fig = px.bar(failure_df, x='count', y='reason', orientation='h',
                                title="Why Similar Trials Failed")
                    fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

            # Phase distribution
            if insights.phase_distribution:
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Phase Distribution")
                    fig = px.pie(values=list(insights.phase_distribution.values()),
                                names=list(insights.phase_distribution.keys()),
                                title="Similar Trials by Phase", hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if insights.sponsor_distribution:
                        st.subheader("Top Sponsors")
                        sponsor_df = pd.DataFrame([
                            {"Sponsor": k, "Trials": v}
                            for k, v in list(insights.sponsor_distribution.items())[:8]
                        ])
                        fig = px.bar(sponsor_df, x='Trials', y='Sponsor', orientation='h')
                        fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

        # Top recommendations
        if top_recommendations:
            st.divider()
            st.subheader("🎯 Top Recommendations from Similar Trials")
            for i, rec in enumerate(top_recommendations[:5], 1):
                st.markdown(f"**{i}.** {rec}")

    # ===== TAB 2: SIMILAR TRIALS DETAIL =====
    with tab2:
        st.subheader("Similar Trials with Detailed Analysis")

        # Filter
        filter_status = st.selectbox(
            "Filter by Status",
            ["All", "Completed", "Terminated", "Active/Recruiting"],
            key="detail_filter"
        )

        if filter_status == "Completed":
            display_matches = completed
        elif filter_status == "Terminated":
            display_matches = terminated
        elif filter_status == "Active/Recruiting":
            display_matches = active
        else:
            display_matches = enhanced_matches

        for match in display_matches[:20]:
            status_icon = "✅" if match.status == "COMPLETED" else "❌" if match.status in ["TERMINATED", "WITHDRAWN"] else "🔄"

            with st.expander(f"{status_icon} {match.nct_id} - {match.title[:60]}..."):
                col1, col2, col3 = st.columns(3)
                col1.metric("Sponsor", match.sponsor[:30] if match.sponsor else "N/A")
                col2.metric("Enrollment", match.enrollment or "N/A")
                col3.metric("Duration", f"{match.duration_months:.0f}mo" if match.duration_months else "N/A")

                # Similarity breakdown if available
                if match.similarity:
                    st.markdown("**Similarity Breakdown:**")
                    sim = match.similarity
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Condition", f"{sim.condition_score:.0f}%")
                    col2.metric("Intervention", f"{sim.intervention_score:.0f}%")
                    col3.metric("Endpoint", f"{sim.endpoint_score:.0f}%")
                    col4.metric("Eligibility", f"{sim.eligibility_score:.0f}%")

                    if sim.condition_details:
                        st.caption(f"**Condition:** {sim.condition_details}")
                    if sim.intervention_details:
                        st.caption(f"**Intervention:** {sim.intervention_details}")

                st.markdown(f"**Conditions:** {match.conditions[:200] if match.conditions else 'N/A'}")
                st.markdown(f"**Interventions:** {match.interventions[:200] if match.interventions else 'N/A'}")

                if match.why_stopped:
                    st.error(f"**Why Stopped:** {match.why_stopped}")

                # Lessons for this trial
                if match.lessons:
                    st.markdown("**Key Lessons:**")
                    for lesson in match.lessons[:3]:
                        st.markdown(f"• *{lesson.category.title()}:* {lesson.lesson}")
                        st.caption(f"  → {lesson.actionable_recommendation}")

    # ===== TAB 3: COMPETITIVE ANALYSIS =====
    with tab3:
        st.subheader("⚔️ Competitive Landscape")

        if competitive_intel:
            # Summary
            col1, col2, col3 = st.columns(3)
            col1.metric("Active Competitors", competitive_intel.total_active_competitors)
            col2.metric("Direct Competitors", len(competitive_intel.direct_competitors))
            col3.metric("Total Competing Enrollment", f"{competitive_intel.total_competing_enrollment:,}")

            # Competitive position
            position = competitive_intel.your_competitive_position
            if position == "favorable":
                st.success(f"**Your Competitive Position:** {position.upper()} ✅")
            elif position == "challenging":
                st.error(f"**Your Competitive Position:** {position.upper()} ⚠️")
            else:
                st.info(f"**Your Competitive Position:** {position.upper()}")

            if competitive_intel.timeline_analysis:
                st.markdown(f"**Timeline Analysis:** {competitive_intel.timeline_analysis}")

            # Key differentiators
            if competitive_intel.key_differentiators:
                st.divider()
                st.subheader("🎯 Your Key Differentiators")
                for diff in competitive_intel.key_differentiators:
                    st.markdown(f"• {diff}")

            # Competitive risks
            if competitive_intel.competitive_risks:
                st.divider()
                st.subheader("⚠️ Strategic Recommendations")
                for risk in competitive_intel.competitive_risks:
                    st.markdown(f"• {risk}")

            # Direct competitors detail
            if competitive_intel.direct_competitors:
                st.divider()
                st.subheader("🎯 Direct Competitors (Same Patient Population)")

                for comp in competitive_intel.direct_competitors[:10]:
                    with st.expander(f"🔄 {comp.nct_id} - {comp.sponsor[:30] if comp.sponsor else 'Unknown'}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Enrollment Target", comp.enrollment or "N/A")
                        col2.metric("Sites", comp.num_sites or "N/A")
                        col3.metric("Patient Overlap", comp.patient_overlap_estimate)

                        st.markdown(f"**Title:** {comp.title[:100]}...")
                        st.markdown(f"**Conditions:** {comp.conditions[:150] if comp.conditions else 'N/A'}")

                        if comp.competitive_advantage:
                            st.success(f"**Your Advantage:** {comp.competitive_advantage}")
                        if comp.competitive_disadvantage:
                            st.warning(f"**Their Advantage:** {comp.competitive_disadvantage}")

        else:
            st.info("No active competitors found or competitive analysis not available.")

    # ===== TAB 4: LESSONS LEARNED =====
    with tab4:
        st.subheader("📚 Lessons from Similar Trials")

        trial_lessons = analysis.get("trial_lessons", {})

        if trial_lessons:
            # Group lessons by category
            lessons_by_category = {}
            for nct_id, lessons in trial_lessons.items():
                for lesson in lessons:
                    if lesson.category not in lessons_by_category:
                        lessons_by_category[lesson.category] = []
                    lessons_by_category[lesson.category].append(lesson)

            # Display by category
            for category in ["design", "enrollment", "endpoint", "safety", "operational", "regulatory"]:
                if category in lessons_by_category:
                    lessons = lessons_by_category[category]
                    st.subheader(f"📌 {category.title()} Lessons ({len(lessons)})")

                    for lesson in lessons[:5]:
                        confidence_icon = "🟢" if lesson.confidence == "high" else "🟡" if lesson.confidence == "medium" else "⚪"
                        st.markdown(f"{confidence_icon} **{lesson.lesson}**")
                        st.caption(f"→ *Recommendation:* {lesson.actionable_recommendation}")
                        st.caption(f"Source: {lesson.source_nct_id}")
                        st.divider()

        else:
            # Fallback to basic terminated trial lessons
            if terminated:
                st.subheader("⚠️ Lessons from Terminated Trials")
                for t in terminated[:10]:
                    if t.why_stopped:
                        st.markdown(f"**{t.nct_id}** - {t.sponsor[:30] if t.sponsor else 'Unknown'}")
                        st.caption(f"Reason: {t.why_stopped}")
                        st.divider()
            else:
                st.info("No lesson data available. Run deep analysis to extract lessons.")


# ============== LEGACY: TERMINATION RISKS (Merged into Risk Assessment) ==============
def render_termination_risks():
    """Feature 6: Termination risk analysis for protocol's area."""
    st.header("5️⃣ Termination Risk Analysis")

    if not require_protocol():
        return

    p = st.session_state.extracted_protocol
    st.markdown(f"Why **{p.condition}** trials fail - learn from history")

    if st.button("🔍 Analyze Termination Patterns", type="primary"):
        db = get_database()
        if not db:
            return

        with st.spinner("Analyzing terminated trials..."):
            from sqlalchemy import text

            query = text("""
                SELECT nct_id, title, phase, sponsor, enrollment, why_stopped
                FROM trials
                WHERE (LOWER(conditions) LIKE :condition OR LOWER(therapeutic_area) LIKE :condition)
                AND status IN ('TERMINATED', 'WITHDRAWN')
                AND why_stopped IS NOT NULL
                ORDER BY enrollment DESC
                LIMIT 200
            """)

            try:
                results = db.execute_raw(query.text, {"condition": f"%{p.condition.lower()}%"})

                if not results:
                    st.success(f"No documented terminations found for {p.condition}!")
                    return

                st.divider()
                st.subheader(f"Termination Patterns in {p.condition}")
                st.caption(f"Analyzed {len(results)} terminated trials")

                # Categorize
                categories = {
                    "Enrollment Issues": 0,
                    "Efficacy/Futility": 0,
                    "Safety Concerns": 0,
                    "Business/Strategic": 0,
                    "Other": 0
                }

                same_phase = 0
                for r in results:
                    if r[2] == p.phase:
                        same_phase += 1
                    reason = (r[5] or "").lower()
                    if any(w in reason for w in ["enroll", "recruit", "accrual", "patient"]):
                        categories["Enrollment Issues"] += 1
                    elif any(w in reason for w in ["efficacy", "futility", "endpoint", "interim"]):
                        categories["Efficacy/Futility"] += 1
                    elif any(w in reason for w in ["safety", "adverse", "toxicity", "risk"]):
                        categories["Safety Concerns"] += 1
                    elif any(w in reason for w in ["business", "strategic", "funding", "sponsor"]):
                        categories["Business/Strategic"] += 1
                    else:
                        categories["Other"] += 1

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Terminated", len(results))
                col2.metric(f"Same Phase ({p.phase})", same_phase)
                col3.metric("Top Risk", max(categories, key=categories.get))

                col1, col2 = st.columns([1, 2])
                with col1:
                    fig = px.pie(values=list(categories.values()), names=list(categories.keys()),
                                title="Termination Reasons", hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Risk Factors for Your Trial")
                    total = sum(categories.values())
                    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                        pct = count / total * 100 if total > 0 else 0
                        if pct > 25:
                            st.error(f"🔴 **{cat}**: {count} ({pct:.0f}%) - HIGH RISK")
                        elif pct > 15:
                            st.warning(f"🟡 **{cat}**: {count} ({pct:.0f}%) - MODERATE RISK")
                        else:
                            st.success(f"🟢 **{cat}**: {count} ({pct:.0f}%) - LOW RISK")

                # Specific examples from same phase
                st.subheader(f"Terminated {p.phase} Trials to Learn From")
                same_phase_trials = [r for r in results if r[2] == p.phase][:10]
                for r in same_phase_trials:
                    with st.expander(f"{r[0]} - {r[3][:30] if r[3] else 'Unknown'}"):
                        st.write(f"**Enrollment:** {r[4] or 'N/A'}")
                        st.error(f"**Reason:** {r[5]}")

            except Exception as e:
                st.error(f"Analysis failed: {e}")


# ============== FEATURE 5: ELIGIBILITY OPTIMIZATION ==============
def render_eligibility_optimization():
    """Feature 5: Comprehensive eligibility criteria optimization."""
    # Hero section - Jeeva style
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 2.25rem; font-weight: 700; margin-bottom: 0.75rem; color: #182A81;">
            Eligibility Criteria Optimization
        </h1>
        <p style="font-size: 1.05rem; color: #4A4A4A; max-width: 650px; line-height: 1.7;">
            Criterion-level analysis with patient pool impact assessment,
            screen failure prediction, and AI-powered optimization suggestions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not require_protocol():
        return

    p = st.session_state.extracted_protocol

    # Protocol context - Jeeva style
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #EAECFB, #F5F5F5); border-radius: 10px; padding: 0.875rem 1.25rem; margin-bottom: 1.5rem;
                display: inline-flex; align-items: center; gap: 0.75rem; border: 1px solid #E5E5E5;">
        <span style="font-size: 0.85rem; color: #6B7280; font-weight: 500;">Analyzing:</span>
        <span style="font-weight: 700; color: #182A81;">{p.condition}</span>
        <span style="color: #CBD5E1; font-weight: 300;">|</span>
        <span style="color: #F45900; font-weight: 600;">{p.phase}</span>
    </div>
    """, unsafe_allow_html=True)

    # Show current eligibility criteria
    if p.eligibility_criteria:
        with st.expander("📋 View Current Eligibility Criteria", expanded=False):
            st.text(p.eligibility_criteria[:2000] + "..." if len(p.eligibility_criteria or "") > 2000 else p.eligibility_criteria)

    if st.button("✅ Optimize Eligibility Criteria", type="primary", use_container_width=True):
        db = get_database()
        if not db:
            return

        with st.spinner("Analyzing criteria, benchmarking, and generating AI suggestions..."):
            try:
                from src.analysis.eligibility_optimizer import EligibilityOptimizer

                optimizer = EligibilityOptimizer(db)
                report = optimizer.optimize(
                    eligibility_text=p.eligibility_criteria or "",
                    condition=p.condition,
                    phase=p.phase,
                    target_enrollment=p.target_enrollment or 200
                )

                st.session_state.eligibility_optimization = report
                st.success("Eligibility optimization analysis complete!")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display results
    if st.session_state.eligibility_optimization:
        _render_eligibility_results()


def _render_eligibility_results():
    """Render eligibility optimization results."""
    report = st.session_state.eligibility_optimization

    st.divider()

    # Overall assessment
    assessment_color = {
        "Criteria need significant optimization": "🔴",
        "Criteria are moderately complex": "🟡",
        "Criteria are well-balanced": "🟢",
    }
    color = "🟡"
    for key, emoji in assessment_color.items():
        if key in report.overall_assessment:
            color = emoji
            break

    st.markdown(f"### {color} Overall Assessment")
    st.info(report.overall_assessment)

    # Key findings
    if report.key_findings:
        st.subheader("🔑 Key Findings")
        for finding in report.key_findings:
            st.markdown(f"- {finding}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Criteria", report.complexity.total_criteria)
    col2.metric("Complexity Score", f"{report.complexity.overall_complexity:.0f}/100")
    col3.metric("Est. Exclusion Rate", f"{report.patient_pool_impact.estimated_exclusion_rate*100:.0f}%")
    col4.metric("Pred. Screen Failure", f"{report.screen_failure_prediction.predicted_screen_failure_rate*100:.0f}%")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Complexity Analysis",
        "👥 Patient Pool Impact",
        "📋 Parsed Criteria",
        "📈 Benchmark Comparison",
        "💡 AI Suggestions"
    ])

    with tab1:
        _render_complexity_tab(report)

    with tab2:
        _render_patient_pool_tab(report)

    with tab3:
        _render_parsed_criteria_tab(report)

    with tab4:
        _render_benchmark_tab(report)

    with tab5:
        _render_suggestions_tab(report)


def _render_complexity_tab(report):
    """Render complexity analysis tab."""
    complexity = report.complexity

    st.subheader("Criteria Complexity Breakdown")

    col1, col2, col3 = st.columns(3)
    col1.metric("Inclusion Criteria", complexity.inclusion_count)
    col2.metric("Exclusion Criteria", complexity.exclusion_count)
    col3.metric("Benchmark Average", f"{complexity.benchmark_criteria_count:.0f}")

    # Comparison to benchmark
    comparison_colors = {"simpler": "🟢", "similar": "🟡", "more_complex": "🔴"}
    st.markdown(f"**vs Benchmark:** {comparison_colors.get(complexity.vs_benchmark, '⚪')} {complexity.vs_benchmark.replace('_', ' ').title()}")

    # Restrictiveness breakdown
    st.subheader("Restrictiveness Profile")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Highly Restrictive", complexity.high_restrictive_count)
    col2.metric("🟡 Medium", complexity.medium_restrictive_count)
    col3.metric("🟢 Low", complexity.low_restrictive_count)

    # Category breakdown chart
    if complexity.by_category:
        st.subheader("Criteria by Category")
        categories = list(complexity.by_category.keys())
        counts = list(complexity.by_category.values())

        fig = px.bar(
            x=counts, y=categories, orientation='h',
            title="Criteria Distribution by Category",
            labels={"x": "Count", "y": "Category"},
            color=counts, color_continuous_scale="Blues"
        )
        fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # Risk factors
    if complexity.risk_factors:
        st.subheader("⚠️ Risk Factors")
        for rf in complexity.risk_factors:
            st.warning(rf)


def _render_patient_pool_tab(report):
    """Render patient pool impact tab."""
    pool = report.patient_pool_impact

    st.subheader("Patient Pool Funnel")

    # Funnel visualization
    funnel_data = [
        ("Initial Pool", pool.estimated_initial_pool),
        ("After Age Criteria", pool.after_age),
        ("After Comorbidities", pool.after_comorbidities),
        ("After Lab Values", pool.after_lab_values),
        ("After Prior Therapy", pool.after_prior_therapy),
        ("Final Eligible Pool", pool.estimated_eligible_pool),
    ]

    fig = go.Figure(go.Funnel(
        y=[d[0] for d in funnel_data],
        x=[d[1] for d in funnel_data],
        textinfo="value+percent initial",
        marker={"color": ["#3498db", "#2980b9", "#1f618d", "#1a5276", "#154360", "#0e3a5c"]}
    ))
    fig.update_layout(title="Patient Pool Impact by Criteria Category", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Top excluders
    if pool.top_excluders:
        st.subheader("Top Excluding Criteria")
        for i, (criterion, rate) in enumerate(pool.top_excluders, 1):
            st.markdown(f"**{i}.** {criterion}... - excludes ~{rate*100:.0f}% of patients")

    # Recommendations
    if pool.pool_recommendations:
        st.subheader("💡 Pool Optimization Recommendations")
        for rec in pool.pool_recommendations:
            st.info(rec)


def _render_parsed_criteria_tab(report):
    """Render parsed criteria details."""
    st.subheader("Inclusion Criteria")

    if report.inclusion_criteria:
        inc_data = []
        for c in report.inclusion_criteria[:20]:
            inc_data.append({
                "Category": c.category.replace("_", " ").title(),
                "Restrictiveness": c.restrictiveness.title(),
                "Est. Exclusion": f"{c.estimated_exclusion_rate*100:.0f}%",
                "Criterion": c.text[:80] + "..." if len(c.text) > 80 else c.text,
            })
        st.dataframe(pd.DataFrame(inc_data), use_container_width=True, hide_index=True)
    else:
        st.caption("No inclusion criteria parsed")

    st.subheader("Exclusion Criteria")

    if report.exclusion_criteria:
        exc_data = []
        for c in report.exclusion_criteria[:20]:
            exc_data.append({
                "Category": c.category.replace("_", " ").title(),
                "Restrictiveness": c.restrictiveness.title(),
                "Est. Exclusion": f"{c.estimated_exclusion_rate*100:.0f}%",
                "Criterion": c.text[:80] + "..." if len(c.text) > 80 else c.text,
            })
        st.dataframe(pd.DataFrame(exc_data), use_container_width=True, hide_index=True)
    else:
        st.caption("No exclusion criteria parsed")


def _render_benchmark_tab(report):
    """Render benchmark comparison tab."""
    benchmark = report.benchmark

    st.subheader("Benchmark Comparison")

    col1, col2, col3 = st.columns(3)
    col1.metric("Trials Analyzed", benchmark.trials_analyzed)
    col2.metric("Completed Trials", benchmark.completed_trials)
    col3.metric("Alignment Score", f"{benchmark.alignment_score:.0f}/100")

    # Alignment visualization
    alignment_colors = {"well_aligned": "🟢", "somewhat_aligned": "🟡", "misaligned": "🔴"}
    st.markdown(f"**Alignment:** {alignment_colors.get(benchmark.benchmark_alignment, '⚪')} {benchmark.benchmark_alignment.replace('_', ' ').title()}")

    # Comparison table
    st.subheader("Your Criteria vs Successful Trials")

    comparison_data = [
        ("Inclusion Criteria", report.complexity.inclusion_count, f"{benchmark.avg_inclusion_count:.1f}", benchmark.user_vs_benchmark.get("inclusions", "N/A")),
        ("Exclusion Criteria", report.complexity.exclusion_count, f"{benchmark.avg_exclusion_count:.1f}", benchmark.user_vs_benchmark.get("exclusions", "N/A")),
        ("Total Criteria", report.complexity.total_criteria, f"{benchmark.avg_total_criteria:.1f}", "-"),
    ]

    df = pd.DataFrame(comparison_data, columns=["Metric", "Your Trial", "Benchmark Avg", "Comparison"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Common criteria in successful trials
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Common in Successful Trials")
        for criterion, prevalence in benchmark.common_inclusions:
            st.markdown(f"- {criterion} ({prevalence*100:.0f}%)")

    with col2:
        st.subheader("⚠️ Correlated with Failure")
        for item in benchmark.failure_correlated:
            st.markdown(f"- {item}")

    # Screen failure prediction
    sf = report.screen_failure_prediction
    st.divider()
    st.subheader("Screen Failure Prediction")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Rate", f"{sf.predicted_screen_failure_rate*100:.0f}%")
    col2.metric("Benchmark Rate", f"{sf.benchmark_screen_failure*100:.0f}%")
    col3.metric("Confidence", sf.confidence.title())

    if sf.high_risk_criteria:
        st.markdown("**High Risk Criteria:**")
        for crit in sf.high_risk_criteria[:3]:
            st.markdown(f"- {crit}...")


def _render_suggestions_tab(report):
    """Render AI optimization suggestions."""
    st.subheader("AI-Powered Optimization Suggestions")

    if not report.optimization_suggestions:
        st.info("No specific suggestions generated. Your criteria appear well-optimized.")
        return

    # Priority actions first
    if report.priority_actions:
        st.markdown("### 🎯 Priority Actions")
        for i, action in enumerate(report.priority_actions, 1):
            st.error(f"**{i}.** {action}")

    st.divider()

    # All suggestions
    priority_emojis = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    for i, suggestion in enumerate(report.optimization_suggestions, 1):
        emoji = priority_emojis.get(suggestion.priority, "⚪")

        with st.expander(f"{emoji} **{suggestion.suggestion}**", expanded=(suggestion.priority == "high")):
            st.markdown(f"**Priority:** {suggestion.priority.title()}")
            st.markdown(f"**Category:** {suggestion.category.title()}")

            if suggestion.criterion_text:
                st.markdown(f"**Specific Criterion:** {suggestion.criterion_text}")

            st.markdown(f"**Rationale:** {suggestion.rationale}")
            st.markdown(f"**Expected Impact:** {suggestion.expected_impact}")

            if suggestion.evidence_strength:
                st.caption(f"Evidence: {suggestion.evidence_strength.title()} | Supporting trials: {suggestion.supporting_trials}")


# ============== FEATURE 6: ENDPOINT BENCHMARKS ==============
def render_endpoint_benchmarks():
    """Feature 6: Comprehensive endpoint benchmarking and optimization."""
    # Hero section - Jeeva style
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 2.25rem; font-weight: 700; margin-bottom: 0.75rem; color: #182A81;">
            Endpoint Benchmarking
        </h1>
        <p style="font-size: 1.05rem; color: #4A4A4A; max-width: 650px; line-height: 1.7;">
            Comprehensive endpoint analysis with classification, regulatory guidance,
            success pattern analysis, and optimization recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not require_protocol():
        return

    p = st.session_state.extracted_protocol

    # Protocol context - Jeeva style
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #EAECFB, #F5F5F5); border-radius: 10px; padding: 0.875rem 1.25rem; margin-bottom: 1.5rem;
                display: inline-flex; align-items: center; gap: 0.75rem; border: 1px solid #E5E5E5;">
        <span style="font-size: 0.85rem; color: #6B7280; font-weight: 500;">Analyzing:</span>
        <span style="font-weight: 700; color: #182A81;">{p.condition}</span>
        <span style="color: #CBD5E1; font-weight: 300;">|</span>
        <span style="color: #F45900; font-weight: 600;">{p.phase}</span>
    </div>
    """, unsafe_allow_html=True)

    # Show current endpoint
    if p.primary_endpoint:
        with st.expander("🎯 View Current Primary Endpoint", expanded=False):
            st.write(p.primary_endpoint)

    if st.button("🎯 Analyze Endpoints", type="primary", use_container_width=True):
        db = get_database()
        if not db:
            return

        with st.spinner("Analyzing endpoints, benchmarks, and generating recommendations..."):
            try:
                from src.analysis.endpoint_benchmarker import EndpointBenchmarker

                benchmarker = EndpointBenchmarker(db=db)
                report = benchmarker.analyze(
                    primary_endpoint=p.primary_endpoint or "",
                    condition=p.condition,
                    phase=p.phase
                )

                st.session_state.endpoint_benchmark = report
                st.success("Endpoint analysis complete!")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display results
    if st.session_state.endpoint_benchmark:
        _render_endpoint_results()


def _render_endpoint_results():
    """Render endpoint benchmark results."""
    report = st.session_state.endpoint_benchmark

    st.divider()

    # Overall assessment
    score = report.primary_endpoint_score
    if score >= 75:
        score_color = "🟢"
    elif score >= 55:
        score_color = "🟡"
    else:
        score_color = "🔴"

    st.subheader(f"{score_color} Endpoint Assessment")

    col1, col2, col3, col4 = st.columns(4)

    if score >= 75:
        col1.success(f"### {score:.0f}/100\n**Endpoint Score**")
    elif score >= 55:
        col1.warning(f"### {score:.0f}/100\n**Endpoint Score**")
    else:
        col1.error(f"### {score:.0f}/100\n**Endpoint Score**")

    if report.primary_endpoint_classification:
        col2.metric("Category", report.primary_endpoint_classification.primary_category.replace("_", " ").title())
        col3.metric("Regulatory Status", report.primary_endpoint_classification.regulatory_status.title())
        col4.metric("Meaningfulness", report.primary_endpoint_classification.clinical_meaningfulness.title())

    st.info(f"**Assessment:** {report.primary_endpoint_assessment}")

    # Key findings
    if report.key_findings:
        st.subheader("🔑 Key Findings")
        for finding in report.key_findings:
            st.markdown(f"- {finding}")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Benchmarks",
        "📋 Classification",
        "⏱️ Timing",
        "💡 Recommendations",
        "📎 Secondary Endpoints"
    ])

    with tab1:
        _render_benchmarks_tab(report)

    with tab2:
        _render_classification_tab(report)

    with tab3:
        _render_timing_tab(report)

    with tab4:
        _render_endpoint_recommendations_tab(report)

    with tab5:
        _render_secondary_endpoints_tab(report)


def _render_benchmarks_tab(report):
    """Render benchmarks tab."""
    st.subheader("Endpoint Success Benchmarks")

    if not report.benchmarks_by_category:
        st.info("Limited benchmark data available")
        return

    # Success rate chart
    categories = list(report.benchmarks_by_category.keys())
    rates = [report.benchmarks_by_category[c].success_rate for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[c.replace("_", " ").title() for c in categories],
        y=rates,
        text=[f"{r:.0f}%" for r in rates],
        textposition='auto',
        marker_color=['#4caf50' if r >= 60 else '#ff9800' if r >= 45 else '#f44336' for r in rates]
    ))
    fig.update_layout(
        title="Success Rate by Endpoint Type",
        yaxis_title="Completion Rate %",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Highlight best/worst
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Best:** {report.best_performing_category.replace('_', ' ').title()}")
    with col2:
        st.error(f"**Worst:** {report.worst_performing_category.replace('_', ' ').title()}")

    # Detailed table
    st.subheader("Benchmark Details")

    benchmark_data = []
    for cat, b in report.benchmarks_by_category.items():
        benchmark_data.append({
            "Endpoint Type": cat.replace("_", " ").title(),
            "Trials": b.trials_analyzed,
            "Completed": b.completed_trials,
            "Success Rate": f"{b.success_rate:.0f}%",
            "Avg Enrollment": f"{b.avg_enrollment:.0f}",
            "Avg Duration": f"{b.avg_duration_months:.0f} mo",
            "FDA Accepted": "✅" if b.fda_accepted else "⚠️",
        })

    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True, hide_index=True)

    # Phase breakdown
    st.subheader("Phase-Specific Performance")
    for cat, b in list(report.benchmarks_by_category.items())[:3]:
        if b.phase_breakdown:
            with st.expander(f"{cat.replace('_', ' ').title()}"):
                for phase, data in b.phase_breakdown.items():
                    st.write(f"**{phase}:** {data['success_rate']:.0f}% success ({data['count']} trials)")


def _render_classification_tab(report):
    """Render classification tab."""
    st.subheader("Endpoint Classification")

    classification = report.primary_endpoint_classification

    if not classification:
        st.warning("Unable to classify endpoint. Please ensure a primary endpoint is defined.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Endpoint Details**")
        st.write(f"**Text:** {classification.endpoint_text}")
        st.write(f"**Category:** {classification.primary_category.replace('_', ' ').title()}")
        st.write(f"**Subcategory:** {classification.subcategory.replace('_', ' ').title()}")
        st.write(f"**Measurement Type:** {classification.measurement_type.replace('_', ' ').title()}")

        if classification.timeframe:
            st.write(f"**Timeframe:** {classification.timeframe}")

    with col2:
        st.markdown("**Quality Assessment**")

        # Regulatory status
        reg_colors = {"established": "🟢", "acceptable": "🟡", "exploratory": "🔴"}
        st.write(f"**Regulatory Status:** {reg_colors.get(classification.regulatory_status, '⚪')} {classification.regulatory_status.title()}")

        # Clinical meaningfulness
        meaning_colors = {"high": "🟢", "moderate": "🟡", "low": "🔴"}
        st.write(f"**Clinical Meaningfulness:** {meaning_colors.get(classification.clinical_meaningfulness, '⚪')} {classification.clinical_meaningfulness.title()}")

    # Regulatory guidance
    st.divider()
    st.subheader("📋 Regulatory Guidance")

    if report.regulatory_guidance:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**FDA Preferred Endpoints:**")
            fda_endpoints = report.regulatory_guidance.get("FDA", [])
            if fda_endpoints:
                for ep in fda_endpoints:
                    st.markdown(f"- {ep}")
            else:
                st.caption("Not specified")

        with col2:
            st.markdown("**EMA Preferred Endpoints:**")
            ema_endpoints = report.regulatory_guidance.get("EMA", [])
            if ema_endpoints:
                for ep in ema_endpoints:
                    st.markdown(f"- {ep}")
            else:
                st.caption("Not specified")

        notes = report.regulatory_guidance.get("Notes", "")
        if notes:
            st.info(f"**Regulatory Notes:** {notes}")

    # Established endpoints
    if report.established_endpoints:
        st.subheader("Established Endpoints for This Indication")
        for ep in report.established_endpoints:
            st.markdown(f"✅ {ep}")


def _render_timing_tab(report):
    """Render timing analysis tab."""
    st.subheader("Endpoint Timing Analysis")

    timing = report.timing_analysis

    if not timing:
        st.info("Timing analysis not available")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Endpoint Type:** {timing.endpoint_type.replace('_', ' ').title()}")
        st.markdown(f"**Recommended Timing:** {timing.recommended_timing}")
        st.markdown(f"**Typical Range:** {timing.typical_range[0]}-{timing.typical_range[1]} weeks")

    with col2:
        st.markdown(f"**Rationale:** {timing.timing_rationale}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**⚠️ Risks of Assessing Too Early:**")
        for risk in timing.too_early_risks:
            st.warning(risk)

    with col2:
        st.markdown("**⚠️ Risks of Assessing Too Late:**")
        for risk in timing.too_late_risks:
            st.warning(risk)

    # Visual timeline
    st.divider()
    st.subheader("Assessment Window")

    min_weeks, max_weeks = timing.typical_range
    optimal = (min_weeks + max_weeks) / 2

    fig = go.Figure()

    # Range bar
    fig.add_trace(go.Bar(
        x=[max_weeks - min_weeks],
        y=["Assessment Window"],
        orientation='h',
        base=[min_weeks],
        marker_color='#e3f2fd',
        name="Typical Range"
    ))

    # Optimal point
    fig.add_trace(go.Scatter(
        x=[optimal],
        y=["Assessment Window"],
        mode='markers',
        marker=dict(size=20, color='#2196f3', symbol='diamond'),
        name="Optimal"
    ))

    fig.update_layout(
        title="Recommended Assessment Timing (Weeks)",
        xaxis_title="Weeks from Baseline",
        height=200,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_endpoint_recommendations_tab(report):
    """Render endpoint recommendations tab."""
    st.subheader("Endpoint Optimization Recommendations")

    if not report.recommendations:
        st.success("No major recommendations - your endpoint selection appears optimal!")
        return

    priority_emojis = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    for rec in report.recommendations:
        emoji = priority_emojis.get(rec.priority, "⚪")

        with st.expander(f"{emoji} **{rec.recommendation_type.replace('_', ' ').title()}**: {rec.recommended_change[:60]}...", expanded=(rec.priority == "high")):
            st.markdown(f"**Priority:** {rec.priority.title()}")
            st.markdown(f"**Type:** {rec.recommendation_type.replace('_', ' ').title()}")

            if rec.current_endpoint:
                st.markdown(f"**Current:** {rec.current_endpoint[:100]}...")

            st.markdown(f"**Recommendation:** {rec.recommended_change}")
            st.markdown(f"**Rationale:** {rec.rationale}")
            st.markdown(f"**Expected Impact:** {rec.expected_impact}")

            if rec.regulatory_support:
                st.caption(f"Regulatory: {rec.regulatory_support}")


def _render_secondary_endpoints_tab(report):
    """Render secondary endpoints suggestions tab."""
    st.subheader("Recommended Secondary Endpoints")

    if not report.secondary_suggestions:
        st.info("No secondary endpoint suggestions available")
        return

    st.markdown("These secondary endpoints would complement your primary endpoint:")

    for i, suggestion in enumerate(report.secondary_suggestions, 1):
        with st.expander(f"**{i}. {suggestion.endpoint}**"):
            st.markdown(f"**Category:** {suggestion.category.replace('_', ' ').title()}")
            st.markdown(f"**Rationale:** {suggestion.rationale}")
            st.markdown(f"**How it complements primary:** {suggestion.complementarity}")
            st.markdown(f"**Regulatory Value:** {suggestion.regulatory_value}")

            feasibility_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
            st.markdown(f"**Feasibility:** {feasibility_colors.get(suggestion.feasibility, '⚪')} {suggestion.feasibility.title()}")


# ============== MAIN ==============
def render_header():
    """Render the main header."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 3rem 0;">
        <div style="display: inline-flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <div style="width: 56px; height: 56px; background: linear-gradient(135deg, #F45900, #FF6B1A);
                        border-radius: 16px; display: flex; align-items: center; justify-content: center;
                        box-shadow: 0 8px 24px rgba(244, 89, 0, 0.3);">
                <span style="color: white; font-size: 1.75rem;">🧬</span>
            </div>
            <div style="text-align: left;">
                <h1 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.75rem; font-weight: 800;
                           margin: 0; background: linear-gradient(135deg, #0F172A, #314DD8);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                           letter-spacing: -0.03em;">Clintelligence</h1>
                <p style="margin: 0; font-size: 1rem; color: #64748B; font-weight: 500;">Clinical Trial Protocol Intelligence</p>
            </div>
        </div>
        <p style="font-size: 1.15rem; color: #475569; max-width: 600px; margin: 0 auto; line-height: 1.7;">
            AI-powered analysis for clinical trial protocols. Enter your protocol once
            and unlock comprehensive insights across all dimensions.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_about():
    """Render the About section."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #F8FAFC, #EEF2FF); border-radius: 20px;
                padding: 3rem; margin-top: 3rem; text-align: center;">
        <h2 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.75rem; font-weight: 700;
                   color: #0F172A; margin-bottom: 1rem;">About Clintelligence</h2>
        <p style="font-size: 1.05rem; color: #475569; max-width: 700px; margin: 0 auto 2rem; line-height: 1.8;">
            Clintelligence is an AI-powered clinical trial protocol intelligence platform developed by
            <strong style="color: #F45900;">Jeeva Clinical Trials</strong>. Using advanced AI and a database of
            <strong>566,000+ clinical trials</strong>, we help sponsors and CROs optimize their protocols,
            reduce risks, and accelerate enrollment.
        </p>

        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 2rem;">
            <div style="text-align: center;">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.5rem; font-weight: 800; color: #F45900;">566K+</div>
                <div style="font-size: 0.9rem; color: #64748B; font-weight: 500;">Clinical Trials</div>
            </div>
            <div style="text-align: center;">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.5rem; font-weight: 800; color: #314DD8;">7</div>
                <div style="font-size: 0.9rem; color: #64748B; font-weight: 500;">Analysis Modules</div>
            </div>
            <div style="text-align: center;">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.5rem; font-weight: 800; color: #10B981;">AI</div>
                <div style="font-size: 0.9rem; color: #64748B; font-weight: 500;">Powered by Claude</div>
            </div>
        </div>

        <div style="margin-top: 2.5rem; padding-top: 2rem; border-top: 1px solid #E2E8F0;">
            <p style="font-size: 0.9rem; color: #64748B;">
                © 2025 Jeeva Clinical Trials ·
                <a href="https://jeevatrials.com" style="color: #F45900; text-decoration: none; font-weight: 600;">jeevatrials.com</a>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application - Single page layout."""
    # Header
    render_header()

    # Main tabs (removed About)
    tab_home, tab_analysis = st.tabs(["🏠 Home", "📊 Analysis Dashboard"])

    with tab_home:
        render_protocol_entry_single_page()

    with tab_analysis:
        if not st.session_state.protocol_analyzed:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📋</div>
                <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; color: #0F172A; margin-bottom: 0.5rem;">No Protocol Loaded</h3>
                <p style="color: #64748B; max-width: 500px; margin: 0 auto; font-size: 1.1rem;">
                    Go to the Home tab and enter your protocol to unlock the full analysis dashboard.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            render_analysis_dashboard()


def render_protocol_entry_single_page():
    """Protocol entry for single page layout."""
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.75rem; font-weight: 700;
                   color: #0F172A; margin-bottom: 0.5rem;">Enter Your Protocol</h2>
        <p style="color: #64748B; font-size: 1.1rem;">
            Paste your protocol synopsis below for AI-powered analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Full width form
    protocol_text = st.text_area(
        "Protocol Synopsis",
        placeholder="""Paste your complete protocol synopsis here. Include:

• Study Title and Condition/Indication
• Phase (1, 2, 3, or 4)
• Target enrollment number
• Primary endpoint(s)
• Key inclusion/exclusion criteria
• Study duration
• Intervention details
• Comparator (if any)
• Number of planned sites

The more detail you provide, the better the analysis.""",
        height=350,
        label_visibility="collapsed"
    )

    # Settings in expander
    with st.expander("⚙️ Advanced Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            max_similar = st.slider("Max candidates", 200, 1000, 500)
        with c2:
            min_similarity = st.slider("Min similarity %", 20, 60, 40)
        with c3:
            rank_top_n = st.slider("Top N for AI", 50, 200, 100)

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    if st.button("Analyze Protocol", type="primary", use_container_width=True):
            if not protocol_text or len(protocol_text.strip()) < 100:
                st.error("Please enter more protocol details (at least 100 characters)")
                return

            if not os.getenv("ANTHROPIC_API_KEY"):
                st.error("ANTHROPIC_API_KEY not configured")
                return

            db = get_database()
            if not db:
                return

            try:
                with st.spinner("🤖 Analyzing your protocol with AI..."):
                    from src.analysis.protocol_analyzer import ProtocolAnalyzer

                    analyzer = ProtocolAnalyzer()
                    results = analyzer.analyze_and_match(
                        protocol_text=protocol_text,
                        db_manager=db,
                        include_site_recommendations=True,
                        min_similarity=min_similarity,
                        max_candidates=max_similar,
                        semantic_rank_top_n=rank_top_n
                    )

                # Store results
                st.session_state.extracted_protocol = results["extracted_protocol"]
                st.session_state.protocol_metrics = results["metrics"]
                st.session_state.risk_assessment = results["risk_assessment"]
                st.session_state.similar_trials = results.get("similar_trials", [])
                st.session_state.site_recommendations = results.get("site_recommendations", [])
                st.session_state.matching_context = results.get("matching_context")
                st.session_state.protocol_analyzed = True
                st.session_state.protocol_text = protocol_text

                st.success("✅ Protocol analyzed! Go to the **Analysis Dashboard** tab to explore insights.")
                st.balloons()

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

        # Show extracted info if available
        if st.session_state.protocol_analyzed and st.session_state.extracted_protocol:
            p = st.session_state.extracted_protocol
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FFF7ED, #FFEDD5); border: 2px solid #F45900;
                        border-radius: 16px; padding: 1.5rem; margin-top: 2rem; text-align: center;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #F45900; text-transform: uppercase;
                           letter-spacing: 0.1em; margin-bottom: 0.75rem;">Protocol Loaded</div>
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.25rem; font-weight: 700;
                           color: #0F172A; margin-bottom: 0.5rem;">{p.condition}</div>
                <div style="color: #64748B;">{p.phase} · {p.target_enrollment:,} patients · {p.intervention_type}</div>
            </div>
            """, unsafe_allow_html=True)


def render_analysis_dashboard():
    """Render the analysis dashboard with all features."""
    p = st.session_state.extracted_protocol

    # Protocol summary bar
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0F172A, #1E293B); border-radius: 16px;
                padding: 1.25rem 2rem; margin-bottom: 2rem; display: flex; align-items: center;
                justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
        <div>
            <div style="font-size: 0.75rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;
                       letter-spacing: 0.1em;">Active Protocol</div>
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 700;
                       color: white;">{p.condition}</div>
        </div>
        <div style="display: flex; gap: 2rem;">
            <div style="text-align: center;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #F45900;">{p.phase}</div>
                <div style="font-size: 0.7rem; color: #94A3B8;">Phase</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #10B981;">{p.target_enrollment:,}</div>
                <div style="font-size: 0.7rem; color: #94A3B8;">Patients</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #3B82F6;">{len(st.session_state.similar_trials)}</div>
                <div style="font-size: 0.7rem; color: #94A3B8;">Similar Trials</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Analysis modules in tabs
    analysis_tabs = st.tabs([
        "⭐ Protocol Optimization",
        "📊 Risk Analysis",
        "🌍 Site Intelligence",
        "📈 Enrollment Forecast",
        "🔍 Similar Trials",
        "✅ Eligibility",
        "🎯 Endpoints"
    ])

    with analysis_tabs[0]:
        render_protocol_optimization_content()

    with analysis_tabs[1]:
        render_risk_assessment_content()

    with analysis_tabs[2]:
        render_site_intelligence_content()

    with analysis_tabs[3]:
        render_enrollment_forecast_content()

    with analysis_tabs[4]:
        render_similar_trials_content()

    with analysis_tabs[5]:
        render_eligibility_content()

    with analysis_tabs[6]:
        render_endpoint_content()


# ===== SIMPLIFIED CONTENT RENDERERS =====
def render_protocol_optimization_content():
    """Protocol optimization content."""
    p = st.session_state.extracted_protocol
    metrics = st.session_state.protocol_metrics
    similar_trials = st.session_state.similar_trials

    st.markdown("""
    <p style="color: #64748B; margin-bottom: 1.5rem;">
        AI-powered comprehensive analysis including design evaluation, regulatory checks, and recommendations.
    </p>
    """, unsafe_allow_html=True)

    if st.button("🚀 Generate Optimization Report", type="primary", key="opt_btn"):
        with st.spinner("Analyzing protocol..."):
            try:
                from src.analysis.enhanced_protocol_optimizer import EnhancedProtocolOptimizer
                db = get_database()
                optimizer = EnhancedProtocolOptimizer(db=db)
                report = optimizer.optimize(
                    extracted_protocol=p,
                    similar_trials=similar_trials,
                    metrics=metrics,
                    matching_context=st.session_state.get("matching_context")
                )
                st.session_state.optimization_report = report
            except Exception as e:
                st.error(f"Failed: {e}")
                return

    if st.session_state.optimization_report:
        _render_enhanced_optimization_report(st.session_state.optimization_report)


def render_risk_assessment_content():
    """Risk assessment content."""
    p = st.session_state.extracted_protocol
    metrics = st.session_state.protocol_metrics

    st.markdown("""
    <p style="color: #64748B; margin-bottom: 1.5rem;">
        Termination pattern analysis, risk identification, and mitigation strategies.
    </p>
    """, unsafe_allow_html=True)

    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Similar Trials", metrics.get('total_similar', 0))
    col2.metric("Success Rate", f"{metrics.get('completion_rate', 0):.0f}%")
    col3.metric("Termination Rate", f"{metrics.get('termination_rate', 0):.0f}%")
    col4.metric("Avg Duration", f"{metrics.get('avg_duration_months', 0):.0f} mo")

    if st.button("🔬 Run Deep Risk Analysis", type="primary", key="risk_btn"):
        db = get_database()
        if db:
            with st.spinner("Analyzing risks..."):
                try:
                    from src.analysis.risk_analyzer import RiskAnalyzer
                    analyzer = RiskAnalyzer(db)
                    assessment = analyzer.analyze(
                        condition=p.condition, phase=p.phase,
                        target_enrollment=p.target_enrollment, num_sites=30,
                        intervention=f"{p.intervention_type}",
                        endpoint=p.primary_endpoint,
                        eligibility_criteria=p.eligibility_criteria or "",
                        use_ai_analysis=True
                    )
                    st.session_state.risk_assessment_detailed = assessment
                except Exception as e:
                    st.error(f"Failed: {e}")

    if st.session_state.get("risk_assessment_detailed"):
        _render_detailed_risk_assessment(st.session_state.risk_assessment_detailed)


def render_site_intelligence_content():
    """Site intelligence content."""
    p = st.session_state.extracted_protocol

    st.markdown("""
    <p style="color: #64748B; margin-bottom: 1.5rem;">
        Site selection analysis with performance scoring and portfolio optimization.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        target_sites = st.number_input("Target Sites", min_value=5, value=30, step=5, key="site_num")
    with col2:
        country_filter = st.selectbox("Country", ["All", "United States", "Germany", "UK", "France", "Canada"], key="site_country")

    if st.button("🌍 Generate Site Intelligence", type="primary", key="site_btn"):
        db = get_database()
        if db:
            with st.spinner("Analyzing sites..."):
                try:
                    from src.analysis.site_intelligence import SiteIntelligenceAnalyzer
                    analyzer = SiteIntelligenceAnalyzer(db)
                    report = analyzer.analyze(
                        condition=p.condition, phase=p.phase,
                        target_enrollment=p.target_enrollment,
                        target_sites=target_sites,
                        country_filter=country_filter if country_filter != "All" else None
                    )
                    st.session_state.site_intelligence = report
                except Exception as e:
                    st.error(f"Failed: {e}")

    if st.session_state.site_intelligence:
        _render_site_intelligence_results()


def render_enrollment_forecast_content():
    """Enrollment forecast content."""
    p = st.session_state.extracted_protocol

    st.markdown("""
    <p style="color: #64748B; margin-bottom: 1.5rem;">
        S-curve modeling with risk-adjusted enrollment projections.
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        target = st.number_input("Target", min_value=10, value=p.target_enrollment or 200, key="enroll_target")
    with col2:
        num_sites = st.number_input("Sites", min_value=1, value=30, key="enroll_sites")
    with col3:
        screen_failure = st.slider("Screen Failure", 0.1, 0.5, 0.25, key="enroll_sf")

    if st.button("📈 Generate Forecast", type="primary", key="enroll_btn"):
        db = get_database()
        if db:
            with st.spinner("Forecasting..."):
                try:
                    from src.analysis.enrollment_forecaster import EnrollmentForecaster
                    forecaster = EnrollmentForecaster(db)
                    forecast = forecaster.forecast(
                        target_enrollment=target, num_sites=num_sites,
                        condition=p.condition, phase=p.phase,
                        screen_failure_rate=screen_failure
                    )
                    st.session_state.enrollment_forecast = forecast
                except Exception as e:
                    st.error(f"Failed: {e}")

    if st.session_state.enrollment_forecast:
        _render_enrollment_forecast_results(st.session_state.enrollment_forecast)


def render_similar_trials_content():
    """Similar trials content."""
    similar = st.session_state.similar_trials

    st.markdown(f"""
    <p style="color: #64748B; margin-bottom: 1.5rem;">
        Found <strong>{len(similar)}</strong> similar trials. Run deep analysis for insights.
    </p>
    """, unsafe_allow_html=True)

    if st.button("🔍 Run Deep Analysis", type="primary", key="similar_btn"):
        with st.spinner("Analyzing..."):
            try:
                from src.analysis.similar_trials_analyzer import SimilarTrialsAnalyzer
                db = get_database()
                analyzer = SimilarTrialsAnalyzer(db)
                analysis = analyzer.analyze(
                    similar_trials=similar,
                    protocol=st.session_state.extracted_protocol,
                    matching_context=st.session_state.matching_context
                )
                st.session_state.enhanced_analysis = analysis
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.session_state.enhanced_analysis:
        _render_enhanced_similar_trials_analysis(st.session_state.enhanced_analysis)
    else:
        _render_similar_trials_quick_view(similar)


def render_eligibility_content():
    """Eligibility content."""
    p = st.session_state.extracted_protocol

    st.markdown("""
    <p style="color: #64748B; margin-bottom: 1.5rem;">
        Criterion-level analysis with patient pool impact and optimization suggestions.
    </p>
    """, unsafe_allow_html=True)

    if p.eligibility_criteria:
        with st.expander("📋 Current Criteria", expanded=False):
            st.text(p.eligibility_criteria[:1500] + "..." if len(p.eligibility_criteria or "") > 1500 else p.eligibility_criteria)

    if st.button("✅ Optimize Eligibility", type="primary", key="elig_btn"):
        db = get_database()
        if db:
            with st.spinner("Optimizing..."):
                try:
                    from src.analysis.eligibility_optimizer import EligibilityOptimizer
                    optimizer = EligibilityOptimizer(db)
                    report = optimizer.optimize(
                        eligibility_text=p.eligibility_criteria or "",
                        condition=p.condition, phase=p.phase,
                        target_enrollment=p.target_enrollment or 200
                    )
                    st.session_state.eligibility_optimization = report
                except Exception as e:
                    st.error(f"Failed: {e}")

    if st.session_state.eligibility_optimization:
        _render_eligibility_results(st.session_state.eligibility_optimization)


def render_endpoint_content():
    """Endpoint content."""
    p = st.session_state.extracted_protocol

    st.markdown("""
    <p style="color: #64748B; margin-bottom: 1.5rem;">
        Endpoint classification, regulatory guidance, and optimization recommendations.
    </p>
    """, unsafe_allow_html=True)

    if p.primary_endpoint:
        with st.expander("🎯 Current Endpoint", expanded=False):
            st.write(p.primary_endpoint)

    if st.button("🎯 Analyze Endpoints", type="primary", key="endpoint_btn"):
        db = get_database()
        if db:
            with st.spinner("Analyzing..."):
                try:
                    from src.analysis.endpoint_benchmarker import EndpointBenchmarker
                    benchmarker = EndpointBenchmarker(db=db)
                    report = benchmarker.benchmark(
                        primary_endpoint=p.primary_endpoint or "",
                        condition=p.condition, phase=p.phase
                    )
                    st.session_state.endpoint_benchmark = report
                except Exception as e:
                    st.error(f"Failed: {e}")

    if st.session_state.endpoint_benchmark:
        _render_endpoint_results(st.session_state.endpoint_benchmark)


if __name__ == "__main__":
    main()
