"""
app.py - Streamlit Application Entry Point
AGENTIC AI IN MEDICAL DIAGNOSIS (M.Tech CSE Final-Year Project)

Main application with 8-page navigation, healthcare CSS theming,
database initialization, and system status indicators.
"""

import os
import streamlit as st

import config
from database.database import init_db
from ui.icons import get_icon_svg, section_header_html, top_clinical_bar_html
from ui.home import render_home_page
from ui.chatbot import render_chatbot_page
from ui.report_view import render_report_page
from ui.hospital_view import render_hospital_page
from ui.appointment_view import render_appointment_page

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title=f"{config.APP_NAME} - Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# GLOBAL HEALTHCARE DESIGN SYSTEM
# ==============================================================================
css_path = os.path.join(os.path.dirname(__file__), "ui", "common.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==============================================================================
# DATABASE INITIALIZATION
# ==============================================================================
try:
    init_db()
except Exception as e:
    st.error(f"Database initialization warning: {e}")

# ==============================================================================
# SIDEBAR — BRAND & NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand-box">
            <div class="sidebar-brand-icon">
                {get_icon_svg("activity", size=20, color="#ffffff")}
            </div>
            <div>
                <div style="color:#172B3A; font-size:1.1rem; font-weight:700; line-height:1.2;">
                    {config.APP_NAME}
                </div>
                <div style="color:#526273; font-size:0.75rem; font-weight:500;">
                    Clinical Decision Support v{config.APP_VERSION}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Page state management
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"

    nav_items = [
        ("Home", ":material/home:"),
        ("Symptom Checker", ":material/stethoscope:"),
        ("Medical Report", ":material/description:"),
        ("Healthcare Center", ":material/local_hospital:"),
        ("Appointments", ":material/event:"),
        ("About", ":material/info:"),
        ("Contact", ":material/mail:"),
    ]

    for page_name, icon_name in nav_items:
        is_active = (
            st.session_state["current_page"] == page_name
            or (page_name == "Healthcare Center" and st.session_state["current_page"] == "Find Hospitals")
        )
        btn_key = f"nav_{page_name.lower().replace(' ', '_')}"
        if st.button(
            page_name,
            key=btn_key,
            icon=icon_name,
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state["current_page"] = page_name
            st.rerun()

    st.markdown(
        "<br><hr style='border:none; border-top:1px solid #DFE5EB; margin:14px 0;'>",
        unsafe_allow_html=True,
    )

    # System Status Indicators
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
            {get_icon_svg("sliders", size=15, color="#176B67")}
            <span style="font-size:0.8rem; font-weight:700; color:#172B3A; text-transform:uppercase; letter-spacing:0.04em;">
                System Status
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_ok = config.DISEASE_MODEL_PATH.exists()
    model_dot = "active" if model_ok else "warning"
    model_label = "ML Model Active (100 Classes)" if model_ok else "Knowledge Base Fallback"

    demo_dot = "warning" if config.DEMO_MODE else "active"
    demo_label = "Demonstration Mode" if config.DEMO_MODE else "Live Operating Mode"

    gemini_dot = "active" if config.GEMINI_API_KEY else "neutral"
    gemini_label = "Gemini Reasoning Online" if config.GEMINI_API_KEY else "Offline Clinical Fallback"

    st.markdown(
        f"""
        <div style="background:#FFFFFF; border:1px solid #DFE5EB; border-radius:6px; padding:12px; font-size:0.78rem; line-height:1.8; color:#172B3A;">
            <div><span class="status-dot {model_dot}"></span><strong>Engine:</strong> {model_label}</div>
            <div><span class="status-dot {gemini_dot}"></span><strong>Reasoning:</strong> {gemini_label}</div>
            <div><span class="status-dot {demo_dot}"></span><strong>Environment:</strong> {demo_label}</div>
            <div><span class="status-dot active"></span><strong>Pipeline:</strong> LangGraph (9 Nodes)</div>
            <div><span class="status-dot active"></span><strong>Database:</strong> SQLite + SQLAlchemy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<br><hr style='border:none; border-top:1px solid #DFE5EB; margin:14px 0;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="notice-box" style="margin:0; padding:10px 12px; font-size:0.75rem;">
            <strong>Notice:</strong> Clinical decision-support prototype for educational and research evaluation.
            Not an autonomous diagnostic practitioner.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# PAGE RENDERERS
# ==============================================================================

def render_how_it_works():
    st.markdown(
        section_header_html(
            "cpu",
            "How It Works — Agentic AI Pipeline Architecture",
            "Coordinated 9-Node LangGraph state machine with deterministic safety guardrails",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        The system processes user symptom descriptions through a coordinated **9-Node LangGraph pipeline**
        integrating deterministic emergency rules, Scikit-Learn clinical classification, and
        structured decision routing.
        """
    )

    st.markdown(
        """
        ```
        User Clinical Presentation (Natural Language Input)
                 |
        [Node 1: Symptom Extraction Agent]
                 Entity extraction, fuzzy matching & synonym normalization
                 |
        [Node 2: Emergency Pre-Check — DETERMINISTIC GUARDRAIL]
                 Red-flag screening (cardiac distress, FAST stroke, acute dyspnea)
                 | (if emergency: bypass follow-up questions)
        [Node 3: Follow-Up Question Agent]
                 Targeted clinical inquiry (symptom duration, severity, fever)
                 Pause/Resume: Graph pauses for patient response
                 |
        [Node 4: Disease Analysis Agent]
                 LogisticRegression inference on 230-feature binary vector
                 Trained on 96,088 clinical observations across 100 conditions
                 |
        [Node 5: Risk Assessment Agent — DETERMINISTIC]
                 LOW / MODERATE / HIGH / CRITICAL stratification
                 |
        [Node 6: Emergency Post-Check — DETERMINISTIC GUARDRAIL]
                 Post-inference audit; enforces safety invariants
                 |
        [Node 7: Recommendations Agent]
                 Evidence-based supportive care and clinical guidance
                 |
        [Node 8: Report Generation Agent]
                 Structured clinical summary + ReportLab PDF compilation
                 |
        [Conditional Routing Gate: Risk Level]
                 LOW / MODERATE --> END
                 HIGH / CRITICAL --> Node 9
                 |
        [Node 9: Hospital Offer]
                 OpenStreetMap/Overpass facility geospatial discovery
        ```
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="clinical-card">
                <div class="clinical-card-header">
                    <span style="font-weight:600; color:#172B3A; display:flex; align-items:center; gap:8px;">
                        {get_icon_svg("shield-check", size=18, color="#176B67")} Safety Architecture
                    </span>
                    <span class="badge badge-brand">Deterministic</span>
                </div>
                <ul style="margin:0; padding-left:20px; font-size:13.5px; color:#526273; line-height:1.7;">
                    <li><strong>Hard Guardrails:</strong> Rule-based screening executes prior to and following LLM inference</li>
                    <li><strong>CRITICAL Risk Lock:</strong> Emergency status cannot be downgraded by generative reasoning</li>
                    <li><strong>Emergency Dispatch Contacts:</strong> Direct routing to 112 (National Emergency) and 108 (Ambulance)</li>
                    <li><strong>Offline Resilience:</strong> Deterministic fallbacks guarantee operation when external APIs are offline</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="clinical-card">
                <div class="clinical-card-header">
                    <span style="font-weight:600; color:#172B3A; display:flex; align-items:center; gap:8px;">
                        {get_icon_svg("cpu", size=18, color="#176B67")} AI Components
                    </span>
                    <span class="badge badge-brand">Hybrid AI</span>
                </div>
                <ul style="margin:0; padding-left:20px; font-size:13.5px; color:#526273; line-height:1.7;">
                    <li><strong>ML Model:</strong> Multi-class LogisticRegression classifier evaluated on 230 features</li>
                    <li><strong>Generative LLM:</strong> Google Gemini for empathetic non-diagnostic explanations</li>
                    <li><strong>Orchestration:</strong> LangGraph StateGraph state machine with cyclic pause/resume</li>
                    <li><strong>Persistence:</strong> SQLAlchemy ORM with SQLite relational schema</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="notice-box">
            <strong>Clinical Disclaimer:</strong> {config.CLINICAL_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about():
    about_header = "".join(
        line.strip()
        for line in section_header_html(
            title="About — Agentic AI in Medical Diagnosis",
            icon="info",
            subtitle="Academic capstone project exploring clinical decision support architecture",
        ).splitlines()
    )
    st.markdown(about_header, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            f"""
            ### {config.APP_NAME}
            **{config.APP_SUBTITLE}**

            **Academic Context:** M.Tech Computer Science & Engineering Final-Year Capstone Project

            #### 1. What is Agentic AI in Medical Diagnosis?
            **Agentic AI in Medical Diagnosis** represents an autonomous, modular multi-agent software architecture
            where specialized artificial intelligence agents collaborate through a typed, deterministic state machine
            (LangGraph) to provide structured clinical decision support. Rather than relying on a single monolithic
            black-box model, the system decouples clinical triage into distinct agents responsible for entity extraction,
            emergency safety gating, empirical statistical classification, risk stratification, clinical guidance,
            automated PDF reporting, and healthcare facility discovery.

            #### 2. Project Purpose & Academic Objectives
            This academic capstone investigates the practical convergence of:
            - **Deterministic Safety Engineering**: Hard rule-based guardrails that supersede probabilistic models.
            - **Empirical Machine Learning**: Transparent, calibrated classification on standardized medical datasets.
            - **Coordinated Agent State Machines**: LangGraph directed acyclic and cyclic graphs with pause/resume capabilities.
            - **Closed-Loop Care Delivery**: Bridging initial symptom reporting with structured documentation and facility geolocation.

            #### 3. How the Clinical Pipeline Operates
            - **Symptom Extraction & Normalization**: Patient natural-language descriptions are tokenized and matched against
              a medical vocabulary of 230 standardized clinical features using fuzzy string matching and synonym dictionaries.
            - **Deterministic Emergency Screening**: Node 2 executes pre-inference red-flag pattern matching (cardiac distress,
              FAST stroke signs, acute respiratory failure). Urgent flags trigger immediate emergency routing.
            - **Targeted Follow-Up Inquiries**: The graph pauses execution to collect critical context (symptom duration,
              progression, fever severity) before resuming inference.
            - **Machine Learning Classification**: A multi-class LogisticRegression classifier evaluates the 230-feature binary
              vector across 100 disease classes, trained and evaluated on 96,088 clinical observations.
            - **Risk Stratification**: Patients are stratified into LOW, MODERATE, HIGH, or CRITICAL risk tiers based on
              symptom severity, persistence, and condition likelihood.
            - **Emergency Post-Check Audit**: Node 6 audits the state post-inference, ensuring CRITICAL risk cannot be
              downgraded by generative LLM reasoning.
            - **Actionable Health Guidance**: Evidence-based supportive care recommendations (diet, hydration, activities to
              avoid, physician consultation urgency) are compiled from clinical knowledge bases and augmented with Gemini LLM.
            - **Automated Report Generation**: ReportLab compiles an auditable 15-section structured clinical report into a
              downloadable PDF document including patient details and disclaimers.
            - **Facility Discovery & Appointments**: OpenStreetMap/Overpass API geolocates verified nearby healthcare facilities,
              allowing patients to initiate consultation booking requests stored locally in SQLite.

            #### 4. System Limitations & Clinical Non-Diagnostic Scope
            - **Not a Diagnostic Device**: The application provides non-diagnostic clinical decision support for educational evaluation.
            - **Statistical Likelihood vs. Certainty**: Model probabilities represent statistical pattern similarity on training data,
              not verified pathology.
            - **In-Person Care Requirement**: The system cannot perform physical examinations, auscultation, or laboratory diagnostics.
              Patients must always consult licensed medical practitioners.
            """
        )
    with col2:
        st.markdown(
            f"""
            <div class="clinical-card">
                <div class="clinical-card-header">
                    <span style="font-weight:600; color:#172B3A; display:flex; align-items:center; gap:8px;">
                        {get_icon_svg("bar-chart", size=18, color="#176B67")} Project Metrics
                    </span>
                </div>
                <table style="width:100%; font-size:13px; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #DFE5EB;"><td style="padding:8px 0; color:#526273;">Dataset Records</td><td style="text-align:right; font-weight:600; color:#172B3A;">96,088</td></tr>
                    <tr style="border-bottom:1px solid #DFE5EB;"><td style="padding:8px 0; color:#526273;">Disease Classes</td><td style="text-align:right; font-weight:600; color:#172B3A;">100</td></tr>
                    <tr style="border-bottom:1px solid #DFE5EB;"><td style="padding:8px 0; color:#526273;">Symptom Features</td><td style="text-align:right; font-weight:600; color:#172B3A;">230</td></tr>
                    <tr style="border-bottom:1px solid #DFE5EB;"><td style="padding:8px 0; color:#526273;">Agent Nodes</td><td style="text-align:right; font-weight:600; color:#172B3A;">9</td></tr>
                    <tr style="border-bottom:1px solid #DFE5EB;"><td style="padding:8px 0; color:#526273;">Database Tables</td><td style="text-align:right; font-weight:600; color:#172B3A;">4</td></tr>
                    <tr><td style="padding:8px 0; color:#526273;">Application Pages</td><td style="text-align:right; font-weight:600; color:#172B3A;">8</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="clinical-card" style="margin-top:15px;">
                <div class="clinical-card-header">
                    <span style="font-weight:600; color:#172B3A; display:flex; align-items:center; gap:8px;">
                        {get_icon_svg("shield-check", size=18, color="#176B67")} Clinical Guardrails
                    </span>
                </div>
                <ul style="margin:0; padding-left:18px; font-size:12.5px; color:#526273; line-height:1.7;">
                    <li>Pre-check & post-check deterministic gates</li>
                    <li>CRITICAL risk lock prevents downgrade</li>
                    <li>Zero storage of sensitive payment credentials</li>
                    <li>Explicit safe medical wording throughout</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="margin:20px 0 10px 0;">
            <div style="font-size:16px; font-weight:600; color:#172B3A; display:flex; align-items:center; gap:8px;">
                {get_icon_svg("layers", size=18, color="#176B67")} Implemented Technology Stack
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tech_cols = st.columns(4)
    techs = [
        ("Python 3.10+", "Core Language Runtime", "cpu"),
        ("Streamlit", "Responsive Web Framework", "activity"),
        ("LangGraph", "Agent State Orchestration", "git-branch"),
        ("Scikit-Learn", "Multi-Class ML Model", "sliders"),
        ("Google Gemini", "Clinical Reasoning LLM", "cpu"),
        ("SQLAlchemy", "ORM Relational Database", "database"),
        ("ReportLab", "Automated PDF Generation", "file-text"),
        ("OpenStreetMap", "Facility Discovery API", "map-pin"),
    ]
    for i, (name, desc, icon_name) in enumerate(techs):
        with tech_cols[i % 4]:
            st.markdown(
                f"""
                <div class="clinical-card" style="padding:14px; text-align:left; margin-bottom:10px;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                        {get_icon_svg(icon_name, size=16, color="#176B67")}
                        <span style="font-weight:600; color:#172B3A; font-size:13.5px;">{name}</span>
                    </div>
                    <div style="color:#526273; font-size:12px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="notice-box">
            <strong>Educational Disclaimer:</strong> {config.CLINICAL_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_contact():
    contact_header = "".join(
        line.strip()
        for line in section_header_html(
            title="Contact & Academic Support",
            icon="contact",
            subtitle="Institutional project information and emergency contact protocols",
        ).splitlines()
    )
    st.markdown(contact_header, unsafe_allow_html=True)

    c_info, c_emerg = st.columns([1.6, 1])

    with c_info:
        st.markdown(
            f"""
            <div class="clinical-card">
                <div class="clinical-card-header">
                    <span style="font-weight:600; color:#172B3A;">Academic Project Metadata</span>
                </div>
                <table style="width:100%; font-size:13.5px; line-height:2.2;">
                    <tr><td style="color:#526273; width:35%;">Project Title</td><td style="font-weight:600; color:#172B3A;">{config.APP_NAME}</td></tr>
                    <tr><td style="color:#526273;">Subtitle</td><td>{config.APP_SUBTITLE}</td></tr>
                    <tr><td style="color:#526273;">Version</td><td>{config.APP_VERSION}</td></tr>
                    <tr><td style="color:#526273;">Academic Level</td><td>M.Tech CSE Capstone Project</td></tr>
                    <tr><td style="color:#526273;">Department</td><td>Computer Science & Engineering</td></tr>
                    <tr><td style="color:#526273;">Primary Focus</td><td>Agentic Clinical Decision Support</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="clinical-card" style="margin-top:16px;">
                <div class="clinical-card-header">
                    <span style="font-weight:600; color:#172B3A;">Academic Project Help & Technical Support</span>
                    <span class="badge badge-brand">Project Support</span>
                </div>
                <div style="font-size:13.5px; color:#334155; line-height:1.7; padding-top:4px;">
                    <p style="margin:0 0 10px 0;">
                        For academic inquiries, codebase inspection, or prototype demonstration feedback, contact the project team:
                    </p>
                    <table style="width:100%; font-size:13.5px; line-height:2.0; margin-bottom:10px;">
                        <tr>
                            <td style="color:#526273; width:35%;"><strong>Support Phone:</strong></td>
                            <td style="color:#0369a1; font-weight:600;">+91-XXXXXXXXXX</td>
                        </tr>
                        <tr>
                            <td style="color:#526273;"><strong>Support Email:</strong></td>
                            <td style="color:#0369a1; font-weight:600;">support@example.com</td>
                        </tr>
                    </table>
                    <div style="background:#f1f5f9; border-left:3px solid #0284c7; padding:8px 12px; border-radius:4px; font-size:12px; color:#475569; margin-bottom:12px;">
                        <em>Note: The phone number and email above are sample placeholders for academic project evaluation. Replace with official institutional contact information prior to public deployment.</em>
                    </div>
                    <div style="color:#475569; font-size:12.5px;">
                        <strong>Supported Query Categories:</strong>
                        <ul style="margin:4px 0 0 0; padding-left:18px;">
                            <li>Academic research methodology & algorithm verification</li>
                            <li>Technical bug reports and Python runtime environment support</li>
                            <li>Feedback on symptom extraction and multi-agent clinical workflow</li>
                            <li>Prototype appointment and facility discovery inquiries</li>
                        </ul>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_emerg:
        st.markdown(
            f"""
            <div class="emergency-banner" style="margin-top:0;">
                <h3>
                    {get_icon_svg("alert-octagon", size=18, color="#B42318")} Acute Medical Emergency
                </h3>
                <p style="font-size:13px; color:#7A1C14; line-height:1.5; margin-bottom:12px;">
                    If you or someone nearby is experiencing acute symptoms (chest pain, stroke signs, difficulty breathing),
                    contact emergency services immediately:
                </p>
                <div style="background:#FFFFFF; border:1px solid #FECDCA; border-radius:4px; padding:10px 14px; margin-bottom:8px;">
                    <div style="font-size:11.5px; color:#526273; text-transform:uppercase; font-weight:600;">National Emergency Services</div>
                    <div style="font-size:18px; font-weight:700; color:#B42318;">112</div>
                </div>
                <div style="background:#FFFFFF; border:1px solid #FECDCA; border-radius:4px; padding:10px 14px;">
                    <div style="font-size:11.5px; color:#526273; text-transform:uppercase; font-weight:600;">Ambulance Dispatch</div>
                    <div style="font-size:18px; font-weight:700; color:#B42318;">108</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="notice-box">
            <strong>Medical Disclaimer:</strong> {config.CLINICAL_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE ROUTING
# ==============================================================================
current = st.session_state["current_page"]

# Render top clinical breadcrumb & context bar
st.markdown(top_clinical_bar_html(current), unsafe_allow_html=True)

if current == "Home":
    render_home_page()
elif current == "Symptom Checker":
    render_chatbot_page()
elif current == "Medical Report":
    render_report_page()
elif current in ("Healthcare Center", "Find Hospitals"):
    render_hospital_page()
elif current == "Appointments":
    render_appointment_page()
elif current == "How It Works":
    render_how_it_works()
elif current == "About":
    render_about()
elif current == "Contact":
    render_contact()

