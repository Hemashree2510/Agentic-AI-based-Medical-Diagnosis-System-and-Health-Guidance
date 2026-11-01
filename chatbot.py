"""
ui/chatbot.py - AI Symptom Checker Streamlit Page
AGENTIC AI IN MEDICAL DIAGNOSIS (M.Tech CSE Final-Year Project)

Connects user input to the LangGraph medical workflow.
Uses MedicalState keys as defined in workflow/state.py.
"""

import os
import streamlit as st

import config
from workflow.graph import run_medical_workflow
from workflow.state import create_initial_state

# Import shared stylesheet
with open(os.path.join(os.path.dirname(__file__), "common.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Remove emojis from page title later in the file (header will be updated)


# ============================================================
# CSS STYLING
# ============================================================

def inject_chatbot_css():
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff !important; }
        .main .block-container { max-width: 1200px; padding-top: 1.5rem; padding-bottom: 3rem; }
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown div { color: #1f2937 !important; }
        p, li, span, label { color: #1f2937 !important; }
        h1, h2, h3, h4, h5, h6 { color: #0f3d56 !important; }
        [data-testid="stChatMessage"] { color: #1f2937 !important; }
        [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] div { color: #1f2937 !important; }
        [data-testid="stChatInput"] {
            background: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 12px !important;
        }
        [data-testid="stChatInput"] textarea { color: #1f2937 !important; background: #ffffff !important; }
        [data-testid="stChatInput"] textarea::placeholder { color: #64748b !important; }
        button[data-baseweb="tab"] { color: #334155 !important; font-weight: 600 !important; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #087f8c !important; }
        .medical-card {
            background: #ffffff !important; border: 1px solid #dbe7ef !important;
            border-radius: 14px !important; padding: 18px 20px !important;
            margin: 10px 0 !important; box-shadow: 0 4px 14px rgba(15,61,86,0.07) !important;
        }
        .condition-card {
            background: #ffffff !important; border: 1px solid #dbe7ef !important;
            border-left: 5px solid #087f8c !important; border-radius: 12px !important;
            padding: 15px 18px !important; margin: 10px 0 !important;
        }
        .section-title { color: #0f3d56 !important; font-size: 1.35rem; font-weight: 800; margin: 18px 0 10px 0; }
        .recommendation-item {
            background: #ffffff !important; border: 1px solid #dbe7ef !important;
            border-radius: 10px !important; padding: 12px 15px !important; margin: 8px 0 !important;
            color: #374151 !important;
        }
        .warning-item {
            background: #fff7ed !important; border: 1px solid #fed7aa !important;
            border-left: 4px solid #f97316 !important; border-radius: 10px !important;
            padding: 12px 15px !important; margin: 8px 0 !important; color: #7c2d12 !important;
        }
        .disclaimer-box {
            background: #eef6fa !important; border: 1px solid #cbdde7 !important;
            border-radius: 12px !important; padding: 15px 18px !important; margin-top: 25px !important;
        }
        .emergency-box {
            background: #fef2f2 !important; border: 2px solid #ef4444 !important;
            border-radius: 14px !important; padding: 20px !important; margin: 15px 0 !important;
        }
        .stButton > button { border-radius: 9px !important; font-weight: 700 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RISK BADGE
# ============================================================

def render_risk_badge(risk_level: str):
    risk = (risk_level or "LOW").upper()
    # Map CRITICAL to EMERGENCY display
    if risk == "CRITICAL":
        risk = "EMERGENCY"

    colors = {
        "LOW": ("#10b981", "#ecfdf5", "LOW RISK", "Mild clinical presentation"),
        "MODERATE": ("#f59e0b", "#fffbeb", "MODERATE RISK", "Physician consultation recommended"),
        "HIGH": ("#f97316", "#fff7ed", "HIGH RISK", "In-person medical evaluation strongly advised"),
        "EMERGENCY": ("#ef4444", "#fef2f2", "EMERGENCY / CRITICAL", "Immediate emergency medical attention required"),
    }

    border, bg, title, subtitle = colors.get(risk, colors["LOW"])
    st.markdown(
        f"""
        <div style="border:2px solid {border}; background:{bg}; border-radius:14px; padding:16px 20px; margin:18px 0;">
            <div style="color:{border}; font-size:1.2rem; font-weight:800;">{title}</div>
            <div style="color:#374151; margin-top:4px; font-size:0.95rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EMERGENCY PANEL
# ============================================================

def render_emergency_panel(state: dict):
    emergency_numbers = state.get("emergency_numbers", config.DEFAULT_EMERGENCY_NUMBERS)
    emergency_reasons = state.get("emergency_reasons", [])
    emergency_message = state.get("emergency_message", "")

    st.markdown(
        """
        <div class="emergency-box">
            <h3 style="color:#dc2626; margin:0 0 10px 0;">🚨 EMERGENCY MEDICAL ALERT</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.error("**Emergency Services: 112**")
    with col_e2:
        st.error("**Ambulance: 108**")

    if emergency_message:
        st.warning(emergency_message)

    if emergency_reasons:
        st.markdown("**Emergency triggers detected:**")
        for reason in emergency_reasons:
            st.markdown(f"- 🚨 **{reason}**")

    st.markdown(
        """
        > ⚕️ **Do NOT wait.** Call emergency services immediately or go to the nearest emergency department.
        > This AI system cannot provide emergency care. Please seek immediate professional medical attention.
        """
    )


# ============================================================
# CONDITIONS DISPLAY
# ============================================================

def render_conditions(top_k_predictions: list):
    if not top_k_predictions:
        return

    st.markdown('<div class="section-title">📊 Possible Conditions (AI Pattern Analysis)</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;">Non-diagnostic statistical estimates — NOT confirmed diagnoses</p>', unsafe_allow_html=True)

    for item in top_k_predictions[:3]:
        # top_k_predictions uses "disease" key from model.predict
        name = item.get("disease") or item.get("condition") or "Unknown"
        prob = float(item.get("probability", 0.0))
        prob_pct = item.get("confidence_percentage") or round(prob * 100, 1)

        st.markdown(
            f"""
            <div class="condition-card">
                <div style="color:#0f3d56; font-size:1.05rem; font-weight:800;">🩺 {name.title()}</div>
                <div style="color:#087f8c; font-weight:700;">AI Estimated Likelihood: {prob_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="color:#64748b; font-size:0.9rem;">⚠️ These are statistical estimates based on symptom patterns and are NOT confirmed medical diagnoses.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# GUIDANCE TABS
# ============================================================

def render_guidance_tabs(state: dict):
    recommendations = state.get("recommendations", {})

    what_to_do = recommendations.get("what_to_do", []) if isinstance(recommendations, dict) else []
    what_to_avoid = recommendations.get("what_to_avoid", []) if isinstance(recommendations, dict) else []
    diet_guidance = recommendations.get("diet_guidance", []) if isinstance(recommendations, dict) else []
    when_doctor = recommendations.get("when_to_see_doctor", []) if isinstance(recommendations, dict) else []

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💡 What To Do",
        "🚫 What To Avoid",
        "🍵 Food & Hydration",
        "⚠️ When to See Doctor",
        "🧠 Analysis Details",
    ])

    with tab1:
        if what_to_do:
            for item in what_to_do:
                st.markdown(f'<div class="recommendation-item">💡 {item}</div>', unsafe_allow_html=True)
        else:
            st.info("No specific self-care recommendations generated.")

    with tab2:
        if what_to_avoid:
            for item in what_to_avoid:
                st.markdown(f'<div class="warning-item">🚫 {item}</div>', unsafe_allow_html=True)
        else:
            st.info("No specific avoidance recommendations generated.")

    with tab3:
        if diet_guidance:
            for item in diet_guidance:
                st.markdown(f'<div class="recommendation-item">🥗 {item}</div>', unsafe_allow_html=True)
        else:
            st.info("No specific dietary recommendations generated.")

    with tab4:
        if when_doctor:
            for item in when_doctor:
                st.markdown(f'<div class="warning-item">⚠️ {item}</div>', unsafe_allow_html=True)
        else:
            st.info("Consult a healthcare professional if symptoms persist or worsen.")

    with tab5:
        explanation = state.get("explanation", "")
        predicted = state.get("predicted_disease")
        active_syms = state.get("extracted_symptoms", [])
        confidence_pct = state.get("confidence_percentage", 0.0)
        expl_source = state.get("explanation_source", "system")

        if predicted:
            st.markdown(f"**Primary Analysis:** {predicted.title()}")
            st.markdown(f"**Model Confidence:** {confidence_pct:.1f}%")
            st.markdown(f"**Explanation Source:** {expl_source}")
        if active_syms:
            st.markdown(f"**Matched Symptoms ({len(active_syms)}):** {', '.join([s.title() for s in active_syms])}")
        if explanation:
            st.markdown("**AI Explanation:**")
            st.info(explanation)

        # Show confidence notice
        st.caption(f"ℹ️ {config.SAFE_CONFIDENCE_NOTICE}")


# ============================================================
# SESSION HELPERS
# ============================================================

# Workflow stages reached once all follow-up questions are finished and the
# full analysis (prediction, risk, recommendations, report) is complete.
_DASHBOARD_STAGES = ("REPORT_GENERATION", "HOSPITAL_OFFER")

# Temporary per-session values created by the Symptom Checker / Medical Report
# pages. Patient records stored in the database are never touched.
_SESSION_KEYS_TO_RESET = (
    "report_pdf_path", "report_pdf_valid",
    "report_patient_name", "report_patient_phone", "report_patient_email",
    "report_patient_address", "report_patient_age", "report_patient_gender",
    "report_patient_ref_id",
    "rep_input_name", "rep_input_phone", "rep_input_email", "rep_input_address",
    "rep_input_age", "rep_input_gender", "rep_input_ref_id",
)


def _dashboard_ready(state: dict) -> bool:
    """True only when follow-up questions are done and the analysis is complete."""
    return (
        bool(state.get("extracted_symptoms"))
        and not state.get("awaiting_followup")
        and state.get("stage") in _DASHBOARD_STAGES
    )


def _start_new_session():
    """Reset temporary answers/results so a fresh symptom-checking session starts."""
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your MediDiagnose AI Health Assistant.\n\n"
                "Please describe what symptoms you are experiencing."
            ),
        }
    ]
    st.session_state["medical_state"] = create_initial_state()
    for key in _SESSION_KEYS_TO_RESET:
        st.session_state.pop(key, None)
    st.rerun()


# ============================================================
# MAIN CHATBOT PAGE
# ============================================================

def render_chatbot_page():
    inject_chatbot_css()

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------
    st.markdown(
        """
        <h2 style="color:#0369a1 !important; margin-top:0; margin-bottom:5px;">
            AI Symptom Checker & Clinical Triage
        </h2>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style="color:#475569 !important; margin-bottom:20px;">
        Describe your symptoms in natural language. Our 10-Agent clinical pipeline will analyze
        your presentation, ask relevant follow-up questions, assess risk, and provide personalized guidance.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # MODEL STATUS INDICATOR
    # --------------------------------------------------------
    if config.DISEASE_MODEL_PATH.exists():
        st.success("ML Engine Active — Disease prediction model loaded (100 disease classes, 230 features).")
    else:
        st.warning(
            "Trained ML model artifacts not found. "
            "Run `python model/train_model.py` to activate ML classification."
        )

    # --------------------------------------------------------
    # INITIALIZE CHAT MESSAGES
    # --------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your MediDiagnose AI Health Assistant. 👋\n\n"
                    "Please describe what symptoms you are experiencing. "
                    "For example: *'I have had fever, cough and fatigue for 3 days.'*\n\n"
                    "I will ask a few follow-up questions to better understand your situation."
                )
            }
        ]

    # --------------------------------------------------------
    # INITIALIZE MEDICAL STATE (LangGraph MedicalState)
    # --------------------------------------------------------
    if "medical_state" not in st.session_state:
        st.session_state["medical_state"] = create_initial_state()

    # --------------------------------------------------------
    # DISPLAY PREVIOUS MESSAGES
    # --------------------------------------------------------
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --------------------------------------------------------
    # RESET BUTTON (slot filled below: once the Clinical Dashboard is
    # visible, its own New Session button replaces this one)
    # --------------------------------------------------------
    reset_slot = st.container()

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------
    user_input = st.chat_input("Describe your symptoms... (e.g. 'I have fever, cough and sore throat')")

    if user_input:
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run LangGraph workflow
        current_state = st.session_state["medical_state"]

        with st.spinner("🤖 Multi-Agent AI Pipeline analyzing your clinical presentation..."):
            try:
                updated_state = run_medical_workflow(current_state, user_input)
                st.session_state["medical_state"] = updated_state
            except Exception as e:
                st.error(f"⚠️ Analysis error: {str(e)}")
                updated_state = current_state

        # -----------------------------------------------
        # BUILD AND DISPLAY ASSISTANT RESPONSE
        # -----------------------------------------------
        _build_and_display_response(updated_state)

    # ========================================================
    # CLINICAL DASHBOARD (shown once, directly after the follow-up
    # questions are completed)
    # ========================================================
    state = st.session_state["medical_state"]
    dashboard_visible = _dashboard_ready(state)

    if not dashboard_visible:
        with reset_slot:
            col_r1, col_r2 = st.columns([5, 1])
            with col_r2:
                if st.button("New Session", key="btn_new_session_top", use_container_width=True):
                    _start_new_session()
    else:
        _render_dashboard(state)

    # ========================================================
    # DISCLAIMER
    # ========================================================
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <p><strong>⚕️ Ethical Disclaimer</strong><br>{config.CLINICAL_DISCLAIMER}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bottom Healthcare Center Navigation
    st.markdown("<br>", unsafe_allow_html=True)
    c_doc_l, c_doc_mid, c_doc_r = st.columns([1.5, 2, 1.5])
    with c_doc_mid:
        if st.button("🏥 Find Doctor", key="btn_chatbot_bottom_find_doctor", use_container_width=True):
            st.session_state["current_page"] = "Healthcare Center"
            st.rerun()


# ============================================================
# BUILD RESPONSE FROM WORKFLOW STATE
# ============================================================

def _build_and_display_response(state: dict):
    """Build assistant response from the updated MedicalState."""

    is_emergency = state.get("is_emergency", False)
    awaiting_followup = state.get("awaiting_followup", False)
    current_followup_q = state.get("current_followup_question")
    extracted_syms = state.get("extracted_symptoms", [])
    stage = state.get("stage", "")

    response_md = ""

    # EMERGENCY FLOW
    if is_emergency:
        # The Clinical Dashboard shows the full emergency panel; avoid repeating it here.
        if _dashboard_ready(state):
            return
        emergency_numbers = state.get("emergency_numbers", config.DEFAULT_EMERGENCY_NUMBERS)
        emergency_reasons = state.get("emergency_reasons", [])
        emergency_message = state.get("emergency_message", "")

        response_md += "### 🚨 EMERGENCY MEDICAL ALERT\n\n"
        response_md += f"**🆘 Emergency Services:** {', '.join(emergency_numbers)}\n\n"

        if emergency_message:
            response_md += f"⚠️ {emergency_message}\n\n"

        if emergency_reasons:
            response_md += "**Emergency indicators detected:**\n"
            for reason in emergency_reasons:
                response_md += f"- 🚨 {reason}\n"
            response_md += "\n"

        response_md += (
            "**⚕️ CRITICAL: Do NOT wait. Call emergency services immediately "
            "or go to the nearest emergency department now.**\n\n"
        )
        response_md += f"*{config.CLINICAL_DISCLAIMER}*"

    # FOLLOW-UP QUESTION
    elif awaiting_followup and current_followup_q:
        response_md += f"❓ **{current_followup_q}**"

    # FULL ANALYSIS RESULT
    elif stage in ("REPORT_GENERATION", "HOSPITAL_OFFER", "FOLLOWUP_SUFFICIENT", "RISK_ASSESSMENT",
                   "RECOMMENDATIONS", "DISEASE_ANALYSIS", "EMERGENCY_POST_CHECK"):
        # Go straight to the Clinical Dashboard; only fall back to a chat summary
        # when no dashboard can be shown.
        if _dashboard_ready(state):
            return
        response_md += _build_analysis_response(state)

    # SYMPTOM EXTRACTION (first message with symptoms)
    elif stage in ("SYMPTOM_EXTRACTION", "EMERGENCY_PRE_CHECK"):
        if extracted_syms:
            formatted = ", ".join([s.title() for s in extracted_syms])
            response_md += f"✅ Symptoms identified: **{formatted}**\n\n"
            response_md += "Processing your clinical information through the AI pipeline..."
        else:
            response_md += (
                "I wasn't able to identify specific medical symptoms from your description. "
                "Could you please describe your symptoms more specifically? "
                "For example: 'I have had fever, cough, and sore throat for 2 days.'"
            )

    else:
        response_md += (
            "Thank you for your message. Please describe any symptoms you are experiencing "
            "and I will analyze them through our AI clinical pipeline."
        )

    # Display response
    if response_md:
        st.session_state["messages"].append({"role": "assistant", "content": response_md})
        with st.chat_message("assistant"):
            st.markdown(response_md)


def _build_analysis_response(state: dict) -> str:
    """Build the full analysis response text."""
    extracted_syms = state.get("extracted_symptoms", [])
    predicted_disease = state.get("predicted_disease")
    confidence_pct = state.get("confidence_percentage", 0.0)
    risk_level = state.get("risk_level", "LOW")
    top_k = state.get("top_k_predictions", [])
    explanation = state.get("explanation", "")

    response_md = "Thank you for sharing your symptoms. Here is your AI-assisted clinical assessment:\n\n"

    if extracted_syms:
        formatted = ", ".join([s.title() for s in extracted_syms])
        response_md += f"**Identified Symptoms:** {formatted}\n\n"

    response_md += f"**Clinical Risk Level:** {risk_level}\n\n"

    if predicted_disease:
        response_md += f"**Primary AI Analysis:** {predicted_disease.title()}\n"
        response_md += f"**Model Confidence:** {confidence_pct:.1f}%\n\n"
        response_md += f"*{config.SAFE_CONFIDENCE_NOTICE}*\n\n"

    if top_k:
        response_md += "**📊 Possible Conditions (Pattern Analysis):**\n"
        for item in top_k[:3]:
            name = (item.get("disease") or item.get("condition") or "Unknown").title()
            prob = item.get("confidence_percentage") or round(float(item.get("probability", 0)) * 100, 1)
            response_md += f"- **{name}** — AI Estimated: `{prob:.1f}%`\n"
        response_md += "\n*⚠️ These are statistical estimates and NOT confirmed diagnoses.*\n\n"

    if explanation:
        response_md += f"**AI Explanation:**\n{explanation}\n\n"

    # Recommendations summary
    recommendations = state.get("recommendations", {})
    if isinstance(recommendations, dict) and recommendations.get("what_to_do"):
        response_md += "**💡 Recommendations:**\n"
        for item in recommendations["what_to_do"][:3]:
            response_md += f"- {item}\n"
        response_md += "\n"

    if risk_level in ("HIGH", "CRITICAL"):
        response_md += "**🏥 Given your risk level, hospital consultation is strongly recommended.**\n\n"

    response_md += f"---\n*{config.CLINICAL_DISCLAIMER}*"

    return response_md


# ============================================================
# CLINICAL DASHBOARD
# ============================================================

def _render_dashboard(state: dict):
    """Render the full clinical dashboard below the chat."""

    st.markdown(
        '<hr style="border:none; border-top:1px solid #dbe7ef; margin:30px 0;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">📋 Clinical Decision-Support Dashboard</div>',
        unsafe_allow_html=True,
    )

    is_emergency = state.get("is_emergency", False)
    risk_level = state.get("risk_level", "LOW")

    # Emergency panel
    if is_emergency:
        render_emergency_panel(state)

    # Risk badge
    render_risk_badge(risk_level)

    # Symptoms identified
    extracted_syms = state.get("extracted_symptoms", [])
    if extracted_syms:
        st.markdown(
            '<div class="section-title">🔍 Identified Symptoms</div>',
            unsafe_allow_html=True,
        )
        sym_html = "".join(
            f'<span style="display:inline-block; background:#e0f2fe; color:#0369a1; padding:4px 10px; '
            f'border-radius:20px; margin:4px; font-size:0.9rem;">{s.title()}</span>'
            for s in extracted_syms
        )
        st.markdown(sym_html, unsafe_allow_html=True)

    # Conditions
    top_k = state.get("top_k_predictions", [])
    render_conditions(top_k)

    # Answered follow-ups summary
    answered = state.get("answered_followups", {})
    if answered:
        with st.expander("📝 Follow-up Answers Summary"):
            for category, answer in answered.items():
                st.markdown(f"- **{category.replace('_', ' ').title()}:** {answer}")

    # Guidance tabs
    st.markdown('<div class="section-title">💡 Personalized Guidance</div>', unsafe_allow_html=True)
    render_guidance_tabs(state)

    # Next Steps
    st.markdown('<div class="section-title">📌 Next Steps</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Opens the Medical Report page, where patient details are confirmed
        # and the PDF is generated and downloaded.
        if st.button("📥 Download Report", key="btn_next_steps_med_report", type="primary", use_container_width=True):
            st.session_state["current_page"] = "Medical Report"
            st.rerun()
    with col2:
        if st.button("🏥 Find Doctor", key="btn_next_steps_find_doctor", use_container_width=True):
            st.session_state["current_page"] = "Healthcare Center"
            st.rerun()
    with col3:
        if st.button("📅 Doctor Appointment", use_container_width=True):
            st.session_state["current_page"] = "Appointments"
            st.rerun()
    with col4:
        if st.button("New Session", key="btn_new_session_dashboard", use_container_width=True):
            _start_new_session()