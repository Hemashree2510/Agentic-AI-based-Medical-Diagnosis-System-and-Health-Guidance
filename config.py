"""
config.py - Core Configuration Module
AGENTIC AI IN MEDICAL DIAGNOSIS (M.Tech CSE Final-Year Project)

Centralized configuration for application settings, file system paths,
safety disclaimers, machine learning artifacts, API endpoints, risk categories,
and runtime execution modes.
"""

import os
from pathlib import Path
# Optional dotenv import; ignore if not installed
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda *args, **kwargs: None

# ==============================================================================
# BASE DIRECTORIES & ENVIRONMENT INITIALIZATION
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent

# Load environment variables from .env file if present
load_dotenv(BASE_DIR / ".env")

# Primary Subdirectories
DATA_DIR: Path = BASE_DIR / "data"
MODEL_DIR: Path = BASE_DIR / "model"
MODEL_ARTIFACTS_DIR: Path = MODEL_DIR / "artifacts"
DATABASE_DIR: Path = BASE_DIR / "database"
SERVICES_DIR: Path = BASE_DIR / "services"
AGENTS_DIR: Path = BASE_DIR / "agents"
WORKFLOW_DIR: Path = BASE_DIR / "workflow"
UTILS_DIR: Path = BASE_DIR / "utils"
UI_DIR: Path = BASE_DIR / "ui"
UI_PAGES_DIR: Path = UI_DIR / "pages"
REPORTS_DIR: Path = BASE_DIR / "reports"
ASSETS_DIR: Path = BASE_DIR / "assets"
TESTS_DIR: Path = BASE_DIR / "tests"

# Ensure essential runtime directories exist
MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# MANDATORY CLINICAL DISCLAIMERS & SAFE MEDICAL WORDING
# ==============================================================================
# Exact clinical disclaimer required by project specifications
CLINICAL_DISCLAIMER: str = (
    "This AI system provides healthcare guidance and clinical decision support "
    "for educational purposes. It does not replace a qualified medical professional. "
    "For serious, persistent, worsening, or emergency symptoms, consult a healthcare "
    "professional immediately."
)

SAFE_CONFIDENCE_NOTICE: str = "Model confidence is not clinical certainty."

OFFLINE_FALLBACK_NOTICE: str = (
    "AI explanation service is currently unavailable. "
    "The system is continuing with rule-based and machine-learning analysis."
)

PAYMENT_PROTOTYPE_NOTICE: str = (
    "Payment handling is a prototype preference only. No real transaction is processed."
)

# ==============================================================================
# APPLICATION METADATA & UI THEME
# ==============================================================================
# Backward-compatible application name used by the Streamlit UI
# ============================================================================
# APPLICATION METADATA
# ============================================================================
APP_TITLE: str = "Agentic AI in Medical Diagnosis"
APP_NAME: str = APP_TITLE
APP_SUBTITLE: str = "AI-Powered Clinical Decision Support & Healthcare Guidance"
APP_VERSION: str = "1.0.0"

# Navigation Pages (Exact 8 required pages)
NAVIGATION_PAGES: list[str] = [
    "Home",
    "AI Symptom Checker",
    "Medical Report",
    "Find Hospitals",
    "Appointments",
    "How It Works",
    "About",
    "Contact",
]

# ==============================================================================
# REGIONAL SETTINGS & EMERGENCY PROTOCOL
# ==============================================================================
DEFAULT_REGION: str = os.getenv("DEFAULT_REGION", "India")
DEFAULT_EMERGENCY_NUMBERS: list[str] = ["112", "108"]

# ==============================================================================
# OPERATING MODE
# ==============================================================================
DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")

# ==============================================================================
# DATASET PATHS
# ==============================================================================
DATASET_PATH: Path = DATA_DIR / "Diseases_and_Symptoms_dataset.csv"

# ==============================================================================
# MACHINE LEARNING ARTIFACT PATHS
# ==============================================================================
MODEL_PATH: Path = MODEL_ARTIFACTS_DIR / "model.joblib"
LABEL_ENCODER_PATH: Path = MODEL_ARTIFACTS_DIR / "label_encoder.joblib"
FEATURE_COLUMNS_PATH: Path = MODEL_ARTIFACTS_DIR / "feature_columns.joblib"
METRICS_PATH: Path = MODEL_ARTIFACTS_DIR / "metrics.json"
# Backward-compatible model path used by the Streamlit UI
DISEASE_MODEL_PATH: Path = MODEL_PATH

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'medical_ai.db'}")

# ==============================================================================
# GEMINI LLM CONFIGURATION (google-genai)
# ==============================================================================
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_PRIMARY_MODEL: str = os.getenv("GEMINI_PRIMARY_MODEL", GEMINI_MODEL)
GEMINI_FALLBACK_MODELS: list[str] = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# Additional dataset paths for clinical guidance and descriptions
DESCRIPTION_PATH: Path = DATA_DIR / "description.csv"
PRECAUTIONS_PATH: Path = DATA_DIR / "precautions.csv"
DIETS_PATH: Path = DATA_DIR / "diets.csv"

# ==============================================================================
# HOSPITAL & GEOCODING APIS
# ==============================================================================
HOSPITAL_API_URL: str = os.getenv(
    "HOSPITAL_API_URL", "https://overpass-api.de/api/interpreter"
)
HOSPITAL_API_TIMEOUT: int = 15
HOSPITAL_SEARCH_RADIUS_METERS: int = 100000

# ==============================================================================
# CLINICAL RISK LEVELS & THRESHOLDS
# ==============================================================================
RISK_LOW: str = "LOW"
RISK_MODERATE: str = "MODERATE"
RISK_HIGH: str = "HIGH"
RISK_CRITICAL: str = "CRITICAL"

VALID_RISK_LEVELS: list[str] = [
    RISK_LOW,
    RISK_MODERATE,
    RISK_HIGH,
    RISK_CRITICAL,
]

# ==============================================================================
# AGENTIC WORKFLOW & CONVERSATIONAL CONSTRAINTS
# ==============================================================================
MAX_FOLLOWUP_QUESTIONS: int = 5

# ==============================================================================
# APPOINTMENT & PAYMENT SPECIFICATIONS
# ==============================================================================
APPOINTMENT_ID_PREFIX: str = "APT-"
PAYMENT_MODES: list[str] = [
    "Credit/Debit Card",
    "UPI",
    "Razorpay",
    "Net Banking",
    "Cash",
    "Card",
    "Online",
    "Insurance",
]
