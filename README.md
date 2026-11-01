# MediDiagnose: Agentic AI in Medical Diagnosis
### An Academic Final-Year Clinical Decision-Support & Healthcare Guidance Prototype

> **CRITICAL MEDICAL & ETHICAL DIRECTIVE**  
> This system is designed solely as an academic prototype for clinical decision-support and patient guidance. **It is NOT a replacement for examination, diagnosis, or treatment by a licensed physician.** The system **NEVER** claims an AI prediction is a confirmed diagnosis and **NEVER** autonomously prescribes prescription medications.

---

## 1. Project Overview & Problem Statement

### The Problem
Traditional healthcare chatbots suffer from severe architectural limitations:
1. **Monolithic LLM Dependency**: Standard single-prompt chatbots hallucinate medical conditions, confuse symptoms with diagnoses, and lack deterministic safety nets.
2. **Absence of Grounded Machine Learning**: Many conversational bots do not utilize empirical clinical datasets, relying instead on generic generative text.
3. **Safety Vulnerabilities**: When a patient reports an acute medical emergency (such as a stroke or myocardial infarction), generative chatbots often continue casual multi-turn conversations rather than executing immediate emergency triage.
4. **Disjointed Care Pathways**: Existing prototypes fail to connect symptom evaluation with tangible healthcare delivery steps such as structured PDF report generation, hospital discovery, and doctor consultation scheduling.

### The Proposed Solution
**MediDiagnose** resolves these shortcomings by introducing an **Agentic AI Architecture** built on **LangGraph**, combining:
- **A 10-Agent Collaborative Pipeline** orchestrated over a shared `MedicalState`.
- **A Locally Trained Machine Learning Classifier** trained on an empirical medical dataset of **96,088 rows × 231 columns** across 100 conditions.
- **Large Language Model (Gemini 3.8 Flash)** clinical reasoning for natural language understanding and empathetic condition explanations.
- **An Auditable Deterministic Safety Net** that immediately intercepts emergency presentations (FAST stroke signs, chest pain, acute respiratory distress).
- **A Complete End-to-End Continuity of Care System** with ReportLab PDF compilation, OpenStreetMap hospital geolocation, and SQLite appointment scheduling.

---

## 2. Multi-Agent Architecture (LangGraph)

The system coordinates 10 specialized agents communicating through a shared typed state (`MedicalState`):

```
                       USER INPUT
                           ↓
             [1. Symptom Analysis Agent]
                           ↓
            [2. Follow-Up Question Agent]
                           ↓
             [3. Disease Analysis Agent] (ML Model Inference)
                           ↓
              [4. Risk Assessment Agent]
                           ↓
             [5. Safety / Emergency Agent]
                    /              \
       (EMERGENCY?)                  (Standard: LOW / MOD / HIGH)
          /                                \
         ↓                                  ↓
[8. Report Agent]              [6. Recommendation Agent]
         ↓                                  ↓
         |                     [7. Medical Knowledge Agent]
         |                                  ↓
         |                     [8. Report Generation Agent]
          \                                /
           \                              /
            →  [Conditional Routing Gate]  ←
               /                        \
      (LOW RISK)                  (MODERATE / HIGH / EMERGENCY)
          ↓                                      ↓
      [Complete]                    [9. Hospital Search Agent]
                                                 ↓
                                    [10. Appointment Agent]
                                                 ↓
                                    [Confirmed Consultation Pass]
```

### Agent Responsibilities
1. **Symptom Analysis Agent**: Natural language entity extraction, clinical synonym normalization, and mapping to the 230 dataset feature columns.
2. **Follow-Up Question Agent**: Context-aware clinical inquiry formulating targeted follow-ups (duration, fever temperature, red flags).
3. **Disease Analysis Agent**: Evaluates 230-dimensional binary symptom vectors through the trained Scikit-Learn classifier to yield Top-$K$ candidate conditions with confidence probabilities.
4. **Risk Assessment Agent**: Computes clinical risk tier (`LOW`, `MODERATE`, `HIGH`, `EMERGENCY`) based on symptom duration, severity, and complexity.
5. **Safety / Emergency Agent**: Deterministic clinical gatekeeper enforcing immediate escalation for life-threatening emergency indicators (FAST stroke signs, chest pain, asphyxia).
6. **Recommendation Agent**: Formulates evidence-based non-pharmacological care plans (rest, hydration, diet, and activities to avoid).
7. **Medical Knowledge Agent**: Generates empathetic, non-diagnostic clinical explanations using Gemini 3.8 Flash (*"may be consistent with"*).
8. **Report Generation Agent**: Synthesizes clinical data into structured summaries and compiles a formal PDF via ReportLab.
9. **Hospital Search Agent**: Locates healthcare facilities matched to recommended clinical specialties via OpenStreetMap Overpass API and verified database records.
10. **Appointment Agent**: Manages physician discovery, time slot selection, patient identity verification, and SQLite database booking.

---

## 3. Technology Stack

- **Primary Language**: Python 3.10+
- **Frontend User Interface**: Streamlit (modular multi-page structure with healthcare styling)
- **Agentic Orchestration**: LangGraph & LangChain-Core
- **LLM Reasoning**: Google Gemini API (`gemini-3.8-flash`)
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Database**: SQLite with SQLAlchemy ORM
- **Visual Mapping**: Plotly Graph Objects / Express
- **PDF Compilation**: ReportLab
- **Geolocation**: OpenStreetMap Overpass API + Safe Demo Provider

---

## 4. Dataset & Machine Learning Pipeline

### Dataset Characteristics
- **File**: `data/Diseases_and_Symptoms_dataset.csv`
- **Dimensions**: 96,088 rows × 231 columns
- **Target Feature**: `diseases` (100 distinct conditions)
- **Input Features**: 230 binary symptom columns ($0$ = absent, $1$ = present)
- **Auxiliary Clinical Tables**: `description.csv`, `precautions.csv`, `diets.csv`, `medications.csv`, `workout.csv`

### ML Training Pipeline (`ml/train.py`)
1. Dynamically inspects CSV columns, strips whitespace, and identifies the target column.
2. Handles missing values and separates features ($X$) and encoded targets ($y$).
3. Splits into 80% training and 20% testing sets using stratified sampling.
4. Trains and benchmarks 4 candidate classifiers:
   - **Logistic Regression**: High multi-class accuracy (~89.85%), compact footprint (~0.17 MB), robust probability calibration.
   - **Multinomial Naive Bayes**: Rapid training, high baseline accuracy (~89.47%), micro-footprint (~0.02 MB).
   - **Random Forest Classifier**: Ensemble decision tree modeling with bounded depth.
   - **Decision Tree Classifier**: Interpretable baseline tree.
5. Selects the highest-performing model based on weighted F1-score and Accuracy.
6. Exports serialized model artifacts:
   - `model/disease_model.pkl` (compressed via joblib)
   - `model/label_encoder.pkl`
   - `model/feature_columns.pkl`
   - `model/metrics.json`

---

## 5. Installation & Execution Guide (Windows / PowerShell)

### Step 1: Open PowerShell and Navigate to Project
```powershell
cd "project path"
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Verify your `.env` file contains your Google Gemini API key:
```ini
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-3.8-flash
DEMO_MODE=True
```

### Step 5: Verify Dataset Placement
Ensure `Diseases_and_Symptoms_dataset.csv` is located in `data/`:
```powershell
Test-Path "data\Diseases_and_Symptoms_dataset.csv"
```

### Step 6: Train Machine Learning Model
Run the automated multi-model training script:
```powershell
python ml/train.py
```
*Artifacts will be generated in `model/` (`disease_model.pkl`, `label_encoder.pkl`, `feature_columns.pkl`, `metrics.json`).*

### Step 7: Launch the Application
```powershell
streamlit run app.py
```
The modern healthcare interface will launch at `http://localhost:8501`.

---

## 6. Clinical Verification Test Scenarios

### Test Scenario 1: Mild Respiratory Symptoms
- **Input**: `"I have a mild cough, runny nose, and sore throat for 2 days."`
- **Agent Workflow**: Symptom Agent extracts `cough`, `sore throat`, `coryza`. Follow-up Agent asks about temperature and breathing. Disease Agent identifies upper respiratory conditions. Risk Agent sets `LOW`.
- **Expected Outcome**: Self-care recommendations, hydration advice, things to avoid, and reassurance.

### Test Scenario 2: Systemic Febrile Syndrome
- **Input**: `"High fever, chills, severe fatigue, and body ache since yesterday."`
- **Agent Workflow**: Symptom Agent extracts `fever`, `chills`, `fatigue`, `ache all over`. Disease Agent outputs candidates with confidence percentages. Risk Agent sets `MODERATE`.
- **Expected Outcome**: Outpatient medical evaluation recommended within 24–48 hours, temperature monitoring, and clinic discovery options.

### Test Scenario 3: Acute Gastrointestinal Syndrome
- **Input**: `"Severe abdominal pain, nausea, and vomiting for 24 hours."`
- **Agent Workflow**: Symptom Agent identifies `sharp abdominal pain`, `nausea`, `vomiting`. Follow-up Agent inquires about fluid tolerance and dehydration. Risk Agent flags `MODERATE/HIGH`.
- **Expected Outcome**: Gastroenterology specialty suggested, hydration caution, and nearby clinic cards displayed.

### Test Scenario 4: Acute Cardiac / Emergency Warning
- **Input**: `"Crushing chest pain radiating to left arm and shortness of breath."`
- **Agent Workflow**: Safety / Emergency Agent intercepts instantly. Deterministic rule `EMERG_CHEST_PAIN` triggers. Risk level elevated to `EMERGENCY`. Standard conversational loops bypassed.
- **Expected Outcome**: Prominent red emergency banner: *"Seek immediate emergency medical care!"*, 24/7 emergency hospital facilities highlighted.

### Test Scenario 5: Acute Neurological / Stroke Indicators
- **Input**: `"Sudden facial drooping, arm weakness, and slurred speech."`
- **Agent Workflow**: Safety / Emergency Agent intercepts via `EMERG_STROKE_FAST`. Risk immediately locked to `EMERGENCY`.
- **Expected Outcome**: Stroke emergency escalation protocol displayed with immediate dispatch recommendation.

---

## 7. Viva Voce & Academic Defense Q&A (20 Questions & Answers)

1. **Q: Why is this project classified as Agentic AI rather than a standard chatbot?**  
   *A:* A chatbot executes a single text generation prompt without separation of concerns. This project decomposes clinical decision-support into 10 autonomous, specialized agents coordinated via a LangGraph state machine, enforcing deterministic safety barriers and conditional routing.

2. **Q: What is the role of the shared `MedicalState`?**  
   *A:* `MedicalState` acts as a centralized, typed blackboard architecture through which agents read context, update findings, and pass execution control downstream.

3. **Q: How does the system ensure an LLM does not hallucinate a confirmed diagnosis?**  
   *A:* System prompts and state constraints strictly enforce non-diagnostic language (*"may be consistent with"*). Predictions are grounded in an empirical ML model, and prominent legal disclaimers are programmatically attached to all outputs.

4. **Q: Why can't the LLM override the Safety / Emergency Agent?**  
   *A:* The Safety Agent operates on deterministic regex rule criteria that execute downstream of preliminary risk scoring. When an emergency pattern triggers, the LangGraph routing function directly forces an emergency state, completely bypassing generative relaxation.

5. **Q: How is the Machine Learning pipeline integrated with unstructured user messages?**  
   *A:* The Symptom Analysis Agent employs a clinical normalization layer that maps free-text symptoms to the exact 230 feature columns of the dataset, producing a binary vector for model inference.

6. **Q: Which ML model was selected and why?**  
   *A:* The pipeline benchmarks Multinomial Naive Bayes, Decision Trees, Random Forests, and Logistic Regression. Multinomial Logistic Regression and Naive Bayes achieve ~89.8% accuracy with well-calibrated class posterior probabilities and a compact disk footprint under 1 MB.

7. **Q: How does the system handle missing symptoms in user input?**  
   *A:* Unmentioned symptoms are initialized to $0$ in the feature vector. If a user presentation has zero overlap with dataset features, the model falls back to a non-specific clinical profile and requests clarification.

8. **Q: What database architecture is utilized?**  
   *A:* SQLite with SQLAlchemy ORM, implementing 7 relational tables: `users`, `symptom_sessions`, `symptom_records`, `medical_reports`, `hospitals`, `doctors`, and `appointments`.

9. **Q: How does DEMO_MODE ensure reliability during offline evaluations?**  
   *A:* When external APIs (Gemini or Overpass OSM) are unavailable or rate-limited, `DEMO_MODE=True` seamlessly supplies verified fallback knowledge bases, demo hospital coordinates, and realistic physician records.

10. **Q: How are medical guidance reports generated for patients?**  
    *A:* The Report Generation Agent compiles clinical findings into structured JSON, which is rendered into a formatted PDF using the ReportLab engine with headers, metadata tables, risk badges, and legal disclaimers.

11. **Q: What prevents the system from prescribing medications?**  
    *A:* Autonomous prescription is strictly disabled at the architectural level. Pharmacological queries return general educational monographs from reference tables accompanied by explicit disclaimers.

12. **Q: How are medical specialties suggested?**  
    *A:* The Diagnosis Agent links candidate conditions to recognized medical specialties (e.g. Pulmonology, Gastroenterology, Cardiology) to guide hospital filtering.

13. **Q: What privacy and security practices are followed?**  
    *A:* Client-side session state isolation, environment variable separation via `.env`, input sanitization, PII masking for phone and email, and zero storage of financial credentials.

14. **Q: How does the Follow-Up Question Agent choose questions dynamically?**  
    *A:* It inspects extracted symptoms and selectively asks domain-specific questions (e.g., chest pain radiation for cardiac complaints, fluid tolerance for gastrointestinal issues).

15. **Q: What is the benefit of using LangGraph over linear LangChain chains?**  
    *A:* LangGraph supports cyclic workflows, conditional branching, state persistence, and explicit edge routing essential for clinical triage pathways.

16. **Q: What is the purpose of the appointment reference number?**  
    *A:* It provides a unique traceability code (`MED-YYYY-XXXX`) registered in the database for tracking, front-desk verification, and lookup.

17. **Q: What happens if a patient enters a rare or unrecognized symptom?**  
    *A:* The Symptom Agent uses substring token overlap and fuzzy word matching against the 230 dataset columns before falling back to general clinical assessment.

18. **Q: How are hospital coordinates displayed?**  
    *A:* Plotly creates an interactive map rendering hospital markers with popups containing address, phone, specialty, and emergency availability.

19. **Q: Can the system be deployed to a hospital intranet?**  
    *A:* Yes, the entire stack (Streamlit, SQLite, Scikit-Learn, LangGraph) is self-contained and can be containerized using Docker for on-premises deployment.

20. **Q: What are the primary future enhancements?**  
    *A:* Real-time EHR (HL7/FHIR) integration, doctor teleconsultation video rooms, multi-lingual support, and multi-modal diagnostic image analysis.

---

## 8. License & Academic Disclaimer
Developed as a capstone engineering project. All medical algorithms, datasets, and decision-support guidance are provided for academic study and non-commercial demonstration only.
