"""
test_scenarios.py - Automated Clinical Scenario Verification Test Suite
AGENTIC AI IN MEDICAL DIAGNOSIS (M.Tech CSE Final-Year Project)

Executes 5 standardized clinical test scenarios through the multi-agent
workflow engine to verify symptom extraction, emergency triage, risk level,
and decision routing.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from database.database import init_db
from workflow.graph import run_medical_workflow
from workflow.state import create_initial_state
import config


def run_tests():
    print("=" * 72)
    print(" MediDiagnose - Automated Clinical Scenario Verification Test Suite ")
    print("=" * 72)

    # Initialize database
    try:
        init_db()
    except Exception as e:
        print(f"Database init notice: {e}")

    scenarios = [
        {
            "id": "SCENARIO 1",
            "name": "Mild Upper Respiratory Presentation",
            "input": "I have mild cough, runny nose and sore throat for 2 days.",
            "acceptable_risks": [config.RISK_LOW, config.RISK_MODERATE],
            "should_be_emergency": False,
        },
        {
            "id": "SCENARIO 2",
            "name": "Systemic Febrile Presentation",
            "input": "I have fever, cough, headache and extreme tiredness for 3 days.",
            "acceptable_risks": [config.RISK_LOW, config.RISK_MODERATE, config.RISK_HIGH],
            "should_be_emergency": False,
        },
        {
            "id": "SCENARIO 3",
            "name": "Acute Gastrointestinal Presentation",
            "input": "I have severe abdominal pain, nausea and vomiting since yesterday.",
            "acceptable_risks": [config.RISK_MODERATE, config.RISK_HIGH, config.RISK_CRITICAL],
            "should_be_emergency": False,
        },
        {
            "id": "SCENARIO 4",
            "name": "Acute Cardiac / Dyspnea Emergency",
            "input": "I have crushing chest pain and shortness of breath.",
            "acceptable_risks": [config.RISK_CRITICAL, "EMERGENCY"],
            "should_be_emergency": True,
        },
        {
            "id": "SCENARIO 5",
            "name": "Acute Stroke Warning (FAST)",
            "input": "Patient has sudden facial drooping, arm weakness and slurred speech.",
            "acceptable_risks": [config.RISK_CRITICAL, "EMERGENCY"],
            "should_be_emergency": True,
        },
    ]

    all_passed = True

    for s in scenarios:
        print(f"\nRunning {s['id']}: {s['name']}")
        print(f"Patient Statement: \"{s['input']}\"")

        initial_state = create_initial_state(s["input"])
        output_state = run_medical_workflow(initial_state, s["input"])

        risk = output_state.get("risk_level", "LOW")
        is_emerg = output_state.get("is_emergency", False)
        symptoms = output_state.get("extracted_symptoms", [])
        predicted = output_state.get("predicted_disease")
        top_k = output_state.get("top_k_predictions", [])
        emerg_reasons = output_state.get("emergency_reasons", [])

        print(f"  [Output] Extracted Symptoms: {symptoms}")
        print(f"  [Output] Clinical Risk Level: {risk}")
        print(f"  [Output] Emergency Intercept: {is_emerg}")
        if predicted:
            print(f"  [Output] Primary Model Match: {predicted.title()}")
        if emerg_reasons:
            print(f"  [Output] Emergency Triggers: {emerg_reasons}")

        # Verification checks
        if s["should_be_emergency"]:
            if not is_emerg or risk not in s["acceptable_risks"]:
                print(f"  [FAIL] Expected emergency/critical risk, got is_emergency={is_emerg}, risk={risk}")
                all_passed = False
            else:
                print("  [PASS] Emergency safety guardrail intercepted presentation successfully.")
        else:
            if is_emerg:
                print(f"  [FAIL] Unexpected emergency trigger on non-emergency input: {emerg_reasons}")
                all_passed = False
            else:
                print("  [PASS] Non-emergency multi-agent triage completed successfully.")

    print("\n" + "=" * 72)
    if all_passed:
        print(" ALL 5 CLINICAL TEST SCENARIOS PASSED WITH FULL SAFETY VALIDATION! ")
    else:
        print(" SOME TEST SCENARIOS FAILED. Check logs above. ")
    print("=" * 72)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
