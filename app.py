"""AI Factory Intelligence Command Center — Stage IX web application.

Integrates the already-built pipeline end to end:
    machine selection / image upload -> Stage II prediction (via Stage V's
    Predictive Maintenance Agent) -> Stage VI XAI -> Stage IV RAG evidence
    -> Stage V multi-agent recommendation -> Stage VII digital twin ->
    Stage VIII human decision -> downloadable PDF report.

This file contains no modeling, prediction, or business logic of its own -
it only calls the real functions already built in Stages I-VIII and
displays their real outputs. No prediction, metric, or recommendation is
computed or hard-coded here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Load ANTHROPIC_API_KEY / OPENAI_API_KEY from a local .env file, if one
# exists, into the process environment. src/rag/generator.py already reads
# these via os.environ.get(...); this just makes a local .env file (never
# committed - see .gitignore) populate that environment automatically, so
# the key never has to be pasted into any source file. If python-dotenv
# isn't installed or there's no .env file, this silently does nothing and
# the app falls back to its existing no-key behaviour.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

sys.path.append(str(Path(__file__).resolve().parent))
from config.config import MACHINE_FLEET, PROJECT_ROOT
from src.agents.orchestrator import run_workflow
from src.xai.gradcam_explainer import explain_uploaded_image
from src.digital_twin.simulator import run_scenarios
from src.hitl.hitl_workflow import submit_decision, HumanDecisionError
from src.hitl import audit_store
from src.reporting.report_generator import generate_report

st.set_page_config(page_title="AI Factory Intelligence Command Center", page_icon="🏭", layout="wide")

MACHINE_IDS = [m["machine_id"] for m in MACHINE_FLEET]

for key in ("ai_output", "hitl_record", "upload_result", "twin_override"):
    st.session_state.setdefault(key, None)


def _status_badge(status: str) -> str:
    return {"ok": "🟢 ok", "unavailable": "🟡 unavailable", "error": "🔴 error"}.get(status, status)


# ---------------------------------------------------------------------------
# Sidebar — input controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏭 AI Factory")
    st.caption("Intelligence Command Center")
    machine_id = st.selectbox("Machine", MACHINE_IDS)

    st.divider()
    st.subheader("Data / file upload")
    uploaded_image = st.file_uploader(
        "Upload an inspection image (optional)", type=["png", "jpg", "jpeg"],
        help="Runs Stage III's real trained CNN + Stage VI's Grad-CAM live on this image, "
             "in addition to the machine's stored inspection result.")
    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("L")
        arr = np.array(img, dtype=np.float64) / 255.0
        st.session_state["upload_result"] = explain_uploaded_image(arr)
        st.image(img, caption="Uploaded image", width=120)

    st.divider()
    horizon = st.slider("Digital twin horizon (shifts)", 2, 24, 6)

    run_clicked = st.button("▶ Run Full Analysis", type="primary", width='stretch')

if run_clicked:
    with st.spinner("Running prediction → XAI → RAG → multi-agent reasoning → digital twin..."):
        st.session_state["ai_output"] = run_workflow(machine_id)
        st.session_state["hitl_record"] = None
        if horizon != 6:
            st.session_state["twin_override"] = run_scenarios(machine_id, horizon_shifts=horizon)
        else:
            st.session_state["twin_override"] = None

ai_output = st.session_state["ai_output"]

st.title("AI Factory Intelligence Command Center")
st.caption("Input → Prediction → XAI → RAG Evidence → Multi-Agent Recommendation → Digital Twin → Human Decision → Report")

if ai_output is None:
    st.info("Select a machine and click **Run Full Analysis** in the sidebar to begin.")
    st.stop()

if ai_output["factory_context"]["machine_id"] != machine_id:
    st.warning("Showing results for a previously analyzed machine. Click **Run Full Analysis** to refresh for the current selection.")

tabs = st.tabs(["📈 Prediction", "🔍 XAI", "📚 RAG Evidence", "🤖 Recommendation",
                 "🔁 Digital Twin", "✅ Human Decision", "📄 Report"])

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
with tabs[0]:
    pm = ai_output["predictive_maintenance_result"]
    vision = ai_output["vision_result"]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predictive Maintenance (Stage II GRU, live inference)")
        st.caption(_status_badge(pm.get("status")))
        if pm.get("status") == "ok":
            m1, m2, m3 = st.columns(3)
            m1.metric("Failure probability", f"{pm['failure_probability']:.2%}")
            m2.metric("Risk level", pm["risk_level"])
            m3.metric("Model", pm["model_name"])
            st.caption("Top signals: " + ", ".join(pm["important_signals"]))
        else:
            st.warning(pm.get("reason", "unavailable"))

    with col2:
        st.subheader("Vision Inspection (Stage III CNN)")
        st.caption(_status_badge(vision.get("status")))
        if vision.get("status") == "ok":
            st.metric("Defect detected", str(vision["defect_detected"]))
            st.metric("Confidence", f"{vision['confidence']:.2%}")
            src_img = PROJECT_ROOT / vision.get("source_image", "")
            if src_img.exists():
                st.image(str(src_img), caption="Most recent inspection image", width=120)
        else:
            st.warning(vision.get("reason", "unavailable"))

    if st.session_state["upload_result"]:
        st.divider()
        st.subheader("Live inference on your uploaded image")
        up = st.session_state["upload_result"]
        if up.get("status") == "ok":
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted class", up["predicted_class"])
            c2.metric("Confidence", f"{up['confidence']:.2%}")
            overlay = PROJECT_ROOT / up["overlay_path"]
            if overlay.exists():
                c3.image(str(overlay), caption="Grad-CAM overlay", width=120)
        else:
            st.warning(up.get("reason", "unavailable"))

# ---------------------------------------------------------------------------
# XAI
# ---------------------------------------------------------------------------
with tabs[1]:
    xai = ai_output["xai_result"]
    st.subheader("Explainable AI (Stage VI)")
    st.write(xai.get("human_summary", "No explanation available."))

    feats = xai.get("feature_explanations", [])
    if feats:
        st.markdown("**Top contributing features**")
        st.dataframe(
            [{"Feature": f["feature"], "Value": round(f["value"], 3),
              "Contribution": round(f["contribution"], 6), "Direction": f["direction"]}
             for f in feats],
            width='stretch', hide_index=True,
        )
        st.caption(f"Method: {xai['method']['tabular']}")
    else:
        st.info("No feature attribution available (predictive-maintenance result was unavailable).")

    vis_xai = xai.get("vision_explanation", {})
    if vis_xai.get("available"):
        st.markdown("**Grad-CAM — which image region drove the vision prediction**")
        overlay = PROJECT_ROOT / vis_xai["overlay_path"]
        if overlay.exists():
            st.image(str(overlay), caption=f"{vis_xai['predicted_class']} ({vis_xai['confidence']:.1%})", width=200)

# ---------------------------------------------------------------------------
# RAG Evidence
# ---------------------------------------------------------------------------
with tabs[2]:
    rag = ai_output["rag_result"]
    st.subheader("Knowledge / RAG Evidence (Stage IV)")
    st.caption(_status_badge(rag.get("status")))
    if rag.get("status") == "ok":
        st.markdown(f"**Question:** {rag['question']}")
        st.write(rag["answer"])
        st.caption(f"Grounded: {rag['grounded']}  |  Live LLM used: {rag['used_live_llm']}")
        if rag.get("sources"):
            st.markdown("**Sources**")
            for s in rag["sources"]:
                with st.expander(f"{s['document']} — p.{s['page']} — {s['section']} (score {s['score']:.3f})"):
                    st.write(s["evidence"])
    else:
        st.warning(rag.get("reason", "unavailable"))

# ---------------------------------------------------------------------------
# Multi-agent recommendation
# ---------------------------------------------------------------------------
with tabs[3]:
    decision = ai_output["decision"]
    st.subheader("Multi-Agent AI Recommendation (Stage V)")
    st.caption(_status_badge(decision.get("status")))
    if decision.get("status") == "ok":
        st.metric("Priority", decision["priority"])
        st.markdown(f"**Recommendation:** {decision['recommendation']}")
        st.markdown("**Reasoning**")
        for line in decision["reasoning"]:
            st.markdown(f"- {line}")
        if decision.get("narrative"):
            st.info(decision["narrative"])

    with st.expander("Agent execution trace"):
        st.dataframe(
            [{"Agent": t["agent"], "Status": t["status"], "Output": t["output_summary"]}
             for t in ai_output["agent_trace"]],
            width='stretch', hide_index=True,
        )

# ---------------------------------------------------------------------------
# Digital twin
# ---------------------------------------------------------------------------
with tabs[4]:
    twin = st.session_state["twin_override"] or ai_output["digital_twin_result"]
    st.subheader("Digital Twin — What-If Simulation (Stage VII)")
    st.caption(_status_badge(twin.get("status")))
    if twin.get("status") == "ok":
        st.caption(f"Horizon: {twin['horizon_shifts']} shifts ({twin['horizon_hours']}h) · "
                   f"Current failure probability: {twin['current_failure_probability']:.2%}")
        st.dataframe(
            [{"Scenario": s["scenario"], "Expected production": s["expected_production_units"],
              "Downtime (h)": s["expected_downtime_hours"], "Failure risk": f"{s['cumulative_failure_risk']:.2%}",
              "Net value (USD)": f"${s['estimated_net_value_usd']:,.2f}"} for s in twin["scenarios"]],
            width='stretch', hide_index=True,
        )
        st.success(f"Recommended: **{twin['recommended_scenario']}** ({twin['recommendation_basis']})")
    else:
        st.warning(twin.get("reason", "unavailable"))

# ---------------------------------------------------------------------------
# Human decision
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Human Supervisor Decision (Stage VIII)")
    st.caption("The AI recommendation above is never final — a human decision is always required.")

    supervisor = st.text_input("Supervisor name (optional)")
    choice = st.radio("Decision", ["APPROVE", "REJECT", "MODIFY"], horizontal=True)
    modified_action = None
    if choice == "MODIFY":
        modified_action = st.text_area("Modified action")
    comment = st.text_area("Reason / comment" + (" (required)" if choice in ("REJECT", "MODIFY") else " (optional)"))

    if st.button("Submit decision"):
        try:
            record = submit_decision(
                ai_output, choice, comment=comment or None,
                modified_action=modified_action or None, supervisor=supervisor or None)
            st.session_state["hitl_record"] = record
            st.success(f"Decision recorded — ID {record['decision_id'][:8]}")
        except HumanDecisionError as exc:
            st.error(str(exc))

    if st.session_state["hitl_record"]:
        st.json(st.session_state["hitl_record"], expanded=False)

    with st.expander("Full audit log for this machine"):
        records = audit_store.records_for_machine(machine_id)
        if records:
            st.dataframe(
                [{"Decision ID": r["decision_id"][:8], "Decision": r["human_decision"],
                  "Timestamp": r["timestamp"], "Supervisor": r.get("supervisor") or "-"}
                 for r in records],
                width='stretch', hide_index=True,
            )
        else:
            st.caption("No decisions recorded yet for this machine.")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Downloadable Report")
    st.caption("Bundles the real prediction, XAI, RAG evidence, recommendation, digital twin, "
               "and (if submitted) human decision into one PDF.")
    if st.button("Generate PDF report"):
        pdf_bytes = generate_report(ai_output, st.session_state["hitl_record"])
        st.download_button(
            "⬇ Download PDF", data=pdf_bytes,
            file_name=f"ai_factory_report_{machine_id}.pdf", mime="application/pdf")
