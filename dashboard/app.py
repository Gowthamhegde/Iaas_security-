"""
Stage 5 — Streamlit Dashboard
Real-time intrusion detection monitoring interface:
  - Live flow classification feed
  - Historical alert trend charts
  - Per-alert feature importance explanation
  - Model performance metrics panel
  - Suricata vs. ML comparison panel
"""

import sys
import time
import queue
import threading
import random
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IaaS Security Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:   #0a0e1a;
    --bg-card:      rgba(16, 24, 48, 0.85);
    --bg-glass:     rgba(255,255,255,0.04);
    --accent-blue:  #3b82f6;
    --accent-cyan:  #06b6d4;
    --accent-green: #10b981;
    --accent-red:   #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple:#a855f7;
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --border:       rgba(99,102,241,0.2);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.97) !important;
    border-right: 1px solid var(--border) !important;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99,102,241,0.5);
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 6px;
}
.alert-row {
    background: rgba(239,68,68,0.08);
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    animation: fadeIn 0.3s ease;
}
.normal-row {
    background: rgba(16,185,129,0.05);
    border-left: 3px solid #10b981;
    border-radius: 8px;
    padding: 8px 14px;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}
@keyframes fadeIn { from {opacity:0; transform:translateX(-8px)} to {opacity:1; transform:none} }

.section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
}
.badge-dos    { background: rgba(239,68,68,0.2);  color:#ef4444; }
.badge-probe  { background: rgba(245,158,11,0.2); color:#f59e0b; }
.badge-r2l    { background: rgba(168,85,247,0.2); color:#a855f7; }
.badge-u2r    { background: rgba(236,72,153,0.2); color:#ec4899; }
.badge-normal { background: rgba(16,185,129,0.2); color:#10b981; }

.stButton>button {
    background: linear-gradient(135deg,#3b82f6,#6366f1) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: white !important;
    transition: all 0.2s !important;
}
.stButton>button:hover { opacity:0.85; transform:translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
HISTORY_MAX = 500   # keep last N events in memory

if "events"         not in st.session_state: st.session_state.events         = deque(maxlen=HISTORY_MAX)
if "alerts"         not in st.session_state: st.session_state.alerts         = deque(maxlen=HISTORY_MAX)
if "service_running"not in st.session_state: st.session_state.service_running= False
if "service"        not in st.session_state: st.session_state.service        = None
if "metrics"        not in st.session_state: st.session_state.metrics        = {}
if "model_loaded"   not in st.session_state: st.session_state.model_loaded   = False
if "tau"            not in st.session_state: st.session_state.tau            = 0.60
if "flow_rate"      not in st.session_state: st.session_state.flow_rate      = 3.0
if "capture_mode"   not in st.session_state: st.session_state.capture_mode   = "Simulate"
if "interface"      not in st.session_state: st.session_state.interface      = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load model artifacts
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model…")
def get_inference_service(tau, rate, simulate, interface, flow_timeout, artifact_fingerprint):
    del artifact_fingerprint
    from inference.service import InferenceService
    return InferenceService(
        simulate=simulate,
        interface=interface or None,
        tau=tau,
        rate=rate,
        flow_timeout=flow_timeout,
    )


def get_artifact_fingerprint(artifacts_path):
    """Return a cache key that changes whenever a model artifact is replaced."""
    names = ("random_forest.pkl", "preprocessor.joblib", "metrics.json")
    files = [artifacts_path / name for name in names]
    if not all(path.exists() for path in files):
        return "incomplete"
    return tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in files)


def load_metrics_cache():
    try:
        from model.train import load_metrics
        return load_metrics()
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Normal": "#10b981",
    "DoS":    "#ef4444",
    "Probe":  "#f59e0b",
    "R2L":    "#a855f7",
    "U2R":    "#ec4899",
}

def badge_html(label):
    key = label.lower().replace("/", "").replace("2", "2")
    cls = f"badge-{label.lower().split('/')[0]}"
    return f'<span class="badge {cls}">{label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
      <div style='font-size:3rem'>🛡️</div>
      <div style='font-size:1.1rem;font-weight:700;color:#f1f5f9'>IaaS Security</div>
      <div style='font-size:0.75rem;color:#94a3b8;margin-top:4px'>Network attack watcher</div>
    </div>
    <hr style='border-color:rgba(99,102,241,0.2);margin:10px 0'>
    """, unsafe_allow_html=True)

    st.info("① Pick mode → ② Adjust threshold → ③ Press **Start**", icon="👉")

    st.markdown("**1. Where does traffic come from?**")
    capture_mode = st.radio(
        "Capture mode",
        ["Simulate", "Live server"],
        horizontal=True,
        help="Simulate = fake traffic to learn the UI. Live server = real NIC packets.",
        label_visibility="collapsed",
    )
    interface = ""
    flow_timeout = 30.0
    if capture_mode == "Live server":
        try:
            from inference.flow_extractor import LiveFlowExtractor
            ifaces = LiveFlowExtractor.list_interfaces()
        except Exception:
            ifaces = []
        interface = st.selectbox(
            "Which network card?",
            options=[""] + ifaces if ifaces else [""],
            format_func=lambda x: x or "(auto / default)",
            help="Windows needs Npcap + Run as Administrator.",
        )
        flow_timeout = st.slider("Wait this long before scoring a quiet connection (sec)", 5.0, 120.0, 30.0, 5.0)
        st.caption("Alerts also saved to logs/alerts.jsonl")

    st.markdown("**2. How strict should alerts be?**")
    tau = st.slider(
        "Alert threshold (τ)", 0.40, 0.99, 0.60, 0.01,
        help="Higher = fewer alerts (only when very sure). Lower = more alerts.",
    )
    st.caption(f"Alert only if confidence ≥ **{tau:.0%}** and class is not Normal.")

    flow_rate = 3.0
    if capture_mode == "Simulate":
        st.markdown("**Simulate speed**")
        flow_rate = st.slider("Fake connections per second", 0.5, 10.0, 3.0, 0.5)

    st.markdown("---")
    st.markdown("**3. Run**")
    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶ Start", use_container_width=True, key="start_btn")
    with col_stop:
        stop_btn  = st.button("■ Stop",  use_container_width=True, key="stop_btn")

    st.markdown("---")
    st.markdown("**System Info**")
    artifacts_path = ROOT / "model" / "artifacts"
    model_exists   = (artifacts_path / "random_forest.pkl").exists()
    metrics_exist  = (artifacts_path / "metrics.json").exists()
    prep_exists    = (artifacts_path / "preprocessor.joblib").exists()

    if model_exists:
        st.success("✓ Model loaded")
    else:
        st.warning("⚠ Model not trained yet")
        st.caption("Run: `python model/train.py`")

    if prep_exists:
        st.success("✓ Preprocessor ready")
    elif model_exists:
        st.info("Preprocessor will rebuild on first start")

    if metrics_exist:
        m = load_metrics_cache()
        if m:
            st.metric("Test Accuracy", f"{m.get('accuracy',0)*100:.1f}%")
            st.metric("F1 (macro)",    f"{m.get('f1_macro',0)*100:.1f}%")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.65rem;color:#475569;text-align:center'>"
        "Algorithm 1 · NSL-KDD · Random Forest<br>"
        "Live server capture · Suricata baseline comparison"
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Start / Stop service
# ─────────────────────────────────────────────────────────────────────────────
if start_btn and not st.session_state.service_running:
    if not model_exists:
        st.sidebar.error("Train the model first: `python model/train.py`")
    else:
        try:
            get_inference_service.clear()
            simulate = capture_mode == "Simulate"
            artifact_fingerprint = get_artifact_fingerprint(artifacts_path)
            svc = get_inference_service(
                tau, flow_rate, simulate, interface, flow_timeout, artifact_fingerprint
            )
            svc.tau = tau
            svc.rate = flow_rate
            svc.start()
            st.session_state.service = svc
            st.session_state.service_running = True
            st.session_state.tau = tau
            st.session_state.capture_mode = capture_mode
            st.session_state.interface = interface
            mode_label = "simulation" if simulate else f"live ({interface or 'default'})"
            st.sidebar.success(f"Pipeline running ({mode_label}) ✓")
        except Exception as e:
            st.sidebar.error(f"Failed to start: {e}")
            st.sidebar.exception(e)

if stop_btn and st.session_state.service_running:
    if st.session_state.service:
        st.session_state.service.stop()
    st.session_state.service_running = False
    st.sidebar.info("Pipeline stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Drain the event queue into session state
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.service_running and st.session_state.service:
    from inference.service import get_event_queue
    eq = get_event_queue()
    drained = 0
    while drained < 50:
        try:
            evt = eq.get_nowait()
            st.session_state.events.append(evt)
            if evt["is_alert"]:
                st.session_state.alerts.append(evt)
            drained += 1
        except queue.Empty:
            break


# ─────────────────────────────────────────────────────────────────────────────
# ── DEMO MODE: inject synthetic events if model isn't available ───────────────
# ─────────────────────────────────────────────────────────────────────────────
def _inject_demo_events(n=20):
    """Generate synthetic events for dashboard demo without a trained model."""
    from inference.service import _generate_synthetic_flow
    classes = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    weights = [0.53, 0.37, 0.09, 0.009, 0.001]
    for _ in range(n):
        true_label = random.choices(classes, weights=weights)[0]
        raw = _generate_synthetic_flow(true_label)
        prob  = random.uniform(0.55, 0.99)
        y_hat = true_label if random.random() > 0.05 else random.choice(classes)
        is_alert = (y_hat != "Normal" and prob >= st.session_state.tau)
        evt = {
            "timestamp":   raw["_timestamp"],
            "src_ip":      raw["_src_ip"],
            "dst_ip":      raw["_dst_ip"],
            "src_port":    raw["_src_port"],
            "dst_port":    raw["_dst_port"],
            "protocol":    raw["protocol_type"],
            "service":     raw["service"],
            "predicted":   y_hat,
            "probability": round(prob, 4),
            "probabilities": {c: round(random.random(), 3) for c in classes},
            "is_alert":    is_alert,
            "true_label":  true_label,
            "top_features": {
                "serror_rate": 0.22, "count": 0.15, "src_bytes": 0.12,
                "dst_bytes": 0.09, "same_srv_rate": 0.08, "duration": 0.07,
                "dst_host_serror_rate": 0.06, "srv_count": 0.05,
                "diff_srv_rate": 0.04, "flag": 0.03,
            },
        }
        st.session_state.events.append(evt)
        if is_alert:
            st.session_state.alerts.append(evt)

if not model_exists and len(st.session_state.events) == 0:
    _inject_demo_events(80)

# Keep demo running
if not model_exists and not st.session_state.service_running:
    _inject_demo_events(5)


# ─────────────────────────────────────────────────────────────────────────────
# Build dataframes for plotting
# ─────────────────────────────────────────────────────────────────────────────
events_list = list(st.session_state.events)
alerts_list = list(st.session_state.alerts)

df_events = pd.DataFrame(events_list) if events_list else pd.DataFrame(
    columns=["timestamp","predicted","probability","is_alert","src_ip","dst_ip"])
df_alerts = pd.DataFrame(alerts_list) if alerts_list else pd.DataFrame(
    columns=["timestamp","predicted","probability","src_ip","dst_ip"])

if not df_events.empty:
    df_events["timestamp"] = pd.to_datetime(df_events["timestamp"])
if not df_alerts.empty:
    df_alerts["timestamp"] = pd.to_datetime(df_alerts["timestamp"])


# ─────────────────────────────────────────────────────────────────────────────
# ── HEADER ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
status_color  = "#10b981" if st.session_state.service_running else "#ef4444"
status_label  = "● LIVE" if st.session_state.service_running else "● STOPPED"
demo_note     = "" if model_exists else "<span style='color:#f59e0b;font-size:0.7rem'> · DEMO MODE</span>"

st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;padding:10px 0 16px'>
  <div>
    <h1 style='margin:0;font-size:1.8rem;font-weight:700;
               background:linear-gradient(135deg,#3b82f6,#06b6d4);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
      IaaS Security Monitor
    </h1>
    <div style='font-size:0.85rem;color:#94a3b8;margin-top:6px'>
      Watches network connections and warns you when traffic looks like an attack.
    </div>
  </div>
  <div style='text-align:right'>
    <div style='color:{status_color};font-size:0.85rem;font-weight:600'>{status_label}{demo_note}</div>
    <div style='color:#475569;font-size:0.7rem'>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("New here? Read this first (plain English)", expanded=not st.session_state.service_running):
    st.markdown("""
**What this app does**

It looks at each network connection (**flow**) and guesses:
- **Normal** — everyday traffic (browsing, email, etc.)
- **DoS** — flood trying to knock a server offline
- **Probe** — scanning / looking for open doors
- **R2L** — remote break-in attempt (login abuse, etc.)
- **U2R** — trying to become admin / root on the machine

An **alert** means: “not Normal” **and** the model is confident enough (see **Alert threshold** in the sidebar).

**How to use it (3 steps)**
1. Left sidebar → leave **Simulate** on (fake traffic for learning) **or** pick **Live server**.
2. Click **▶ Start**.
3. Watch the numbers and the red rows — those are the warnings.

**What the numbers mean**
| You see | Meaning |
|---|---|
| Total Flows | How many connections were checked |
| Alerts Raised | How many looked dangerous |
| Alert Rate | Alerts ÷ flows (higher = noisier) |
| Normal Flows | Connections judged safe |
| Top Threat Class | Most common attack type so far |

**Feed vs charts**
- **Live feed** = latest connections, one by one (red = alert).
- **Trend chart** = how attack types change over time.
- **Pie chart** = mix of Normal vs attack types.
- **Feature bar** = which traffic clues the model trusts most (e.g. byte counts).
- **Recent Alerts** = table of only the dangerous ones.
- **Bottom metrics** = how good the model was on a standard test dataset (not live traffic).

Tip: if nothing moves, click **▶ Start** in the sidebar.
""")



# ─────────────────────────────────────────────────────────────────────────────
# ── ROW 1: KPI Cards ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
total_flows  = len(df_events)
total_alerts = len(df_alerts)
alert_rate   = f"{(total_alerts/total_flows*100):.1f}%" if total_flows else "0%"
class_counts = Counter(df_events["predicted"].tolist()) if not df_events.empty else {}
top_threat   = max(
    [(k,v) for k,v in class_counts.items() if k != "Normal"],
    key=lambda x: x[1], default=("—", 0)
)[0]

k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, value, label, color="#3b82f6"):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-value" style="background:linear-gradient(135deg,{color},#06b6d4);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent">{value}</div>
      <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, f"{total_flows:,}",  "Connections checked")
kpi(k2, f"{total_alerts:,}", "Warnings (alerts)",   "#ef4444")
kpi(k3, alert_rate,          "% that look dangerous", "#f59e0b")
kpi(k4, class_counts.get("Normal", 0), "Safe (Normal)",  "#10b981")
kpi(k5, top_threat,          "Most common attack", "#a855f7")

st.caption("These five boxes summarize what the AI saw since you pressed Start.")
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Real-time investigation panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Real-time investigation</div>', unsafe_allow_html=True)
st.caption("Use the latest flow and rolling one-minute counters to decide whether traffic needs attention.")

now = pd.Timestamp.now()
recent_window = df_events[df_events["timestamp"] >= now - pd.Timedelta(minutes=1)] if not df_events.empty else df_events
latest_event = events_list[-1] if events_list else None
window_flows = len(recent_window)
window_alerts = int(recent_window["is_alert"].sum()) if not recent_window.empty else 0
window_alert_rate = (window_alerts / window_flows) if window_flows else 0
window_confidence = float(recent_window["probability"].mean()) if not recent_window.empty else 0

rt1, rt2, rt3, rt4 = st.columns(4)
kpi(rt1, f"{window_flows:,}", "Flows in last minute", "#06b6d4")
kpi(rt2, f"{window_alerts:,}", "Alerts in last minute", "#ef4444")
kpi(rt3, f"{window_alert_rate:.1%}", "Rolling alert rate", "#f59e0b")
kpi(rt4, f"{window_confidence:.1%}", "Mean confidence", "#a855f7")

if latest_event:
    detail_col, prob_col = st.columns([1, 1])
    with detail_col:
        latest_label = latest_event.get("predicted", "Unknown")
        latest_color = CLASS_COLORS.get(latest_label, "#64748b")
        latest_status = "ALERT" if latest_event.get("is_alert") else "NORMAL"
        st.markdown(
            f"""
            <div class="metric-card" style="border-color:{latest_color}80">
              <div style="font-size:0.7rem;color:#94a3b8;letter-spacing:0.08em">LATEST DECISION</div>
              <div style="font-size:1.8rem;font-weight:700;color:{latest_color}">{latest_label}</div>
              <div style="color:#cbd5e1">Confidence: <b>{latest_event.get("probability", 0):.1%}</b>
                · <b>{latest_status}</b></div>
              <div style="font-size:0.78rem;color:#94a3b8;margin-top:8px">
                {latest_event.get("src_ip", "?")}:{latest_event.get("src_port", 0)}
                → {latest_event.get("dst_ip", "?")}:{latest_event.get("dst_port", 0)}
                · {latest_event.get("protocol", "?")} · {latest_event.get("service", "?")}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with prob_col:
        probabilities = latest_event.get("probabilities", {})
        if probabilities:
            probability_df = pd.DataFrame(
                {"Class": list(probabilities), "Probability": list(probabilities.values())}
            )
            fig_prob = px.bar(
                probability_df,
                x="Probability",
                y="Class",
                orientation="h",
                range_x=[0, 1],
                color="Class",
                color_discrete_map=CLASS_COLORS,
                title="Latest model probabilities",
                template="plotly_dark",
            )
            fig_prob.update_layout(
                height=190,
                margin=dict(l=0, r=0, t=35, b=0),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_prob, use_container_width=True)

    raw_features = latest_event.get("raw_features", {})
    if raw_features:
        with st.expander("Inspect latest flow features", expanded=False):
            feature_df = pd.DataFrame(
                [{"Feature": key, "Value": value} for key, value in raw_features.items()]
            )
            st.dataframe(feature_df, use_container_width=True, hide_index=True, height=220)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── ROW 2: Live Feed  +  Alert Trend ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
col_feed, col_trend = st.columns([1, 1.6])

with col_feed:
    st.markdown('<div class="section-title">Live feed — each connection</div>', unsafe_allow_html=True)
    st.caption("Green-ish rows = safe. Red rows = alert. Number after the label = confidence (0–1).")
    feed_container = st.container(height=320)
    with feed_container:
        recent = list(st.session_state.events)[-30:][::-1]
        if not recent:
            st.caption("Nothing yet. Click ▶ Start in the left sidebar.")
        for evt in recent:
            ts   = evt["timestamp"][:19] if isinstance(evt["timestamp"], str) else str(evt["timestamp"])[:19]
            pred = evt["predicted"]
            prob = evt["probability"]
            color = CLASS_COLORS.get(pred, "#64748b")
            row_cls = "alert-row" if evt.get("is_alert") else "normal-row"
            st.markdown(f"""
            <div class="{row_cls}">
              <span style='color:{color};font-weight:600'>{pred}</span>
              <span style='color:#94a3b8'> · confidence {prob:.0%} · </span>
              <span style='color:#64748b'>{ts}</span><br>
              <span style='color:#475569'>{evt.get('src_ip','?')} → {evt.get('dst_ip','?')}</span>
            </div>""", unsafe_allow_html=True)

with col_trend:
    st.markdown('<div class="section-title">Trend — attacks over time</div>', unsafe_allow_html=True)
    st.caption("Stacked area: how many connections of each type appeared recently.")
    if not df_events.empty:
        # Bin by 5-second windows
        df_plot = df_events.copy()
        df_plot["bin"] = df_plot["timestamp"].dt.floor("5s")
        trend = df_plot.groupby(["bin","predicted"]).size().reset_index(name="count")
        fig = px.area(
            trend, x="bin", y="count", color="predicted",
            color_discrete_map=CLASS_COLORS,
            template="plotly_dark",
            labels={"bin":"Time","count":"Flows","predicted":"Class"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=0,r=0,t=0,b=30),
            height=300,
            font=dict(family="Inter"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No data yet.")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── ROW 3: Class Distribution  +  Feature Importance ─────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
col_dist, col_feat = st.columns(2)

with col_dist:
    st.markdown('<div class="section-title">Mix of traffic types</div>', unsafe_allow_html=True)
    st.caption("Pie chart: share of Normal vs each attack class.")
    if class_counts:
        fig_pie = go.Figure(go.Pie(
            labels=list(class_counts.keys()),
            values=list(class_counts.values()),
            hole=0.55,
            marker_colors=[CLASS_COLORS.get(k, "#64748b") for k in class_counts],
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#f1f5f9"),
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=0,r=0,t=10,b=0),
            height=280,
            annotations=[dict(text=f"{total_flows}<br><span style='font-size:10px'>flows</span>",
                              showarrow=False, font_size=18, font_color="#f1f5f9")],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.caption("No data yet.")

with col_feat:
    st.markdown('<div class="section-title">Why the model decides (top clues)</div>', unsafe_allow_html=True)
    st.caption("Longer bars = clues the AI trusts more (e.g. how many bytes were sent).")
    # Use last alert's feature importances if available, else load from model metrics
    feat_imp = None
    if alerts_list:
        feat_imp = alerts_list[-1].get("top_features")
    if feat_imp is None:
        m = load_metrics_cache()
        feat_imp = m.get("feature_importances") or {
            "serror_rate":0.22,"count":0.15,"src_bytes":0.12,
            "dst_bytes":0.09,"same_srv_rate":0.08,"duration":0.07,
            "dst_host_serror_rate":0.06,"srv_count":0.05,
            "diff_srv_rate":0.04,"flag":0.03,
        }
    if feat_imp:
        top10 = dict(list(feat_imp.items())[:10])
        fig_bar = go.Figure(go.Bar(
            x=list(top10.values())[::-1],
            y=list(top10.keys())[::-1],
            orientation="h",
            marker=dict(
                color=list(top10.values())[::-1],
                colorscale=[[0,"#1e3a5f"],[1,"#3b82f6"]],
                showscale=False,
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#f1f5f9"),
            margin=dict(l=0,r=0,t=0,b=0),
            height=280,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Importance"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── ROW 4: Recent Alerts Table ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Alert list — only the dangerous ones</div>', unsafe_allow_html=True)
st.caption("Source IP → Dest IP is who talked to whom. Confidence closer to 100% = more sure.")
if not df_alerts.empty:
    display_cols = ["timestamp","predicted","probability","src_ip","dst_ip","dst_port","protocol","service"]
    display_cols = [c for c in display_cols if c in df_alerts.columns]
    st.dataframe(
        df_alerts[display_cols].tail(50).sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp":   st.column_config.DatetimeColumn("Time",    format="HH:mm:ss"),
            "probability": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
            "predicted":   st.column_config.TextColumn("Class"),
            "src_ip":      st.column_config.TextColumn("Source IP"),
            "dst_ip":      st.column_config.TextColumn("Dest IP"),
        },
        height=300,
    )
else:
    st.info("No alerts raised yet. Flows are classified as Normal.", icon="ℹ️")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── ROW 5: Suricata vs. ML Comparison ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Our AI vs old-style rules (Suricata) — demo comparison</div>', unsafe_allow_html=True)
st.caption("Illustrative only: shows why ML can catch attacks that simple signature rules miss.")
col_ml, col_sur, col_diff = st.columns(3)

# Simulate Suricata-style detection (signature-based limitations)
ml_detected  = total_alerts
# Suricata is good at DoS but weaker at R2L/U2R (novel attacks)
sur_detected = int(
    class_counts.get("DoS",0) * 0.92 +
    class_counts.get("Probe",0) * 0.70 +
    class_counts.get("R2L",0) * 0.30 +
    class_counts.get("U2R",0) * 0.10
)
advantage = ml_detected - sur_detected

with col_ml:
    st.markdown(f"""
    <div class="metric-card" style="border-color:rgba(59,130,246,0.4)">
      <div style='font-size:0.65rem;color:#3b82f6;font-weight:600;letter-spacing:0.1em;margin-bottom:8px'>ML MODEL (RANDOM FOREST)</div>
      <div style='font-size:2rem;font-weight:700;color:#3b82f6'>{ml_detected:,}</div>
      <div class="metric-label">Alerts Detected</div>
      <div style='margin-top:12px;font-size:0.75rem;color:#94a3b8'>
        ✓ Novel attack detection<br>✓ Statistical anomalies<br>✓ R2L / U2R coverage
      </div>
    </div>""", unsafe_allow_html=True)

with col_sur:
    st.markdown(f"""
    <div class="metric-card" style="border-color:rgba(245,158,11,0.4)">
      <div style='font-size:0.65rem;color:#f59e0b;font-weight:600;letter-spacing:0.1em;margin-bottom:8px'>SURICATA (EMERGING THREATS)</div>
      <div style='font-size:2rem;font-weight:700;color:#f59e0b'>{sur_detected:,}</div>
      <div class="metric-label">Alerts Detected</div>
      <div style='margin-top:12px;font-size:0.75rem;color:#94a3b8'>
        ✓ Known DoS / Probe patterns<br>✗ Novel R2L / U2R attacks<br>✗ Obfuscated traffic
      </div>
    </div>""", unsafe_allow_html=True)

with col_diff:
    color = "#10b981" if advantage >= 0 else "#ef4444"
    sign  = "+" if advantage >= 0 else ""
    st.markdown(f"""
    <div class="metric-card" style="border-color:rgba(16,185,129,0.4)">
      <div style='font-size:0.65rem;color:{color};font-weight:600;letter-spacing:0.1em;margin-bottom:8px'>ML ADVANTAGE</div>
      <div style='font-size:2rem;font-weight:700;color:{color}'>{sign}{advantage:,}</div>
      <div class="metric-label">Additional Detections</div>
      <div style='margin-top:12px;font-size:0.75rem;color:#94a3b8'>
        ML catches attacks that<br>signature rules miss —<br>especially R2L and U2R.
      </div>
    </div>""", unsafe_allow_html=True)

# Grouped bar: per-class ML vs Suricata
sur_map = {
    "Normal": 0,
    "DoS":    int(class_counts.get("DoS",0) * 0.92),
    "Probe":  int(class_counts.get("Probe",0) * 0.70),
    "R2L":    int(class_counts.get("R2L",0) * 0.30),
    "U2R":    int(class_counts.get("U2R",0) * 0.10),
}
ml_map = {k: v for k, v in class_counts.items()}

cats = ["DoS","Probe","R2L","U2R"]
fig_cmp = go.Figure([
    go.Bar(name="ML Model",  x=cats, y=[ml_map.get(c,0)  for c in cats], marker_color="#3b82f6"),
    go.Bar(name="Suricata",  x=cats, y=[sur_map.get(c,0) for c in cats], marker_color="#f59e0b"),
])
fig_cmp.update_layout(
    barmode="group",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#f1f5f9"),
    legend=dict(orientation="h", y=1.1),
    margin=dict(l=0,r=0,t=20,b=0),
    height=220,
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Detections"),
)
st.plotly_chart(fig_cmp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── ROW 6: Model Performance Metrics ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
m = load_metrics_cache()
if m:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">How good was the model in training? (lab test scores)</div>',
                unsafe_allow_html=True)
    st.caption("These scores are from the NSL-KDD test dataset — not from your live traffic right now.")
    col_acc, col_f1, col_cv, col_par = st.columns(4)

    kpi(col_acc, f"{m.get('accuracy',0)*100:.1f}%",  "Overall correctness", "#10b981")
    kpi(col_f1,  f"{m.get('f1_macro',0)*100:.1f}%",  "Balance across attack types", "#3b82f6")
    kpi(col_cv,  f"{m.get('best_cv_f1',0)*100:.1f}%","Score during training", "#a855f7")
    kpi(col_par, str(m.get("best_params",{}).get("n_estimators","—")),
                                                       "Trees in the forest", "#f59e0b")

    # Per-class metrics table
    cr = m.get("classification_report", {})
    rows = []
    for cls in ["Normal","DoS","Probe","R2L","U2R"]:
        if cls in cr:
            c = cr[cls]
            rows.append({
                "Class":     cls,
                "Precision": round(c.get("precision",0),4),
                "Recall":    round(c.get("recall",0),4),
                "F1-Score":  round(c.get("f1-score",0),4),
                "Support":   int(c.get("support",0)),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Confusion matrix heatmap
    cm = m.get("confusion_matrix")
    classes = m.get("classes", [])
    if cm and classes:
        total_test = sum(sum(row) for row in cm)
        correct_test = sum(cm[i][i] for i in range(min(len(cm), len(classes))))
        reported_accuracy = m.get("accuracy")
        if reported_accuracy is not None:
            st.caption(
                f"Test accuracy: **{reported_accuracy:.2%}** "
                f"({correct_test:,} correct out of {total_test:,} test samples)."
            )
        fig_cm = px.imshow(
            cm, x=classes, y=classes,
            color_continuous_scale="Blues",
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            title=(
                f"Confusion Matrix — Test Accuracy: "
                f"{reported_accuracy:.2%}"
                if reported_accuracy is not None
                else "Confusion Matrix"
            ),
            template="plotly_dark",
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#f1f5f9"),
            height=350,
            margin=dict(l=0,r=0,t=40,b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh when pipeline is running
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.service_running:
    time.sleep(1)
    st.rerun()
