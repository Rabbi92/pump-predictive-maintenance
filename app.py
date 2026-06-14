"""
Ngezi Pump Station - Predictive Condition Monitoring & Control System
=====================================================================
Demonstration system for BEng dissertation defence.
Author: Kimberly T. Mandikiyana

Upgrades in this version:
  * Engineering-themed page styling (header, status cards, KPI strip)
  * Auto-generate now writes back into the slider widgets so the bars move
  * Manual sliders still fully usable
  * Severity-ranked maintenance plan + station service-priority ranking
  * Altair charts with threshold reference lines + RUL gauge
  * Feature-importance chart that echoes the FMEA
"""

import os
import io
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import altair as alt
import streamlit as st

# --------------------------------------------------------------------------
# Page config + engineering theme
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Ngezi Pump Station - Predictive Maintenance",
    page_icon="P",
    layout="wide",
)

HERE = os.path.dirname(os.path.abspath(__file__))

st.markdown(
    """
    <style>
      .stApp { background-color: #0f1419; }
      .main .block-container { padding-top: 1.2rem; max-width: 1400px; }
      .scada-header {
          background: linear-gradient(90deg, #15314b 0%, #1b4b6b 100%);
          border-radius: 10px; padding: 16px 22px; margin-bottom: 14px;
          border-left: 6px solid #2f9e8f; color: #eaf2f8;
      }
      .scada-header h1 { font-size: 24px; margin: 0; color: #ffffff; }
      .scada-header p  { margin: 2px 0 0; color: #b9cad6; font-size: 13px; }
      .pump-card {
          border-radius: 10px; padding: 12px 16px; margin-bottom: 6px;
          background: #18222e; color: #e8eef4;
      }
      .pump-card .pname { font-size: 16px; font-weight: 700; color:#cfe0ec; }
      .pump-card .pstate{ font-size: 22px; font-weight: 800; }
      .pump-card .pmeta { font-size: 13px; color:#9fb3c2; }
      .kpi { background:#18222e; border-radius:10px; padding:10px 14px;
             text-align:center; color:#e8eef4; }
      .kpi .v { font-size:26px; font-weight:800; }
      .kpi .l { font-size:12px; color:#9fb3c2; text-transform:uppercase;
                letter-spacing:.05em; }
      h2, h3 { color:#dbe7f0 !important; }
      hr { border-color:#243240; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    clf = joblib.load(os.path.join(HERE, "pump_state_clf.pkl"))
    reg = joblib.load(os.path.join(HERE, "pump_rul_reg.pkl"))
    with open(os.path.join(HERE, "pump_meta.json")) as f:
        meta = json.load(f)
    return clf, reg, meta


try:
    CLF, REG, META = load_artifacts()
except Exception as e:
    st.error(
        "Could not load model files. Ensure pump_state_clf.pkl, "
        "pump_rul_reg.pkl and pump_meta.json sit next to app.py.\n\n"
        f"Details: {e}"
    )
    st.stop()

FEATURES = META["features"]
RANGES = META["ranges"]
TH = META["thresholds"]
STATE_NAMES = META["state_names"]
SERVICE_LIFE_H = META["service_life_h"]
RESERVOIR_MIN = TH.get("reservoir_min", 45.0)
PUMPS = ["PUM-01", "PUM-02", "PUM-03"]

LABELS = {
    "motor_temp_C": ("Motor temperature", "deg C"),
    "bearing_temp_C": ("Bearing temperature", "deg C"),
    "vibration_mm_s": ("Vibration (RMS)", "mm/s"),
    "pressure_bar": ("Discharge pressure", "bar"),
    "flow_rate_m3h": ("Flow rate", "m3/h"),
    "motor_current_A": ("Motor current", "A"),
    "runtime_hours": ("Runtime", "hours"),
    "ambient_temp_C": ("Ambient temperature", "deg C"),
    "reservoir_level_percent": ("Reservoir level", "%"),
}

STATE_COLOR = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}


def default_reading():
    return {f: float(RANGES[f][2]) for f in FEATURES}


if "readings" not in st.session_state:
    st.session_state.readings = {p: default_reading() for p in PUMPS}
if "log" not in st.session_state:
    st.session_state.log = []
if "auto" not in st.session_state:
    st.session_state.auto = False
if "tick" not in st.session_state:
    st.session_state.tick = 0
if "last_update" not in st.session_state:
    st.session_state.last_update = "-"


def widget_key(pump, feat):
    return f"sld_{pump}_{feat}"


def push_reading_to_widgets(pump, reading):
    """Write a reading into the slider widget keys so the bars visually move."""
    for f in FEATURES:
        st.session_state[widget_key(pump, f)] = float(reading[f])


def predict(reading):
    x = pd.DataFrame([reading])[FEATURES]
    state = int(CLF.predict(x)[0])
    proba = CLF.predict_proba(x)[0]
    pf = np.zeros(len(STATE_NAMES))
    for c, p in zip(CLF.classes_, proba):
        pf[int(c)] = p
    rul = max(0.0, float(REG.predict(x)[0]))
    return state, pf, rul


def maintenance_plan(reading, state, rul):
    actions = []  # (rank, severity, component, text)
    bt, vb, pr = reading["bearing_temp_C"], reading["vibration_mm_s"], reading["pressure_bar"]
    mt, res = reading["motor_temp_C"], reading["reservoir_level_percent"]

    if bt >= TH["bearing_crit"]:
        actions.append((0, "Critical", "Bearing", "Bearing temperature in failure "
                        "range. Schedule immediate bearing inspection, relubrication "
                        "or replacement."))
    elif bt >= TH["bearing_warn"]:
        actions.append((2, "Warning", "Bearing", "Bearing temperature elevated. "
                        "Check lubrication and plan a bearing inspection."))

    if vb >= TH["vib_crit"]:
        actions.append((0, "Critical", "Vibration", "Vibration in failure range. "
                        "Check shaft alignment, balancing and bearing wear "
                        "(possible imbalance / cavitation)."))
    elif vb >= TH["vib_warn"]:
        actions.append((2, "Warning", "Vibration", "Vibration rising. Inspect "
                        "alignment and balance at the next opportunity."))

    if res < RESERVOIR_MIN:
        actions.append((0, "Critical", "Reservoir", f"Reservoir below the "
                        f"{RESERVOIR_MIN:.0f}% minimum operating level. Initiate "
                        "duty/standby changeover to restore level."))

    if pr <= TH["press_low"]:
        actions.append((2, "Warning", "Hydraulics", "Discharge pressure low. "
                        "Inspect mechanical seal and check for cavitation or "
                        "impeller erosion."))

    if mt >= 88:
        actions.append((2, "Warning", "Motor", "Motor temperature high. Check "
                        "cooling, load and electrical condition."))

    if not actions:
        actions.append((5, "Normal", "General", "All indicators within normal "
                        "limits. Continue routine condition monitoring."))

    actions.sort(key=lambda a: a[0])

    days = rul / 24.0
    if rul <= 500:
        sched = f"Act now - RUL ~ {rul:.0f} h ({days:.0f} days)."
    elif rul <= 2000:
        sched = f"Schedule soon - RUL ~ {rul:.0f} h ({days:.0f} days)."
    else:
        sched = f"Routine - RUL ~ {rul:.0f} h ({days:.0f} days)."
    return actions, sched


def severity_score(state, rul):
    return state * 100000 + (SERVICE_LIFE_H - rul)


def drift_reading(reading, inject_fault=False):
    new = dict(reading)
    rng = np.random.default_rng()
    for f in FEATURES:
        lo, hi, _ = RANGES[f]
        new[f] = float(np.clip(reading[f] + rng.normal(0, (hi - lo) * 0.02), lo, hi))
    new["runtime_hours"] = float(min(RANGES["runtime_hours"][1],
                                     reading["runtime_hours"] + rng.integers(1, 4)))
    if inject_fault or rng.random() < 0.12:
        bbt = rng.uniform(6, 12) if inject_fault else rng.uniform(3, 8)
        bvb = rng.uniform(1.0, 2.2) if inject_fault else rng.uniform(0.4, 1.2)
        dpr = rng.uniform(0.4, 0.9) if inject_fault else rng.uniform(0.2, 0.6)
        new["bearing_temp_C"] = float(min(RANGES["bearing_temp_C"][1], new["bearing_temp_C"] + bbt))
        new["vibration_mm_s"] = float(min(RANGES["vibration_mm_s"][1], new["vibration_mm_s"] + bvb))
        new["pressure_bar"] = float(max(RANGES["pressure_bar"][0], new["pressure_bar"] - dpr))
    return new


def log_reading(pump, reading, state, rul):
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "pump_id": pump}
    row.update({f: round(reading[f], 4) for f in FEATURES})
    row["failure"] = 1 if state == 2 else 0
    row["predicted_state"] = STATE_NAMES[state]
    row["predicted_RUL_hours"] = round(rul, 1)
    st.session_state.log.append(row)


def step_all(inject=False):
    for p in PUMPS:
        new = drift_reading(st.session_state.readings[p], inject)
        st.session_state.readings[p] = new
        push_reading_to_widgets(p, new)
        s, _, r = predict(new)
        log_reading(p, new, s, r)
    st.session_state.tick += 1
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Control panel")
st.sidebar.caption("Ngezi Pump Station demonstrator")

st.sidebar.subheader("Random operation generator")
c1, c2 = st.sidebar.columns(2)
gen_once = c1.button("Generate step", use_container_width=True)
fault_btn = c2.button("Inject fault", use_container_width=True)

st.session_state.auto = st.sidebar.toggle("Auto-run (live)", value=st.session_state.auto)
auto_delay = st.sidebar.slider("Auto-run interval (s)", 1, 10, 3)

st.sidebar.divider()
if st.sidebar.button("Reset to normal", use_container_width=True):
    for p in PUMPS:
        st.session_state.readings[p] = default_reading()
        push_reading_to_widgets(p, st.session_state.readings[p])
    st.rerun()
if st.sidebar.button("Clear data log", use_container_width=True):
    st.session_state.log = []
    st.rerun()

st.sidebar.divider()
st.sidebar.metric("Readings logged", len(st.session_state.log))
st.sidebar.metric("Generator ticks", st.session_state.tick)
st.sidebar.caption(f"Last update: {st.session_state.last_update}")

if gen_once:
    step_all(False)
if fault_btn:
    step_all(True)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="scada-header">
      <h1>Ngezi Pump Station &mdash; Predictive Condition Monitoring &amp; Control</h1>
      <p>Random Forest condition classifier + RUL regressor &nbsp;|&nbsp;
         3 critical pumps &nbsp;|&nbsp; {datetime.now():%Y-%m-%d %H:%M}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

pump_status = {}
for p in PUMPS:
    s, pr, r = predict(st.session_state.readings[p])
    pump_status[p] = {"state": s, "proba": pr, "rul": r}

n_healthy = sum(1 for v in pump_status.values() if v["state"] == 0)
n_warn = sum(1 for v in pump_status.values() if v["state"] == 1)
n_crit = sum(1 for v in pump_status.values() if v["state"] == 2)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='kpi'><div class='v' style='color:#2ecc71'>{n_healthy}</div>"
            f"<div class='l'>Healthy</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi'><div class='v' style='color:#f1c40f'>{n_warn}</div>"
            f"<div class='l'>Warning</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi'><div class='v' style='color:#e74c3c'>{n_crit}</div>"
            f"<div class='l'>Critical</div></div>", unsafe_allow_html=True)
min_rul = min(v["rul"] for v in pump_status.values())
k4.markdown(f"<div class='kpi'><div class='v'>{min_rul:,.0f} h</div>"
            f"<div class='l'>Lowest RUL</div></div>", unsafe_allow_html=True)

st.write("")

cards = st.columns(3)
for col, p in zip(cards, PUMPS):
    s = pump_status[p]["state"]
    r = pump_status[p]["rul"]
    pr = pump_status[p]["proba"]
    col.markdown(
        f"<div class='pump-card' style='border-left:6px solid {STATE_COLOR[s]}'>"
        f"<div class='pname'>{p}</div>"
        f"<div class='pstate' style='color:{STATE_COLOR[s]}'>{STATE_NAMES[s]}</div>"
        f"<div class='pmeta'>RUL ~ {r:,.0f} h ({r/24:,.0f} days) &nbsp;|&nbsp; "
        f"conf {pr[s]*100:.0f}%</div></div>",
        unsafe_allow_html=True,
    )

ranked = sorted(PUMPS, key=lambda p: -severity_score(pump_status[p]["state"],
                                                      pump_status[p]["rul"]))
worst = max(v["state"] for v in pump_status.values())
top = ranked[0]
if worst == 2:
    st.error(f"STATION ALERT - service priority: **{top}** "
             f"({STATE_NAMES[pump_status[top]['state']]}, "
             f"RUL {pump_status[top]['rul']:,.0f} h). Order: " + " > ".join(ranked))
elif worst == 1:
    st.warning(f"STATION NOTICE - watch **{top}** first. Order: " + " > ".join(ranked))
else:
    st.success("All pumps within normal limits. Continue routine monitoring.")

st.divider()


def trend_chart(plog, col, label, unit, warn=None, crit=None, color="#3498db"):
    d = plog.reset_index().rename(columns={"index": "step"})
    base = alt.Chart(d).mark_line(point=True, color=color).encode(
        x=alt.X("step:Q", title="reading #"),
        y=alt.Y(f"{col}:Q", title=f"{label} ({unit})"),
        tooltip=[col],
    )
    layers = [base]
    if warn is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [warn]})).mark_rule(
            color="#f1c40f", strokeDash=[6, 4]).encode(y="y:Q"))
    if crit is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [crit]})).mark_rule(
            color="#e74c3c", strokeDash=[6, 4]).encode(y="y:Q"))
    return alt.layer(*layers).properties(height=240).interactive()


def rul_gauge(rul):
    pct = max(0.0, min(1.0, rul / SERVICE_LIFE_H))
    color = "#e74c3c" if pct < 0.05 else "#f1c40f" if pct < 0.2 else "#2ecc71"
    d = pd.DataFrame({"k": ["RUL"], "v": [pct]})
    bar = alt.Chart(d).mark_bar(color=color, size=40).encode(
        x=alt.X("v:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title="RUL vs service life")),
        y=alt.Y("k:N", title=None),
    )
    return bar.properties(height=90)


tabs = st.tabs(PUMPS)
for tab, pump in zip(tabs, PUMPS):
    with tab:
        for f in FEATURES:
            st.session_state.setdefault(widget_key(pump, f),
                                        float(st.session_state.readings[pump][f]))

        left, right = st.columns([1, 1.15])

        with left:
            st.subheader("Sensor inputs")
            st.caption("Slide to simulate readings. Auto-run also moves these bars.")
            reading = {}
            for f in FEATURES:
                lo, hi, _ = RANGES[f]
                label, unit = LABELS[f]
                step = (hi - lo) / 200.0
                reading[f] = st.slider(
                    f"{label} ({unit})",
                    min_value=float(round(lo, 2)),
                    max_value=float(round(hi, 2)),
                    step=float(round(step, 3)) if step > 0 else 0.1,
                    key=widget_key(pump, f),
                )
            st.session_state.readings[pump] = reading

        with right:
            s, pr, r = predict(reading)
            st.subheader("Prediction")
            m1, m2, m3 = st.columns(3)
            m1.metric("State", STATE_NAMES[s])
            m2.metric("RUL (hours)", f"{r:,.0f}")
            m3.metric("RUL (days)", f"{r/24:,.0f}")

            st.altair_chart(rul_gauge(r), use_container_width=True)

            st.write("**State probabilities**")
            pcols = st.columns(3)
            pcols[0].metric("Healthy", f"{pr[0]*100:.0f}%")
            pcols[1].metric("Warning", f"{pr[1]*100:.0f}%")
            pcols[2].metric("Critical", f"{pr[2]*100:.0f}%")

            st.subheader("Maintenance plan")
            actions, sched = maintenance_plan(reading, s, r)
            st.write(f"**Scheduling:** {sched}")
            for _, sev, comp, text in actions:
                if sev == "Critical":
                    st.error(f"**[{comp}]** {text}")
                elif sev == "Warning":
                    st.warning(f"**[{comp}]** {text}")
                else:
                    st.info(f"**[{comp}]** {text}")

            if st.button(f"Log current reading for {pump}", key=f"log_{pump}"):
                log_reading(pump, reading, s, r)
                st.success("Reading logged.")

        st.subheader("Operating trends")
        plog = pd.DataFrame([x for x in st.session_state.log if x["pump_id"] == pump])
        if len(plog) >= 2:
            g1, g2 = st.columns(2)
            with g1:
                st.altair_chart(
                    trend_chart(plog, "bearing_temp_C", "Bearing temp", "deg C",
                                warn=TH["bearing_warn"], crit=TH["bearing_crit"],
                                color="#e67e22"),
                    use_container_width=True)
                st.altair_chart(
                    trend_chart(plog, "vibration_mm_s", "Vibration", "mm/s",
                                warn=TH["vib_warn"], crit=TH["vib_crit"],
                                color="#9b59b6"),
                    use_container_width=True)
            with g2:
                st.altair_chart(
                    trend_chart(plog, "pressure_bar", "Pressure", "bar",
                                warn=TH["press_low"], color="#3498db"),
                    use_container_width=True)
                st.altair_chart(
                    trend_chart(plog, "predicted_RUL_hours", "Predicted RUL", "hours",
                                color="#2ecc71"),
                    use_container_width=True)
        else:
            st.caption("Generate or log at least two readings to see trends.")

st.divider()

with st.expander("Model insight - what drives the prediction (FMEA confirmation)"):
    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": CLF.feature_importances_,
    }).sort_values("importance", ascending=False)
    chart = alt.Chart(imp).mark_bar(color="#2f9e8f").encode(
        x=alt.X("importance:Q", title="importance"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", "importance"],
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Bearing temperature and vibration dominate the model - an "
               "independent, data-driven confirmation of the FMEA ranking in "
               "Chapter 5.")

st.subheader("Collected data log")
if st.session_state.log:
    log_df = pd.DataFrame(st.session_state.log)
    st.dataframe(log_df, use_container_width=True, height=240)
    buf = io.StringIO()
    log_df.to_csv(buf, index=False)
    st.download_button(
        "Download session data as CSV",
        data=buf.getvalue(),
        file_name=f"pump_session_log_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
    )
else:
    st.caption("No data collected yet. Use the generator or log a reading.")

if st.session_state.auto:
    time.sleep(auto_delay)
    step_all(False)
    st.rerun()
