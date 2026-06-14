"""
Ngezi Pump Station - Predictive Condition Monitoring & Control System
Demonstration system for BEng dissertation defence.
Author: Kimberly T. Mandikiyana
"""
import os, io, json, time
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Ngezi Pump Station - Predictive Maintenance",
                   page_icon="P", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))

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
    st.error("Could not load model files. Ensure pump_state_clf.pkl, "
             "pump_rul_reg.pkl and pump_meta.json sit next to app.py.\n\n"
             f"Details: {e}")
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
STATE_COLOR = {0: "#1D9E75", 1: "#EF9F27", 2: "#E24B4A"}

def default_reading():
    return {f: float(RANGES[f][2]) for f in FEATURES}

if "readings" not in st.session_state:
    st.session_state.readings = {p: default_reading() for p in PUMPS}
if "log" not in st.session_state:
    st.session_state.log = []
if "auto" not in st.session_state:
    st.session_state.auto = False

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
    actions = []
    bt, vb, pr = reading["bearing_temp_C"], reading["vibration_mm_s"], reading["pressure_bar"]
    mt, res = reading["motor_temp_C"], reading["reservoir_level_percent"]
    if bt >= TH["bearing_crit"]:
        actions.append(("Critical", "Bearing", "Bearing temperature in failure range. "
                        "Schedule immediate bearing inspection, relubrication or replacement."))
    elif bt >= TH["bearing_warn"]:
        actions.append(("Warning", "Bearing", "Bearing temperature elevated. "
                        "Check lubrication and plan a bearing inspection."))
    if vb >= TH["vib_crit"]:
        actions.append(("Critical", "Vibration", "Vibration in failure range. Check shaft "
                        "alignment, balancing and bearing wear (possible imbalance / cavitation)."))
    elif vb >= TH["vib_warn"]:
        actions.append(("Warning", "Vibration", "Vibration rising. Inspect alignment and "
                        "balance at the next opportunity."))
    if pr <= TH["press_low"]:
        actions.append(("Warning", "Hydraulics", "Discharge pressure low. Inspect mechanical "
                        "seal and check for cavitation or impeller erosion."))
    if mt >= 88:
        actions.append(("Warning", "Motor", "Motor temperature high. Check cooling, load "
                        "and electrical condition."))
    if res < RESERVOIR_MIN:
        actions.append(("Critical", "Reservoir", f"Reservoir below the {RESERVOIR_MIN:.0f}% "
                        "minimum operating level. Initiate duty/standby changeover."))
    if not actions:
        actions.append(("Normal", "General", "All indicators within normal limits. "
                        "Continue routine condition monitoring."))
    days = rul / 24.0
    if rul <= 500:
        sched = f"Plan maintenance immediately (RUL ~ {rul:.0f} h / {days:.0f} days)."
    elif rul <= 2000:
        sched = f"Schedule maintenance soon (RUL ~ {rul:.0f} h / {days:.0f} days)."
    else:
        sched = f"No urgent action (RUL ~ {rul:.0f} h / {days:.0f} days)."
    return actions, sched

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
        st.session_state.readings[p] = drift_reading(st.session_state.readings[p], inject)
    for p in PUMPS:
        s, _, r = predict(st.session_state.readings[p])
        log_reading(p, st.session_state.readings[p], s, r)

st.sidebar.title("Controls")
st.sidebar.caption("Ngezi Pump Station demonstrator")
st.sidebar.subheader("Random operation generator")
c1, c2 = st.sidebar.columns(2)
gen_once = c1.button("Generate step", use_container_width=True)
fault_btn = c2.button("Inject fault", use_container_width=True)
st.session_state.auto = st.sidebar.toggle("Auto-run (live)", value=st.session_state.auto)
auto_delay = st.sidebar.slider("Auto-run interval (seconds)", 1, 10, 3)
st.sidebar.divider()
if st.sidebar.button("Reset sliders to normal", use_container_width=True):
    st.session_state.readings = {p: default_reading() for p in PUMPS}
    st.rerun()
if st.sidebar.button("Clear data log", use_container_width=True):
    st.session_state.log = []
    st.rerun()
st.sidebar.divider()
st.sidebar.metric("Readings logged", len(st.session_state.log))

if gen_once:
    step_all(False)
if fault_btn:
    step_all(True)

st.title("Ngezi Pump Station - Predictive Condition Monitoring & Control")
st.caption("Three critical pumps monitored by a Random Forest condition classifier and RUL "
           "regressor. Move the sliders or use the random generator in the sidebar.")

overview = st.columns(3)
station_states = {}
for col, pump in zip(overview, PUMPS):
    s, pr, r = predict(st.session_state.readings[pump])
    station_states[pump] = s
    with col:
        st.markdown(
            f"<div style='border:1px solid {STATE_COLOR[s]};border-radius:10px;padding:12px 14px'>"
            f"<div style='font-size:18px;font-weight:600'>{pump}</div>"
            f"<div style='color:{STATE_COLOR[s]};font-size:22px;font-weight:700'>{STATE_NAMES[s]}</div>"
            f"<div style='color:#666'>RUL ~ {r:,.0f} h ({r/24:,.0f} days)</div>"
            f"<div style='color:#666'>Confidence {pr[s]*100:.0f}%</div></div>",
            unsafe_allow_html=True)

worst = max(station_states.values())
if worst == 2:
    st.error("STATION ALERT: at least one pump is CRITICAL. Review the maintenance plan below.")
elif worst == 1:
    st.warning("STATION NOTICE: at least one pump shows early degradation (Warning).")
else:
    st.success("All pumps operating within normal limits.")
st.divider()

tabs = st.tabs(PUMPS)
for tab, pump in zip(tabs, PUMPS):
    with tab:
        reading = st.session_state.readings[pump]
        left, right = st.columns([1, 1.1])
        with left:
            st.subheader("Sensor inputs")
            st.caption("Slide to simulate live sensor readings.")
            new_reading = {}
            for f in FEATURES:
                lo, hi, _ = RANGES[f]
                label, unit = LABELS[f]
                step = (hi - lo) / 200.0
                new_reading[f] = st.slider(f"{label} ({unit})",
                    min_value=float(round(lo, 2)), max_value=float(round(hi, 2)),
                    value=float(reading[f]),
                    step=float(round(step, 3)) if step > 0 else 0.1,
                    key=f"{pump}_{f}")
            st.session_state.readings[pump] = new_reading
            reading = new_reading
        with right:
            s, pr, r = predict(reading)
            st.subheader("Prediction")
            m1, m2, m3 = st.columns(3)
            m1.metric("State", STATE_NAMES[s])
            m2.metric("RUL (hours)", f"{r:,.0f}")
            m3.metric("RUL (days)", f"{r/24:,.0f}")
            st.write("**State probabilities**")
            st.progress(int(pr[0]*100), text=f"Healthy {pr[0]*100:.0f}%")
            st.progress(int(pr[1]*100), text=f"Warning {pr[1]*100:.0f}%")
            st.progress(int(pr[2]*100), text=f"Critical {pr[2]*100:.0f}%")
            st.subheader("Recommended maintenance plan")
            actions, sched = maintenance_plan(reading, s, r)
            st.write(f"**Scheduling:** {sched}")
            for sev, comp, text in actions:
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
            plog = plog.reset_index(drop=True)
            g1, g2 = st.columns(2)
            with g1:
                st.caption("Bearing temperature & vibration")
                st.line_chart(plog[["bearing_temp_C", "vibration_mm_s"]])
            with g2:
                st.caption("Discharge pressure & predicted RUL")
                st.line_chart(plog[["pressure_bar", "predicted_RUL_hours"]])
        else:
            st.caption("Generate or log at least two readings to see trends.")
st.divider()

st.subheader("Collected data log")
if st.session_state.log:
    log_df = pd.DataFrame(st.session_state.log)
    st.dataframe(log_df, use_container_width=True, height=260)
    buf = io.StringIO()
    log_df.to_csv(buf, index=False)
    st.download_button("Download session data as CSV", data=buf.getvalue(),
        file_name=f"pump_session_log_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")
else:
    st.caption("No data collected yet. Use the generator or log a reading.")

if st.session_state.auto:
    time.sleep(auto_delay)
    step_all(False)
    st.rerun()
