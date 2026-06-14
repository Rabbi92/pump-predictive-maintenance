# Ngezi Pump Station - Predictive Maintenance Demonstrator

Self-contained web app for the MEng dissertation defence. Monitors three
critical pumps (PUM-01, PUM-02, PUM-03) without external sensors. Sensor
readings come from sliders or a built-in random generator; the system
predicts pump state, estimates Remaining Useful Life (RUL), raises
three-level alerts, recommends a maintenance plan, plots live trends, and
logs every reading to a downloadable CSV.

## Files (all must be in the same folder)
- app.py                 : the Streamlit application
- pump_state_clf.pkl     : trained Random Forest state classifier
- pump_rul_reg.pkl       : trained Random Forest RUL regressor
- pump_meta.json         : features, slider ranges, thresholds, service life
- requirements.txt       : Python dependencies (scikit-learn pinned to 1.6.1)

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy free on Streamlit Community Cloud
1. Create a free account at https://streamlit.io/cloud (sign in with GitHub).
2. Create a public GitHub repo and upload all five files above.
3. On Streamlit Cloud: New app -> select the repo -> main file app.py -> Deploy.
4. You get a public URL like https://your-app.streamlit.app for the defence.

## scikit-learn version
The models were trained with scikit-learn 1.6.1, which is pinned in
requirements.txt so the .pkl files load without a version-mismatch warning.

## Using it during the defence
- Sliders (per-pump tab): move any sensor; prediction, RUL, alert and the
  maintenance plan update instantly.
- Random generator (sidebar): "Generate step" = one realistic drift across
  all pumps; "Inject fault" = strong degradation that drives a pump toward
  Critical; "Auto-run (live)" = steps automatically on a timer.
- Trends: bearing temperature, vibration, pressure and predicted RUL plot
  live once two or more readings exist for a pump.
- Data log: every reading is collected in the bottom table; download it as
  CSV in the training-dataset schema plus predicted state and RUL.

## Defence talking points
- Feature importance ranks bearing temperature and vibration highest - an
  independent, data-driven confirmation of the FMEA in Chapter 5.
- RUL regressor: R2 = 0.98, MAE ~ 186 hours on the held-out test set.
- The three-level alert and 45% reservoir minimum operating level mirror the
  control scheme described in the dissertation.
