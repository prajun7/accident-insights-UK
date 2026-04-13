# Running the accident severity predict server

This project serves a small web UI (`index.html`) and a JSON API (`POST /predict`) from **`predict_server.py`**, using the trained Random Forest bundle in **`output/7_rf_model.joblib`**.

## Prerequisites

- **Python 3** with dependencies from `requirements.txt` (at minimum `joblib`, `numpy`, `scikit-learn`, `pandas` for the export script).
- **`output/X_final.csv`** and **`output/y_final.csv`** if you need to (re)build the model file (produced by `src/feature_selection.py` in the usual pipeline).

## 1. Create the model file (once per retrain)

The server does not train on startup; it only loads **`output/7_rf_model.joblib`**.

From the **project root** (`accident-insights-UK/`):

```bash
python3 export_7_rf_model.py
```

This trains a `StandardScaler` + `RandomForestClassifier` (same settings as `src/classification.py`) on `X_final.csv` / `y_final.csv` and writes **`output/7_rf_model.joblib`**. Training on the full dataset can take a long time.

If you already have a compatible `7_rf_model.joblib` (dict with `scaler`, `model`, `feature_columns`), you can skip this step.

## 2. Start the HTTP server

From the **project root**:

```bash
python3 predict_server.py
```

Defaults:

- **URL:** [http://127.0.0.1:8765/](http://127.0.0.1:8765/)
- The page injects the predict URL automatically; the browser calls **`POST /predict`** on the same host and port.

### Options

| Flag | Default | Meaning |
|------|---------|--------|
| `--host` | `127.0.0.1` | Address to bind |
| `--port` | `8765` | TCP port |

Example on another port:

```bash
python3 predict_server.py --port 8766
```

Then open **http://127.0.0.1:8766/**.

Stop the server with **Ctrl+C** in the terminal.

## 3. If the port is already in use

You may see `Address already in use` if another process (often an old `predict_server.py`) is still bound to that port.

- Use a different port: `python3 predict_server.py --port 8766`
- Or find and stop the process, e.g. on macOS/Linux: `lsof -i :8765` then stop the listed PID.

## 4. Quick API check (optional)

With the server running:

```bash
curl -s -X POST http://127.0.0.1:8765/predict \
  -H 'Content-Type: application/json' \
  -d '{"Speed_limit":30,"Number_of_Vehicles":1,"Vehicle_Manoeuvre":17,"Road_Type":6,"IsNight":0,"Urban_or_Rural_Area":1,"Sex_of_Driver":1,"Junction_Detail":0,"Age_Band_of_Driver":6,"Light_Conditions":1}'
```

You should get JSON with `prediction`, `label` (Fatal / Serious / Slight), and `probabilities`.

## Files involved

| File | Role |
|------|------|
| `predict_server.py` | Stdlib HTTP server: `GET /` → `index.html` (with predict URL injected), `POST /predict` |
| `index.html` | Form for the ten features used in `X_final.csv` |
| `export_7_rf_model.py` | Builds `output/7_rf_model.joblib` from `X_final.csv` / `y_final.csv` |
| `output/7_rf_model.joblib` | `joblib` dict: `scaler`, `model`, `feature_columns` |
