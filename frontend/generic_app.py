"""
frontend/generic_app.py  --  Generic AutoML Platform
Dataset-agnostic sibling of frontend/app.py (the Titanic-only demo).

Any CSV in, any CSV out: upload -> pick target -> auto-detect task type ->
train with FLAML AutoML -> point serving/generic_app.py at the result ->
predict through a form built dynamically from the serving API's /schema.

Build order (each section is a try/except island, same pattern as app.py):
  [1] Dataset      -- CSV upload + preview
  [2] Target        -- column dropdown + task-type auto-detection
  [3] Train         -- generic_prep + generic_train, shows FLAML's result
  [4] Prediction    -- GET /schema -> dynamic form -> POST /predict
"""
import contextlib
import functools
import http.server
import importlib
import io
import pathlib
import sys
import threading
import uuid

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup so `from generic_prep import ...` / `from generic_train import ...` work
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from generic_prep import detect_task_type, prep as run_prep    # noqa: E402
from generic_train import train as run_train                   # noqa: E402

UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SERVING_URL = "http://localhost:8001"
KFP_URL = "http://localhost:8082"
CSV_SERVER_PORT = 8765

# Plain-English explanations for the model families FLAML commonly picks.
# Matched by substring against FLAML's estimator name (e.g. "xgb_limitdepth"
# contains "xgb"), so exact FLAML naming quirks don't need to be enumerated.
MODEL_EXPLANATIONS = {
    # NOTE: "xgb_limitdepth" must stay ahead of the plain "xgb"/"xgboost"
    # entries below -- lookup matches the first substring hit, and
    # "xgb_limitdepth" contains "xgb", so a more specific key only wins if
    # it's checked first.
    "xgb_limitdepth": "XGBoost with limited tree depth: gradient boosting constrained to shallower trees, which reduces overfitting.",
    "lightgbm": "LightGBM: a fast gradient-boosting method that builds many small decision trees in sequence, each correcting the previous one's errors.",
    "lgbm": "LightGBM: a fast gradient-boosting method that builds many small decision trees in sequence, each correcting the previous one's errors.",
    "xgboost": "XGBoost: a powerful gradient-boosting method — builds decision trees in sequence, each fixing the last one's mistakes. Very accurate on structured data.",
    "xgb": "XGBoost: a powerful gradient-boosting method — builds decision trees in sequence, each fixing the last one's mistakes. Very accurate on structured data.",
    "histgb": "Histogram-based Gradient Boosting: bins feature values into histograms before building trees, making it fast on larger datasets.",
    "random_forest": "Random Forest: builds many independent decision trees and averages them for a stable prediction.",
    "rf": "Random Forest: builds many independent decision trees and averages them for a stable prediction.",
    "extra_trees": "Extra Trees: like Random Forest but with more randomness in how trees split — fast and robust.",
    "extra_tree": "Extra Trees: like Random Forest but with more randomness in how trees split — fast and robust.",
    "lrl1": "Logistic Regression: a simple, fast linear model — often best when the classes are cleanly separable.",
    "lrl2": "Logistic Regression: a simple, fast linear model — often best when the classes are cleanly separable.",
    "logistic": "Logistic Regression: a simple, fast linear model — often best when the classes are cleanly separable.",
    "catboost": "CatBoost: a gradient-boosting method that handles categorical data especially well.",
    "kneighbor": "K-Nearest Neighbors: predicts based on the most similar examples in the training data.",
    "knn": "K-Nearest Neighbors: predicts based on the most similar examples in the training data.",
    "sgd": "SGD Regressor: a linear model trained with stochastic gradient descent — fast and effective on larger or noisy datasets.",
    "enet": "Elastic Net: a linear regression model that blends two regularization penalties to handle correlated features and resist overfitting.",
    "lassolars": "Lasso (LARS): a linear regression model that automatically zeroes out less useful features, leaving only the most important ones.",
    "svc": "Support Vector Classifier: finds the boundary that best separates classes — effective on smaller or cleanly-separable datasets.",
}
MODEL_EXPLANATION_FALLBACK = "An automatically selected machine-learning model."


def _explain_model(name: str) -> str:
    _lname = (name or "").lower()
    for _key, _text in MODEL_EXPLANATIONS.items():
        if _key in _lname:
            return _text
    return MODEL_EXPLANATION_FALLBACK

# ---------------------------------------------------------------------------
# Page config -- MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Generic AutoML Platform",
    page_icon="\U0001F9E9",   # puzzle piece -- any dataset snaps in
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "df": None,                # uploaded DataFrame
    "csv_name": None,          # original filename
    "raw_path": None,          # saved copy on disk
    "target_col": None,
    "train_result": None,      # {"ok": bool, "meta": {...}, "log": str} | {"ok": False, "error": str}
    "kfp_run": None,           # {"ok": bool, "run_id": str} | {"ok": False, "error": str} | None (not attempted)
    "serve_load_result": None, # {"ok": bool, "data": {...}} | {"ok": False, "error": str} | None (not attempted)
    "serving_url": DEFAULT_SERVING_URL,
    "schema": None,            # {"ok": bool, "data": {...}} | {"ok": False, "kind": str}
    "prediction": None,        # {"ok": bool, "data": ..., "payload": ...} | error dict
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Global CSS tweaks -- same look as the Titanic app for a consistent platform feel
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }

[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: .75rem 1rem !important;
    box-shadow: 0 1px 6px rgba(15,23,42,.08) !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #2563EB !important;
}
[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-size: .78rem !important;
    text-transform: uppercase;
    letter-spacing: .05em;
}

button[kind="primary"] {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(37,99,235,.25) !important;
    transition: background .15s ease, box-shadow .15s ease, transform .12s ease !important;
}
button[kind="primary"]:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 6px 18px rgba(37,99,235,.4) !important;
    transform: translateY(-1px) !important;
}
button[kind="primary"]:active { transform: translateY(0) !important; }

button[kind="secondary"] {
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
    border: 1.5px solid #2563EB !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: background .15s ease, box-shadow .15s ease !important;
}
button[kind="secondary"]:hover {
    background: #DBEAFE !important;
    box-shadow: 0 4px 14px rgba(37,99,235,.22) !important;
}

[data-testid="stAlert"] {
    border-radius: 8px !important;
    box-shadow: 0 1px 5px rgba(15,23,42,.08) !important;
    border-left-width: 4px !important;
}

hr { border-color: #E2E8F0 !important; opacity: 1 !important; }

[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
}

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 10px !important;
    background: #F8FAFC !important;
    transition: border-color .18s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #2563EB !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# HEADER
# ===========================================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1D4ED8 0%, #4F46E5 100%);
    border-radius: 12px;
    padding: 1.6rem 2.2rem;
    color: white;
    margin-bottom: 1.2rem;">
  <h1 style="margin:0;color:white;font-size:2rem;letter-spacing:-0.4px;">
    Generic AutoML Platform
  </h1>
  <p style="margin:.35rem 0 0;opacity:.83;font-size:.92rem;">
    Any CSV &nbsp;&middot;&nbsp;
    Auto data cleaning &nbsp;&middot;&nbsp;
    FLAML AutoML &nbsp;&middot;&nbsp;
    Dynamic FastAPI serving
  </p>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# [1]  DATASET UPLOADER
# ===========================================================================
st.divider()
st.subheader("1. Dataset")
st.caption("Drag and drop any CSV. Rows, columns, types, and missing values are inspected immediately.")

try:
    _uploaded = st.file_uploader(
        "Drag and drop a CSV here, or click Browse",
        type=["csv"],
        help="Works with any tabular CSV -- Titanic, Iris, housing prices, or your own data.",
    )

    if _uploaded is not None:
        if st.session_state.csv_name != _uploaded.name:
            # New file -- reset everything downstream so stale results don't linger.
            st.session_state.df = pd.read_csv(_uploaded)
            st.session_state.csv_name = _uploaded.name
            st.session_state.target_col = None
            st.session_state.train_result = None

            _slug = f"{pathlib.Path(_uploaded.name).stem}_{uuid.uuid4().hex[:8]}.csv"
            _raw_path = UPLOAD_DIR / _slug
            st.session_state.df.to_csv(_raw_path, index=False)
            st.session_state.raw_path = str(_raw_path)

        _df = st.session_state.df

        _u1, _u2 = st.columns([3, 1], gap="large")

        with _u1:
            st.markdown(f"**Preview** -- first {min(10, len(_df))} of {len(_df)} rows")
            st.dataframe(_df.head(10), use_container_width=True, hide_index=False)

        with _u2:
            st.metric("Rows", f"{len(_df):,}")
            st.metric("Columns", len(_df.columns))

            st.markdown("**Columns**")
            for _col in _df.columns:
                _dtype = str(_df[_col].dtype)
                _missing = int(_df[_col].isna().sum())
                _miss_str = f"  ({_missing} missing)" if _missing else ""
                st.markdown(
                    f"- `{_col}` &nbsp; <span style='color:#64748B;font-size:.85rem;'>"
                    f"{_dtype}{_miss_str}</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No file uploaded yet. Drop any CSV above to get started.")

except Exception as _e:
    st.error("Dataset panel unavailable")
    st.exception(_e)


# ===========================================================================
# [2]  TARGET COLUMN + TASK-TYPE DETECTION
# ===========================================================================
st.divider()
st.subheader("2. Target column")
st.caption("Pick the column to predict. Task type is auto-detected the same way generic_prep.py does.")

_detected_task = None

try:
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    else:
        _df = st.session_state.df
        _options = ["-- select target --"] + list(_df.columns)
        _prev_idx = _options.index(st.session_state.target_col) if st.session_state.target_col in _options else 0
        _choice = st.selectbox("Target column", _options, index=_prev_idx)

        if _choice != "-- select target --":
            if _choice != st.session_state.target_col:
                st.session_state.target_col = _choice
                st.session_state.train_result = None  # target changed -- old result is stale

            _detected_task, _reason = detect_task_type(_df[_choice])

            if _detected_task == "classification":
                st.success(f"**Task type: CLASSIFICATION**\n\n{_reason}")
            else:
                st.success(f"**Task type: REGRESSION**\n\n{_reason}")
        else:
            st.session_state.target_col = None

except Exception as _e:
    st.error("Target panel unavailable")
    st.exception(_e)


# ===========================================================================
# KFP bonus-path helpers -- local training above is the primary, fast path.
# Submitting the same run to Kubeflow is a non-blocking bonus: any failure
# here (cluster offline, submission error) must never break the local result.
# ===========================================================================

@st.cache_resource(show_spinner=False)
def _start_csv_server():
    """Serves data/uploads/ over HTTP so a KFP pod (inside minikube) can pull
    a locally uploaded CSV via host.minikube.internal. Starts once per
    Streamlit server process and stays up for its lifetime -- the pipeline
    run fetches the file asynchronously, sometime after this call returns,
    so the server can't be torn down right after submission."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(UPLOAD_DIR))
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", CSV_SERVER_PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _load_model_on_serving_api(serving_url: str, model_path: pathlib.Path) -> dict:
    """POST /load on the serving API so it hot-swaps to the just-trained
    model in memory -- no uvicorn restart needed. Returns
    {"ok": True, "data": {...}} or {"ok": False, "error": str}; never raises."""
    try:
        r = requests.post(
            f"{serving_url.rstrip('/')}/load",
            json={"model_path": str(model_path)},
            timeout=5,
        )
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _submit_kfp_run(raw_path: str, target_column: str, time_budget: int) -> dict:
    """Returns {"ok": True, "run_id": str} or {"ok": False, "error": str}."""
    try:
        _start_csv_server()
        csv_filename = pathlib.Path(raw_path).name
        csv_url = f"http://host.minikube.internal:{CSV_SERVER_PORT}/{csv_filename}"

        import kfp                                              # noqa: PLC0415
        import pipelines.generic_pipeline as _gp_mod             # noqa: PLC0415

        # Force-reload so an edited pipeline definition is picked up even
        # though Streamlit only hot-reloads this file, not its imports.
        importlib.reload(_gp_mod)

        client = kfp.Client(host=KFP_URL)
        run = client.create_run_from_pipeline_func(
            _gp_mod.generic_pipeline,
            arguments={
                "csv_url": csv_url,
                "target_column": target_column,
                "drift_mode": False,
                "time_budget": time_budget,
            },
            experiment_name="generic-automl",
            run_name=f"streamlit-{pathlib.Path(raw_path).stem}",
        )
        return {"ok": True, "run_id": run.run_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ===========================================================================
# [3]  TRAIN
# ===========================================================================
st.divider()
st.subheader("3. Train")
st.caption(
    "Runs generic_prep.py (clean + encode) then generic_train.py (FLAML AutoML) "
    "on the uploaded data, entirely on this machine."
)

try:
    if st.session_state.df is None or st.session_state.target_col is None:
        st.info("Upload a dataset and pick a target column first.")
    else:
        _time_budget = st.slider(
            "FLAML time budget (seconds)", 10, 120, 30, step=5,
            help="How long FLAML is allowed to search model families and hyperparameters.",
        )

        if st.button("Train", type="primary", use_container_width=True):
            _log_buf = io.StringIO()
            try:
                with st.spinner("Cleaning data and running FLAML AutoML..."):
                    with contextlib.redirect_stdout(_log_buf):
                        _clean_df = run_prep(st.session_state.raw_path, st.session_state.target_col)
                        _clean_path = str(pathlib.Path(st.session_state.raw_path).with_name(
                            pathlib.Path(st.session_state.raw_path).stem + "_clean.csv"
                        ))
                        _meta = run_train(
                            _clean_path,
                            st.session_state.target_col,
                            time_budget=_time_budget,
                        )
                st.session_state.train_result = {"ok": True, "meta": _meta, "log": _log_buf.getvalue()}

                # Local training succeeded -- also try to hot-swap the serving
                # API to this model and Kubeflow to submit the same run, both
                # as non-blocking bonuses. Wrapped so a dead/unreachable
                # serving API or cluster can never take down the local result.
                _model_file = UPLOAD_DIR / f"{pathlib.Path(st.session_state.raw_path).stem}_clean_model.pkl"
                st.session_state.serve_load_result = _load_model_on_serving_api(
                    st.session_state.serving_url, _model_file
                )
                if st.session_state.serve_load_result.get("ok"):
                    # Refresh the prediction panel's schema automatically so
                    # it's ready to go without a manual "Load schema" click.
                    try:
                        _sr = requests.get(f"{st.session_state.serving_url.rstrip('/')}/schema", timeout=3)
                        _sr.raise_for_status()
                        st.session_state.schema = {"ok": True, "data": _sr.json()}
                        st.session_state.prediction = None  # stale from whatever was loaded before
                    except Exception:
                        pass  # non-critical -- the panel's own "Load schema" button still works

                with st.spinner("Submitting the same run to Kubeflow Pipelines..."):
                    st.session_state.kfp_run = _submit_kfp_run(
                        st.session_state.raw_path, st.session_state.target_col, _time_budget
                    )
            except Exception as _exc:
                st.session_state.train_result = {
                    "ok": False, "error": str(_exc), "log": _log_buf.getvalue(),
                }
                st.session_state.kfp_run = None
                st.session_state.serve_load_result = None

        _result = st.session_state.train_result
        if _result is None:
            st.caption("No training run yet this session.")
        elif not _result["ok"]:
            st.error(f"Training failed: {_result['error']}")
            with st.expander("Log"):
                st.code(_result["log"] or "(empty)")
        else:
            _meta = _result["meta"]
            st.success("Training complete.")

            _m1, _m2, _m3 = st.columns(3)
            _m1.metric("Task type", _meta["task_type"])
            _m2.metric("Best model (FLAML)", _meta["best_model"])
            if _meta["task_type"] == "classification":
                _m3.metric("Accuracy", f"{_meta['metrics']['accuracy']:.4f}")
            else:
                _m3.metric("R²", f"{_meta['metrics']['r2']:.4f}")
                st.metric("RMSE", f"{_meta['metrics']['rmse']:.4f}")

            st.caption(_explain_model(_meta["best_model"]))

            _serve_load = st.session_state.serve_load_result
            if _serve_load and _serve_load.get("ok"):
                st.caption(f"Serving API ({st.session_state.serving_url}) auto-switched to this model.")
            else:
                st.caption(
                    f"Could not auto-switch the serving API at {st.session_state.serving_url} "
                    "(offline?) -- the prediction panel's own Load schema button still works."
                )

            _kfp_run = st.session_state.kfp_run
            if _kfp_run and _kfp_run.get("ok"):
                _run_url = f"{KFP_URL}/#/runs/details/{_kfp_run['run_id']}"
                st.markdown(f"[View this run in Kubeflow →]({_run_url})")
            else:
                st.caption("KFP pipeline not available (cluster offline)")

            with st.expander("Preprocessing + training log"):
                st.code(_result["log"] or "(empty)")

except Exception as _e:
    st.error("Train panel unavailable")
    st.exception(_e)


# ===========================================================================
# [4]  PREDICTION -- DYNAMIC FORM FROM /schema
# ===========================================================================
st.divider()
st.subheader("4. Live prediction")
st.caption(
    "Calls GET /schema on the serving API to discover which features the currently "
    "loaded model needs, builds one input per feature, then POSTs to /predict."
)

try:
    st.session_state.serving_url = st.text_input(
        "Serving API URL", value=st.session_state.serving_url,
        help="Point this at whichever generic_app.py instance is running "
             "(e.g. http://localhost:8001 for Iris, 8002 for housing).",
    )
    _serving_url = st.session_state.serving_url.rstrip("/")

    _c_form, _spacer, _c_result = st.columns([5, 1, 5])

    predict_btn = False
    _feature_values = {}

    with _c_form:
        if st.button("Load schema", type="secondary"):
            try:
                _r = requests.get(f"{_serving_url}/schema", timeout=3)
                _r.raise_for_status()
                st.session_state.schema = {"ok": True, "data": _r.json()}
            except requests.exceptions.ConnectionError:
                st.session_state.schema = {"ok": False, "kind": "connection"}
            except Exception as _exc:
                st.session_state.schema = {"ok": False, "kind": "other", "msg": str(_exc)}

        _schema = st.session_state.schema
        if _schema is None:
            st.info("Click **Load schema** to fetch the feature list from the serving API.")
        elif not _schema["ok"]:
            if _schema.get("kind") == "connection":
                st.warning(
                    f"Cannot reach the serving API at `{_serving_url}`.\n\n"
                    "Start it in another terminal, e.g.:\n"
                    "```\nMODEL_NAME=iris_clean uvicorn serving.generic_app:app --port 8001\n```"
                )
            else:
                st.error(f"Schema fetch failed: {_schema.get('msg', 'unknown error')}")
        else:
            _sd = _schema["data"]
            st.markdown(
                f"**Model:** `{_sd['best_model']}` &nbsp;|&nbsp; "
                f"**Task:** `{_sd['task_type']}` &nbsp;|&nbsp; "
                f"**Target:** `{_sd['target_col']}`"
            )

            _feature_values = {}
            _cols = st.columns(2)
            for _i, _feat in enumerate(_sd["feature_cols"]):
                with _cols[_i % 2]:
                    _feature_values[_feat] = st.number_input(_feat, value=0.0, format="%.4f", key=f"feat_{_feat}")

            predict_btn = st.button("Predict", type="primary", use_container_width=True)

    if predict_btn and st.session_state.schema and st.session_state.schema.get("ok"):
        try:
            _resp = requests.post(f"{_serving_url}/predict", json=_feature_values, timeout=5)
            _resp.raise_for_status()
            st.session_state.prediction = {
                "ok": True, "data": _resp.json(), "payload": _feature_values,
                "task_type": st.session_state.schema["data"]["task_type"],
            }
        except requests.exceptions.ConnectionError:
            st.session_state.prediction = {"ok": False, "kind": "connection"}
        except requests.exceptions.Timeout:
            st.session_state.prediction = {"ok": False, "kind": "timeout"}
        except Exception as _exc:
            st.session_state.prediction = {"ok": False, "kind": "other", "msg": str(_exc)}

    with _c_result:
        _pred = st.session_state.prediction
        if _pred is None:
            st.caption("Load the schema, fill in the feature values, and press **Predict**.")
        elif not _pred["ok"]:
            _kind = _pred.get("kind", "other")
            if _kind == "connection":
                st.warning(f"Cannot reach the serving API at `{_serving_url}`.")
            elif _kind == "timeout":
                st.error("Serving API timed out after 5s.")
            else:
                st.error(f"Prediction failed: {_pred.get('msg', 'unknown error')}")
        else:
            _res = _pred["data"]
            if _pred["task_type"] == "classification":
                _cls = _res["prediction"]
                _prob = _res["probability"]
                st.markdown(
                    f'<div style="background:#EFF6FF;border:2px solid #2563EB;'
                    f'border-radius:10px;padding:1.5rem;text-align:center;'
                    f'margin-bottom:1.1rem;">'
                    f'<div style="font-size:1.9rem;font-weight:700;color:#1D4ED8;">'
                    f'Class {_cls}</div>'
                    f'<div style="color:#1D4ED8;font-size:1rem;margin-top:.4rem;">'
                    f'Probability: <strong>{_prob:.1%}</strong></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.progress(_prob)
                st.markdown("**Class probabilities**")
                st.json(_res["class_probabilities"])
            else:
                st.metric("Predicted value", f"{_res['prediction']:.4f}")

            with st.expander("Request payload"):
                st.json(_pred["payload"])
            with st.expander("Raw API response"):
                st.json(_res)

except Exception as _e:
    st.error("Prediction panel unavailable")
    st.exception(_e)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Generic AutoML demo  |  "
    "generic_prep.py + generic_train.py (FLAML)  |  "
    "serving/generic_app.py (FastAPI)  |  "
    "works with any CSV"
)
