"""
frontend/app.py  --  Titanic MLOps Platform
Single-page Streamlit demo wiring together the full MLOps stack.

Build order (each section is a try/except island):
  [3] Prediction   -- POST to FastAPI serving endpoint   (this pass)
  [1] Dataset      -- CSV upload + preview
  [2] Run Pipeline -- submit clean KFP run
  [4] Simulate Drift -- submit drifted KFP run
"""
import pathlib
import sys

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup so `from pipelines.titanic_pipeline import ...` works later
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SERVING_URL = "http://localhost:8001"
KFP_URL = "http://localhost:8082"

# ---------------------------------------------------------------------------
# Page config -- MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic MLOps Platform",
    page_icon="\U0001f6a2",   # ship -- unicode escape avoids cp1252 source issues
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "prediction": None,    # {"ok": bool, "data": ..., "payload": ...} | error dict
    "clean_run":  None,    # {"ok": bool, "run_id": str} | {"ok": False, "error": str}
    "drift_run":  None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Global CSS tweaks -- minimal, mostly relies on Streamlit defaults
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Layout ──────────────────────────────────────────────── */
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }

/* ── Metric cards ─────────────────────────────────────────── */
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

/* ── Primary buttons ──────────────────────────────────────── */
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

/* ── Secondary buttons ────────────────────────────────────── */
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

/* ── Alert boxes (status badges) ──────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    box-shadow: 0 1px 5px rgba(15,23,42,.08) !important;
    border-left-width: 4px !important;
}

/* ── Dividers ─────────────────────────────────────────────── */
hr { border-color: #E2E8F0 !important; opacity: 1 !important; }

/* ── Expanders ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
}

/* ── File uploader drop zone ──────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 10px !important;
    background: #F8FAFC !important;
    transition: border-color .18s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #2563EB !important;
}

/* ── Dataframe ────────────────────────────────────────────── */
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
    Titanic MLOps Platform
  </h1>
  <p style="margin:.35rem 0 0;opacity:.83;font-size:.92rem;">
    Kubeflow Pipelines &nbsp;&middot;&nbsp;
    FastAPI serving &nbsp;&middot;&nbsp;
    KS drift detection &nbsp;&middot;&nbsp;
    RandomForest classifier
  </p>
</div>
""", unsafe_allow_html=True)

# Status badges row
_b1, _b2, _b3, _b4 = st.columns(4)
with _b1:
    try:
        _r = requests.get(f"{SERVING_URL}/health", timeout=2)
        if _r.ok:
            st.success("Model API: Live")
        else:
            st.error("Model API: Error")
    except Exception:
        st.error("Model API: Offline  --  run port-forward on 8001")
with _b2:
    st.info(f"KFP UI: {KFP_URL}")
with _b3:
    st.info("RandomForest  |  76.97% acc")
with _b4:
    st.info("Titanic dataset  |  889 rows")


# ===========================================================================
# [1]  DATASET UPLOADER
# ===========================================================================
st.divider()
st.subheader("Dataset")
st.caption(
    "Upload a Titanic CSV to inspect it before sending a pipeline run. "
    "The standard dataset has 891 rows and 12 columns."
)

try:
    _uploaded = st.file_uploader(
        "Drag and drop a Titanic CSV here, or click Browse",
        type=["csv"],
        help="Expected columns: Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, ...",
    )

    if _uploaded is not None:
        import pandas as _pd

        _df = _pd.read_csv(_uploaded)

        _u1, _u2 = st.columns([3, 1], gap="large")

        with _u1:
            st.markdown(f"**Preview** -- first {min(10, len(_df))} of {len(_df)} rows")
            st.dataframe(
                _df.head(10),
                use_container_width=True,
                hide_index=False,
            )
            st.markdown(
                "<div style='font-size:.8rem;color:#475569;line-height:1.7;"
                "margin-top:.4rem;padding:.45rem .75rem;"
                "background:#F8FAFC;border-left:3px solid #CBD5E1;"
                "border-radius:0 4px 4px 0;'>"
                "<strong style='color:#374151;'>Column guide</strong><br>"
                "<b>SibSp</b> &mdash; siblings / spouses aboard &emsp;"
                "<b>Parch</b> &mdash; parents / children aboard<br>"
                "<b>Fare</b> &mdash; ticket price in British pounds (wealth indicator) &emsp;"
                "<b>Embarked</b> &mdash; boarding port: "
                "S&nbsp;=&nbsp;Southampton &middot; C&nbsp;=&nbsp;Cherbourg &middot; Q&nbsp;=&nbsp;Queenstown"
                "</div>",
                unsafe_allow_html=True,
            )

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

            # Survival breakdown if column exists
            if "Survived" in _df.columns:
                st.divider()
                _n_surv = int(_df["Survived"].sum())
                _n_total = len(_df.dropna(subset=["Survived"]))
                st.metric(
                    "Survival rate",
                    f"{_n_surv / _n_total:.1%}" if _n_total else "N/A",
                    help=f"{_n_surv} survived of {_n_total}",
                )
    else:
        st.info(
            "No file uploaded yet.  "
            "Download the dataset from the "
            "[pandas docs GitHub](https://raw.githubusercontent.com/pandas-dev/"
            "pandas/main/doc/data/titanic.csv) and drag it in above."
        )

except Exception as _e:
    st.error("Dataset panel unavailable")
    st.exception(_e)


# ===========================================================================
# [3]  PASSENGER SURVIVAL PREDICTION
# ===========================================================================
st.divider()
st.subheader("Live Survival Prediction")
st.caption(
    "Sends passenger features to the RandomForest model deployed as a FastAPI "
    "service on Minikube.  Results are returned in real time."
)

_c_form, _spacer, _c_result = st.columns([5, 1, 5])

# ── Input form ──────────────────────────────────────────────────────────────
with _c_form:
    try:
        pclass = st.selectbox(
            "Passenger class",
            [1, 2, 3],
            index=2,
            format_func=lambda x: {1: "1 -- First (Upper)", 2: "2 -- Second (Middle)",
                                    3: "3 -- Third  (Lower)"}[x],
        )
        sex = st.radio("Sex", ["male", "female"], horizontal=True)
        age = st.slider("Age (years)", 1, 80, 22)

        _f1, _f2, _f3 = st.columns(3)
        with _f1:
            sibsp = st.number_input("SibSp", 0, 8, 0,
                                    help="Siblings / spouses aboard")
        with _f2:
            parch = st.number_input("Parch", 0, 6, 0,
                                    help="Parents / children aboard")
        with _f3:
            fare = st.number_input("Fare (GBP)", 0.0, 600.0, 7.25, step=5.0)

        embarked = st.radio(
            "Port of embarkation",
            ["S", "C", "Q"],
            format_func=lambda x: {"S": "Southampton (S)",
                                    "C": "Cherbourg (C)",
                                    "Q": "Queenstown (Q)"}[x],
            horizontal=True,
        )
        predict_btn = st.button(
            "Predict Survival", type="primary", use_container_width=True
        )
    except Exception as _e:
        st.error(f"Form error: {_e}")
        predict_btn = False

# ── Handle button press ──────────────────────────────────────────────────────
if predict_btn:
    _payload = {
        "Pclass":   int(pclass),
        "Sex":      sex,
        "Age":      float(age),
        "SibSp":    int(sibsp),
        "Parch":    int(parch),
        "Fare":     float(fare),
        "Embarked": embarked,
    }
    try:
        _resp = requests.post(f"{SERVING_URL}/predict", json=_payload, timeout=5)
        _resp.raise_for_status()
        st.session_state.prediction = {
            "ok": True, "data": _resp.json(), "payload": _payload,
        }
    except requests.exceptions.ConnectionError:
        st.session_state.prediction = {"ok": False, "kind": "connection"}
    except requests.exceptions.Timeout:
        st.session_state.prediction = {"ok": False, "kind": "timeout"}
    except Exception as _e:
        st.session_state.prediction = {"ok": False, "kind": "other", "msg": str(_e)}

# ── Result panel ─────────────────────────────────────────────────────────────
with _c_result:
    try:
        _pred = st.session_state.prediction

        if _pred is None:
            # placeholder before first prediction
            st.markdown("### Try these classic passengers")
            st.table({
                "Passenger": ["Jack", "Rose"],
                "Pclass":    [3, 1],
                "Sex":       ["male", "female"],
                "Age":       [22, 29],
                "Fare":      [7.25, 100.0],
                "Expected":  ["Does not survive", "Survives"],
            })
            st.caption("Set the sliders and press **Predict Survival**.")

        elif not _pred["ok"]:
            _kind = _pred.get("kind", "other")
            if _kind == "connection":
                st.warning(
                    "**Cannot reach the model API** at `localhost:8001`.\n\n"
                    "Start the port-forward in another terminal:\n"
                    "```\nkubectl port-forward -n serving "
                    "svc/titanic-serving 8001:80\n```"
                )
            elif _kind == "timeout":
                st.error("Model API timed out after 5 s.  The pod may be starting up.")
            else:
                st.error(f"Prediction failed: {_pred.get('msg', 'unknown error')}")

        else:
            _result   = _pred["data"]
            _survived = _result["prediction"] == 1
            _prob     = _result["probability"]

            if _survived:
                _col, _bg, _verdict = "#065F46", "#D1FAE5", "SURVIVED"
            else:
                _col, _bg, _verdict = "#7F1D1D", "#FEE2E2", "DID NOT SURVIVE"

            # Big verdict box
            st.markdown(
                f'<div style="background:{_bg};border:2px solid {_col};'
                f'border-radius:10px;padding:1.5rem;text-align:center;'
                f'margin-bottom:1.1rem;">'
                f'<div style="font-size:1.9rem;font-weight:700;color:{_col};">'
                f'{_verdict}</div>'
                f'<div style="color:{_col};font-size:1rem;margin-top:.4rem;">'
                f'Survival probability: <strong>{_prob:.1%}</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Probability bar
            st.progress(_prob)
            st.caption(f"P(survived) = {_prob:.4f}")

            # Metrics row
            _m1, _m2 = st.columns(2)
            _m1.metric("Prediction", "Survived" if _survived else "Not Survived")
            _m2.metric("Confidence", f"{max(_prob, 1 - _prob):.1%}")

            # Expandable raw details
            with st.expander("Request payload"):
                st.json(_pred["payload"])
            with st.expander("Raw API response"):
                st.json(_result)

    except Exception as _e:
        st.error("Prediction panel error")
        st.exception(_e)


# ===========================================================================
# [2] + [4]  KUBEFLOW PIPELINE CONTROLS
# ===========================================================================
st.divider()
st.subheader("Kubeflow Pipeline Controls")
st.caption(
    f"Submits runs to the KFP instance at {KFP_URL}.  "
    "Both buttons use the same `titanic_pipeline` function; "
    "`drift_mode=True` injects Age+30 and Fare*8+100 so the KS test fires."
)

# -- Lazy KFP import (so prediction panel works even if kfp is missing) ------
def _submit_kfp_run(
    drift_mode: bool,
    n_estimators: int = 100,
    max_depth: int = 0,
    min_samples_split: int = 2,
) -> dict:
    """Returns {"ok": True, "run_id": str} or {"ok": False, "error": str}."""
    try:
        import importlib                                            # noqa: PLC0415
        import kfp                                                 # noqa: PLC0415
        import pipelines.titanic_pipeline as _tp_mod               # noqa: PLC0415

        # Force-reload so Streamlit's module cache never serves a stale version
        # of the pipeline function (Streamlit only hot-reloads app.py, not
        # transitive imports, so without this the pre-HP-change function object
        # would be compiled and KFP would reject the unknown argument names).
        importlib.reload(_tp_mod)
        _pipeline_fn = _tp_mod.titanic_pipeline

        _client = kfp.Client(host=KFP_URL)
        _run_name = (
            "streamlit-drifted-demo" if drift_mode else "streamlit-clean-run"
        )
        _run = _client.create_run_from_pipeline_func(
            _pipeline_fn,
            arguments={
                "drift_mode": drift_mode,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
            },
            experiment_name="titanic",
            run_name=_run_name,
        )
        return {"ok": True, "run_id": _run.run_id}
    except Exception as _exc:
        return {"ok": False, "error": str(_exc)}


def _run_status_widget(run_state: dict | None, label: str) -> None:
    """Render stored run result (or nothing if not yet submitted)."""
    if run_state is None:
        st.caption(f"No {label} run submitted this session.")
        return
    if run_state["ok"]:
        _rid = run_state["run_id"]
        st.success(f"Run submitted: `{_rid}`")
        st.markdown(
            f"[Open in KFP UI]({KFP_URL}/#/runs/details/{_rid})",
        )
    else:
        st.error(f"Submission failed: {run_state['error']}")


try:
    _p2, _gap, _p4 = st.columns([5, 1, 5])

    # ── [2] Clean pipeline run ────────────────────────────────────────────
    with _p2:
        st.markdown(
            '<div style="background:#EFF6FF;border:1px solid #BFDBFE;'
            'border-radius:10px;padding:1.2rem 1.4rem;margin-bottom:.8rem;'
            'box-shadow:0 1px 5px rgba(37,99,235,.1);">'
            '<strong style="color:#1E40AF;font-size:1.05rem;">Clean pipeline run</strong><br>'
            '<span style="color:#374151;font-size:.9rem;">'
            'Runs all four steps: generate reference stats &rarr; data prep '
            '&rarr; drift check (passes) &rarr; train.  '
            'The validation gate opens and the RandomForest model is trained.'
            '</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size:.82rem;font-weight:600;color:#374151;"
            "margin:.7rem 0 .25rem;'>Model hyperparameters</div>",
            unsafe_allow_html=True,
        )
        _hp1, _hp2, _hp3 = st.columns(3)
        with _hp1:
            _n_estimators = st.slider(
                "Trees", 10, 300, 100, step=10, key="hp_n_est",
                help="n_estimators: number of trees in the forest",
            )
            st.markdown(
                "<span style='font-size:.76rem;color:#64748B;'>"
                "More trees = more stable. 100&ndash;200 is the sweet spot; "
                "diminishing returns after."
                "</span>",
                unsafe_allow_html=True,
            )
        with _hp2:
            _unlimited = st.checkbox(
                "Unlimited depth", value=True, key="hp_unlimited",
                help="Unchecked: set max_depth via slider",
            )
            _max_depth = 0 if _unlimited else st.slider(
                "Max depth", 1, 20, 5, key="hp_depth",
                help="max_depth: maximum depth of each tree",
            )
            st.markdown(
                "<span style='font-size:.76rem;color:#64748B;'>"
                "Controls complexity. 5&ndash;10 avoids overfitting on this "
                "small dataset; unlimited can overfit."
                "</span>",
                unsafe_allow_html=True,
            )
        with _hp3:
            _min_samples = st.slider(
                "Min split", 2, 10, 2, key="hp_mss",
                help="min_samples_split: min samples required to split a node",
            )
            st.markdown(
                "<span style='font-size:.76rem;color:#64748B;'>"
                "Higher = simpler trees. "
                "5&ndash;10 can reduce overfitting."
                "</span>",
                unsafe_allow_html=True,
            )

        if st.button(
            "Run Clean Pipeline",
            type="primary",
            use_container_width=True,
            key="btn_clean",
        ):
            with st.spinner("Submitting to KFP ..."):
                st.session_state.clean_run = _submit_kfp_run(
                    drift_mode=False,
                    n_estimators=_n_estimators,
                    max_depth=_max_depth,
                    min_samples_split=_min_samples,
                )

        _run_status_widget(st.session_state.clean_run, "clean")

        with st.expander("What this does"):
            st.markdown("""
1. `generate_reference_stats` — downloads clean Titanic CSV, computes per-column
   mean/std/values as a JSON artifact (may use KFP cache).
2. `data_prep` — same CSV, standard preprocessing.
3. `drift_check` — KS test passes (p=1.0, shift=0 std for all columns).
4. Validation gate **OPENS** → `train` runs, model artifact saved.
""")

    # ── [4] Drifted pipeline run ──────────────────────────────────────────
    with _p4:
        st.markdown(
            '<div style="background:#FFF7ED;border:1px solid #FED7AA;'
            'border-radius:10px;padding:1.2rem 1.4rem;margin-bottom:.8rem;'
            'box-shadow:0 1px 5px rgba(234,88,12,.1);">'
            '<strong style="color:#9A3412;font-size:1.05rem;">Simulate drift</strong><br>'
            '<span style="color:#374151;font-size:.9rem;">'
            'Injects synthetic drift into the incoming data: '
            '<strong>Age + 30 years</strong> and <strong>Fare &times; 8 + 100</strong>. '
            'The KS test fires on both columns and the validation gate '
            '<strong>blocks training entirely</strong>.'
            '</span></div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "Simulate Drift  (gate will BLOCK training)",
            type="secondary",
            use_container_width=True,
            key="btn_drift",
        ):
            with st.spinner("Submitting drifted run to KFP ..."):
                st.session_state.drift_run = _submit_kfp_run(drift_mode=True)

        _run_status_widget(st.session_state.drift_run, "drifted")

        with st.expander("What the drift report will show"):
            st.markdown("""
| Column | KS stat | p-value | Mean shift | Verdict |
|--------|---------|---------|------------|---------|
| Pclass | 0.000 | 1.000 | 0.00 std | [OK] |
| Sex | 0.000 | 1.000 | 0.00 std | [OK] |
| **Age** | **0.794** | **~0** | **2.31 std** | **[DRIFT]** |
| SibSp | 0.000 | 1.000 | 0.00 std | [OK] |
| Parch | 0.000 | 1.000 | 0.00 std | [OK] |
| **Fare** | **0.951** | **~0** | **6.53 std** | **[DRIFT]** |
| Embarked | 0.000 | 1.000 | 0.00 std | [OK] |

Drift check returns `False` &rarr; `dsl.If` condition not met &rarr; `train` is **skipped**.
Click the drift-check node in the KFP UI and open the **Artifacts** tab to see the
full Markdown report.
""")

except Exception as _e:
    st.error("Pipeline controls panel unavailable")
    st.exception(_e)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Titanic MLOps demo  |  "
    "Kubeflow Pipelines 2.16.1  |  "
    "scikit-learn RandomForest  |  "
    "FastAPI on Minikube  |  "
    "KS drift detection via scipy"
)
