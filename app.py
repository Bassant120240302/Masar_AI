"""
Masar AI  |  مسار AI
=====================
Streamlit deployment for the Thanaweya Amma faculty-recommendation model.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

See README.md for: how the hybrid recommender works, where to drop extra
classifier models so they show up as choices, and how the RAG assistant
tab is wired up.
"""

import io
import os
import re
import sys
import json
import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import db
import model_utils
from model_utils import SUBJECT_COLS as DEFAULT_SUBJECT_COLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

ASSET_PATHS = {
    "config": os.path.join(BASE_DIR, "recommender_config.pkl"),
    "label_enc": os.path.join(BASE_DIR, "label_enc.pkl"),
    "profile_pipeline": os.path.join(BASE_DIR, "profile_pipeline.pkl"),
    "faculty_profiles": os.path.join(BASE_DIR, "faculty_profiles.pkl"),
    
}
TANSIQ_CSV_PATH = os.path.join(BASE_DIR, "tansiq_cutoffs_2022.csv")
STUDENT_DATASET_PATH = os.path.join(BASE_DIR, "thanaweya_final_with_profiles_modified.csv")
KNOWN_ASSET_FILENAMES = {os.path.basename(p) for p in ASSET_PATHS.values()}

db.init_db()

st.set_page_config(
    page_title="Masar AI | مسار",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

TRACK_OPTIONS = ["Scientific - Science Track", "Scientific - Math Track", "Literary Track"]

# The trained pipelines (profile_pipeline.pkl and the classifier) were fit on a
# 'Track' column containing these short codes ('Science' / 'Math' / 'Literacy'),
# NOT the friendly labels shown in the UI dropdown above. TrackOneHotEncoder
# (see model_utils.py) does `X['Track'] == cat` for each of its learned
# categories — if the value it receives doesn't match one of those exact
# strings, every Track_* column silently comes out as 0, i.e. the model gets
# NO track information at all, which is what was causing science-track
# students to get math-track-looking results (and vice versa). This map
# translates the dropdown label to the code the pipelines actually expect.
TRACK_DISPLAY_TO_CODE = {
    "Scientific - Science Track": "Science",
    "Scientific - Math Track": "Math",
    "Literary Track": "Literacy",
}

# Standard Egyptian Thanaweya Amma tansiq eligibility: which broad faculty
# categories each Track can actually apply to. Needed as a hard filter
# because content-similarity mode (no classifier loaded) has NO track
# information in its feature vector at all (full_skill_interest_cols is
# Skills/Interests only) — without this, it can freely recommend e.g.
# Engineering to a Literary-track student. Even with a classifier loaded,
# this acts as a safety net against a wrong/low-confidence prediction.
SCIENCE_ONLY_FACULTIES = {
    "Medicine", "Medicine Dentistry", "Medicine Veterinary", "Pharmacy",
    "Nursing", "Physical Therapy", "Applied Health Technology",
    "Technical Health Institute", "Agriculture", "Aquaculture and Fisheries",
    "Science",
}
SCIENCE_AND_MATH_FACULTIES = {"Engineering", "Computers and Information"}
ALL_TRACKS_FACULTIES = {
    "Arts", "Commerce", "Law", "Economics", "Education",
    "Early Childhood Education", "Mass Communication",
    "Languages (Al-Alsun)", "Dar Al-Uloom", "Archaeology",
    "Social Service", "Tourism and Hotels", "Disability and Rehabilitation",
    "Sports Education", "Other",
}
LITERARY_ONLY_FACULTIES = {"Technical Commercial"}
TRACK_ELIGIBLE_FACULTIES = {
    "Science": SCIENCE_ONLY_FACULTIES | SCIENCE_AND_MATH_FACULTIES | ALL_TRACKS_FACULTIES,
    "Math": SCIENCE_AND_MATH_FACULTIES | ALL_TRACKS_FACULTIES,
    "Literacy": ALL_TRACKS_FACULTIES | LITERARY_ONLY_FACULTIES,
}

# tansiq_cutoffs_2022.csv has data-entry duplicates: programs like Medicine,
# Medicine Dentistry, Medicine Veterinary, Pharmacy, Nursing, Agriculture,
# Aquaculture/Fisheries, Technical Health, and even the plain "Science"
# faculty are each listed under BOTH 'Scientific - Science Track' (correct)
# AND 'Scientific - Math Track' (wrong — same college, same cutoff score,
# duplicated), which is why a Math-track student's RAG search could surface
# e.g. "Medicine Cairo" even filtered to their own track. Since the CSV's
# own Track_EN column can't be trusted for these rows, this keyword-based
# classifier gives a second, hard-coded opinion on eligibility, used as an
# extra filter on top of (not instead of) the Track_EN column.
_RAG_SCIENCE_ONLY_KEYWORDS = [
    "medicine", "pharmacy", "nursing", "agricultur", "aquaculture",
    "fisheries", "technical health", "applied health", "health applied",
    "therap",  # Physical Therapy — was missing, let literary/math track
               # students see "Therapy Physical (therapy) ..." rows unfiltered
]
_RAG_SCIENCE_AND_MATH_KEYWORDS = ["engineering", "computer"]
_RAG_LITERARY_ONLY_KEYWORDS = ["technical commercial"]


def rag_track_eligible(college_name: str, track_code: str) -> bool:
    """True if `college_name` (a raw College_University_EN value from
    tansiq_cutoffs_2022.csv) is plausibly open to `track_code`
    ('Science' / 'Math' / 'Literacy'), per the keyword rules above.
    Defaults to eligible (True) for anything that doesn't match a
    restrictive keyword, so this only ever narrows results, never
    invents new exclusions for faculties it doesn't recognize."""
    name = str(college_name).lower()
    if any(kw in name for kw in _RAG_SCIENCE_ONLY_KEYWORDS) or name.startswith("science "):
        category = "science_only"
    elif any(kw in name for kw in _RAG_SCIENCE_AND_MATH_KEYWORDS):
        category = "science_and_math"
    elif any(kw in name for kw in _RAG_LITERARY_ONLY_KEYWORDS):
        category = "literary_only"
    else:
        return True  # not a recognized restricted category — allow it

    if category == "science_and_math":
        return track_code in ("Science", "Math")
    if category == "science_only":
        return track_code == "Science"
    if category == "literary_only":
        return track_code == "Literacy"
    return True

THEMES = {
    "🌙 Dark": {
        "bg": "#0f1117", "bg2": "#171a24", "card": "#1c2030", "text": "#eef1f8",
        "muted": "#9aa3b8", "accent": "#7c6cf0", "accent2": "#22d3ee", "border": "#2a2f42",
    },
    "☀️ Light": {
        "bg": "#f7f8fc", "bg2": "#ffffff", "card": "#ffffff", "text": "#161a23",
        "muted": "#5b6376", "accent": "#5b4bdb", "accent2": "#0891b2", "border": "#e4e6f0",
    },
    "🩵 Sky Blue": {
        "bg": "#eaf6ff", "bg2": "#f5fbff", "card": "#ffffff", "text": "#0b2540",
        "muted": "#4a6580", "accent": "#0284c7", "accent2": "#06b6d4", "border": "#cdeaff",
    },
}


def inject_css(theme_name: str):
    t = THEMES[theme_name]
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Cairo:wght@600;700;800&display=swap');
    html, body, [class*="css"], .stApp {{ font-family:'Poppins','Cairo',sans-serif !important; color:{t['text']}; }}
    .stApp {{ background: linear-gradient(180deg, {t['bg']} 0%, {t['bg2']} 100%); }}
    h1,h2,h3,h4 {{ font-weight:800 !important; letter-spacing:-0.01em; }}
    p,li,label,span,div {{ font-weight:500; }}
    .masar-hero {{ background: linear-gradient(120deg, {t['accent']} 0%, {t['accent2']} 100%);
        padding:28px 32px; border-radius:20px; margin-bottom:22px; box-shadow:0 10px 30px rgba(0,0,0,0.18); }}
    .masar-hero h1 {{ color:white !important; margin:0; font-size:2.1rem; font-weight:800 !important; }}
    .masar-hero p {{ color:rgba(255,255,255,0.92) !important; margin-top:6px; font-size:1.02rem; font-weight:600; }}
    .masar-card {{ background:{t['card']}; border:1px solid {t['border']}; border-radius:16px;
        padding:20px 22px; margin-bottom:14px; box-shadow:0 4px 14px rgba(0,0,0,0.06); }}
    .masar-badge {{ display:inline-block; padding:4px 12px; border-radius:999px; font-size:0.78rem;
        font-weight:700; background:{t['accent']}22; color:{t['accent']}; margin-right:6px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:6px; }}
    .stTabs [data-baseweb="tab"] {{ font-weight:700 !important; border-radius:10px 10px 0 0; padding:10px 18px; }}
    div.stButton > button {{ font-weight:700; border-radius:12px; border:none;
        background: linear-gradient(120deg, {t['accent']} 0%, {t['accent2']} 100%);
        color:white; padding:10px 22px; box-shadow:0 4px 14px {t['accent']}55; }}
    div.stButton > button:hover {{ filter:brightness(1.08); }}
    section[data-testid="stSidebar"] {{ background:{t['bg2']}; border-right:1px solid {t['border']}; }}
    .rank-card {{ background:{t['card']}; border:1px solid {t['border']}; border-radius:14px;
        padding:16px 18px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center; }}
    .rank-num {{ font-size:1.4rem; font-weight:800; color:{t['accent']}; margin-right:12px; }}
    .rag-card {{ background:{t['card']}; border:1px solid {t['border']}; border-radius:14px;
        padding:14px 16px; margin-bottom:10px; }}
    .rag-score {{ float:right; font-size:.75rem; font-weight:700; color:{t['accent']}; }}
    </style>
    """, unsafe_allow_html=True)


# ==========================================================================
# Asset / model loading helpers
# ==========================================================================
@st.cache_resource
def load_recommender_assets():
    """Loads the 4 bundled recommender files (config, label encoder, profile
    pipeline, faculty profiles). Returns (assets_dict, missing_list)."""
    model_utils.register_classes_on_main()
    assets, missing = {}, []
    for key, path in ASSET_PATHS.items():
        if os.path.exists(path):
            try:
                assets[key] = joblib.load(path)
            except Exception as e:
                missing.append(f"{os.path.basename(path)} (failed to load: {e})")
        else:
            missing.append(os.path.basename(path))
    return assets, missing


def _tansiq_csv_signature():
    """mtime+size of tansiq_cutoffs_2022.csv, used as a cache key below so the
    dataframe reloads if the file is added/changed after the app already ran
    once — without this, load_tansiq_data() would permanently cache `None`
    from a run where the CSV was missing, and never notice it later showed up."""
    if os.path.exists(TANSIQ_CSV_PATH):
        stat = os.stat(TANSIQ_CSV_PATH)
        return f"{stat.st_mtime_ns}-{stat.st_size}"
    return "missing"


@st.cache_resource
def load_tansiq_data(_sig: str):
    if os.path.exists(TANSIQ_CSV_PATH):
        try:
            return pd.read_csv(TANSIQ_CSV_PATH, encoding="utf-8-sig")
        except Exception as e:
            st.sidebar.error(f"Failed to read tansiq_cutoffs_2022.csv: {e}")
            return None
    return None


def _student_dataset_signature():
    """Same mtime+size cache-busting pattern as _tansiq_csv_signature(), for
    the optional bundled real student dataset
    (thanaweya_final_with_profiles_modified.csv). Not required for the app
    to run — the Data/EDA/Visualization tabs fall back to a small synthetic
    sample if it's absent — but if present, it becomes the default dataset
    for those tabs instead of the synthetic sample, with no upload needed."""
    if os.path.exists(STUDENT_DATASET_PATH):
        stat = os.stat(STUDENT_DATASET_PATH)
        return f"{stat.st_mtime_ns}-{stat.st_size}"
    return "missing"


@st.cache_data(show_spinner="Loading bundled student dataset…")
def load_bundled_student_dataset(_sig: str):
    if os.path.exists(STUDENT_DATASET_PATH):
        try:
            return pd.read_csv(STUDENT_DATASET_PATH, encoding="utf-8-sig")
        except Exception as e:
            st.sidebar.error(f"Failed to read thanaweya_final_with_profiles_modified.csv: {e}")
            return None
    return None


@st.cache_resource
def build_tansiq_index(_df_marker: str):
    """TF-IDF index over the tansiq cutoffs table. _df_marker is just the row
    count, used as a cache key so this rebuilds if the dataframe changes."""
    df = load_tansiq_data(_tansiq_csv_signature())
    if df is None:
        return None, None
    text = (df["College_University_EN"].fillna("") + " " +
            df["Track_EN"].fillna("") + " " +
            df.get("KeySkillsRequired", "").fillna(""))
    # char n-grams (not word-level) so a partial word like "alex" still matches
    # "Alexandria" — word-level TF-IDF gave that combo ZERO similarity with
    # every single row, since "alex" never appears as its own token, which
    # made query+track+score searches silently ignore the query entirely.
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vec.fit_transform(text)
    return vec, matrix


def discover_classifier_files():
    """Scans BASE_DIR and BASE_DIR/models for .pkl/.joblib files that aren't
    the known recommender asset files — each becomes a selectable model."""
    found = {}
    for d in (BASE_DIR, MODELS_DIR):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn in KNOWN_ASSET_FILENAMES:
                continue
            if fn.lower().endswith((".pkl", ".joblib")):
                label = os.path.splitext(fn)[0].replace("_", " ").title()
                found[label] = os.path.join(d, fn)
    return found


@st.cache_resource
def load_classifier_from_path(path: str):
    model_utils.register_classes_on_main()
    return joblib.load(path)


def transform_through(steps, X):
    Xt = X
    for _, step in steps:
        Xt = step.transform(Xt)
    return Xt


def classifier_uses_skills_interests(pipeline) -> bool:
    """True if `pipeline` has a skills/interests multi-hot step (i.e. it's one
    of the 3-signal Track+grades+Skills/Interests models, not the original
    Track+grades-only final_model_pipeline.pkl which drops them)."""
    return hasattr(pipeline, "named_steps") and "skills_mh" in pipeline.named_steps


def classifier_predict_proba(pipeline, X):
    """Manually chains .transform() through every step but the last, then
    calls .predict_proba() on the final estimator directly. Avoids an
    overly strict check_is_fitted() that some sklearn versions raise on
    Pipeline.predict_proba() for custom transformer steps that don't set a
    trailing-underscore attribute."""
    if hasattr(pipeline, "steps"):
        *pre_steps, (_, last_step) = pipeline.steps
        Xt = transform_through(pre_steps, X)
        return last_step.predict_proba(Xt), last_step
    # Bare estimator (not a Pipeline)
    return pipeline.predict_proba(X), pipeline


# ==========================================================================
# Session state defaults
# ==========================================================================
defaults = {
    "logged_in": False, "username": None, "theme": "🌙 Dark",
    "dataset": None, "uploaded_models": {},
    # None = "not decided yet this session" so the sidebar block below can
    # auto-pick a bundled classifier the first time; once the user (or the
    # auto-pick) sets a real value it's remembered for the rest of the session.
    "selected_classifier": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

inject_css(st.session_state.theme)
assets, missing_assets = load_recommender_assets()
tansiq_df = load_tansiq_data(_tansiq_csv_signature())


# ==========================================================================
# Login page
# ==========================================================================
def login_page():
    st.markdown("""
    <div class="masar-hero" style="text-align:center; margin-top:40px;">
        <h1>🎓 Masar AI — مسار</h1>
        <p>Your AI compass from Thanaweya Amma results to the right faculty</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.container(border=True):
            tab_in, tab_up = st.tabs(["🔐 Sign in", "🆕 Create account"])
            with tab_in:
                u = st.text_input("Username", key="login_user")
                p = st.text_input("Password", type="password", key="login_pass")
                if st.button("Login", use_container_width=True):
                    if db.verify_user(u, p):
                        db.touch_login(u)
                        db.log_event(u, "login")
                        st.session_state.logged_in = True
                        st.session_state.username = u
                        st.rerun()
                    else:
                        db.log_event(u or "(empty)", "login_failed")
                        st.error("Invalid username or password.")
            with tab_up:
                nu = st.text_input("Choose a username", key="signup_user")
                np1 = st.text_input("Choose a password", type="password", key="signup_pass")
                np2 = st.text_input("Confirm password", type="password", key="signup_pass2")
                if st.button("Create account", use_container_width=True):
                    if not nu:
                        st.error("Pick a username.")
                    elif len(np1) < 4:
                        st.error("Password needs at least 4 characters.")
                    elif np1 != np2:
                        st.error("Passwords don't match.")
                    elif db.user_exists(nu):
                        st.error("That username is already taken.")
                    else:
                        db.add_user(nu, np1)
                        db.touch_login(nu)
                        db.log_event(nu, "signup")
                        st.session_state.logged_in = True
                        st.session_state.username = nu
                        st.rerun()
                st.caption("Your account is saved in a real database (masar_ai.db) — "
                           "it's still there next time you sign in.")


if not st.session_state.logged_in:
    login_page()
    st.stop()

# ==========================================================================
# Sidebar
# ==========================================================================
with st.sidebar:
    st.markdown(f"### 👋 Welcome, {st.session_state.username}")
    st.session_state.theme = st.selectbox("🎨 Theme", list(THEMES.keys()),
                                           index=list(THEMES.keys()).index(st.session_state.theme))
    inject_css(st.session_state.theme)

    st.markdown("---")
    st.markdown("### 🧬 Recommender assets")
    if not missing_assets:
        st.success("✅ config / label encoder / profile pipeline / faculty profiles all loaded.")
    else:
        st.warning("⚠️ Missing: " + ", ".join(missing_assets))
    st.caption("These power the content-based (Skills/Interests) half of the recommender. "
               "They live next to app.py as fixed filenames — see README.md.")

    st.markdown("---")
    st.markdown("### 🤖 Classifier model")
    discovered = discover_classifier_files()
    model_files = st.file_uploader("Add model(s) (.pkl / .joblib)", type=["pkl", "joblib"],
                                    accept_multiple_files=True, key="model_uploader")
    if model_files:
        for f in model_files:
            label = os.path.splitext(f.name)[0].replace("_", " ").title()
            if label not in st.session_state.uploaded_models:
                try:
                    model_utils.register_classes_on_main()
                    obj = joblib.load(io.BytesIO(f.getvalue()))
                    st.session_state.uploaded_models[label] = obj
                    db.log_event(st.session_state.username, "model_upload", f.name)
                    st.success(f"Loaded '{label}' ✅")
                except Exception as e:
                    db.log_event(st.session_state.username, "model_upload_failed", f"{f.name}: {e}")
                    st.error(f"Couldn't load {f.name}: {e}")

    options = ["— None (similarity-only) —"] + list(discovered.keys()) + list(st.session_state.uploaded_models.keys())
    if st.session_state.selected_classifier is None:
        # First render this session: if a classifier file is bundled on the
        # server (models/ or next to app.py), make it the default active
        # classifier automatically — no manual selection needed. Falls back
        # to similarity-only if nothing is bundled.
        st.session_state.selected_classifier = next(iter(discovered), options[0])
    if st.session_state.selected_classifier not in options:
        st.session_state.selected_classifier = options[0]
    st.session_state.selected_classifier = st.selectbox(
        "Active classifier", options,
        index=options.index(st.session_state.selected_classifier))
    if discovered:
        st.caption("Auto-detected next to app.py (default classifier): " + ", ".join(discovered.keys()))
    st.caption("Drop `final_model_pipeline.pkl` (or `best_rf.pkl` / `cat_pipeline.pkl`) into "
               "`models/` next to app.py and it becomes the default active classifier for "
               "every visitor automatically — no manual selection needed.")

    st.markdown("---")
    st.markdown("### 📁 Dataset (optional)")
    data_file = st.file_uploader("Upload CSV for Data/EDA/Viz tabs", type=["csv"], key="data_uploader")
    if data_file is not None:
        try:
            st.session_state.dataset = pd.read_csv(data_file)
            db.log_event(st.session_state.username, "dataset_upload",
                         f"{data_file.name} ({st.session_state.dataset.shape[0]} rows)")
            st.success(f"Dataset loaded: {st.session_state.dataset.shape[0]:,} rows")
        except Exception as e:
            db.log_event(st.session_state.username, "dataset_upload_failed", str(e))
            st.error(f"Couldn't read CSV: {e}")

    st.markdown("---")
    if st.button("Log out", use_container_width=True):
        db.log_event(st.session_state.username, "logout")
        st.session_state.logged_in = False
        st.rerun()


def get_active_classifier():
    label = st.session_state.selected_classifier
    if label == "— None (similarity-only) —":
        return None, None
    if label in st.session_state.uploaded_models:
        return st.session_state.uploaded_models[label], label
    if label in discovered:
        return load_classifier_from_path(discovered[label]), label
    return None, None


# ==========================================================================
# Hero
# ==========================================================================
st.markdown("""
<div class="masar-hero">
    <h1>🎓 Masar AI — مسار</h1>
    <p>Explore the data, understand the pipeline, get AI-powered faculty recommendations,
    and ask the RAG assistant about real 2022 tansiq cutoffs.</p>
</div>
""", unsafe_allow_html=True)


FALLBACK_FACULTY_CATS = ["Engineering", "Medicine", "Commerce", "Arts", "Science",
                          "Economics", "Education", "Other"]


def sample_dataset(n=800, seed=42):
    rng = np.random.default_rng(seed)
    branches = rng.choice(["أدبي", "علمي علوم", "علمي رياضة"], n, p=[0.35, 0.35, 0.3])
    tracks = np.where(branches == "أدبي", "Literary Track",
                       np.where(branches == "علمي علوم", "Scientific - Science Track",
                                "Scientific - Math Track"))
    df = pd.DataFrame({"branch": branches, "Track": tracks})
    for c in DEFAULT_SUBJECT_COLS:
        df[c] = np.where(rng.random(n) < 0.45, np.nan, rng.integers(0, 61, n))
    df["Percentage"] = rng.uniform(40, 99, n).round(2)
    df["status"] = rng.choice(["ناجح", "دور ثاني"], n, p=[0.85, 0.15])
    cats = list(assets["label_enc"].classes_) if "label_enc" in assets else FALLBACK_FACULTY_CATS
    df["Faculty_Category"] = rng.choice(cats, n)
    skills_vocab = _skill_vocab()
    interests_vocab = _interest_vocab()
    df["Skills"] = rng.choice(skills_vocab, n) if skills_vocab else "Communication"
    df["Interests"] = rng.choice(interests_vocab, n) if interests_vocab else "Learning"
    return df


def get_dataset():
    if st.session_state.dataset is not None:
        return st.session_state.dataset
    bundled = load_bundled_student_dataset(_student_dataset_signature())
    if bundled is not None:
        return bundled
    return sample_dataset()


def using_real_dataset() -> bool:
    """True if get_dataset() is returning either an uploaded CSV or the
    bundled real dataset — i.e. NOT the synthetic fallback sample."""
    if st.session_state.dataset is not None:
        return True
    return load_bundled_student_dataset(_student_dataset_signature()) is not None


def _skill_vocab():
    pp = assets.get("profile_pipeline")
    if pp is None:
        return []
    try:
        return dict(pp.steps)["skills_mh"].vocab_
    except Exception:
        return []


def _interest_vocab():
    pp = assets.get("profile_pipeline")
    if pp is None:
        return []
    try:
        return dict(pp.steps)["interests_mh"].vocab_
    except Exception:
        return []


# ==========================================================================
# Tabs
# ==========================================================================
tab_data, tab_eda, tab_model, tab_rag, tab_viz, tab_logs = st.tabs(
    ["📊 Data", "🔬 EDA & Preprocessing", "🤖 Model & Recommendation",
     "🧠 RAG Assistant", "📈 Visualization", "🗂️ Logs"]
)

# ---- TAB 1: DATA ------------------------------------------------------------
with tab_data:
    df = get_dataset()
    if st.session_state.dataset is not None:
        st.success(f"Using your uploaded dataset ({df.shape[0]:,} rows).")
    elif using_real_dataset():
        st.success(f"Using the bundled real student dataset "
                   f"(thanaweya_final_with_profiles_modified.csv, {df.shape[0]:,} rows).")
    else:
        st.info("No dataset found — showing a small synthetic sample so you can preview the layout. "
                 "Upload your real CSV from the sidebar, or bundle "
                 "thanaweya_final_with_profiles_modified.csv next to app.py, to explore actual results.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Faculty categories", df["Faculty_Category"].nunique() if "Faculty_Category" in df else "—")

    st.markdown("#### Preview")
    st.dataframe(df.head(50), use_container_width=True, height=320)

    with st.expander("Column types & missing values"):
        info = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
        })
        st.dataframe(info, use_container_width=True)

# ---- TAB 2: EDA & PREPROCESSING --------------------------------------------
with tab_eda:
    df = get_dataset()
    left, right = st.columns(2)
    with left:
        if "Percentage" in df.columns:
            fig = px.histogram(df, x="Percentage", nbins=40, title="Percentage distribution",
                                color_discrete_sequence=[THEMES[st.session_state.theme]["accent"]])
            st.plotly_chart(fig, use_container_width=True)
        if "branch" in df.columns:
            fig = px.pie(df, names="branch", title="Branch split", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if "status" in df.columns:
            fig = px.bar(df["status"].value_counts().reset_index(),
                         x="status", y="count", title="Pass / retake status")
            st.plotly_chart(fig, use_container_width=True)
        num_cols = [c for c in DEFAULT_SUBJECT_COLS if c in df.columns]
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                             title="Subject-score correlation", color_continuous_scale="Tealrose")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🧪 Preprocessing pipeline (as built in the notebook)")
    st.markdown("""
    <div class="masar-card">
    <span class="masar-badge">Step 1</span> <b>Structural imputation</b> — a missing subject score means the
    student didn't take that subject, so it's filled with 0.<br><br>
    <span class="masar-badge">Step 2</span> <b>Feature engineering</b> — <code>total</code>,
    <code>total_pct</code>, <code>avg_subject_score</code>, and per-group averages (science/humanities/language)
    are computed from the raw subject scores.<br><br>
    <span class="masar-badge">Step 3</span> <b>Track encoding</b> — the 3-way <code>Track</code> column is
    one-hot encoded.<br><br>
    <span class="masar-badge">Step 4</span> <b>Two parallel branches</b> — the classifier branch drops
    Skills/Interests entirely; a separate <b>profile pipeline</b> keeps them, multi-hot encoding both into
    <code>skill_*</code>/<code>interest_*</code> columns for content-based similarity.<br><br>
    <span class="masar-badge">Step 5</span> <b>Selective Yeo-Johnson + feature selection</b> — near-zero-variance
    and highly correlated columns dropped from the classifier branch (CatBoost's branch skips scaling/power
    transforms, since it doesn't need them).
    </div>
    """, unsafe_allow_html=True)

# ---- TAB 3: MODEL & RECOMMENDATION -----------------------------------------
with tab_model:
    classifier, classifier_label = get_active_classifier()
    have_profile = "profile_pipeline" in assets and "faculty_profiles" in assets
    label_enc = assets.get("label_enc")
    cfg = assets.get("config")

    if classifier is None and not have_profile:
        st.warning("No classifier selected and no recommender assets found — add a model in the sidebar, "
                   "or make sure recommender_config.pkl / profile_pipeline.pkl / faculty_profiles.pkl / "
                   "label_enc.pkl sit next to app.py.")
    else:
        uses_skills_note = classifier is not None and classifier_uses_skills_interests(classifier)
        mode_note = ("hybrid (3-signal model + content similarity)" if classifier and have_profile and uses_skills_note else
                     "hybrid (model + content similarity)" if classifier and have_profile else
                     "model-only, 3-signal (Track + grades + Skills/Interests)" if classifier and uses_skills_note else
                     "model-only (no recommender assets found)" if classifier else
                     "content-similarity-only (no classifier selected — Skills/Interests drive this, "
                     "exam scores don't)")
        st.info(f"Active mode: **{mode_note}**" + (f" · classifier: **{classifier_label}**" if classifier_label else ""))
        if uses_skills_note:
            st.caption("This classifier was trained on Track + exam scores + Skills/Interests together — "
                       "not just exam scores + Track.")

        st.markdown("#### 🧾 Student profile")
        with st.form("predict_form"):
            c1, c2 = st.columns([1, 2])
            with c1:
                track = st.selectbox("Track", TRACK_OPTIONS)
                skills = st.multiselect("Skills", _skill_vocab())
                interests = st.multiselect("Interests", _interest_vocab())
                if classifier and have_profile:
                    mw = st.slider("Model weight", 0.0, 1.0, 0.7, 0.05)
                    sw = round(1 - mw, 2)
                    st.caption(f"Similarity weight: {sw}")
                else:
                    mw, sw = 1.0, 1.0
                top_n = st.slider("How many faculties to show", 1, 29, 3)
            with c2:
                st.caption("Enter subject scores (leave at 0 for subjects not taken)")
                cols = st.columns(3)
                scores = {}
                for i, subj in enumerate(DEFAULT_SUBJECT_COLS):
                    with cols[i % 3]:
                        scores[subj] = st.number_input(subj.replace("_", " ").title(),
                                                         min_value=0, max_value=100, value=0, step=1,
                                                         key=f"score_{subj}")
            submitted = st.form_submit_button(f"🎯 Get top-{top_n} recommendations", use_container_width=True)

        if submitted:
            try:
                student_raw = dict(scores)
                student_raw["Track"] = track
                student_raw["Skills"] = ", ".join(skills)
                student_raw["Interests"] = ", ".join(interests)

                track_code = TRACK_DISPLAY_TO_CODE[track]

                model_scores = None
                if classifier is not None:
                    uses_skills = classifier_uses_skills_interests(classifier)
                    if uses_skills:
                        # 3-signal model (Track + grades + Skills/Interests) — feed it
                        # the raw comma-joined Skills/Interests text too, so its own
                        # skills_mh/interests_mh steps can multi-hot encode them same
                        # as during training, instead of the model only ever seeing
                        # exam scores + Track.
                        base_row = {c: scores.get(c, np.nan) for c in DEFAULT_SUBJECT_COLS}
                        base_row["Track"] = track_code
                        base_row["Skills"] = student_raw["Skills"]
                        base_row["Interests"] = student_raw["Interests"]
                        Xc = pd.DataFrame([base_row])
                    else:
                        feature_cols_raw = (cfg or {}).get("feature_cols_raw",
                                                            DEFAULT_SUBJECT_COLS + ["Track", "total"])
                        base_row = {c: scores.get(c, np.nan) for c in DEFAULT_SUBJECT_COLS}
                        base_row["Track"] = track_code
                        base_row["total"] = float(sum(scores.values()))
                        Xc = pd.DataFrame([base_row])[[c for c in feature_cols_raw if c in base_row]]
                    proba, last_step = classifier_predict_proba(classifier, Xc)
                    proba = proba[0]
                    raw_classes = getattr(last_step, "classes_", np.arange(len(proba)))
                    class_names = (label_enc.inverse_transform(raw_classes.astype(int))
                                   if label_enc is not None else raw_classes)
                    model_scores = pd.Series(proba, index=class_names)

                sim_scores = None
                if have_profile:
                    full_skill_interest_cols = cfg["full_skill_interest_cols"]
                    prow = {c: scores.get(c, np.nan) for c in DEFAULT_SUBJECT_COLS}
                    prow["Track"] = track_code
                    prow["total"] = np.nan
                    prow["Skills"] = student_raw["Skills"]
                    prow["Interests"] = student_raw["Interests"]
                    Xp = pd.DataFrame([prow])
                    Xt = transform_through(assets["profile_pipeline"].steps, Xp)
                    vec = Xt[full_skill_interest_cols].values
                    fac_profiles = assets["faculty_profiles"]
                    sims = cosine_similarity(vec, fac_profiles.values)[0]
                    sim_scores = pd.Series(sims, index=fac_profiles.index)
                    if sim_scores.max() > 0:
                        sim_scores = sim_scores / sim_scores.max()

                if model_scores is not None and sim_scores is not None:
                    all_f = model_scores.index.union(sim_scores.index)
                    model_scores = model_scores.reindex(all_f, fill_value=0)
                    sim_scores = sim_scores.reindex(all_f, fill_value=0)
                    combined = (mw * model_scores + sw * sim_scores).sort_values(ascending=False)
                elif model_scores is not None:
                    combined = model_scores.sort_values(ascending=False)
                else:
                    combined = sim_scores.sort_values(ascending=False)

                eligible = TRACK_ELIGIBLE_FACULTIES.get(track_code, set())
                blocked = [f for f in combined.index if f not in eligible and combined[f] > 0]
                combined = combined[combined.index.isin(eligible)].sort_values(ascending=False)

                result = combined.head(top_n)
                db.log_event(st.session_state.username, "recommendation_run",
                             f"Track={track}, classifier={classifier_label}, top{top_n}={list(result.index)}")

                st.markdown(f"#### 🏆 Top {top_n} recommended faculty categories")
                if blocked:
                    st.caption(f"🚫 Filtered out {len(blocked)} faculties not open to this track "
                               f"(e.g. {', '.join(blocked[:3])}).")
                palette = [THEMES[st.session_state.theme]["accent"], THEMES[st.session_state.theme]["accent2"], "#a78bfa"]
                colors = [palette[i % len(palette)] for i in range(len(result))]
                for rank, (name, score) in enumerate(result.items(), start=1):
                    st.markdown(f"""
                    <div class="rank-card">
                        <div><span class="rank-num">#{rank}</span><b>{name}</b></div>
                        <div style="font-weight:700; color:{colors[rank-1]};">{score*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                fig = go.Figure(go.Bar(
                    x=result.values[::-1] * 100, y=result.index[::-1], orientation="h",
                    marker_color=colors[:len(result)][::-1]))
                fig.update_layout(title="Confidence by recommendation", xaxis_title="Score (%)",
                                   height=280, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                db.log_event(st.session_state.username, "recommendation_failed", str(e))
                st.error(f"Prediction failed: {e}")

# ---- TAB 4: RAG ASSISTANT ---------------------------------------------------
with tab_rag:
    st.markdown("#### 🧠 Ask about real 2022 tansiq (coordination) cutoffs")
    if tansiq_df is None:
        st.warning("tansiq_cutoffs_2022.csv wasn't found next to app.py — the RAG assistant needs it.")
    else:
        st.caption(f"Retrieval runs over {len(tansiq_df):,} real 2022 college programs "
                   "(TF-IDF search — no data leaves your machine for this part).")

        with st.expander("🔑 Optional: generate a natural-language answer with an LLM"):
            st.caption("Leave this empty to just see the matching rows (extractive). Paste an Anthropic "
                       "API key here to have a model write a short answer grounded in those rows. The key "
                       "is used only for this request and isn't stored anywhere.")
            api_key = st.text_input("Anthropic API key", type="password", key="rag_api_key")

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            query = st.text_input("Ask in English or Arabic", placeholder="e.g. medicine colleges in Cairo")
        with c2:
            track_filter = st.selectbox("Track filter", ["Any"] + sorted(tansiq_df["Track_EN"].dropna().unique().tolist()))
        with c3:
            my_score = st.number_input("My coordination score (optional)", min_value=0, max_value=410, value=0, step=1)
        with c4:
            max_results = st.number_input("Max results to show", min_value=1, max_value=200, value=8, step=1)

        if st.button("🔍 Search", use_container_width=False):
            work_df = tansiq_df.copy()
            blocked_count = 0
            if track_filter != "Any":
                work_df = work_df[work_df["Track_EN"] == track_filter]
                track_code = TRACK_DISPLAY_TO_CODE.get(track_filter)
                if track_code:
                    eligible_mask = work_df["College_University_EN"].apply(
                        lambda name: rag_track_eligible(name, track_code))
                    blocked_count = int((~eligible_mask).sum())
                    work_df = work_df[eligible_mask]
            if my_score > 0:
                work_df = work_df[work_df["MinScore_2022_out_of_410"] <= my_score]

            if query.strip():
                vec, matrix = build_tansiq_index(str(len(tansiq_df)))
                q_vec = vec.transform([query])
                # restrict cosine search to the filtered subset's rows
                sub_idx = work_df.index
                sims = cosine_similarity(q_vec, matrix[sub_idx])[0] if len(sub_idx) else np.array([])
                work_df = work_df.assign(_score=sims)
                # Keep only rows that genuinely match the query text — without
                # this, a query with ~0 similarity to everything (e.g. a typo)
                # would just re-show the full track/score-filtered set in its
                # original order, silently ignoring what was typed.
                work_df = work_df[work_df["_score"] > 0]
                # Exact substring hits (e.g. typing a governorate name) always
                # outrank fuzzy character-ngram noise, so an obvious literal
                # match never gets buried under loosely-similar rows.
                q_lower = query.strip().lower()
                work_df = work_df.assign(
                    _exact=(work_df["College_University_EN"].str.lower().str.contains(q_lower, regex=False) |
                            work_df["Track_EN"].str.lower().str.contains(q_lower, regex=False)))
                work_df = work_df.sort_values(["_exact", "_score"], ascending=[False, False])
            else:
                work_df = work_df.sort_values("MinScore_2022_out_of_410", ascending=False)

            total_matches = len(work_df)
            results = work_df.head(int(max_results))
            db.log_event(st.session_state.username, "rag_query",
                         f"q='{query}', track={track_filter}, my_score={my_score}, hits={len(results)}")

            active_filters = []
            active_filters.append(f"Track = {track_filter}" if track_filter != "Any" else "Track = Any")
            active_filters.append(f"Score ≤ {my_score}" if my_score > 0 else "Score = (not set)")
            active_filters.append(f"Query = \"{query.strip()}\"" if query.strip() else "Query = (empty)")
            st.caption("🔎 Filters applied together (AND): " + " · ".join(active_filters))

            if results.empty:
                st.info("No matching programs found — try loosening the filters." +
                        (" Your search text didn't match any program in the current "
                         "track/score filter — try fewer or different words."
                         if query.strip() else ""))
            else:
                st.markdown(f"##### 📋 Retrieved programs (showing {len(results)} of {total_matches:,} matches)")
                if blocked_count:
                    st.caption(f"🚫 Filtered out {blocked_count} program(s) that the raw data mistakenly listed "
                               f"under {track_filter} but that actually require a different track (e.g. "
                               f"Medicine/Pharmacy/Nursing/Agriculture require the Science track specifically).")
                for _, r in results.iterrows():
                    st.markdown(f"""
                    <div class="rag-card">
                        <b>{r['College_University_EN']}</b> <span class="rag-score">{r['Track_EN']}</span><br>
                        <span style="opacity:.8; font-size:.85rem;">Min score: {r['MinScore_2022_out_of_410']:.0f}/410
                        ({r['Percentage']:.1f}%)</span><br>
                        <span style="opacity:.7; font-size:.8rem;">{r.get('KeySkillsRequired','')}</span>
                    </div>
                    """, unsafe_allow_html=True)

                if api_key:
                    with st.spinner("Asking the model..."):
                        try:
                            import urllib.request
                            context = "\n".join(
                                f"- {r['College_University_EN']} ({r['Track_EN']}): min score "
                                f"{r['MinScore_2022_out_of_410']:.0f}/410 ({r['Percentage']:.1f}%). "
                                f"Key skills: {r.get('KeySkillsRequired','')}"
                                for _, r in results.iterrows()
                            )
                            prompt = (
                                "You are a concise academic-advising assistant for Egyptian Thanaweya Amma "
                                "students. Using ONLY the context below, answer the student's question in "
                                "2-4 short sentences. If the context doesn't fully answer it, say what's "
                                "missing.\n\nContext:\n" + context + f"\n\nQuestion: {query}"
                            )
                            payload = json.dumps({
                                "model": "claude-sonnet-4-6",
                                "max_tokens": 400,
                                "messages": [{"role": "user", "content": prompt}],
                            }).encode()
                            req = urllib.request.Request(
                                "https://api.anthropic.com/v1/messages", data=payload,
                                headers={"content-type": "application/json", "x-api-key": api_key,
                                         "anthropic-version": "2023-06-01"})
                            with urllib.request.urlopen(req, timeout=30) as resp:
                                data = json.loads(resp.read())
                            answer = "".join(b.get("text", "") for b in data.get("content", []))
                            st.markdown("##### 💬 Answer")
                            st.markdown(f'<div class="masar-card">{answer}</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"LLM call failed: {e}")

# ---- TAB 5: VISUALIZATION ---------------------------------------------------
with tab_viz:
    df = get_dataset()
    if "Faculty_Category" in df.columns:
        fig = px.bar(df["Faculty_Category"].value_counts().reset_index(),
                     x="Faculty_Category", y="count", title="Faculty category distribution",
                     color="Faculty_Category")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if {"Track", "Percentage"}.issubset(df.columns):
            fig = px.box(df, x="Track", y="Percentage", color="Track", title="Percentage by Track")
            st.plotly_chart(fig, use_container_width=True)
        if "Skills" in df.columns:
            fig = px.bar(df["Skills"].value_counts().reset_index().head(10),
                         x="Skills", y="count", title="Top skills")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "Track" in df.columns:
            fig = px.pie(df, names="Track", title="Track distribution", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        if "Interests" in df.columns:
            fig = px.bar(df["Interests"].value_counts().reset_index().head(10),
                         x="Interests", y="count", title="Top interests")
            st.plotly_chart(fig, use_container_width=True)

# ---- TAB 6: LOGS -------------------------------------------------------------
with tab_logs:
    if st.session_state.username != db.ADMIN_USERNAME:
        st.warning("🔒 This tab is restricted to admin accounts.")
    else:
        st.markdown("#### 👥 Registered users")
        st.dataframe(db.get_all_users(), use_container_width=True, height=220)

        st.markdown("#### 🗂️ Activity log")
        logs_df = db.get_all_logs()
        c1, c2 = st.columns([1, 3])
        with c1:
            action_filter = st.multiselect("Filter by action", sorted(logs_df["action"].unique().tolist()))
        filtered = logs_df[logs_df["action"].isin(action_filter)] if action_filter else logs_df
        st.dataframe(filtered, use_container_width=True, height=380)
        st.download_button("Download logs as CSV", filtered.to_csv(index=False),
                            file_name="masar_ai_logs.csv", mime="text/csv")

st.markdown("---")
st.caption("Masar AI · Built for Thanaweya Amma students · Recommendations are guidance, not a guarantee.")
