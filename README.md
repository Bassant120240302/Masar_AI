# 🎓 Masar AI | مسار

**Masar AI** is a Streamlit web app that recommends university faculty categories to Egyptian **Thanaweya Amma** (high school) students, based on their track, subject scores, skills, and interests. It combines a trained classifier, a content-based similarity recommender, and a retrieval-based assistant over real 2022 tansiq (coordination) cutoff data.

> Recommendations are guidance, not a guarantee.

---

##  Features

- **Theming** — switch between Dark, Light, and Sky Blue color themes live from the sidebar (see [Themes](#-themes) below).
- **Accounts & auth** — simple username/password login backed by SQLite, with a seeded admin account.
- **Data tab** — preview the bundled/uploaded student dataset, with row/column/missing-value stats.
- **EDA & Preprocessing tab** — score distributions, track/branch breakdowns, correlation heatmap, and a walkthrough of the feature-engineering pipeline.
- **Model & Recommendation tab** — enter a student's track, subject scores, skills, and interests to get top-*N* recommended faculty categories. Supports three modes depending on which assets are available:
  - **Hybrid** — classifier probabilities blended with content-based cosine similarity (adjustable weight).
  - **Model-only** — classifier predictions alone.
  - **Content-similarity-only** — skills/interests similarity when no classifier is loaded.
  - A track-eligibility filter acts as a safety net so, e.g., a Literary-track student is never recommended Engineering.
- **RAG Assistant tab** — TF-IDF retrieval over real 2022 tansiq cutoff data (English/Arabic query support), with an optional bring-your-own-key call to the Anthropic API for a natural-language, context-grounded answer.
- **Visualization tab** — faculty, track, skills, and interests distributions via Plotly.
- **Logs tab** — registered users and an append-only activity trail (logins, predictions, RAG queries), downloadable as CSV.
- **Pluggable classifiers** — drop any additional trained pipeline into `models/` and it's auto-discovered as a selectable model in the sidebar, no code changes needed.

---

##  How it works

The app is built around a scikit-learn pipeline shared by both the classifier and the recommender:

1. **Structural imputation** — a missing subject score means the student didn't take that subject, so it's filled with `0`.
2. **Feature engineering** — `total`, `total_pct`, `avg_subject_score`, and per-group averages (science / humanities / language) are derived from raw subject scores.
3. **Track encoding** — the 3-way `Track` column (Science / Math / Literacy) is one-hot encoded.
4. **Two parallel branches**:
   - The **classifier branch** predicts a faculty category from Track + grades (+ Skills/Interests, for the 3-signal model).
   - The **profile branch** keeps Skills/Interests, multi-hot encoding them for content-based similarity against precomputed faculty profiles.
5. **Selective Yeo-Johnson + feature selection** — near-zero-variance and highly correlated columns are dropped from the classifier branch before training.

Custom transformer classes live in [`model_utils.py`](model_utils.py) and are re-registered onto `__main__` at load time so pipelines pickled from a notebook's `__main__` namespace can be unpickled inside the Streamlit process.

### Available classifiers

`app.py` auto-discovers any `.pkl`/`.joblib` file next to `app.py` or inside `models/` (other than the known recommender assets) and lists it as a selectable classifier in the sidebar. This repo ships with two:

| File | Algorithm | Signals used | Source |
|---|---|---|---|
| `final_model_pipeline.pkl` | **CatBoost** | Track + grades only (drops Skills/Interests) | Exported directly from the capstone notebook |
| `models/track_grades_skills_model.pkl` | **HistGradientBoostingClassifier** | Track + grades + Skills/Interests | Produced by [`train_3signal.py`](train_3signal.py) (see below) |

Whichever one is selected in the sidebar (or both combined with content similarity, in hybrid mode) is what actually drives the recommendations at runtime — there's no single hardcoded "the model," it's whatever's loaded from these files.

### Retraining the 3-signal model

[`train_3signal.py`](train_3signal.py) retrains a classifier that uses **Track + grades + Skills/Interests together** (the original `final_model_pipeline.pkl` drops Skills/Interests before the classifier ever sees them). It reuses the exact preprocessing steps from `model_utils.py`, trains a `HistGradientBoostingClassifier` on a stratified subsample, and saves the result to `models/track_grades_skills_model.pkl`, where `app.py` picks it up automatically. See [`train_3signal_report.txt`](train_3signal_report.txt) for the latest evaluation metrics.

---

## Project structure

```
.
├── app.py                                   # Streamlit application (entry point)
├── db.py                                    # SQLite persistence: users + activity logs
├── model_utils.py                           # Custom sklearn transformers used by the pipelines
├── train_3signal.py                         # Retrains the Track+grades+Skills/Interests classifier
├── train_3signal_report.txt                 # Latest evaluation report for the 3-signal model
├── requirements.txt
├── recommender_config.pkl                   # Config for the hybrid recommender
├── label_enc.pkl                            # Label encoder for faculty categories
├── profile_pipeline.pkl                     # Feature pipeline for content-based similarity
├── faculty_profiles.pkl                     # Precomputed faculty profile vectors
├── final_model_pipeline.pkl                 # Original notebook classifier (CatBoost, Track+grades only)
├── tansiq_cutoffs_2022.csv                  # Real 2022 tansiq cutoff data (RAG assistant source)
├── thanaweya_final_with_profiles_modified.csv  # Bundled dataset (stratified ~40k-row sample, see note below)
├── models/                                  # Drop additional trained classifier pipelines here
│   └── track_grades_skills_model.pkl        # Optional: 3-signal classifier from train_3signal.py
└── masar_ai.db                              # SQLite database (created on first run)

> **Which classifier runs by default?** `app.py` scans the repo root before `models/`, so if `final_model_pipeline.pkl` is present, it's auto-selected as the active classifier the first time anyone opens the app (see [Available classifiers](#available-classifiers) below for how to change this).

---

## Themes

The sidebar theme selector switches the whole UI's color palette live, no restart needed:

| Theme | Vibe |
|---|---|
| 🌙 **Dark** | Deep navy/charcoal background, violet + cyan accents — default theme |
| ☀️ **Light** | Clean white background, indigo + teal accents |
| 🩵 **Sky Blue** | Soft light-blue background, blue + cyan accents |

Theme choice is stored per-session (`st.session_state.theme`) and applied via `inject_css()` in `app.py`.

---

## Getting started

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone https://github.com/<your-username>/masar-ai.git
cd masar-ai
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Optional: LLM-powered RAG answers

The RAG Assistant tab works fully offline by default (TF-IDF retrieval over the tansiq CSV, no data leaves your machine). To also get a natural-language answer grounded in the retrieved rows, paste an [Anthropic API key](https://console.anthropic.com/) into the tab's expander — the key is used only for that single request and is never stored.

---

## Security notes

This project was built as an ML capstone and includes a few things you should harden before any real deployment:

- **Change the seeded admin password** (`ADMIN_PASSWORD` in `db.py`), or replace it with your own account-creation flow.
- **Restrict the Logs tab** to admin accounts only — it's currently visible to any logged-in user. Gate it on `st.session_state.username` in `app.py`.
- **Persistent storage**: `masar_ai.db` lives next to `app.py` and survives restarts on a persistent filesystem (your own server, a VM, Docker with a volume). On ephemeral hosting (some free tiers wipe the filesystem on every redeploy) it will reset — swap in a managed database (Postgres, Supabase, etc.) if you need it to survive redeploys.
- **API keys**: never commit real Anthropic API keys; the RAG tab is designed for users to paste their own key per-session.

---

## Data & model files

Several files in this repo are binary artifacts (.pkl, .db) produced by the accompanying capstone notebook (ML_Final_Capstone_Project_Masar_AI.ipynb). If you're forking this repo:

thanaweya_final_with_profiles_modified.csv is a stratified sample (~40k rows, ~8.4MB), not the full dataset. The full dataset used for actual model training has ~667k rows (~140MB). Shipping the full file with the deployed app caused Streamlit Community Cloud to throttle the app on cold start (reading + indexing 140MB on first load is expensive on the free tier). The sample preserves every Faculty_Category's proportion (with a floor per category) so the Data/EDA/Visualization tabs look representative, just lighter to load. Model training/retraining (train_3signal.py, the notebook) should still use the full dataset, kept outside the deployed app.
.pkl files must be regenerated with the same scikit-learn version pinned in requirements.txt to avoid unpickling errors.

---

## Roadmap ideas

- Admin-only gating for the Logs tab.
- Swap SQLite for a managed database for multi-instance deployments.
- Add automated CI to retrain and validate `models/*.pkl` on data updates.

---

##  License

Add a license of your choice (e.g. MIT) before publishing publicly.

---

## 🙏 Acknowledgements

Built as an ML capstone project analyzing Egyptian Thanaweya Amma outcomes and 2022 tansiq coordination data, to help students explore faculty options aligned with their academic profile, skills, and interests.


