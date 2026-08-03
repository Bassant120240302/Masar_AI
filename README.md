# Masar AI — مسار

Streamlit deployment for the Thanaweya Amma faculty-recommendation model,
wired up to your real capstone artifacts — **including the trained
classifier**, so recommendations now use exam scores + Track, not just
Skills/Interests similarity.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — the full site (login/sign-up, theme switcher, 6 tabs, sidebar model picker)
- `db.py` — SQLite persistence: user accounts (hashed passwords) + an activity log
- `model_utils.py` — custom preprocessing classes matching
  `ML_Final_Capstone_Project_Masar_AI.ipynb` (needed to unpickle any of the files below)
- `requirements.txt` — dependencies
- **Bundled recommender assets** (loaded automatically, next to `app.py`):
  - `recommender_config.pkl` — feature column lists + the full skill/interest vocabulary
  - `label_enc.pkl` — decodes predicted class indices back into faculty names
  - `profile_pipeline.pkl` — encodes a student's Skills/Interests for content-based similarity
  - `faculty_profiles.pkl` — each faculty's average Skills/Interests profile (29 faculties)
  - `tansiq_cutoffs_2022.csv` — 1,286 real 2022 college programs with tansiq (coordination)
    cutoff scores, used by the RAG Assistant tab *(bring your own copy — not part of this bundle)*
- `models/final_model_pipeline.pkl` — **the trained classifier, bundled and confirmed working.**
  It's a scikit-learn `Pipeline` (`feature_engineer` → `track_ohe` → `drop_skill_interest` →
  `drop_low_var` → `classifier`) whose final estimator is a **CatBoostClassifier** — hence
  `catboost` being a hard requirement in `requirements.txt`, not an optional extra.
  Auto-selected as the active classifier the moment the app starts (still overridable, or
  swappable for another model, from the sidebar).
- `masar_ai.db` — SQLite database (accounts + activity log). Included pre-seeded; on an
  ephemeral host (see "Accounts, database, and logs" below) it resets on redeploy.
- `thanaweya_final_with_profiles_modified.csv` *(optional, ~146 MB, 666,932 rows)* — the real
  student dataset behind the notebook. **Not required** — the Model, RAG, and Logs tabs don't
  touch it at all, and the Data / EDA & Preprocessing / Visualization tabs fall back to a small
  synthetic sample if it's absent. If it *is* present next to `app.py`, those three tabs use it
  automatically as the default (no manual upload needed each session) — you can still override
  with your own CSV from the sidebar uploader at any time. See "About the bundled student
  dataset" below before deciding whether to commit it to your repo.

## Verified end-to-end
All bundled `.pkl` files were loaded together and run through a real prediction in this
environment: `model_utils.register_classes_on_main()` → load classifier + recommender assets →
build a student feature row → `classifier.predict_proba()` → decode via `label_enc` → combine
with Skills/Interests cosine similarity → track-eligibility filter. It produced a sane ranked
list (e.g. a Science-track student with strong scores topped out at Medicine on the model side,
Science on the similarity side) — so the pipeline is confirmed compatible end to end, not just
theoretically wired up.

One thing to know: the `.pkl` files were originally saved with **scikit-learn 1.6.1**. Loading
them on a newer scikit-learn (e.g. 1.8) still works but throws an `InconsistentVersionWarning`.
`requirements.txt` pins `scikit-learn==1.6.1` to match training exactly and avoid that — if you
ever retrain, keep this pin in sync with whatever version the notebook used to `joblib.dump()`.

## What changed in the last revision
-1. **RAG tab recommending science-only programs to Math/Literary-track
   students.** `tansiq_cutoffs_2022.csv` itself contains data-entry
   duplicates: Medicine, Medicine Dentistry, Medicine Veterinary, Pharmacy,
   Nursing, Agriculture, Aquaculture/Fisheries, Technical Health, and the
   plain "Science" faculty are each listed under **both**
   `Scientific - Science Track` (correct) **and**
   `Scientific - Math Track` (wrong — same college, same cutoff score,
   duplicated) — which is why filtering the RAG tab to Math track could
   still surface e.g. "Medicine Cairo". Since the CSV's own `Track_EN`
   column can't be fully trusted for these rows, the RAG tab now applies a
   second, keyword-based eligibility check (`rag_track_eligible()`) on top
   of the `Track_EN` filter, using the same Science-only /
   Science-and-Math / Literary-only / all-tracks convention already used
   by the Model & Recommendation tab's `TRACK_ELIGIBLE_FACULTIES`. When a
   specific track is selected, mis-tagged rows are dropped and a caption
   reports how many were filtered — mirroring the caption the Model tab
   already shows. Verified directly against the data: filtering to
   `Scientific - Math Track` used to return 60 Medicine-family rows; it
   now returns 0, while Science-track results are unaffected.
0. **RAG tab stale-cache bug.** `load_tansiq_data()` was decorated with
   `@st.cache_resource` and took no arguments, so if the app ever ran once
   before `tansiq_cutoffs_2022.csv` existed next to `app.py`, it permanently
   cached `None` and kept serving that even after the CSV was added later —
   the RAG Assistant tab would keep showing "wasn't found" forever, on that
   process, regardless of the file actually being there. The loader is now
   keyed on the CSV's mtime + size, so it reloads automatically whenever the
   file is added or changed, and it reads with `encoding="utf-8-sig"` to be
   safe against a leading BOM.
1. **Track-encoding bug (root cause of mixed-up Science/Math/Literary
   results).** The UI's dropdown sent `"Scientific - Science Track"` etc. to
   the pipelines, but both `profile_pipeline.pkl` and the classifier were
   trained on the short codes `'Science'` / `'Math'` / `'Literacy'`
   (confirmed straight from the notebook's own `track_mapping` dict).
   `TrackOneHotEncoder` silently zeroed out every `Track_*` column on a
   mismatch, so Track information was dropped entirely. `app.py` now
   translates the dropdown label to the correct code before calling either
   pipeline.
2. **Hard track-eligibility filter.** Even with the Track fix, the
   similarity-only signal has *no* Track information in its feature vector
   by design (`full_skill_interest_cols` is Skills/Interests only) — so it
   could still recommend e.g. Engineering to a Literary-track student.
   `app.py` hard-filters the ranked results against a standard Egyptian
   tansiq eligibility table before showing the top picks, regardless of
   which mode is active.
3. **Auto-selected default classifier.** `final_model_pipeline.pkl` sits in
   `models/` and is picked as the **active classifier automatically** the
   first time the app loads — no manual dropdown selection needed. (Still
   overridable live from the sidebar.)
4. **Adjustable result counts.** The Model & Recommendation tab has a
   1–29 slider (button label now matches the chosen count, e.g.
   "Get top-5 recommendations"). The RAG Assistant tab has a numeric "Max
   results to show" field, and the app tells you how many total matches
   existed vs. how many are displayed.

## Choosing between models
Any `.pkl`/`.joblib` file in `models/` (or next to `app.py`) — other than
the recommender asset files above — shows up automatically as an option in
the sidebar's **"Active classifier"** dropdown, and the first one found
(`final_model_pipeline.pkl`, currently) is pre-selected by default. Add
another file (e.g. `best_rf.pkl`) alongside it and you can switch between
them live, or pick **"— None (similarity-only) —"** to see recommendations
from Skills/Interests alone. You can also upload extra models on the fly
from the same sidebar section without restarting the app — uploads are
session-only, while files in `models/` become permanent defaults for
everyone.

## How a recommendation is actually computed
Same hybrid logic as `recommend_top_faculties()` in the notebook, plus a
track-eligibility filter applied at the end:
- **Model score**: `classifier.predict_proba()` on the exam scores + Track,
  decoded back to faculty names via `label_enc`.
- **Similarity score**: cosine similarity between the student's
  Skills/Interests (encoded by `profile_pipeline`) and each faculty's
  average profile (`faculty_profiles`).
- **Combined** = `model_weight × model_score + similarity_weight × similarity_score`
  — adjustable live via the slider in the Model & Recommendation tab
  (defaults to 0.7 / 0.3, matching the notebook).
- **Track-eligibility filter**: faculties the chosen Track can't actually
  apply to (per standard tansiq rules) are dropped from the ranked list
  before the top-N is shown, with a caption noting how many were filtered.

## About the bundled student dataset
`thanaweya_final_with_profiles_modified.csv` is **optional**, and only affects three tabs
(Data, EDA & Preprocessing, Visualization) — it plays no role in generating recommendations;
that's entirely `final_model_pipeline.pkl` + the recommender assets. A few things worth
weighing before committing it to your repo:

- **Size.** It's ~146 MB / 666,932 rows. GitHub blocks regular (non-LFS) pushes over 100 MB,
  so a plain `git add` will fail — you'd need [Git LFS](https://git-lfs.com) or a release
  asset, similar to how `final_model_pipeline.pkl` was too large to attach directly here.
- **Load time.** Reading it took ~5 seconds in testing, once per session (cached after that
  via `st.cache_data`) — noticeable but not painful.
- **What you get for it.** The Data tab's row/column/missing-cell metrics, the EDA tab's
  distributions, and the Visualization tab's charts all reflect your actual 666,932 real
  student records instead of an 800-row synthetic stand-in — useful if you want those tabs to
  demonstrate the real project, less useful if they're just a UI backdrop for the
  recommendation flow.
- **Alternative:** if repo size is a concern, consider committing a random sample (e.g.
  `df.sample(20000, random_state=42).to_csv(...)`) under the same filename instead of the
  full file — the Data/EDA/Viz tabs work identically on a sample, just with smaller numbers.

## RAG Assistant tab
Retrieval runs entirely locally: a TF-IDF index over the 1,286 real tansiq
program rows (college name, track, key skills), filterable by track and by
"my coordination score" (only shows programs you'd actually qualify for),
with an adjustable max-results field. That's the "R" in RAG. For the "G" (a
written answer instead of a raw list), paste an Anthropic API key in the
expander — it's used only for that one request, sent straight from your
browser session to Anthropic's API, never stored. Leave it empty and you
still get the ranked list of matching programs, just without the generated
paragraph. **Needs `tansiq_cutoffs_2022.csv` next to `app.py`** — not part
of this bundle, so add your own copy or the tab will show a warning instead
of erroring.

## Skills & Interests vocabulary
The 30 skills and 30 interests offered in the Model tab's multiselects come
directly from the fitted `profile_pipeline` (`skills_mh.vocab_` /
`interests_mh.vocab_`), so they exactly match what the model was trained on
— no need to keep a separate list in sync.

## Accounts, database, and logs
Real SQLite accounts with hashed passwords, the admin account (`admin` /
`masar2026`) seeded on first run but not shown on the login page, and a
🗂️ Logs tab with every registered user plus a full, downloadable activity
trail (logins, uploads, recommendations, RAG queries). **On an ephemeral
host** (some free tiers wipe the filesystem on every redeploy/sleep-wake),
new accounts will disappear when the filesystem resets — swap
`db.DB_PATH` for a managed database (Postgres/Supabase) if you need
accounts to survive that too.

## New: 3-signal model (Track + grades + Skills/Interests)
`final_model_pipeline.pkl` (CatBoost) has a `drop_skill_interest` step —
Skills/Interests never reach that classifier; they only fed the separate
similarity score, then got blended in afterward. That's two independent
signals averaged, not one model reasoning over all three.

`models/track_grades_skills_model.pkl` is a new classifier that actually
takes Track + exam scores + Skills/Interests together as input features
(same `feature_engineer → track_ohe → skills_mh/interests_mh` steps as
`profile_pipeline.pkl`, plus a classifier at the end — see
`train_3signal.py`). It shows up automatically as a second option in the
sidebar's "Active classifier" dropdown; `final_model_pipeline.pkl` is
untouched and still the default. `app.py` detects which kind of classifier
is active (`classifier_uses_skills_interests()`) and builds the right input
row for either one.

Two things worth knowing:
- **No internet in the build sandbox** meant no `catboost` — this one is a
  scikit-learn `HistGradientBoostingClassifier` instead, trained on a
  stratified 180k-row sample of the 666,932-row dataset (single-CPU sandbox;
  see `train_3signal_report.txt` for the held-out test metrics).
- **The dataset's Skills/Interests are near-deterministic per
  `Faculty_Category`** (e.g. every one of the 269,066 "Other" rows has the
  *exact same* Skills/Interests string), so test accuracy comes out ~99.9%.
  That's the dataset, not a leak I introduced — but it does mean, on this
  data, Skills/Interests end up as a very strong signal once the model is
  allowed to see them. The two-signal-only `final_model_pipeline.pkl` is
  still there for comparison. Also note the multi-hot vocabulary is capped
  at the top-30 most frequent Skills/Interests tokens (same cap
  `profile_pipeline.pkl` already used) — a real student whose skills fall
  outside that top-30 will get weaker signal from this feature, same
  limitation the original bundle already had.

## Fixed: RAG tab missed one science-only keyword
`rag_track_eligible()`'s keyword list was missing anything for **Physical
Therapy** — `"Therapy Physical (therapy) ..."` rows in
`tansiq_cutoffs_2022.csv` weren't being caught by the science-only filter,
so Literary/Math-track students could see Physical Therapy programs in RAG
search results even though real tansiq rules require the Science track for
that faculty. Added a `"therap"` keyword to
`_RAG_SCIENCE_ONLY_KEYWORDS` to close that gap. Every other faculty
category in `TRACK_ELIGIBLE_FACULTIES` (all 29, matching `label_enc.pkl`
exactly) was checked against the raw `College_University_EN` values and had
adequate keyword coverage already.

## Themes
Switch between 🌙 Dark, ☀️ Light, and 🩵 Sky Blue from the sidebar at any time.
