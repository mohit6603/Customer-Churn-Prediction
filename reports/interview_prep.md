# Interview Preparation — Customer Churn Prediction

Phase-by-phase questions an interviewer will actually ask, the answer this
project supports, and *why* the interviewer asks it. Numbers in answers come
from `reports/metrics.json` and `reports/model_comparison.csv`.

---

## 1. Business understanding

**Q: What business problem does this project solve?**
A telco loses ~26.5% of customers and only finds out after they leave.
Acquiring a replacement costs 5–7× more than retaining one. The model produces
a monthly ranked list of at-risk customers so the retention team can intervene
(discounts, contract upgrades) while there's still time.
*Why they ask:* screens out candidates who jump to `model.fit()` without
knowing who uses the output.

**Q: How do you define churn?**
A precise, time-bound, observable event: the customer terminated service
within the last month (`Churn Value` = 1). Voluntary vs involuntary churn
matters — you can only *prevent* voluntary churn — and "will they churn next
month" is actionable while "will they ever churn" is not.
*Why they ask:* vague target definitions sink real projects.

**Q: What do wrong predictions cost?**
Asymmetric costs: a false negative loses the customer's remaining lifetime
value (thousands); a false positive wastes a small retention offer (tens).
This asymmetry drives the metric choice, the class weighting and the tuned
threshold — one fact, three design decisions.
*Why they ask:* connects ML choices to money; the #1 senior-vs-junior signal.

## 2. Data understanding & leakage

**Q: What is data leakage, and where was it in this dataset?**
Leakage = training on information unavailable at prediction time. The IBM
33-column export contains four leaky columns: `Churn Label` (duplicate of the
target), `Churn Score` (the output of IBM's *own* churn model), `Churn Reason`
(only exists after a customer churns — it's null for all 5,174 non-churners),
and `CLTV` (IBM-computed, undocumented, may embed the outcome). Using them
gives spectacular offline metrics and a useless production model.
*Why they ask:* the single most common cause of ML projects that die in
production. This dataset is a famous trap.

**Q: How do you test a feature for leakage?**
Ask: "Would this value exist, with this exact value, at the moment of
prediction — before the customer decides to leave?" Also suspicious: features
that are null for exactly one class (`Churn Reason`), and features with
implausibly high univariate predictive power.

**Q: Why drop geography (City, Zip, Lat/Long)?**
1,100+ categories on 7,043 rows → high overfit risk, no business lever, and
the dataset is California-only so it can't generalise anyway. Dropping is a
*feature selection* decision made from data understanding, not from a
selector algorithm.

## 3. Cleaning & EDA

**Q: What data quality issues did you find?**
`Total Charges` arrives as text with 11 blank values — all tenure-0 customers
who haven't been billed yet. Coerced to numeric and filled with 0.0, which is
the *semantically correct* value, not an arbitrary imputation. Also collapsed
"No internet service"/"No phone service" to "No" since the parent column
already carries that information.
*Why they ask:* do you investigate missingness or blindly impute the mean?

**Q: Top EDA insights?**
(1) Month-to-month contracts churn ~15× more than two-year contracts;
(2) churn concentrates in the first months of tenure (early-life churn);
(3) churners skew to higher monthly bills — fiber-optic-without-contract is
the classic risk profile; (4) protective add-ons (tech support, security)
correlate with staying. Each insight either motivated an engineered feature
or set an expectation later verified against feature importance.

## 4. Feature engineering & selection

**Q: What features did you engineer and why?**
Four, each computable from a single row at prediction time (no leakage):
`num_addon_services` (ecosystem lock-in), `avg_monthly_spend` (lifetime
average bill), `charge_growth` (current bill − lifetime average: recent price
increases trigger churn), `tenure_bucket` (gives linear models the non-linear
first-year churn cliff for free).

**Q: How did you prevent preprocessing leakage?**
Strict separation: stateless row-wise transforms (cleaning, engineered
features) are pure functions applied anywhere; *fitted* transforms (scaler,
one-hot encoder) live inside the sklearn Pipeline, so cross-validation
re-fits them on each training fold. Fitting a scaler on all data before
splitting leaks test-set statistics into training.
*Why they ask:* pipeline-level leakage is subtler than column-level leakage
and catches most mid-level candidates.

**Q: Total Charges ≈ tenure × monthly charges — collinearity. Why keep it?**
Trees are unaffected by collinearity; the logistic model is L2-regularised,
which handles it. Dropping would also lose the small independent signal.
The cost of collinearity here is only coefficient interpretability, which we
recover via permutation importance on raw features.

**Q: Why no automated feature selector?**
23 features on 5,634 training rows is a comfortable ratio. Selection happened
where it matters: leakage removal and information screening (constants, IDs,
geo). Regularisation and tree feature-sampling handle redundancy better than
a hard filter; permutation importance verified nothing kept is dead weight.

## 5. Modelling

**Q: Which models and why those three?**
Logistic regression (interpretable linear baseline — any complex model must
beat it), random forest (non-linear interactions, robust defaults), XGBoost
(usually the strongest on tabular data). All compared under identical 5-fold
stratified CV with the full pipeline inside the fold.

**Q: Logistic regression won (0.859 vs 0.846 RF, 0.840 XGB). Why?**
The dataset is small (7k rows), clean, and its signal is mostly additive —
contract type, tenure, charges combine roughly linearly in log-odds. Boosting
earns its complexity on large datasets with deep interactions; here it only
found noise to overfit. Lesson: baseline first, complexity must pay rent.
*Why they ask:* tests intellectual honesty — candidates who force XGBoost
everywhere fail this.

**Q: How did you handle class imbalance?**
`class_weight='balanced'` (sklearn) / `scale_pos_weight` (XGBoost) — cost
weighting re-weights the minority class in the loss. Chosen over SMOTE/
resampling because it's simpler, leakage-free (SMOTE inside CV is easy to get
wrong), and equivalent in effect for these models. And 26.5% is only *mildly*
imbalanced — the bigger sin is using accuracy.

**Q: Random vs grid search?**
RandomizedSearchCV, 40 candidates × 5 folds. At a fixed compute budget random
search covers continuous spaces far better than a coarse grid, and important
parameters get 40 distinct values instead of 3.

## 6. Evaluation

**Q: Why not accuracy?**
Predicting "nobody churns" scores 73.5% accuracy and catches zero churners.
Used instead: ROC-AUC for model comparison (threshold-independent), and
recall/precision/F2 at the chosen operating point for the business decision.

**Q: Explain your threshold choice.**
Default 0.5 assumes symmetric error costs — false here. Tuned to maximise F2
(recall weighted 2× precision) on *out-of-fold training predictions*, giving
0.29. On test: recall rises 79.1% → 91.7% while precision falls 51.8% → 43.4%.
In business terms: the team contacts ~2.3 customers per real churner caught —
a trade the cost asymmetry easily justifies.
*Why they ask:* the threshold is where ML meets business; tuning it on the
test set (a leak) is a classic mistake — ours comes from training data only.

**Q: The test set — how was it used?**
Carved out by stratified split before any modelling, then touched exactly
once, at the very end, for the final numbers. Every intermediate decision
(model family, hyperparameters, threshold) used cross-validation on the
training set only.

## 7. Interpretation

**Q: How do you explain the model?**
Three layers: (1) permutation importance on the *whole pipeline* with raw
input columns — so importance lands on business features, not one-hot
dummies; (2) SHAP for per-customer attributions ("why is *this* customer at
risk?") which tells the retention team *which* offer to make; (3) sanity
check: top drivers (contract, tenure, charges, internet service) match the
EDA story — a model that contradicts EDA usually means a bug or leakage.

## 8. Deployment & engineering

**Q: How does the model get to production?**
The artifact (`models/churn_model.joblib`) bundles the fitted pipeline, the
tuned threshold, expected input columns, model version and metadata. A
FastAPI service (`api/main.py`) loads it once at startup and exposes
`/predict` and `/predict/batch`; both go through `src/inference.py`, which
replays the *identical* cleaning and feature-engineering functions used in
training — one code path, no train/serve skew. Batch scoring returns a
ranked list with a recommended action per customer, matching how the
retention team consumes it. The API ships as a slim, non-root Docker image
with a health check; the Streamlit app is a second front end on the same
inference code.

**Q: Why put Pydantic schemas in front of the model?**
One-hot encoding silently maps an unseen category (e.g. "Month to month"
without hyphens) to all zeros — the model would still return a confident,
wrong score. Validating categories and ranges at the API edge turns that
into a 422 error the caller can fix. Same principle as the training-time
schema validator: bad data should fail loudly, never silently.

**Q: How would you know the model is degrading in production?**
Implemented: every prediction is logged to SQLite with its inputs, score and
model version. Training saves a reference profile of every feature's
distribution; `/monitoring/drift` compares recent traffic against it with
the Population Stability Index (< 0.1 stable, 0.1–0.25 moderate, > 0.25
significant → retrain). `/monitoring/stats` tracks the flagged rate and
risk-band mix — a sudden jump is prediction drift. Once churn outcomes
arrive a month later, realised recall/precision is the ground-truth check.

**Q: Why PSI rather than a KS test for drift?**
KS p-values become "significant" for tiny, harmless shifts once samples are
large, and KS only works for numeric features. PSI is a magnitude, works on
binned numeric *and* categorical features with one formula, and has
industry-standard action thresholds that are easy to explain to the business.

**Q: How do you version models and roll back?**
Every training run writes an immutable `models/churn_model_<version>.joblib`
(version = training timestamp) and appends a row with its metrics to
`reports/model_registry.csv`; `churn_model.joblib` is always the latest.
Rolling back is copying an older version over the latest and restarting the
API. Every API response carries `model_version`, so any score is traceable
to the exact model. Runs are also logged to MLflow (params, metrics, plots)
for side-by-side comparison.

**Q: What does your CI do?**
GitHub Actions on every push/PR: ruff lint → 49 pytest tests (on synthetic
data, so CI never needs the real dataset or model) → Docker image build.
A broken API contract or a schema change fails the build before it ships.

**Q: What would you improve with more time?**
Real CLTV data to convert the threshold from F2 to a true expected-value
optimisation; probability calibration; time-based validation if snapshots
across months existed; champion/challenger deployment; feature store for
serving parity at scale; automated retraining triggered by drift alerts;
SHAP reasons per customer in the API response.

## Rapid-fire facts to remember

- 7,043 customers · 33 raw columns · 26.5% churn · 80/20 stratified split
- 4 leakage columns dropped · 19 raw features kept · +4 engineered = 23 inputs
- Winner: logistic regression, C≈2.64 · CV ROC-AUC 0.859 · test ROC-AUC 0.849
- Threshold 0.29 (F2-optimal, out-of-fold) · test recall 91.7% · precision 43.4%
- Top drivers: contract, tenure, monthly charges, internet service
- Served by FastAPI (6 endpoints) · Pydantic validation → 422 on bad input
- PSI drift: < 0.1 stable · 0.1–0.25 moderate · > 0.25 retrain
- Versioned models + CSV registry + MLflow · CI: ruff → pytest → Docker
