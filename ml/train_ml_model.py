import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

#importing data from the csv files
matches_df = pd.read_csv('Raw_Data_Kaggle/matches.csv')
deliveries_df = pd.read_csv('Raw_Data_Kaggle/deliveries.csv')

print(matches_df.head())
print(deliveries_df.head())

print(matches_df.columns.tolist())


# making primary key consistent
matches_df["match_id"] = matches_df["id"]


#  Remove matches with no result or tie

if "result" in matches_df.columns:
    before = len(matches_df)
    matches_df = matches_df[~matches_df["result"].isin(["no result", "tie"])]
    after = len(matches_df)
    print(f"Removed {before - after} matches with no result or tie.")



# cleaning for text fields
string_cols = [
    "team1", "team2", "winner", "venue", "city",
    "toss_winner", "toss_decision", "match_type"
]
for col in string_cols:
    if col in matches_df.columns:
        matches_df[col] = matches_df[col].astype(str).str.strip()

for col in ["batting_team", "bowling_team", "batter", "bowler", "non_striker"]:
    if col in deliveries_df.columns:
        deliveries_df[col] = deliveries_df[col].astype(str).str.strip()    



# Merge deliveries + matches
merged_df = deliveries_df.merge(
    matches_df,
    on="match_id",
    how="left",
    suffixes=("_delivery", "_match"),
)
print(f"Merged shape: {merged_df.shape}")   



# Keep relevant columns only
keep_cols = [
    "match_id", "season", "date", "city", "venue", "team1", "team2",
    "winner", "match_type", "toss_winner", "toss_decision",
    "inning", "over", "ball", "batting_team", "bowling_team",
    "batter", "bowler", "non_striker",
    "batsman_runs", "extra_runs", "total_runs",
    "is_wicket", "player_dismissed", "dismissal_kind", "fielder"
]
keep_cols_present = [c for c in keep_cols if c in merged_df.columns]
merged_df = merged_df[keep_cols_present].copy()

# Drop rows with missing essentials
critical_cols = ["match_id", "inning", "over", "ball", "batting_team", "bowling_team", "total_runs"]
before_drop = len(merged_df)
merged_df.dropna(subset=critical_cols, inplace=True)
after_drop = len(merged_df)
print(f"Dropped {before_drop - after_drop} rows with missing critical values.")



# Convert numeric columns
numeric_cols = ["inning", "over", "ball", "batsman_runs", "extra_runs", "total_runs", "is_wicket"]
for col in numeric_cols:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna(0).astype(int)



# Sort and remove duplicates
merged_df.drop_duplicates(inplace=True)
merged_df.sort_values(by=["match_id", "inning", "over", "ball"], inplace=True)
merged_df.reset_index(drop=True, inplace=True)

print("\nSample rows:")
print(merged_df.head(10))

print(f"\nTotal deliveries after cleaning: {merged_df.shape[0]}")
print(f"Unique matches: {merged_df['match_id'].nunique()}")
print(f"Columns retained: {merged_df.columns.tolist()}")



####################################################################################################################

# Feature Engineering 


print("Input data shape:", merged_df.shape)

# Keeping only 1st and 2nd innings deliveries
merged_df = merged_df[merged_df["inning"].isin([1, 2])]

# Compute cumulative runs and wickets
merged_df["cumulative_runs"] = merged_df.groupby(["match_id", "inning"])["total_runs"].cumsum()
merged_df["cumulative_wkts"] = merged_df.groupby(["match_id", "inning"])["is_wicket"].cumsum()


# Compute balls bowled and remaining
merged_df["ball_number"] = (merged_df["over"] - 1) * 6 + merged_df["ball"]
merged_df["balls_bowled"] = merged_df.groupby(["match_id", "inning"])["ball_number"].rank(method="first")
merged_df["balls_remaining"] = 120 - merged_df["balls_bowled"]


# Derive target runs from first innings
target_df = (
    merged_df[merged_df["inning"] == 1]
    .groupby("match_id")[["cumulative_runs"]]
    .max()
    .reset_index()
    .rename(columns={"cumulative_runs": "target_runs"})
)
target_df["target_runs"] = target_df["target_runs"] + 1  # +1 run to win


#Merge target and winner info
match_targets = merged_df[["match_id", "team1", "team2", "winner"]].drop_duplicates("match_id")
target_df = target_df.merge(match_targets, on="match_id", how="left")


# Merge target info into main data
merged_df = merged_df.merge(target_df, on="match_id", how="left")

# Keep only second innings (for chase-based prediction)
merged_df = merged_df[merged_df["inning"] == 2].copy()


# Compute cricket match situation features
merged_df["runs_remaining"] = merged_df["target_runs"] - merged_df["cumulative_runs"]
merged_df["current_run_rate"] = merged_df["cumulative_runs"] / (merged_df["balls_bowled"] / 6)
merged_df["required_run_rate"] = (merged_df["runs_remaining"] / merged_df["balls_remaining"]) * 6
merged_df["wickets_left"] = 10 - merged_df["cumulative_wkts"]

# Handle division-by-zero or inf values
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
merged_df.dropna(subset=["required_run_rate"], inplace=True)



# Ensure correct winner column is available
if "winner" not in merged_df.columns:
    if "winner_x" in merged_df.columns:
        merged_df.rename(columns={"winner_x": "winner"}, inplace=True)
    elif "winner_y" in merged_df.columns:
        merged_df.rename(columns={"winner_y": "winner"}, inplace=True)
    else:
        raise KeyError("Winner column not found after merge.")


#Target label: whether batting team won
merged_df["batting_team_won"] = np.where(merged_df["batting_team"] == merged_df["winner"], 1, 0)

# Keep only relevant columns for ML
feature_cols = [
    "match_id", "season", "city", "venue", "batting_team", "bowling_team",
    "runs_remaining", "balls_remaining", "wickets_left",
    "current_run_rate", "required_run_rate", "batting_team_won"
]
features_df = merged_df[feature_cols].copy()

print("dataset shape:", features_df.shape)


#############################################################################################################################

# Feature Preprocessing

# Define feature groups and label
numeric_features = [
    "runs_remaining",
    "balls_remaining",
    "wickets_left",
    "current_run_rate",
    "required_run_rate",
]

categorical_features = [
    "batting_team",
    "bowling_team",
    "venue",
]

label_col = "batting_team_won"

# Defensive check
missing_cols = [c for c in numeric_features + categorical_features + [label_col] if c not in features_df.columns]
if missing_cols:
    raise KeyError(f"Missing columns for preprocessing: {missing_cols}")

# Extract year for sorting
def season_to_key(s):
    try:
        return int(str(s)[:4])
    except:
        return 0

features_df["season_key"] = features_df["season"].apply(season_to_key)
unique_seasons = sorted(features_df["season_key"].unique())

if len(unique_seasons) > 1:
    test_season = unique_seasons[-1]
    test_df = features_df[features_df["season_key"] == test_season]
    train_val_df = features_df[features_df["season_key"] != test_season]
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1, random_state=42, stratify=train_val_df[label_col]
    )
    print(f"🧭 Using latest season {test_season} for test split.")
else:
    # fallback: single season, random split
    train_val_df, test_df = train_test_split(features_df, test_size=0.1, random_state=42, stratify=features_df[label_col])
    train_df, val_df = train_test_split(train_val_df, test_size=0.111, random_state=42, stratify=train_val_df[label_col])
    print("🧭 Single season found, used random stratified split.")

print("Split sizes -> Train:", len(train_df), "| Val:", len(val_df), "| Test:", len(test_df))

#Build preprocessing pipeline
num_scaler = StandardScaler()
cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_scaler, numeric_features),
        ("cat", cat_encoder, categorical_features),
    ],
    remainder="drop",
)

pipeline = Pipeline([("preprocessor", preprocessor)])


# Fit on training data only
X_train = train_df[numeric_features + categorical_features]
y_train = train_df[label_col].astype(int)

pipeline.fit(X_train, y_train)


# Transform all splits
def transform_split(df):
    X = df[numeric_features + categorical_features]
    X_arr = pipeline.transform(X)
    y_arr = df[label_col].astype(int).values
    return X_arr, y_arr

X_train, y_train = transform_split(train_df)
X_val, y_val = transform_split(val_df)
X_test, y_test = transform_split(test_df)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")



##############################################################################################################################

#Model Training & Evaluation


# Train XGBoost Model

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)

print("\n XGBoost Results:")
print(f"Accuracy: {xgb_acc:.4f}")
print(f"AUC: {xgb_auc:.4f}")
print(classification_report(y_test, y_pred_xgb))

# Save the XGBoost model
joblib.dump(xgb_model, "../models/xgb_model.pkl")


########################################################################################################################


# Model Evaluation & Interpretation

# Load the final model

final_model = joblib.load("../models/xgb_model.pkl")

# Confusion Matrix

y_pred = final_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
labels = ["Lost (0)", "Won (1)"]

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix Match Outcome Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve
if hasattr(final_model, "predict_proba"):
    y_proba = final_model.predict_proba(X_test)[:, 1]
else:
    y_proba = final_model.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
