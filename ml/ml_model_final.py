import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import pickle
 
# Define the file path
FILE_PATH = 'data/processed/ipl_cleaned.csv'
 
############################################################################################################################
 
#1. Data Loading and Initial Cleaning
try:
    df = pd.read_csv(FILE_PATH, low_memory=False)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    raise
 
#Dropping unnecessary columns
COLUMNS_TO_DROP = [
    'date', 'match_type', 'player_of_match', 'target_overs', 'super_over',
    'season_start', 'season_end', 'team1', 'team2'
]
df.drop(columns=COLUMNS_TO_DROP, inplace=True, errors='ignore')
 
#Handle missing values
df['city'] = df['city'].fillna('Unknown')
df['venue'] = df['venue'].fillna('Unknown')
df.dropna(subset=['winner', 'toss_winner', 'toss_decision'], inplace=True)
df.sort_values(by=["match_id", "inning", "over", "ball"], inplace=True)
df.reset_index(drop=True, inplace=True)
 
print(f"Data after cleaning and dropping columns: {df.shape}")
 
#################################################################################################################################
 
#2. Feature Engineering
 
# Keeping only 2nd innings
df_2nd_inning = df[df["inning"] == 2].copy()
print(f"Filtered for 2nd innings: {df_2nd_inning.shape}")
 
 
#Cumulative Runs and Wickets - new features
df_2nd_inning["cumulative_runs"] = df_2nd_inning.groupby("match_id")["total_runs"].cumsum()
df_2nd_inning["cumulative_wkts"] = df_2nd_inning.groupby("match_id")["is_wicket"].cumsum()
 
#Balls Bowled and Remaining
 
# Ball number calculation: (over * 6) + ball_in_over, adjusted for 0-indexing
df_2nd_inning["balls_bowled"] = (df_2nd_inning["over"] * 6) + df_2nd_inning["ball"]
# Total balls available in a standard T20 chase is 120 (20 overs * 6 balls)
df_2nd_inning["balls_remaining"] = 120 - df_2nd_inning["balls_bowled"]
# Ensure balls_remaining is not negative, though data quality should prevent this
df_2nd_inning["balls_remaining"] = df_2nd_inning["balls_remaining"].clip(lower=0)
 
 
#Runs and Wickets Left - key features
df_2nd_inning["runs_remaining"] = df_2nd_inning["target_runs"] - df_2nd_inning["cumulative_runs"]
df_2nd_inning["wickets_left"] = 10 - df_2nd_inning["cumulative_wkts"]
df_2nd_inning["wickets_left"] = df_2nd_inning["wickets_left"].clip(lower=0)
 
 
# Run Rate Metrics (Momentum Features)
# Current Run Rate (CRR): Total runs / Overs bowled
df_2nd_inning["overs_bowled"] = df_2nd_inning["balls_bowled"] / 6
df_2nd_inning["current_run_rate"] = np.where(
    df_2nd_inning["overs_bowled"] > 0,
    df_2nd_inning["cumulative_runs"] / df_2nd_inning["overs_bowled"],
    0
)
 
# Required Run Rate (RRR): Runs remaining / Overs remaining * 6
df_2nd_inning["overs_remaining"] = df_2nd_inning["balls_remaining"] / 6
df_2nd_inning["required_run_rate"] = np.where(
    df_2nd_inning["overs_remaining"] > 0,
    (df_2nd_inning["runs_remaining"] / df_2nd_inning["overs_remaining"]),
    np.where(df_2nd_inning["runs_remaining"] > 0, np.inf, 0)
)
 
 
#Toss Advantage
df_2nd_inning['toss_advantage'] = np.where(df_2nd_inning['batting_team'] == df_2nd_inning['toss_winner'], 1, 0)
 
 
# 6. Over Phase
df_2nd_inning['over_phase'] = pd.cut(df_2nd_inning['over'],
                                     bins=[-1, 5, 10, 15, 20],
                                     labels=['Powerplay', 'Middle_1', 'Middle_2', 'Death'],
                                     right=True)
 
 
################################################################################################################################
 
#3. Target Variable Creation
 
# Target label: whether batting team won
df_2nd_inning["batting_team_won"] = np.where(
    df_2nd_inning["batting_team"] == df_2nd_inning["winner"], 1, 0
)
 
# Handle division-by-zero or inf values created in RRR (replace inf with a large number for safety)
df_2nd_inning.replace([np.inf, -np.inf], 999, inplace=True)
df_2nd_inning.dropna(subset=["required_run_rate", "current_run_rate"], inplace=True) # Drop any remaining NaNs
 
# Keep only relevant columns for ML
feature_cols = [
    "match_id", "season", "city", "venue", "batting_team", "bowling_team",
    "runs_remaining", "balls_remaining", "wickets_left",
    "current_run_rate", "required_run_rate", "toss_advantage", "over_phase",
    "batting_team_won"
]
features_df = df_2nd_inning[feature_cols].copy()
 
print("\nFinal feature set ready.")
print("Dataset shape:", features_df.shape)
 
###############################################################################################################################
 
#4. Feature Preprocessing and Time-Based Split
 
# Define feature groups and label
numeric_features = [
    "runs_remaining",
    "balls_remaining",
    "wickets_left",
    "current_run_rate",
    "required_run_rate",
    "toss_advantage"
]
 
categorical_features = [
    "batting_team",
    "bowling_team",
    "venue",
    "city",
    "over_phase"
]
 
label_col = "batting_team_won"
 
 
print("\n Splitting Data (70% Train, 20% Test, 10% Validation)")
 
 
train_val_df, test_df = train_test_split(
    features_df,
    test_size=0.20,
    random_state=42,
    stratify=features_df[label_col]
)
 
#Split (Train + Validation) into Train and Validation
 
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.125,
    random_state=42,
    stratify=train_val_df[label_col]
)
 
print(f"Total Samples: {len(features_df)}")
print(f"Train Size (70%): {len(train_df)}")
print(f"Test Size  (20%): {len(test_df)}")
print(f"Val Size   (10%): {len(val_df)}")
 
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
 
#Combine Preprocessor and Model into a full pipeline for XGBoost Classifier
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,     
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    ))
])
 
 
############################################################################################################################
#5. Model Training and Evaluation
 
 
X_train = train_df.drop(columns=[label_col, 'match_id', 'season'])
y_train = train_df[label_col].astype(int)
 
 
X_val = val_df.drop(columns=[label_col, 'match_id', 'season'])
y_val = val_df[label_col].astype(int)
 
X_test = test_df.drop(columns=[label_col, 'match_id', 'season'])
y_test = test_df[label_col].astype(int)
 
print("\nTraining XGBoost Classifier")
full_pipeline.fit(X_train, y_train)
 
y_pred_xgb = full_pipeline.predict(X_test)
y_pred_proba_xgb = full_pipeline.predict_proba(X_test)[:, 1]
 
# Evaluation
xgb_acc = accuracy_score(y_test, y_pred_xgb)
 
# Safe AUC Calculation
if len(np.unique(y_test)) > 1:
    xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)
    auc_text = f"AUC: {xgb_auc:.4f}"
else:
    xgb_auc = None
    auc_text = "AUC: Not defined (Only one class in test set)"
 
print(f"Accuracy: {xgb_acc:.4f}")
print(auc_text)
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
 
 
#############################################################################################################################
#6. Model Evaluation Visualizations
 
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_xgb)
labels = ["Lost (0)", "Won (1)"]
 
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix Match Outcome Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
 
# ROC Curve (Only plot if AUC is defined)
if xgb_auc is not None:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_xgb)
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
else:
    print("\nSkipping ROC Curve plot because only one class is present in the test set.")

#############################################################################################################################

#8. Model Persistence (Save Model and LabelEncoder) ---

MODEL_PATH = 'models/ipl_winner_xgb_powerplay_model.pkl'
ENCODER_PATH = 'models/ipl_winner_label_encoder.pkl'

print(f"\nSaving Trained Model and LabelEncoder")

try:
    # Save the entire pipeline (including preprocessing steps)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(full_pipeline, file)
    print(f"Trained Pipeline saved successfully to {MODEL_PATH}")

    # Save the LabelEncoder (CRITICAL for decoding future predictions)
    with open(ENCODER_PATH, 'wb') as file:
        pickle.dump(le, file)
    print(f"LabelEncoder saved successfully to {ENCODER_PATH}")

except Exception as e:
    print(f"An error occurred while saving files: {e}")

# --- End of ipl_winner_pipeline.py ---    