import pandas as pd
import numpy as np


def clean_match_record(match: dict) -> dict:
    match = match.copy()

    unwanted_match_cols = ["method", "umpire1", "umpire2"]
    for c in unwanted_match_cols:
        match.pop(c, None)

    for col in ["result_margin", "target_runs", "target_overs"]:
        if col in match:
            try:
                match[col] = int(match[col]) if match[col] not in [None, "", "null"] else None
            except:
                match[col] = None

    for col in ["result_margin", "target_runs", "target_overs"]:
        if match.get(col) is None:
            match[col] = 0

    season = match.get("season")
    if season and "/" in season:
        p = season.split("/")
        match["season_start"] = int(p[0])
        end = p[1]
        if len(end) == 2:
            end = "20" + end
        match["season_end"] = int(end)
    else:
        match["season_start"] = None
        match["season_end"] = None

    mapping = {
        'delhi daredevils': 'delhi capitals',
        'kings xi punjab': 'punjab kings',
        'rising pune supergiants': 'pune warriors',
        'rising pune supergiant': 'pune warriors',
        'gujarat lions': 'gujarat titans',
        'deccan chargers': 'sunrisers hyderabad',
        'royal challengers bengaluru': 'royal challengers bangalore'
    }

    def fix_team(name: str):
        if not name:
            return name
        name_l = name.lower()
        return mapping.get(name_l, name_l)

    for col in ["team1", "team2", "toss_winner", "winner"]:
        if col in match:
            match[col] = fix_team(match[col])

    for k, v in match.items():
        if isinstance(v, str):
            match[k] = v.lower()

    if "id" in match:
        match["match_id"] = match.pop("id")

    return match


def clean_delivery_record(delivery: dict) -> dict:
    delivery = delivery.copy()

    unwanted_delivery_cols = ["player_dismissed", "dismissal_kind", "fielder"]
    for c in unwanted_delivery_cols:
        delivery.pop(c, None)

    for k, v in delivery.items():
        if isinstance(v, str):
            delivery[k] = v.lower()

    return delivery


def merge_match_delivery(match: dict, delivery: dict) -> pd.DataFrame:
    """
    Merge single delivery + single match just like Spark join.
    Keep only necessary delivery features.
    """
    ball_features = [
        "match_id", "inning", "over", "ball",
        "batting_team", "bowling_team",
        "batsman_runs", "extra_runs", "total_runs", "is_wicket"
    ]

    merged = {}

    # Add allowed delivery fields
    for key in ball_features:
        if key in delivery:
            merged[key] = delivery[key]

    # Merge match fields (Spark join)
    for k, v in match.items():
        merged[k] = v

    # Drop leakage fields
    for c in ["win_by_runs", "win_by_wickets", "result", "result_margin"]:
        merged.pop(c, None)

    return pd.DataFrame([merged])


def generate_features(df):
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
    
    
    #Cumulative Runs and Wickets - new features
    df["cumulative_runs"] = df.groupby("match_id")["total_runs"].cumsum()
    df["cumulative_wkts"] = df.groupby("match_id")["is_wicket"].cumsum()
    
    #Balls Bowled and Remaining
    
    # Ball number calculation: (over * 6) + ball_in_over, adjusted for 0-indexing
    df["balls_bowled"] = (df["over"] * 6) + df["ball"]
    # Total balls available in a standard T20 chase is 120 (20 overs * 6 balls)
    df["balls_remaining"] = 120 - df["balls_bowled"]
    # Ensure balls_remaining is not negative, though data quality should prevent this
    df["balls_remaining"] = df["balls_remaining"].clip(lower=0)
    
    
    #Runs and Wickets Left - key features
    df["runs_remaining"] = df["target_runs"] - df["cumulative_runs"]
    df["wickets_left"] = 10 - df["cumulative_wkts"]
    df["wickets_left"] = df["wickets_left"].clip(lower=0)
    
    
    # Run Rate Metrics (Momentum Features)
    # Current Run Rate (CRR): Total runs / Overs bowled
    df["overs_bowled"] = df["balls_bowled"] / 6
    df["current_run_rate"] = np.where(
        df["overs_bowled"] > 0,
        df["cumulative_runs"] / df["overs_bowled"],
        0
    )
    
    # Required Run Rate (RRR): Runs remaining / Overs remaining * 6
    df["overs_remaining"] = df["balls_remaining"] / 6
    df["required_run_rate"] = np.where(
        df["overs_remaining"] > 0,
        (df["runs_remaining"] / df["overs_remaining"]),
        np.where(df["runs_remaining"] > 0, np.inf, 0)
    )
    
    
    #Toss Advantage
    df['toss_advantage'] = np.where(df['batting_team'] == df['toss_winner'], 1, 0)
    
    
    # 6. Over Phase
    df['over_phase'] = pd.cut(df['over'],
                                        bins=[-1, 5, 10, 15, 20],
                                        labels=['Powerplay', 'Middle_1', 'Middle_2', 'Death'],
                                        right=True)
    
    
    ################################################################################################################################
    
    #3. Target Variable Creation
    
    # Target label: whether batting team won
    df["batting_team_won"] = np.where(
        df["batting_team"] == df["winner"], 1, 0
    )
    
    # Handle division-by-zero or inf values created in RRR (replace inf with a large number for safety)
    df.replace([np.inf, -np.inf], 999, inplace=True)
    df.dropna(subset=["required_run_rate", "current_run_rate"], inplace=True) # Drop any remaining NaNs
    
    # Keep only relevant columns for ML
    feature_cols = [
        "match_id", "season", "city", "venue", "batting_team", "bowling_team",
        "runs_remaining", "balls_remaining", "wickets_left",
        "current_run_rate", "required_run_rate", "toss_advantage", "over_phase",
        "batting_team_won"
    ]
    features_df = df[feature_cols].copy()
    
    print("\nFinal feature set ready.")
    print("Dataset shape:", features_df.shape)
    return features_df