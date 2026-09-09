import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------------------------------------
# HIT140 Assessment 2 — FIFA World Cup 2026
# Analytical Question 2: Discipline
# Question:
# Do teams with higher foul rates receive significantly more
# yellow cards per match than teams with lower foul rates?
# ------------------------------------------------------------

EVENTS_FILE = "match_events.csv"
MATCH_STATS_FILE = "match_team_stats.csv"
TEAMS_FILE = "teams.csv"

# 1. LOAD DATA
events = pd.read_csv(EVENTS_FILE)
match_stats = pd.read_csv(MATCH_STATS_FILE)
teams = pd.read_csv(TEAMS_FILE)

print("Rows in match_events.csv:", len(events))
print("Rows in match_team_stats.csv:", len(match_stats))
print("Teams in teams.csv:", len(teams))

# 2. PREPARE TEAM-LEVEL DISCIPLINE DATA
team_match = (
    match_stats.groupby("team_id")
    .agg(
        matches_played=("match_id", "count"),
        avg_fouls_per_match=("fouls", "mean")
    )
    .reset_index()
)

yellow = (
    events.loc[events["event_type"] == "Yellow Card"]
    .groupby("team_id")
    .size()
    .rename("yellow_cards_total")
)

red = (
    events.loc[events["event_type"] == "Red Card"]
    .groupby("team_id")
    .size()
    .rename("red_cards_total")
)

clean = (
    teams[["team_id", "team_name", "fifa_code"]]
    .merge(team_match, on="team_id", how="left")
    .merge(yellow, on="team_id", how="left")
    .merge(red, on="team_id", how="left")
)

clean[["yellow_cards_total", "red_cards_total"]] = (
    clean[["yellow_cards_total", "red_cards_total"]].fillna(0)
)

clean["yellow_cards_total"] = clean["yellow_cards_total"].astype(int)
clean["red_cards_total"] = clean["red_cards_total"].astype(int)

clean["yellow_cards_per_match"] = (
    clean["yellow_cards_total"] / clean["matches_played"]
)
clean["red_cards_per_match"] = (
    clean["red_cards_total"] / clean["matches_played"]
)

# Split teams into high-foul and low-foul groups using the median.
median_fouls = clean["avg_fouls_per_match"].median()

clean["foul_group"] = np.where(
    clean["avg_fouls_per_match"] >= median_fouls,
    "High fouls",
    "Low fouls"
)

# 3. SAVE CLEANED DATASET
clean.to_csv(
    "Saroj_WC2026_Discipline_Data_Cleaned.csv",
    index=False
)

# 4. DESCRIPTIVE STATISTICS
high = clean.loc[
    clean["foul_group"] == "High fouls",
    "yellow_cards_per_match"
]

low = clean.loc[
    clean["foul_group"] == "Low fouls",
    "yellow_cards_per_match"
]

print("\nMEDIAN FOUL RATE USED TO SPLIT GROUPS")
print(f"{median_fouls:.3f} fouls per match")

print("\nHIGH-FOUL GROUP")
print(f"n = {len(high)}")
print(f"Mean = {high.mean():.3f}")
print(f"Median = {high.median():.3f}")
print(f"SD = {high.std(ddof=1):.3f}")
print(f"Min = {high.min():.3f}")
print(f"Max = {high.max():.3f}")

print("\nLOW-FOUL GROUP")
print(f"n = {len(low)}")
print(f"Mean = {low.mean():.3f}")
print(f"Median = {low.median():.3f}")
print(f"SD = {low.std(ddof=1):.3f}")
print(f"Min = {low.min():.3f}")
print(f"Max = {low.max():.3f}")

# 5. 95% CONFIDENCE INTERVAL FOR DIFFERENCE IN MEANS
difference = high.mean() - low.mean()

standard_error = math.sqrt(
    high.var(ddof=1) / len(high)
    + low.var(ddof=1) / len(low)
)

welch_df = (
    (high.var(ddof=1) / len(high) + low.var(ddof=1) / len(low)) ** 2
    /
    (
        (high.var(ddof=1) / len(high)) ** 2 / (len(high) - 1)
        + (low.var(ddof=1) / len(low)) ** 2 / (len(low) - 1)
    )
)

critical_t = stats.t.ppf(0.975, df=welch_df)

ci_lower = difference - critical_t * standard_error
ci_upper = difference + critical_t * standard_error

print("\n95% CONFIDENCE INTERVAL FOR MEAN DIFFERENCE")
print(f"{ci_lower:.3f} to {ci_upper:.3f}")

# 6. WELCH INDEPENDENT TWO-SAMPLE T-TEST
# H0: mean yellow-card rate is equal between groups
# H1: mean yellow-card rate differs between groups

test = stats.ttest_ind(high, low, equal_var=False)

print("\nWELCH TWO-SAMPLE T-TEST")
print("H0: mu_high = mu_low")
print("H1: mu_high != mu_low")
print(f"t({test.df:.2f}) = {test.statistic:.3f}")
print(f"p-value = {test.pvalue:.4f}")

if test.pvalue < 0.05:
    print("Decision: Reject H0.")
else:
    print("Decision: Fail to reject H0.")

# 7. BOXPLOT
plt.figure(figsize=(8, 6))
plt.boxplot(
    [high, low],
    tick_labels=["High fouls", "Low fouls"]
)
plt.title("Yellow Cards per Match by Team Foul Group")
plt.ylabel("Yellow cards per match")
plt.tight_layout()
plt.savefig(
    "Saroj_WC2026_Discipline_Boxplot.png",
    dpi=200
)
plt.show()

# 8. BAR CHART
plt.figure(figsize=(7, 6))
plt.bar(
    ["High fouls", "Low fouls"],
    [high.mean(), low.mean()],
    yerr=[
        high.std(ddof=1) / math.sqrt(len(high)),
        low.std(ddof=1) / math.sqrt(len(low))
    ],
    capsize=6
)
plt.title("Mean Yellow Cards per Match by Foul Group")
plt.ylabel("Mean yellow cards per match")
plt.tight_layout()
plt.savefig(
    "Saroj_WC2026_Discipline_BarChart.png",
    dpi=200
)
plt.show()

# 9. HISTOGRAM
plt.figure(figsize=(8, 6))
plt.hist(high, bins=6, alpha=0.6, label="High fouls")
plt.hist(low, bins=6, alpha=0.6, label="Low fouls")
plt.title("Distribution of Yellow Cards per Match")
plt.xlabel("Yellow cards per match")
plt.ylabel("Number of teams")
plt.legend()
plt.tight_layout()
plt.savefig(
    "Saroj_WC2026_Discipline_Histogram.png",
    dpi=200
)
plt.show()
