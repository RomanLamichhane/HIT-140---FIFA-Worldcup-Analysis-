import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# HIT140 Assessment 2 — FIFA World Cup 2026
# Analytical Question 3: Attacking Contribution
# Do higher-ranked teams record significantly more attacking
# contributions per match than lower-ranked teams?

players = pd.read_csv("player_stats.csv")
match_stats = pd.read_csv("match_team_stats.csv")
teams = pd.read_csv("teams.csv")

team_matches = (
    match_stats.groupby("team_id")["match_id"]
    .nunique()
    .rename("matches_played")
)

team_attack = (
    players.groupby("team_id")
    .agg(
        total_goals=("goals", "sum"),
        total_assists=("assists", "sum")
    )
    .join(team_matches)
    .reset_index()
)

df = teams[
    ["team_id", "team_name", "fifa_code",
     "fifa_ranking_pre_tournament"]
].merge(team_attack, on="team_id", how="left")

df["attacking_contributions"] = (
    df["total_goals"] + df["total_assists"]
)

df["attacking_contributions_per_match"] = (
    df["attacking_contributions"] / df["matches_played"]
)

median_rank = df["fifa_ranking_pre_tournament"].median()

df["ranking_group"] = np.where(
    df["fifa_ranking_pre_tournament"] <= median_rank,
    "Higher-ranked",
    "Lower-ranked"
)

df.to_csv(
    "Saroj_WC2026_Attacking_Contribution_Data_Cleaned.csv",
    index=False
)

high = df.loc[
    df["ranking_group"] == "Higher-ranked",
    "attacking_contributions_per_match"
]

low = df.loc[
    df["ranking_group"] == "Lower-ranked",
    "attacking_contributions_per_match"
]

print("Median FIFA ranking:", median_rank)
print("\nHigher-ranked group")
print(high.describe())
print("\nLower-ranked group")
print(low.describe())

# 95% CI for the difference in means
difference = high.mean() - low.mean()
se = math.sqrt(
    high.var(ddof=1)/len(high) +
    low.var(ddof=1)/len(low)
)

test = stats.ttest_ind(high, low, equal_var=False)
t_critical = stats.t.ppf(0.975, test.df)
ci_lower = difference - t_critical * se
ci_upper = difference + t_critical * se

print("\n95% CI for mean difference:")
print(f"{ci_lower:.3f} to {ci_upper:.3f}")

# Welch two-sample t-test
print("\nH0: mu_high = mu_low")
print("H1: mu_high != mu_low")
print(f"t({test.df:.2f}) = {test.statistic:.3f}")
print(f"p-value = {test.pvalue:.4f}")

if test.pvalue < 0.05:
    print("Decision: Reject H0.")
else:
    print("Decision: Fail to reject H0.")

# Boxplot
plt.figure(figsize=(8,6))
plt.boxplot(
    [high, low],
    tick_labels=["Higher-ranked", "Lower-ranked"]
)
plt.title(
    "Attacking Contributions per Match by FIFA Ranking Group"
)
plt.ylabel("Goals + assists per match")
plt.tight_layout()
plt.savefig(
    "Saroj_WC2026_Attacking_Contribution_Boxplot.png",
    dpi=200
)
plt.show()

# Bar chart
plt.figure(figsize=(7,6))
plt.bar(
    ["Higher-ranked", "Lower-ranked"],
    [high.mean(), low.mean()],
    yerr=[
        high.std(ddof=1)/math.sqrt(len(high)),
        low.std(ddof=1)/math.sqrt(len(low))
    ],
    capsize=6
)
plt.title("Mean Attacking Contributions per Match")
plt.ylabel("Goals + assists per match")
plt.tight_layout()
plt.savefig(
    "Saroj_WC2026_Attacking_Contribution_BarChart.png",
    dpi=200
)
plt.show()
