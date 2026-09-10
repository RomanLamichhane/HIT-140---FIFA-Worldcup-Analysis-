import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# HIT140 Assessment 2 — FIFA World Cup 2026
# Analytical Question 4: Squad Age
# Do higher-ranked teams have a significantly different
# average squad age than lower-ranked teams?

squads = pd.read_csv("squads_and_players.csv")
teams = pd.read_csv("teams.csv")

reference_date = pd.Timestamp("2026-06-11")
squads["date_of_birth"] = pd.to_datetime(
    squads["date_of_birth"]
)

squads["age_years"] = (
    (reference_date - squads["date_of_birth"]).dt.days
    / 365.2425
)

team_age = (
    squads.groupby("team_id")
    .agg(
        squad_size=("player_id", "count"),
        average_squad_age=("age_years", "mean"),
        median_squad_age=("age_years", "median"),
        min_squad_age=("age_years", "min"),
        max_squad_age=("age_years", "max")
    )
    .reset_index()
)

df = teams[
    ["team_id", "team_name", "fifa_code",
     "fifa_ranking_pre_tournament"]
].merge(team_age, on="team_id", how="left")

median_rank = df["fifa_ranking_pre_tournament"].median()

df["ranking_group"] = np.where(
    df["fifa_ranking_pre_tournament"] <= median_rank,
    "Higher-ranked",
    "Lower-ranked"
)

df.to_csv(
    "Abhinav_WC2026_Squad_Age_Data_Cleaned.csv",
    index=False
)

high = df.loc[
    df["ranking_group"] == "Higher-ranked",
    "average_squad_age"
]

low = df.loc[
    df["ranking_group"] == "Lower-ranked",
    "average_squad_age"
]

print("Median FIFA ranking:", median_rank)
print("\nHigher-ranked teams")
print(high.describe())
print("\nLower-ranked teams")
print(low.describe())

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
print(f"{ci_lower:.2f} to {ci_upper:.2f} years")

print("\nH0: mu_high = mu_low")
print("H1: mu_high != mu_low")
print(f"t({test.df:.2f}) = {test.statistic:.3f}")
print(f"p-value = {test.pvalue:.4f}")

if test.pvalue < 0.05:
    print("Decision: Reject H0.")
else:
    print("Decision: Fail to reject H0.")

plt.figure(figsize=(8,6))
plt.boxplot(
    [high, low],
    tick_labels=["Higher-ranked", "Lower-ranked"]
)
plt.title("Average Squad Age by FIFA Ranking Group")
plt.ylabel("Average squad age (years)")
plt.tight_layout()
plt.savefig(
    "Abhinav_WC2026_Squad_Age_Boxplot.png",
    dpi=200
)
plt.show()

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
plt.title("Mean Squad Age by FIFA Ranking Group")
plt.ylabel("Mean squad age (years)")
plt.tight_layout()
plt.savefig(
    "Abhinav_WC2026_Squad_Age_BarChart.png",
    dpi=200
)
plt.show()
