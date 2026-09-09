import csv
import math
import statistics
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------------------------------------
# HIT140 Assessment 2 — FIFA World Cup 2026
# Analytical Question 1: Ball Possession
# Question: Does average possession among teams differ from 50%?
# ------------------------------------------------------------

MATCH_STATS_FILE = "match_team_stats.csv"
TEAMS_FILE = "teams.csv"

# 1. LOAD DATA
with open(TEAMS_FILE, newline="", encoding="utf-8-sig") as f:
    team_lookup = {row["team_id"]: row for row in csv.DictReader(f)}

with open(MATCH_STATS_FILE, newline="", encoding="utf-8-sig") as f:
    match_stats = list(csv.DictReader(f))

print("Rows in match_team_stats.csv:", len(match_stats))
print("Teams in teams.csv:", len(team_lookup))

# 2. CLEAN AND PREPARE
# Collect each team's possession percentage from every match.
team_possessions = defaultdict(list)

for row in match_stats:
    possession = row["possession_pct"].strip()
    if possession:
        team_possessions[row["team_id"]].append(float(possession))

# Create one observation per national team by averaging match possession.
team_rows = []

for team_id, values in team_possessions.items():
    info = team_lookup[team_id]

    team_rows.append(
        {
            "team_id": int(team_id),
            "team_name": info["team_name"],
            "fifa_code": info["fifa_code"],
            "matches_observed": len(values),
            "average_possession_pct": sum(values) / len(values),
            "median_possession_pct": statistics.median(values),
            "min_possession_pct": min(values),
            "max_possession_pct": max(values),
        }
    )

team_rows.sort(
    key=lambda row: row["average_possession_pct"],
    reverse=True
)

# 3. SAVE CLEANED DATASET
cleaned_file = "Raman_WC2026_Ball_Possession_Data_Cleaned.csv"

fieldnames = [
    "team_id",
    "team_name",
    "fifa_code",
    "matches_observed",
    "average_possession_pct",
    "median_possession_pct",
    "min_possession_pct",
    "max_possession_pct",
]

with open(cleaned_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in team_rows:
        output_row = dict(row)
        output_row["average_possession_pct"] = round(
            output_row["average_possession_pct"], 3
        )
        output_row["median_possession_pct"] = round(
            output_row["median_possession_pct"], 3
        )
        output_row["min_possession_pct"] = round(
            output_row["min_possession_pct"], 3
        )
        output_row["max_possession_pct"] = round(
            output_row["max_possession_pct"], 3
        )
        writer.writerow(output_row)

# 4. DESCRIPTIVE STATISTICS
possession = np.array(
    [row["average_possession_pct"] for row in team_rows],
    dtype=float
)

n = len(possession)
mean = np.mean(possession)
median = np.median(possession)
std_dev = np.std(possession, ddof=1)
minimum = np.min(possession)
q1 = np.percentile(possession, 25)
q3 = np.percentile(possession, 75)
maximum = np.max(possession)

print("\nDESCRIPTIVE STATISTICS")
print("----------------------")
print(f"n: {n}")
print(f"Mean: {mean:.2f}%")
print(f"Median: {median:.2f}%")
print(f"Standard deviation: {std_dev:.2f}")
print(f"Minimum: {minimum:.2f}%")
print(f"Q1: {q1:.2f}%")
print(f"Q3: {q3:.2f}%")
print(f"Maximum: {maximum:.2f}%")

# 5. 95% CONFIDENCE INTERVAL
standard_error = std_dev / math.sqrt(n)
t_critical = stats.t.ppf(0.975, df=n - 1)

ci_lower = mean - t_critical * standard_error
ci_upper = mean + t_critical * standard_error

print("\n95% CONFIDENCE INTERVAL")
print("-----------------------")
print(f"{ci_lower:.2f}% to {ci_upper:.2f}%")

# 6. ONE-SAMPLE T-TEST
# H0: population mean possession = 50%
# H1: population mean possession != 50%

test = stats.ttest_1samp(possession, popmean=50)

t_statistic = test.statistic
p_value = test.pvalue

print("\nONE-SAMPLE T-TEST")
print("-----------------")
print("H0: mu = 50%")
print("H1: mu != 50%")
print(f"t({n - 1}) = {t_statistic:.3f}")
print(f"p-value = {p_value:.4f}")

if p_value < 0.05:
    print("Decision: Reject H0.")
    print(
        "Conclusion: Mean team possession differs "
        "significantly from 50%."
    )
else:
    print("Decision: Fail to reject H0.")
    print(
        "Conclusion: There is insufficient evidence that "
        "mean team possession differs from 50%."
    )

# 7. BAR CHART
team_names = [row["team_name"] for row in team_rows]
team_averages = [
    row["average_possession_pct"] for row in team_rows
]

plt.figure(figsize=(13, 7))
plt.bar(team_names, team_averages)
plt.axhline(50, linestyle="--", label="50% reference")
plt.title(
    "Average Ball Possession by Team — FIFA World Cup 2026"
)
plt.xlabel("Team")
plt.ylabel("Average Possession (%)")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig(
    "Raman_WC2026_Ball_Possession_BarChart.png",
    dpi=200
)
plt.show()

# 8. BOXPLOT
plt.figure(figsize=(7, 6))
plt.boxplot(possession)
plt.axhline(50, linestyle="--", label="50% reference")
plt.title("Distribution of Team Average Ball Possession")
plt.ylabel("Average Possession (%)")
plt.xticks([1], ["48 teams"])
plt.legend()
plt.tight_layout()
plt.savefig(
    "Raman_WC2026_Ball_Possession_Boxplot.png",
    dpi=200
)
plt.show()

# 9. HISTOGRAM
plt.figure(figsize=(8, 6))
plt.hist(possession, bins=10, edgecolor="black")
plt.axvline(50, linestyle="--", label="50% reference")
plt.title("Distribution of Team Average Ball Possession")
plt.xlabel("Average Possession (%)")
plt.ylabel("Number of Teams")
plt.legend()
plt.tight_layout()
plt.savefig(
    "Raman_WC2026_Ball_Possession_Histogram.png",
    dpi=200
)
plt.show()
