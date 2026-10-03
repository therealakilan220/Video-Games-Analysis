"""
Generate Visualizations for Video Game Clustering Project
Big Data Machine Learning Deliverable
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup plotting style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports"
FIGURES_DIR = EXPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data
elbow_df = pd.read_csv(EXPORTS_DIR / "elbow_cost.csv")
games_df = pd.read_csv(EXPORTS_DIR / "clustered_games.csv")
cluster_stats_df = pd.read_csv(EXPORTS_DIR / "cluster_statistics.csv")
cluster_sizes_df = pd.read_csv(EXPORTS_DIR / "cluster_sizes.csv")

cluster_names = {
    0: "Cluster 0: Budget & F2P Hits",
    1: "Cluster 1: Mid-Tier Indie Hits",
    2: "Cluster 2: Poorly-Rated Games",
    3: "Cluster 3: AAA Blockbusters & Time-Sinks",
}
games_df["cluster_label"] = games_df["cluster_id"].map(cluster_names)

# ==========================================
# 1. Elbow & Silhouette Evaluation Plot
# ==========================================
fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

color1 = "#1f77b4"
ax1.set_xlabel("Number of Clusters (K)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Within-Cluster Sum of Squares (WSSSE Cost)", color=color1, fontsize=12, fontweight="bold")
line1 = ax1.plot(elbow_df["K"], elbow_df["WSSSE_Cost"], marker="o", linewidth=2.5, color=color1, label="WSSSE Cost (Elbow)")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_xticks(elbow_df["K"])

# Secondary axis for Silhouette
ax2 = ax1.twinx()
color2 = "#2ca02c"
ax2.set_ylabel("Silhouette Score", color=color2, fontsize=12, fontweight="bold")
line2 = ax2.plot(elbow_df["K"], elbow_df["Silhouette_Score"], marker="s", linewidth=2.5, linestyle="--", color=color2, label="Silhouette Score")
ax2.tick_params(axis="y", labelcolor=color2)
ax2.grid(False)

# Highlight optimal K=4
ax1.axvline(x=4, color="red", linestyle=":", linewidth=2, alpha=0.8)
ax1.annotate("Optimal K = 4\n(Max Silhouette = 0.4663)", xy=(4, 537.02), xytext=(5.2, 700),
             arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
             fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

plt.title("Model Selection: Elbow Method & Silhouette Score vs. K", fontsize=14, fontweight="bold", pad=15)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right", frameon=True)
fig.savefig(FIGURES_DIR / "01_elbow_and_silhouette_plot.png", dpi=300)
plt.close(fig)
print("-> Saved 01_elbow_and_silhouette_plot.png")

# ==========================================
# 2. 2D Cluster Scatter Plot with Game Labels
# ==========================================
fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

palette = ["#2b83ba", "#fdae61", "#d7191c", "#abdda4"]
scatter = sns.scatterplot(
    data=games_df,
    x="rating_percent",
    y="playtime_hours",
    hue="cluster_label",
    size="popularity",
    sizes=(40, 400),
    palette="tab10",
    alpha=0.85,
    edgecolor="black",
    linewidth=0.5,
    ax=ax,
)

# Annotate prominent representative games
notable_games = [
    "The Elder Scrolls V: Skyrim",
    "Grand Theft Auto V",
    "Among Us",
    "Portal 2",
    "Terraria",
    "Artifact",
    "NBA 2K18",
    "South Park: The Stick of Truth",
    "A Hat in Time",
    "PUBG: BATTLEGROUNDS",
]

for _, row in games_df.iterrows():
    if any(name.lower() in row["game_name"].lower() for name in ["skyrim", "grand theft auto v", "among us", "portal 2", "terraria", "artifact", "nba 2k18", "hat in time"]):
        ax.annotate(
            row["game_name"],
            xy=(row["rating_percent"], row["playtime_hours"]),
            xytext=(row["rating_percent"] - 2, row["playtime_hours"] + 15),
            fontsize=8.5,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="black", lw=0.8),
        )

ax.set_title("2D Video Game Clusters: Rating (%) vs. Average Playtime (Hours)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Rating Reception (% Positive Reviews)", fontsize=12, fontweight="bold")
ax.set_ylabel("Average Playtime (Hours)", fontsize=12, fontweight="bold")
ax.set_yscale("log")
ax.set_ylabel("Average Playtime (Hours, Log Scale)", fontsize=12, fontweight="bold")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

fig.savefig(FIGURES_DIR / "02_clusters_2d_scatter_plot.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("-> Saved 02_clusters_2d_scatter_plot.png")

# ==========================================
# 3. Cluster Feature Averages Comparison Bar Chart
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)

metrics = [
    ("avg_price", "Average Price ($)", "Blues_d", axes[0, 0]),
    ("avg_rating_percent", "Average Rating Reception (%)", "Greens_d", axes[0, 1]),
    ("avg_popularity_reviews", "Average Popularity (Reviews)", "Oranges_d", axes[1, 0]),
    ("avg_playtime_hours", "Average Playtime (Hours)", "Purples_d", axes[1, 1]),
]

short_labels = [
    "C0:\nBudget & F2P",
    "C1:\nMid-Tier Indie",
    "C2:\nPoorly-Rated",
    "C3:\nAAA Blockbusters",
]

for col, title, pal, ax in metrics:
    sns.barplot(x=short_labels, y=cluster_stats_df[col], hue=short_labels, palette=pal, legend=False, ax=ax, edgecolor="black")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    for p in ax.patches:
        height = p.get_height()
        if height > 1000:
            txt = f"{height:,.0f}"
        elif height > 10:
            txt = f"{height:.1f}"
        else:
            txt = f"${height:.2f}" if "Price" in title else f"{height:.1f}"
        ax.annotate(txt, (p.get_x() + p.get_width() / 2.0, height),
                    ha="center", va="bottom", fontsize=10, fontweight="bold", xytext=(0, 3),
                    textcoords="offset points")

plt.suptitle("Cluster Feature Profiles: Price, Rating, Popularity, and Playtime", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "03_cluster_feature_averages_barchart.png", dpi=300)
plt.close(fig)
print("-> Saved 03_cluster_feature_averages_barchart.png")

# ==========================================
# 4. Cluster Distribution Pie Chart
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
colors = sns.color_palette("Set2", 4)
wedges, texts, autotexts = ax.pie(
    cluster_sizes_df["game_count"],
    labels=[f"{cluster_names[i]}\n({count} games)" for i, count in zip(cluster_sizes_df["cluster_id"], cluster_sizes_df["game_count"])],
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    textprops=dict(fontweight="bold"),
    wedgeprops=dict(edgecolor="black", linewidth=1),
)
for autotext in autotexts:
    autotext.set_color("black")
    autotext.set_fontsize(11)

ax.set_title("Cluster Size Distribution (Total Games = 318)", fontsize=14, fontweight="bold", pad=15)
fig.savefig(FIGURES_DIR / "04_cluster_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("-> Saved 04_cluster_distribution.png")

print(f"\nALL 4 HIGH-RESOLUTION PLOTS GENERATED IN: {FIGURES_DIR}")
