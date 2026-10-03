# Video Game Clustering Analysis using PySpark (K-Means)

[![PySpark](https://img.shields.io/badge/PySpark-4.2.0-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()

**Big Data Engineering & Machine Learning Project**  
Unsupervised clustering of video games across Price, Player Ratings, Popularity, and Playtime Engagement using Apache PySpark and distributed K-Means ML.

---

## 🖥️ Interactive Web Dashboard & Frontend

The repository includes a web frontend dashboard for interactive data exploration, 3D/2D feature spaces, and real-time cluster inference.

### How to Access the Dashboard:

#### 1. Quick Local Launch (1-Click)
- **Windows:** Double-click `launch_dashboard.bat` or open `index.html` directly in your browser.
- **Terminal (Cross-Platform):**
  ```bash
  python serve_dashboard.py
  ```
  This immediately launches the local server and opens `http://localhost:8000/index.html`.

#### 2. Free Cloud Hosting (GitHub Pages / Vercel / Netlify)
- **GitHub Pages:** Go to **Repo Settings** ➔ **Pages** ➔ Set Source to **GitHub Actions** (or Deploy from `main` branch `/docs` folder).
- **Vercel / Netlify:** Import the repository &mdash; zero build configuration required (static web app).

---

## 1. Problem Statement
Group video games into meaningful clusters based on **Price**, **Rating Reception**, **Popularity**, and **Playtime** using unsupervised learning (**K-Means Clustering** via Spark MLlib).

---

## 2. Big Data Dataset & Ingestion
- **Raw Reviews Dataset:** Steam Reviews Dataset 2021 ([najzeko/steam-reviews-2021](https://www.kaggle.com/datasets/najzeko/steam-reviews-2021))
  - **Raw Size:** ~5.32 GB archive / 8.17 GB uncompressed
  - **Total Raw Review Records:** 40,848,659 reviews
- **Metadata Dataset:** Steam Store Games Clean Dataset ([nikdavis/steam-store-games](https://www.kaggle.com/datasets/nikdavis/steam-store-games))
  - **Total Games Metadata:** 27,075 games
- **Processing Engine:** Apache PySpark 4.2.0 (distributed DataFrame API & Spark MLlib).

---

## 3. Data Cleaning & Feature Engineering
1. **Filtering & Cleaning:**
   - Handled non-integer and null `app_id`s with `try_cast`.
   - Extracted binary recommendation flag from `recommended` (`voted_up`).
   - Filtered `author.playtime_forever` to range $[0, 600000]$ minutes ($\le 10,000$ hours) to eliminate corrupted unix timestamp anomalies.
2. **Aggregations by `app_id`:**
   - $\text{total\_reviews}$ (count of reviews, used as popularity proxy).
   - $\text{positive\_reviews}$ (count where `recommended == True`).
   - $\text{rating\_percent} = \frac{\text{positive\_reviews}}{\text{total\_reviews}} \times 100$.
   - $\text{playtime\_hours} = \frac{\text{avg\_playtime\_minutes}}{60.0}$ (documented in hours).
3. **Metadata Join:** Joined on `appid` to extract `game_name` and `price`. Free games and missing prices are mapped to \$0.00.
4. **Final ML Table:** 4 clustering features:
   - `price`
   - `rating_percent`
   - `popularity` (total reviews)
   - `playtime_hours`

---

## 4. Preprocessing & Scaling
- **Log Transformation:** Applied `log1p` to `popularity`, `playtime_hours`, and `price` to mitigate high positive skewness across multiple orders of magnitude.
- **StandardScaler:** Standardized all 4 features to mean = 0, standard deviation = 1 using Spark ML `StandardScaler`.

---

## 5. Model Selection & Evaluation
- **Elbow Method:** Evaluated Within-Set Sum of Squared Errors (WSSSE Cost) for $K = 2 \text{ to } 10$.
- **Silhouette Score:** Evaluated using Spark ML `ClusteringEvaluator` (squared Euclidean distance).
- **Optimal K:** Selected **$K = 4$** with peak Silhouette score of **0.4663** and clear elbow cost stabilization (537.02).

### Elbow & Silhouette Summary Table
| K | WSSSE Cost | Silhouette Score |
|---|------------|------------------|
| 2 | 923.85 | 0.3633 |
| 3 | 719.91 | 0.4176 |
| **4** | **537.02** | **0.4663** (Best) |
| 5 | 463.13 | 0.4122 |
| 6 | 413.94 | 0.4227 |
| 7 | 389.76 | 0.4345 |
| 8 | 350.98 | 0.3818 |
| 9 | 309.91 | 0.4038 |
| 10| 288.26 | 0.4128 |

---

## 6. Discovered Cluster Profiles ($K=4$)

| Cluster ID | Semantic Archetype | Game Count | Avg Price ($) | Avg Rating (%) | Avg Reviews (Pop) | Avg Playtime (h) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **0** | **Budget & Free-to-Play Mass Hits** | 56 (17.6%) | $3.12 | 95.63% | 43,873 | 36.91 h |
| **1** | **Mid-Tier & Niche Indie Favorites** | 117 (36.8%) | $22.10 | 86.41% | 6,639 | 53.95 h |
| **2** | **Underperforming / Poorly-Received Titles** | 30 (9.4%) | $7.23 | 37.15% | 4,399 | 59.31 h |
| **3** | **AAA Blockbusters & High-Engagement Time-Sinks** | 115 (36.2%) | $24.96 | 89.69% | 116,564 | 224.23 h |

---

## 7. Analysis & Deliverables
Located in [`exports/`](./exports/):
- [`exports/Cluster_Interpretation_and_Inference_Report.md`](./exports/Cluster_Interpretation_and_Inference_Report.md): Full analysis report, cluster profiles, and strategic takeaways.
- [`exports/elbow_cost.csv`](./exports/elbow_cost.csv): $K$ vs WSSSE cost and Silhouette score.
- [`exports/cluster_sizes.csv`](./exports/cluster_sizes.csv): Game count per cluster.
- [`exports/cluster_statistics.csv`](./exports/cluster_statistics.csv): Mean and median for all 4 features per cluster.
- [`exports/clustered_games.csv`](./exports/clustered_games.csv): Full game table with features and assigned `cluster_id`.
- [`exports/figures/`](./exports/figures/): High-resolution visualization figures.

---

## 8. Reproducibility
To run the full PySpark pipeline from terminal:
```bash
python src/pipeline.py
```
Or open and execute the Jupyter Notebook:
```bash
jupyter notebook notebooks/VideoGame_Clustering_PySpark_Pipeline.ipynb
```
