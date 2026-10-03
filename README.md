# Video Game Clustering Analysis using PySpark (K-Means)

[![Live Dashboard](https://img.shields.io/badge/🌐_Live_Frontend_Dashboard-Open_Online_App-1999e3?style=for-the-badge&logo=google-chrome&logoColor=white)](https://therealakilan220.github.io/Video-Games-Analysis/)
[![PySpark](https://img.shields.io/badge/PySpark-4.2.0-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![GitHub Pages](https://img.shields.io/badge/GitHub_Pages-Active-success?style=for-the-badge&logo=github)](https://therealakilan220.github.io/Video-Games-Analysis/)

**Big Data Engineering & Machine Learning Project**  
- **Akilan**: Data Engineering + Feature Engineering + Model Training & Evaluation (Heavy Compute Lead)  
- **Arvind**: Visualization + Cluster Interpretation + Inference + Limitations (Light Compute Lead)

---

## 🌐 Live Interactive Frontend Dashboard
**Direct URL:** 👉 **[https://therealakilan220.github.io/Video-Games-Analysis/](https://therealakilan220.github.io/Video-Games-Analysis/)**

The repository features an online frontend dashboard featuring:
- 📊 **Executive Overview & KPI Cards:** Real-time metrics across 40.8M reviews and 318 games.
- 🌌 **2D & 3D Interactive Feature Space:** WebGL 3D scatter plot (Plotly) + customizable 2D bi-variable scatter explorer.
- 🎮 **Searchable Game Catalog:** Filter, search, and sort 318 games by price, rating %, popularity, and playtime with instant CSV export.
- 🧪 **Live Cluster Predictor (Simulator):** Test custom game parameters with dynamic `StandardScaler` centroid distance calculations.
- 📑 **Arvind's Inference & Presentation Script:** Key insights, limitations, and full slide script.
- ⚙️ **PySpark Pipeline Architecture:** End-to-end distributed data engineering diagrams.


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

| Cluster ID | Game Count | Avg Price ($) | Median Price ($) | Avg Rating (%) | Median Rating (%) | Avg Reviews (Pop) | Median Reviews | Avg Playtime (h) | Median Playtime (h) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | 56 | $3.12 | $0.00 | 95.63% | 97.16% | 43,873 | 29,982 | 36.91 h | 21.63 h |
| **1** | 117 | $22.10 | $18.99 | 86.41% | 89.76% | 6,639 | 4,570 | 53.95 h | 38.48 h |
| **2** | 30 | $7.23 | $0.00 | 37.15% | 38.06% | 4,399 | 2,860 | 59.31 h | 21.60 h |
| **3** | 115 | $24.96 | $22.99 | 89.69% | 91.82% | 116,564 | 53,546 | 224.23 h | 156.36 h |

---

## 7. Exported Handoff Deliverables for Arvind
Located in [`exports/`](./exports/):
- [`exports/elbow_cost.csv`](./exports/elbow_cost.csv): $K$ vs WSSSE cost and Silhouette score.
- [`exports/cluster_sizes.csv`](./exports/cluster_sizes.csv): Game count per cluster.
- [`exports/cluster_statistics.csv`](./exports/cluster_statistics.csv): Mean and median for all 4 features per cluster.
- [`exports/clustered_games.csv`](./exports/clustered_games.csv): Full game table with features and assigned `cluster_id`.
- [`exports/plot_sample.csv`](./exports/plot_sample.csv): Ready-to-plot sample for 2D scatter plots.
- [`exports/metrics_summary.txt`](./exports/metrics_summary.txt): Full text summary for the presentation.

---

## 8. Reproducibility
To run the full PySpark pipeline from terminal:
```bash
python src/pipeline.py
```
Or open and execute the Jupyter Notebook:
```bash
jupyter notebook notebooks/VideoGame_Clustering_Akilan_Pipeline.ipynb
```
