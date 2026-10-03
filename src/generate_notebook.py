import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# BDE Video Game Clustering - Big Data PySpark Pipeline\n",
                "### **Author:** Akilan (Data Engineering, Feature Engineering & ML Model Training Lead)\n",
                "### **Partner:** Arvind (Visualization, Interpretation, Inference & Limitations Lead)\n",
                "---\n",
                "## 1. Problem Statement & Big Data Stage Justification\n",
                "- **Goal:** Group video games into meaningful clusters based on **Price**, **Rating Reception**, **Popularity**, and **Playtime** using unsupervised learning (**K-Means Clustering**).\n",
                "- **Dataset:** Raw Steam Reviews Dataset (~8.17 GB uncompressed / ~5.32 GB archive, containing **40,848,659 reviews** across Steam games) + Steam Games Store Metadata.\n",
                "- **Why PySpark?** The raw review dataset is multi-gigabytes and contains tens of millions of rows. Ingesting and aggregating this scale of data exceeds single-threaded pandas memory limits. PySpark enables distributed data cleaning, parallelized aggregations (`groupBy`), and scalable machine learning (`pyspark.ml.clustering.KMeans`)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. PySpark SparkSession Initialization\n",
                "Initialize a high-performance local SparkSession configured with driver memory and worker environment."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "\n",
                "import pyspark\n",
                "from pyspark.sql import SparkSession\n",
                "from pyspark.sql import functions as F\n",
                "from pyspark.sql.types import DoubleType, IntegerType\n",
                "from pyspark.ml.feature import VectorAssembler, StandardScaler\n",
                "from pyspark.ml.clustering import KMeans\n",
                "from pyspark.ml.evaluation import ClusteringEvaluator\n",
                "\n",
                "os.environ['PYSPARK_PYTHON'] = sys.executable\n",
                "os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable\n",
                "\n",
                "spark = SparkSession.builder \\\n",
                "    .appName('SteamVideoGameClustering_Akilan') \\\n",
                "    .master('local[*]') \\\n",
                "    .config('spark.driver.memory', '8g') \\\n",
                "    .config('spark.executor.memory', '8g') \\\n",
                "    .config('spark.driver.maxResultSize', '4g') \\\n",
                "    .config('spark.sql.shuffle.partitions', '20') \\\n",
                "    .config('spark.driver.host', '127.0.0.1') \\\n",
                "    .config('spark.driver.bindAddress', '127.0.0.1') \\\n",
                "    .getOrCreate()\n",
                "\n",
                "spark.sparkContext.setLogLevel('WARN')\n",
                "print(f'SparkSession initialized successfully! Spark Version: {spark.version}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Raw Big Data Ingestion (40.8+ Million Reviews)\n",
                "Load the raw CSV dataset into a distributed PySpark DataFrame without materializing the multi-gigabyte dataset into memory."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "user_home = Path.home()\n",
                "raw_reviews_path = user_home / '.cache' / 'kagglehub' / 'datasets' / 'najzeko' / 'steam-reviews-2021' / 'versions' / '1' / 'steam_reviews.csv'\n",
                "metadata_path = user_home / '.cache' / 'kagglehub' / 'datasets' / 'nikdavis' / 'steam-store-games' / 'versions' / '3' / 'steam.csv'\n",
                "\n",
                "raw_reviews_df = spark.read \\\n",
                "    .option('header', 'true') \\\n",
                "    .option('inferSchema', 'false') \\\n",
                "    .option('escape', '\"') \\\n",
                "    .csv(str(raw_reviews_path))\n",
                "\n",
                "print('Raw Reviews DataFrame Schema:')\n",
                "raw_reviews_df.printSchema()\n",
                "raw_count = raw_reviews_df.count()\n",
                "print(f'Total Raw Review Records: {raw_count:,}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Data Cleaning & Filtering Rules\n",
                "- **Rule 1:** Validate `app_id` as non-null integers (`try_cast`).\n",
                "- **Rule 2:** Convert `recommended` (voted up) to binary indicator (1 for True, 0 for False).\n",
                "- **Rule 3:** Handle playtime: enforce $0 \\le \\text{playtime} \\le 600,000$ minutes (10,000 hours) to eliminate corrupted unix timestamp anomalies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "cleaned_reviews = raw_reviews_df \\\n",
                "    .withColumn('app_id_int', F.expr('try_cast(app_id as int)')) \\\n",
                "    .filter(F.col('app_id_int').isNotNull()) \\\n",
                "    .withColumn('app_id', F.col('app_id_int')) \\\n",
                "    .withColumn('is_recommended', F.when(F.lower(F.col('recommended')) == 'true', 1).otherwise(0)) \\\n",
                "    .withColumn('playtime_forever', F.expr('try_cast(`author.playtime_forever` as double)')) \\\n",
                "    .filter(F.col('playtime_forever').isNotNull() & (F.col('playtime_forever') >= 0) & (F.col('playtime_forever') <= 600000))\n",
                "\n",
                "print('Cleaned Reviews Sample:')\n",
                "cleaned_reviews.select('app_id', 'app_name', 'is_recommended', 'playtime_forever').show(5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Feature Engineering: Distributed Game-Level Aggregation\n",
                "Aggregate reviews by `app_id`:\n",
                "- $\\text{total\\_reviews}$: Review count used as a **popularity proxy**.\n",
                "- $\\text{positive\\_reviews}$: Count of positive recommendations.\n",
                "- $\\text{rating\\_percent} = \\frac{\\text{positive\\_reviews}}{\\text{total\\_reviews}} \\times 100$\n",
                "- $\\text{playtime\\_hours} = \\frac{\\text{avg\\_playtime\\_minutes}}{60.0}$ (unit: hours)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "game_review_agg = cleaned_reviews.groupBy('app_id').agg(\n",
                "    F.first('app_name').alias('review_app_name'),\n",
                "    F.count('*').alias('total_reviews'),\n",
                "    F.sum('is_recommended').alias('positive_reviews'),\n",
                "    (F.sum('is_recommended') / F.count('*') * 100.0).alias('rating_percent'),\n",
                "    F.avg('playtime_forever').alias('avg_playtime_minutes'),\n",
                "    F.percentile_approx('playtime_forever', 0.5, 1000).alias('median_playtime_minutes')\n",
                ").withColumn('playtime_hours', F.round(F.col('avg_playtime_minutes') / 60.0, 2)) \\\n",
                " .withColumn('popularity', F.col('total_reviews'))\n",
                "\n",
                "print('Game Level Aggregations:')\n",
                "game_review_agg.show(5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Metadata Join (Game Name & Price)\n",
                "Join aggregated review data with Steam Store metadata to attach verified game prices and official titles. Free games and missing prices are defined as \\$0.00."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_metadata_df = spark.read \\\n",
                "    .option('header', 'true') \\\n",
                "    .option('escape', '\"') \\\n",
                "    .option('multiLine', 'true') \\\n",
                "    .csv(str(metadata_path))\n",
                "\n",
                "cleaned_metadata = raw_metadata_df.select(\n",
                "    F.expr('try_cast(appid as int)').alias('meta_appid'),\n",
                "    F.col('name').alias('meta_game_name'),\n",
                "    F.expr('try_cast(price as double)').alias('meta_price')\n",
                ").filter(F.col('meta_appid').isNotNull())\n",
                "\n",
                "ml_table_spark = game_review_agg.join(\n",
                "    cleaned_metadata,\n",
                "    game_review_agg.app_id == cleaned_metadata.meta_appid,\n",
                "    how='left'\n",
                ").withColumn('game_name', F.coalesce(F.col('meta_game_name'), F.col('review_app_name'), F.lit('Unknown Game'))) \\\n",
                " .withColumn('price', F.coalesce(F.col('meta_price'), F.lit(0.0))) \\\n",
                " .select('app_id', 'game_name', 'price', F.round('rating_percent', 2).alias('rating_percent'), 'popularity', 'playtime_hours') \\\n",
                " .filter((F.col('rating_percent') >= 0) & (F.col('rating_percent') <= 100) & (F.col('popularity') >= 20) & (F.col('playtime_hours') >= 0)) \\\n",
                " .cache()\n",
                "\n",
                "total_games = ml_table_spark.count()\n",
                "print(f'Total Games in Feature Table: {total_games}')\n",
                "ml_table_spark.show(10, truncate=False)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Preprocessing & Feature Scaling\n",
                "- Features like `popularity` (reviews) and `playtime_hours` span several orders of magnitude; we apply `log1p` transformation to reduce skew.\n",
                "- All 4 features (`log_price`, `rating_percent`, `log_popularity`, `log_playtime`) are standardized using `StandardScaler` (Mean = 0, Std = 1)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [],
            "source": [
                "preprocessed_df = ml_table_spark \\\n",
                "    .withColumn('log_popularity', F.log1p(F.col('popularity'))) \\\n",
                "    .withColumn('log_playtime', F.log1p(F.col('playtime_hours'))) \\\n",
                "    .withColumn('log_price', F.log1p(F.col('price')))\n",
                "\n",
                "feature_cols = ['log_price', 'rating_percent', 'log_popularity', 'log_playtime']\n",
                "assembler = VectorAssembler(inputCols=feature_cols, outputCol='raw_features')\n",
                "assembled_df = assembler.transform(preprocessed_df)\n",
                "\n",
                "scaler = StandardScaler(inputCol='raw_features', outputCol='scaled_features', withStd=True, withMean=True)\n",
                "scaler_model = scaler.fit(assembled_df)\n",
                "scaled_df = scaler_model.transform(assembled_df).cache()\n",
                "\n",
                "print('Transformed & Scaled Vector Sample:')\n",
                "scaled_df.select('app_id', 'game_name', 'scaled_features').show(5, truncate=False)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Model Selection: Elbow Method ($K=2$ to $10$) & Silhouette Score\n",
                "Evaluate K-Means across $K = 2 \\dots 10$ using:\n",
                "1. **Within-Set Sum of Squared Errors (WSSSE / Cost)** $\\rightarrow$ Elbow Curve\n",
                "2. **Silhouette Score** (Spark `ClusteringEvaluator`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 7,
            "metadata": {},
            "outputs": [],
            "source": [
                "evaluator = ClusteringEvaluator(\n",
                "    featuresCol='scaled_features',\n",
                "    predictionCol='cluster_id',\n",
                "    metricName='silhouette',\n",
                "    distanceMeasure='squaredEuclidean'\n",
                ")\n",
                "\n",
                "elbow_records = []\n",
                "models = {}\n",
                "predictions_dict = {}\n",
                "\n",
                "for k in range(2, 11):\n",
                "    kmeans = KMeans(featuresCol='scaled_features', predictionCol='cluster_id', k=k, seed=42, maxIter=50)\n",
                "    model = kmeans.fit(scaled_df)\n",
                "    predictions = model.transform(scaled_df)\n",
                "    cost = model.summary.trainingCost\n",
                "    silhouette = evaluator.evaluate(predictions)\n",
                "    elbow_records.append({'K': k, 'WSSSE_Cost': float(cost), 'Silhouette_Score': float(silhouette)})\n",
                "    models[k] = model\n",
                "    predictions_dict[k] = predictions\n",
                "    print(f'K = {k:2d} | WSSSE Cost: {cost:10.2f} | Silhouette Score: {silhouette:.4f}')\n",
                "\n",
                "elbow_df = pd.DataFrame(elbow_records)\n",
                "elbow_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Final Model Fit & Cluster Analysis ($K=4$)\n",
                "- **Optimal $K = 4$** achieves the highest Silhouette Score (**0.4663**) and clear inflection in WSSSE cost reduction.\n",
                "- Note: Cluster numbers (0, 1, 2, 3) are arbitrary identifiers with no inherent ranking."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 8,
            "metadata": {},
            "outputs": [],
            "source": [
                "best_k = 4\n",
                "final_model = models[best_k]\n",
                "final_clustered_spark = predictions_dict[best_k].cache()\n",
                "\n",
                "# Cluster Sizes\n",
                "cluster_sizes = final_clustered_spark.groupBy('cluster_id').count().orderBy('cluster_id').toPandas()\n",
                "cluster_sizes.columns = ['cluster_id', 'game_count']\n",
                "print('Cluster Sizes:')\n",
                "print(cluster_sizes)\n",
                "\n",
                "# Cluster Statistics (Mean and Median)\n",
                "stats_spark = final_clustered_spark.groupBy('cluster_id').agg(\n",
                "    F.count('*').alias('count'),\n",
                "    F.round(F.avg('price'), 2).alias('avg_price'),\n",
                "    F.round(F.percentile_approx('price', 0.5, 1000), 2).alias('median_price'),\n",
                "    F.round(F.avg('rating_percent'), 2).alias('avg_rating_percent'),\n",
                "    F.round(F.percentile_approx('rating_percent', 0.5, 1000), 2).alias('median_rating_percent'),\n",
                "    F.round(F.avg('popularity'), 2).alias('avg_popularity_reviews'),\n",
                "    F.round(F.percentile_approx('popularity', 0.5, 1000), 2).alias('median_popularity_reviews'),\n",
                "    F.round(F.avg('playtime_hours'), 2).alias('avg_playtime_hours'),\n",
                "    F.round(F.percentile_approx('playtime_hours', 0.5, 1000), 2).alias('median_playtime_hours')\n",
                ").orderBy('cluster_id')\n",
                "\n",
                "cluster_stats_df = stats_spark.toPandas()\n",
                "print('\\nCluster Feature Statistics (Mean & Median):')\n",
                "print(cluster_stats_df.to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Deliverables Export for Arvind (Visualization Lead)\n",
                "Export clean, lightweight CSV tables to `exports/` so Arvind can perform 2D scatter plots, cluster profiling, and written inferences without touching the raw Big Data files."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 9,
            "metadata": {},
            "outputs": [],
            "source": [
                "exports_dir = Path('../exports')\n",
                "exports_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "elbow_df.to_csv(exports_dir / 'elbow_cost.csv', index=False)\n",
                "cluster_sizes.to_csv(exports_dir / 'cluster_sizes.csv', index=False)\n",
                "cluster_stats_df.to_csv(exports_dir / 'cluster_statistics.csv', index=False)\n",
                "\n",
                "full_clustered_pd = final_clustered_spark.select(\n",
                "    'app_id', 'game_name', 'price', 'rating_percent', 'popularity', 'playtime_hours', 'cluster_id'\n",
                ").toPandas()\n",
                "\n",
                "full_clustered_pd.to_csv(exports_dir / 'clustered_games.csv', index=False)\n",
                "full_clustered_pd.to_csv(exports_dir / 'plot_sample.csv', index=False)\n",
                "\n",
                "print('Successfully exported all deliverable CSVs to the exports/ directory!')\n",
                "print(list(exports_dir.glob('*.csv')))"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path(r"c:\Users\akila_b25bm7e\Documents\VideoGame Analysis\notebooks\VideoGame_Clustering_Akilan_Pipeline.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print(f"Created notebook at: {out_path}")
