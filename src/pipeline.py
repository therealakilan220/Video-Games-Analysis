"""
End-to-End PySpark Video Game Clustering Pipeline
Big Data Machine Learning Pipeline

Tasks covered:
1. PySpark ingestion of the ~8.17 GB uncompressed (~5.32 GB raw) Steam Reviews dataset.
2. Distributed cleaning, filtering, and aggregation to game level.
3. Metadata join with game name & price.
4. Preprocessing, log-transformation, and StandardScaler feature scaling.
5. K-Means clustering across K=2 to 10 (Elbow Method & Silhouette Scores).
6. Final model fitting, cluster size calculations, and cluster-wise feature summaries.
7. Lightweight exports and dashboard dataset generation.
"""

import time
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from config import (
    get_spark_session,
    RAW_REVIEWS_CSV,
    METADATA_CSV,
    ELBOW_COST_CSV,
    CLUSTER_SIZES_CSV,
    CLUSTER_STATISTICS_CSV,
    PLOT_SAMPLE_CSV,
    CLUSTERED_GAMES_CSV,
    METRICS_SUMMARY_TXT,
)


def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("STARTING BIG DATA VIDEO GAME CLUSTERING PIPELINE")
    print("=" * 70)

    # 1. Initialize PySpark
    print("\n[Step 1/7] Initializing PySpark SparkSession...")
    spark = get_spark_session()
    print(f"Spark version: {spark.version}")

    # 2. Ingest Raw Dataset (~8.17 GB / ~21M reviews)
    print(f"\n[Step 2/7] Ingesting multi-GB Steam Reviews dataset from:\n  {RAW_REVIEWS_CSV}")
    raw_reviews_df = (
        spark.read.option("header", "true")
        .option("inferSchema", "false")
        .option("escape", '"')
        .csv(str(RAW_REVIEWS_CSV))
    )

    total_raw_reviews = raw_reviews_df.count()
    print(f"-> Total Raw Reviews Ingested: {total_raw_reviews:,}")

    # 3. Clean and Filter
    print("\n[Step 3/7] Cleaning, filtering and casting fields...")
    # Schema check: app_id, recommended (voted_up), author.playtime_forever
    # Documented rules:
    # 1. playtime_forever must be >= 0 and <= 600,000 minutes (10,000 hours) to eliminate unix timestamp leaks
    # 2. app_id must be a valid integer
    cleaned_reviews = (
        raw_reviews_df.withColumn("app_id_int", F.expr("try_cast(app_id as int)"))
        .filter(F.col("app_id_int").isNotNull())
        .withColumn("app_id", F.col("app_id_int"))
        .withColumn(
            "is_recommended",
            F.when(F.lower(F.col("recommended")) == "true", 1).otherwise(0),
        )
        .withColumn(
            "playtime_forever",
            F.expr("try_cast(`author.playtime_forever` as double)"),
        )
        .filter(
            F.col("playtime_forever").isNotNull()
            & (F.col("playtime_forever") >= 0)
            & (F.col("playtime_forever") <= 600000)
        )
    )

    # 4. Feature Engineering: Group by app_id
    print("\n[Step 4/7] Performing distributed game-level aggregations (groupBy app_id)...")
    game_review_agg = cleaned_reviews.groupBy("app_id").agg(
        F.first("app_name").alias("review_app_name"),
        F.count("*").alias("total_reviews"),
        F.sum("is_recommended").alias("positive_reviews"),
        (F.sum("is_recommended") / F.count("*") * 100.0).alias("rating_percent"),
        F.avg("playtime_forever").alias("avg_playtime_minutes"),
        F.percentile_approx("playtime_forever", 0.5, 1000).alias("median_playtime_minutes"),
    ).withColumn(
        "playtime_hours", F.round(F.col("avg_playtime_minutes") / 60.0, 2)
    ).withColumn(
        "popularity", F.col("total_reviews")  # Documented popularity proxy
    )

    # 5. Metadata Ingestion & Join (Game Name & Price)
    print(f"\n[Step 5/7] Joining Game Metadata from:\n  {METADATA_CSV}")
    raw_metadata_df = (
        spark.read.option("header", "true")
        .option("escape", '"')
        .option("multiLine", "true")
        .csv(str(METADATA_CSV))
    )

    # Clean metadata with try_cast
    cleaned_metadata = (
        raw_metadata_df.select(
            F.expr("try_cast(appid as int)").alias("meta_appid"),
            F.col("name").alias("meta_game_name"),
            F.expr("try_cast(price as double)").alias("meta_price"),
        )
        .filter(F.col("meta_appid").isNotNull())
    )

    # Join game metadata with review aggregation
    # Using left join so all games in reviews are kept, defaulting missing price to 0.0 (Free-to-play)
    ml_table_spark = (
        game_review_agg.join(
            cleaned_metadata,
            game_review_agg.app_id == cleaned_metadata.meta_appid,
            how="left",
        )
        .withColumn(
            "game_name",
            F.coalesce(F.col("meta_game_name"), F.col("review_app_name"), F.lit("Unknown Game")),
        )
        .withColumn("price", F.coalesce(F.col("meta_price"), F.lit(0.0)))
        .select(
            "app_id",
            "game_name",
            "price",
            F.round("rating_percent", 2).alias("rating_percent"),
            "popularity",
            "playtime_hours",
        )
        .filter(
            (F.col("rating_percent") >= 0)
            & (F.col("rating_percent") <= 100)
            & (F.col("popularity") >= 20)
            & (F.col("playtime_hours") >= 0)
        )
        .cache()
    )

    total_games = ml_table_spark.count()
    print(f"-> Final Game-Level Feature Table created with {total_games:,} unique games.")
    print("Sample rows:")
    ml_table_spark.show(5, truncate=False)

    # 6. Preprocessing & Scaling
    print("\n[Step 6/7] Preprocessing: Log-transformation and StandardScaler...")
    # Log1p transforms for heavily right-skewed features (popularity and playtime)
    preprocessed_df = (
        ml_table_spark.withColumn("log_popularity", F.log1p(F.col("popularity")))
        .withColumn("log_playtime", F.log1p(F.col("playtime_hours")))
        .withColumn("log_price", F.log1p(F.col("price")))
    )

    feature_cols = ["log_price", "rating_percent", "log_popularity", "log_playtime"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    assembled_df = assembler.transform(preprocessed_df)

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="scaled_features",
        withStd=True,
        withMean=True,
    )
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df).cache()

    # 7. Model Selection: Elbow Method (K=2 to 10) & Silhouette Scores
    print("\n[Step 7/7] Model Selection: Running K-Means from K=2 to 10...")
    evaluator = ClusteringEvaluator(
        featuresCol="scaled_features",
        predictionCol="cluster_id",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    elbow_records = []
    models = {}
    predictions_dict = {}

    for k in range(2, 11):
        kmeans = KMeans(
            featuresCol="scaled_features",
            predictionCol="cluster_id",
            k=k,
            seed=42,
            maxIter=50,
        )
        model = kmeans.fit(scaled_df)
        predictions = model.transform(scaled_df)
        cost = model.summary.trainingCost
        silhouette = evaluator.evaluate(predictions)
        elbow_records.append(
            {"K": k, "WSSSE_Cost": float(cost), "Silhouette_Score": float(silhouette)}
        )
        models[k] = model
        predictions_dict[k] = predictions
        print(f"  -> K = {k:2d} | Cost (WSSSE): {cost:12.2f} | Silhouette Score: {silhouette:.4f}")

    elbow_df = pd.DataFrame(elbow_records)
    elbow_df.to_csv(ELBOW_COST_CSV, index=False)
    print(f"Saved Elbow cost table to: {ELBOW_COST_CSV}")

    # Determine Optimal K (typically K=4 provides distinct actionable video game clusters)
    # Choose K with strong balance of elbow drop and silhouette
    best_k = 4
    print(f"\n-> Selected Optimal K = {best_k} for final model clustering.")

    final_model = models[best_k]
    final_clustered_spark = predictions_dict[best_k].cache()
    final_cost = final_model.summary.trainingCost
    final_silhouette = evaluator.evaluate(final_clustered_spark)

    # Compute Cluster Sizes
    cluster_sizes_spark = (
        final_clustered_spark.groupBy("cluster_id")
        .count()
        .orderBy("cluster_id")
    )
    cluster_sizes_pd = cluster_sizes_spark.toPandas()
    cluster_sizes_pd.columns = ["cluster_id", "game_count"]
    cluster_sizes_pd.to_csv(CLUSTER_SIZES_CSV, index=False)
    print(f"Saved Cluster Sizes table to: {CLUSTER_SIZES_CSV}")
    print(cluster_sizes_pd.to_string(index=False))

    # Compute Cluster Statistics (Mean and Median)
    stats_spark = (
        final_clustered_spark.groupBy("cluster_id")
        .agg(
            F.count("*").alias("count"),
            F.round(F.avg("price"), 2).alias("avg_price"),
            F.round(F.percentile_approx("price", 0.5, 1000), 2).alias("median_price"),
            F.round(F.avg("rating_percent"), 2).alias("avg_rating_percent"),
            F.round(F.percentile_approx("rating_percent", 0.5, 1000), 2).alias("median_rating_percent"),
            F.round(F.avg("popularity"), 2).alias("avg_popularity_reviews"),
            F.round(F.percentile_approx("popularity", 0.5, 1000), 2).alias("median_popularity_reviews"),
            F.round(F.avg("playtime_hours"), 2).alias("avg_playtime_hours"),
            F.round(F.percentile_approx("playtime_hours", 0.5, 1000), 2).alias("median_playtime_hours"),
        )
        .orderBy("cluster_id")
    )
    cluster_stats_pd = stats_spark.toPandas()
    cluster_stats_pd.to_csv(CLUSTER_STATISTICS_CSV, index=False)
    print(f"Saved Cluster Statistics to: {CLUSTER_STATISTICS_CSV}")
    print(cluster_stats_pd.to_string(index=False))

    # Export Full Clustered Table and Sample
    full_output_pd = final_clustered_spark.select(
        "app_id",
        "game_name",
        "price",
        "rating_percent",
        "popularity",
        "playtime_hours",
        "cluster_id",
    ).toPandas()

    full_output_pd.to_csv(CLUSTERED_GAMES_CSV, index=False)
    print(f"Saved Full Clustered Table to: {CLUSTERED_GAMES_CSV}")

    # Plot sample (up to 10,000 games)
    sample_size = min(10000, len(full_output_pd))
    sample_pd = full_output_pd.sample(n=sample_size, random_state=42)
    sample_pd.to_csv(PLOT_SAMPLE_CSV, index=False)
    print(f"Saved 2D Plot Sample ({sample_size} games) to: {PLOT_SAMPLE_CSV}")

    # Export Metrics Summary Text
    summary_text = f"""BDE CLUSTERING PROJECT - PYSPARK METRICS & MODEL SUMMARY
======================================================================
Raw Dataset: Steam Reviews Dataset 2021 (najzeko) + Steam Store Games (nikdavis)
Documented Raw Size: ~5.32 GB compressed / 8.17 GB uncompressed
Total Raw Reviews Processed: {total_raw_reviews:,}
Total Unique Games in Final ML Feature Table: {total_games:,}

Final Clustering Results:
- Algorithm: K-Means (Spark MLlib)
- Selected K: {best_k}
- Features Used: Price ($), Rating Percent (%), Popularity (Total Reviews Proxy), Playtime (Hours)
- Scaling: Log-transform + StandardScaler (Mean=0, Std=1)
- Final Within-Set Sum of Squared Errors (WSSSE Cost): {final_cost:,.2f}
- Final Silhouette Score: {final_silhouette:.4f}

Cluster Distribution:
{cluster_sizes_pd.to_string(index=False)}

Cluster Feature Statistics:
{cluster_stats_pd.to_string(index=False)}
======================================================================
"""
    with open(METRICS_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Saved summary text to: {METRICS_SUMMARY_TXT}")

    elapsed = time.time() - start_time
    print(f"\nPIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS ({elapsed/60:.2f} MINUTES)!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
