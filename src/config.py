"""
Configuration and Spark Session Setup for Video Game Clustering Pipeline
Author: Akilan (Data Engineering & ML Lead)
"""

import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession

# Base Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = PROJECT_ROOT / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets downloaded via KaggleHub
USER_HOME = Path.home()
RAW_REVIEWS_CSV = (
    USER_HOME
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "najzeko"
    / "steam-reviews-2021"
    / "versions"
    / "1"
    / "steam_reviews.csv"
)

METADATA_CSV = (
    USER_HOME
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "nikdavis"
    / "steam-store-games"
    / "versions"
    / "3"
    / "steam.csv"
)

# Pipeline Output File Paths
ELBOW_COST_CSV = EXPORTS_DIR / "elbow_cost.csv"
CLUSTER_SIZES_CSV = EXPORTS_DIR / "cluster_sizes.csv"
CLUSTER_STATISTICS_CSV = EXPORTS_DIR / "cluster_statistics.csv"
PLOT_SAMPLE_CSV = EXPORTS_DIR / "plot_sample.csv"
CLUSTERED_GAMES_CSV = EXPORTS_DIR / "clustered_games.csv"
METRICS_SUMMARY_TXT = EXPORTS_DIR / "metrics_summary.txt"


def get_spark_session(app_name="SteamVideoGameClustering"):
    """
    Initializes and returns an optimized local SparkSession with proper memory allocations
    and Windows environment variables.
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
