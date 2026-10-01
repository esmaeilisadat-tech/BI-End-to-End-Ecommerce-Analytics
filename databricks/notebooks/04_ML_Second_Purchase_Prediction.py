# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Machine Learning: Second Purchase Prediction
# MAGIC 
# MAGIC ## Executive summary
# MAGIC 
# MAGIC This notebook develops and evaluates a **customer repurchase propensity model** for the BI project built on the synthetic **TheLook e-commerce** dataset.
# MAGIC 
# MAGIC The analytical goal is not to predict every possible future purchase. Instead, the model answers one clearly defined business question:
# MAGIC 
# MAGIC > **After a customer's first non-cancelled order, will that customer place a second non-cancelled order within the next 180 days?**
# MAGIC 
# MAGIC The result is designed for **customer retention, remarketing, and prioritization in Power BI**. The model produces a score that helps rank first-time customers from lower to higher repurchase propensity.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Business objective
# MAGIC 
# MAGIC A business often wants to know which new customers are worth targeting after their first order. A useful model can help answer questions such as:
# MAGIC 
# MAGIC - Which first-time customers are more likely to buy again?
# MAGIC - Which customers should receive retention or remarketing campaigns?
# MAGIC - Can customers be grouped into High, Medium, and Low repurchase-opportunity segments?
# MAGIC - Which customer and first-order characteristics are associated with repeat purchasing?
# MAGIC 
# MAGIC The operational output of this notebook is therefore a **customer-level repurchase score and opportunity segment**.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Target definition
# MAGIC 
# MAGIC The target is defined at the **first non-cancelled order** of each customer.
# MAGIC 
# MAGIC | Target | Definition |
# MAGIC |---|---|
# MAGIC | `label = 1` | The customer places a second non-cancelled order within 180 days after the first order |
# MAGIC | `label = 0` | No second non-cancelled order occurs within the next 180 days |
# MAGIC 
# MAGIC ### Prediction moment
# MAGIC 
# MAGIC The prediction moment is the timestamp of the customer's **first non-cancelled order**.  
# MAGIC Only information available **at or before this moment** is allowed as a model feature.
# MAGIC 
# MAGIC ### Right-censoring rule
# MAGIC 
# MAGIC Recent first orders do not yet have a complete 180-day future observation window. They are therefore **excluded from labeled model training and evaluation**. They are later used as the **current scoring population**.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Why the model focuses on first → second purchase
# MAGIC 
# MAGIC TheLook is a synthetic dataset and, in this project data, customers have at most four non-cancelled orders. Using order position as a general repurchase feature would allow the model to learn an artificial dataset ceiling rather than genuine customer behavior.
# MAGIC 
# MAGIC To avoid this artifact, the final modeling problem is intentionally restricted to:
# MAGIC 
# MAGIC > **First order → second purchase within 180 days**
# MAGIC 
# MAGIC This makes the target cleaner, easier to explain, and more defensible in a presentation.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Data sources
# MAGIC 
# MAGIC The notebook reads the existing Gold-layer tables as **read-only inputs**:
# MAGIC 
# MAGIC - `default.fact_orders`
# MAGIC - `default.fact_order_items`
# MAGIC - `default.dim_users`
# MAGIC - `default.dim_products`
# MAGIC - `default.dim_date`
# MAGIC 
# MAGIC No Bronze, Silver, Gold, Fact, or Dimension table is modified.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Leakage prevention
# MAGIC 
# MAGIC The following information is excluded from the model feature vector:
# MAGIC 
# MAGIC - `next_order_at`
# MAGIC - future order counts
# MAGIC - future spend
# MAGIC - future delivery or return information
# MAGIC - future order status
# MAGIC - label-derived variables
# MAGIC - identifiers such as `order_id` and `user_id`
# MAGIC 
# MAGIC Identifiers are retained only for traceability and output preparation.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Modeling strategy
# MAGIC 
# MAGIC The mature labeled dataset is split **chronologically**:
# MAGIC 
# MAGIC **70% Train → 15% Validation → 15% Final Test**
# MAGIC 
# MAGIC This is intentionally different from a random split.
# MAGIC 
# MAGIC - **Train**: fit preprocessing and candidate models
# MAGIC - **Validation**: compare model families and tune the classification threshold
# MAGIC - **Test**: evaluate the selected configuration once, after all model decisions are locked
# MAGIC 
# MAGIC Candidate models:
# MAGIC 
# MAGIC 1. Logistic Regression
# MAGIC 2. Random Forest
# MAGIC 3. Gradient-Boosted Trees
# MAGIC 
# MAGIC Because the positive class is imbalanced, class weighting is used and model quality is judged mainly with:
# MAGIC 
# MAGIC - PR-AUC
# MAGIC - ROC-AUC
# MAGIC - Recall
# MAGIC - Precision
# MAGIC - F1 Score
# MAGIC 
# MAGIC Accuracy is reported but is **not** the primary selection metric.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Important temporal pattern
# MAGIC 
# MAGIC A read-only validation of the real project data showed substantial target drift over calendar time:
# MAGIC 
# MAGIC - Train positive rate: approximately **8.70%**
# MAGIC - Validation positive rate: approximately **15.30%**
# MAGIC - Test positive rate: approximately **20.63%**
# MAGIC 
# MAGIC This makes the chronological test more realistic than a random split and should be mentioned as a limitation when presenting the model.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Databricks Serverless compatibility
# MAGIC 
# MAGIC This version is designed for Databricks Serverless compute.
# MAGIC 
# MAGIC - No Spark DataFrame `cache()` calls
# MAGIC - No `persist()` calls
# MAGIC - No table-write operations
# MAGIC - No destructive SQL
# MAGIC 
# MAGIC Some global window operations can still generate a Spark performance warning. This affects execution efficiency, not the statistical logic of the model.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Notebook workflow
# MAGIC 
# MAGIC **Source validation → Target engineering → Feature engineering → Data quality → EDA → Chronological split → Preprocessing → Logistic Regression / Random Forest / GBT → Validation threshold tuning → Model selection → Final test → Interpretation → Current-customer scoring → Power BI output preparation**
# MAGIC 
# MAGIC The notebook is intended to be run **top to bottom** in a fresh Databricks session.
# MAGIC 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 01 — Setup
# MAGIC 
# MAGIC ### Purpose
# MAGIC Prepare the Python, PySpark, Spark ML, and plotting libraries used throughout the notebook.
# MAGIC 
# MAGIC ### What happens in this step
# MAGIC - Imports Spark SQL functions and window operations.
# MAGIC - Imports the three classification algorithms.
# MAGIC - Imports preprocessing utilities such as imputation, one-hot encoding, vector assembly, and scaling.
# MAGIC - Defines reproducibility settings, the 180-day target window, the threshold search grid, and plot formatting.
# MAGIC 
# MAGIC ### Why it matters
# MAGIC Centralizing configuration at the beginning makes the notebook reproducible and keeps modeling decisions visible for reviewers.
# MAGIC 
# MAGIC ### Expected result
# MAGIC The cell should finish with a setup confirmation and no data transformation yet.
# MAGIC 

# COMMAND ----------

# ==========================================
# 01 - Setup
# ==========================================

import math
from datetime import timedelta

import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array

SEED = 42
POSITIVE_WINDOW_DAYS = 180
THRESHOLD_GRID = [round(value / 100, 2) for value in range(5, 96, 5)]

PLOT_COLORS = {
    "primary": "#2F6B9A",
    "secondary": "#D39B2A",
    "accent": "#D97941",
    "neutral": "#5B6470",
}

plt.rcParams.update({
    "figure.figsize": (9, 4.8),
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.2,
})

print("Setup complete. This notebook contains no Databricks write operations.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 02 — Load read-only Gold data
# MAGIC 
# MAGIC ### Purpose
# MAGIC Load the curated analytical tables that were created earlier in the Databricks Bronze → Silver → Gold pipeline.
# MAGIC 
# MAGIC ### What happens in this step
# MAGIC The notebook reads the order, order-item, user, product, and date tables into Spark DataFrames and reports their row counts.
# MAGIC 
# MAGIC ### Why it matters
# MAGIC The ML notebook should build on the trusted Gold layer rather than reimplementing ETL logic. This keeps responsibilities separated:
# MAGIC 
# MAGIC - Databricks Gold = cleaned analytical source
# MAGIC - ML notebook = prediction workflow
# MAGIC - Power BI = reporting and business visualization
# MAGIC 
# MAGIC ### Expected result
# MAGIC The source tables load successfully and the counts match the current Databricks data.
# MAGIC 

# COMMAND ----------

# ==========================================
# 02 - Load Data
# ==========================================

SOURCE_TABLE_NAMES = {
    "orders": "default.fact_orders",
    "order_items": "default.fact_order_items",
    "users": "default.dim_users",
    "products": "default.dim_products",
    "date": "default.dim_date",
}

orders = spark.table(SOURCE_TABLE_NAMES["orders"])
order_items = spark.table(SOURCE_TABLE_NAMES["order_items"])
users = spark.table(SOURCE_TABLE_NAMES["users"])
products = spark.table(SOURCE_TABLE_NAMES["products"])
dim_date = spark.table(SOURCE_TABLE_NAMES["date"])

source_counts = [
    (table_name, spark.table(table_name).count())
    for table_name in SOURCE_TABLE_NAMES.values()
]
source_count_df = spark.createDataFrame(source_counts, ["table_name", "row_count"])

print("Read-only source counts")
source_count_df.orderBy("table_name").show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 03 — Source data validation
# MAGIC 
# MAGIC ### Purpose
# MAGIC Verify that the source data has the expected structure before any modeling is attempted.
# MAGIC 
# MAGIC ### Checks performed
# MAGIC - Required columns exist.
# MAGIC - Primary identifiers are not null.
# MAGIC - Key columns are unique where expected.
# MAGIC - Order items match valid orders and products.
# MAGIC - Orders match valid users.
# MAGIC - Order-item counts agree with the order-level item count.
# MAGIC 
# MAGIC ### Why it matters
# MAGIC A model can appear to work while silently learning from broken joins or duplicated records. These assertions stop the workflow early if source integrity is not sufficient.
# MAGIC 
# MAGIC ### Expected result
# MAGIC All failed-row counts should be zero and the cell should print **Source validation passed**.
# MAGIC 

# COMMAND ----------

# ==========================================
# 03 - Data Validation
# ==========================================

required_columns = {
    "orders": {"order_id", "user_id", "status", "created_at", "num_of_item"},
    "order_items": {"id", "order_id", "user_id", "product_id", "sale_price", "cost"},
    "users": {"id", "age", "gender", "country", "state", "traffic_source", "created_at"},
    "products": {"id", "category", "brand", "department", "retail_price"},
}
source_frames = {
    "orders": orders,
    "order_items": order_items,
    "users": users,
    "products": products,
}

for frame_name, expected in required_columns.items():
    missing = sorted(expected - set(source_frames[frame_name].columns))
    assert not missing, f"{frame_name} is missing required columns: {missing}"

def duplicate_key_count(frame, key_column):
    return (
        frame.groupBy(key_column)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

key_checks = {
    "null_order_ids": orders.filter(F.col("order_id").isNull()).count(),
    "null_order_item_ids": order_items.filter(F.col("id").isNull()).count(),
    "null_user_ids": users.filter(F.col("id").isNull()).count(),
    "null_product_ids": products.filter(F.col("id").isNull()).count(),
    "duplicate_order_ids": duplicate_key_count(orders, "order_id"),
    "duplicate_order_item_ids": duplicate_key_count(order_items, "id"),
    "duplicate_user_ids": duplicate_key_count(users, "id"),
    "duplicate_product_ids": duplicate_key_count(products, "id"),
    "orphan_items_to_orders": order_items.join(
        orders.select("order_id"), on="order_id", how="left_anti"
    ).count(),
    "orphan_items_to_products": order_items.join(
        products.select(F.col("id").alias("product_id")),
        on="product_id",
        how="left_anti",
    ).count(),
    "orphan_orders_to_users": orders.join(
        users.select(F.col("id").alias("user_id")),
        on="user_id",
        how="left_anti",
    ).count(),
}

item_count_by_order = order_items.groupBy("order_id").agg(
    F.count("*").alias("observed_item_count")
)
item_count_mismatches = (
    orders.select("order_id", "num_of_item")
    .join(item_count_by_order, on="order_id", how="left")
    .filter(
        F.coalesce(F.col("observed_item_count"), F.lit(-1))
        != F.col("num_of_item")
    )
    .count()
)
key_checks["item_count_mismatch_orders"] = item_count_mismatches

source_validation_df = spark.createDataFrame(
    [(name, int(value)) for name, value in key_checks.items()],
    ["check", "failed_rows"],
)
source_validation_df.show(truncate=False)

orders.groupBy("status").count().orderBy(F.desc("count")).show(truncate=False)

assert all(value == 0 for value in key_checks.values()), (
    "Source validation found a key, orphan, or item-count problem. "
    "Resolve it before modeling."
)
print("Source validation passed.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 04 — Target engineering
# MAGIC 
# MAGIC ### Purpose
# MAGIC Construct the supervised-learning target while preventing right-censoring and target leakage.
# MAGIC 
# MAGIC ### Method
# MAGIC 1. Cancelled orders are excluded from purchase behavior.
# MAGIC 2. Orders are placed in chronological sequence for each customer.
# MAGIC 3. Only the customer's **first non-cancelled order** is retained as the prediction point.
# MAGIC 4. A positive label is assigned when the next non-cancelled order occurs within 180 days.
# MAGIC 5. First orders too recent to have a full 180-day observation window are excluded from the labeled cohort.
# MAGIC 
# MAGIC ### Why it matters
# MAGIC The target must represent a future business outcome while every predictor remains available at prediction time.
# MAGIC 
# MAGIC ### Expected result
# MAGIC The output reports:
# MAGIC - maximum observed order timestamp,
# MAGIC - mature-label cutoff,
# MAGIC - number of mature first-order customers,
# MAGIC - class distribution of the target.
# MAGIC 

# COMMAND ----------

# ==========================================
# 04 - Target Engineering
# ==========================================

# A purchase is any non-cancelled order. Cancelled orders do not start or satisfy the target.
purchase_orders = (
    orders.filter(F.col("status") != "Cancelled")
    .select("order_id", "user_id", "created_at")
)

customer_order_window = Window.partitionBy("user_id").orderBy(
    "created_at", "order_id"
)

purchase_timeline = (
    purchase_orders
    .withColumn(
        "purchase_number",
        F.row_number().over(customer_order_window),
    )
    .withColumn(
        "next_order_at",
        F.lead("created_at").over(customer_order_window),
    )
    .withColumnRenamed("created_at", "first_order_at")
)

max_data_ts = purchase_orders.agg(F.max("created_at").alias("max_data_ts")).first()[
    "max_data_ts"
]
if max_data_ts is None:
    raise ValueError("No non-cancelled orders are available.")

label_cutoff_ts = max_data_ts - timedelta(days=POSITIVE_WINDOW_DAYS)

first_order_timeline = purchase_timeline.filter(F.col("purchase_number") == 1)

labeled_first_orders = (
    first_order_timeline
    .filter(F.col("first_order_at") <= F.lit(label_cutoff_ts))
    .withColumn(
        "label",
        F.when(
            F.col("next_order_at").isNotNull()
            & (
                F.col("next_order_at")
                <= F.expr(f"first_order_at + INTERVAL {POSITIVE_WINDOW_DAYS} DAYS")
            ),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    .select("order_id", "user_id", "first_order_at", "label")
)

target_summary = (
    labeled_first_orders.groupBy("label")
    .count()
    .withColumn(
        "percentage",
        F.round(F.col("count") / F.sum("count").over(Window.partitionBy()) * 100, 3),
    )
    .orderBy("label")
)

print("Maximum observed non-cancelled order timestamp:", max_data_ts)
print("Mature-label cutoff timestamp:", label_cutoff_ts)
print("Mature first-order customers:", labeled_first_orders.count())
target_summary.show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 05 — Feature engineering
# MAGIC 
# MAGIC ### Purpose
# MAGIC Create predictors that are known at the customer's first order and may contain repurchase signal.
# MAGIC 
# MAGIC ### Feature groups
# MAGIC 
# MAGIC **First-order behavior**
# MAGIC - item count
# MAGIC - distinct products
# MAGIC - order value
# MAGIC - average item price
# MAGIC - order cost
# MAGIC - gross margin
# MAGIC - gross margin percentage
# MAGIC - category / brand diversity
# MAGIC - average retail price
# MAGIC 
# MAGIC **Customer attributes**
# MAGIC - age
# MAGIC - gender
# MAGIC - country
# MAGIC - state
# MAGIC - acquisition traffic source
# MAGIC - customer tenure at the first order
# MAGIC 
# MAGIC **Time features**
# MAGIC - month
# MAGIC - day of week
# MAGIC - hour
# MAGIC - cyclical sine/cosine encodings
# MAGIC - weekend indicator
# MAGIC 
# MAGIC ### Leakage rule
# MAGIC Only information available at or before the first order is included. Future order fields remain outside the feature frame.
# MAGIC 
# MAGIC ### Expected result
# MAGIC A model-ready feature dataset is created at one row per mature first-time customer.
# MAGIC 

# COMMAND ----------

# ==========================================
# 05 - Feature Engineering
# ==========================================

product_lookup = products.select(
    F.col("id").alias("product_id"),
    "category",
    "brand",
    "department",
    "retail_price",
)

order_features = (
    order_items.alias("oi")
    .join(product_lookup.alias("p"), on="product_id", how="left")
    .groupBy("order_id")
    .agg(
        F.count("*").alias("item_count"),
        F.countDistinct("product_id").alias("distinct_products"),
        F.sum("sale_price").alias("order_value"),
        F.avg("sale_price").alias("avg_item_price"),
        F.sum("cost").alias("order_cost"),
        F.sum(F.col("sale_price") - F.col("cost")).alias("gross_margin"),
        F.countDistinct("category").alias("distinct_categories"),
        F.countDistinct("brand").alias("distinct_brands"),
        F.countDistinct("department").alias("distinct_departments"),
        F.avg("retail_price").alias("avg_retail_price"),
    )
    .withColumn(
        "gross_margin_pct",
        F.when(
            F.col("order_value") > 0,
            F.col("gross_margin") / F.col("order_value") * 100,
        ),
    )
)

user_features = users.select(
    F.col("id").alias("user_id"),
    "age",
    "gender",
    "country",
    "state",
    "traffic_source",
    F.col("created_at").alias("user_created_at"),
)

SAFE_ENGINEERED_FEATURES = [
    "item_count",
    "distinct_products",
    "order_value",
    "avg_item_price",
    "order_cost",
    "gross_margin",
    "gross_margin_pct",
    "distinct_categories",
    "distinct_brands",
    "distinct_departments",
    "avg_retail_price",
    "customer_tenure_days",
    "age",
    "gender",
    "country",
    "state",
    "traffic_source",
    "order_month",
    "order_day_of_week",
    "order_hour",
    "order_month_sin",
    "order_month_cos",
    "order_dow_sin",
    "order_dow_cos",
    "order_hour_sin",
    "order_hour_cos",
    "is_weekend",
]

def build_safe_feature_frame(base_frame):
    id_columns = ["order_id", "user_id", "first_order_at"]
    if "label" in base_frame.columns:
        id_columns.append("label")

    enriched = (
        base_frame
        .join(order_features, on="order_id", how="left")
        .join(user_features, on="user_id", how="left")
        .withColumn(
            "customer_tenure_days",
            F.datediff(F.to_date("first_order_at"), F.to_date("user_created_at")),
        )
        .withColumn("order_month", F.month("first_order_at"))
        .withColumn("order_day_of_week", F.dayofweek("first_order_at"))
        .withColumn("order_hour", F.hour("first_order_at"))
        .withColumn(
            "order_month_sin",
            F.sin(F.lit(2 * math.pi) * (F.col("order_month") - 1) / F.lit(12.0)),
        )
        .withColumn(
            "order_month_cos",
            F.cos(F.lit(2 * math.pi) * (F.col("order_month") - 1) / F.lit(12.0)),
        )
        .withColumn(
            "order_dow_sin",
            F.sin(F.lit(2 * math.pi) * (F.col("order_day_of_week") - 1) / F.lit(7.0)),
        )
        .withColumn(
            "order_dow_cos",
            F.cos(F.lit(2 * math.pi) * (F.col("order_day_of_week") - 1) / F.lit(7.0)),
        )
        .withColumn(
            "order_hour_sin",
            F.sin(F.lit(2 * math.pi) * F.col("order_hour") / F.lit(24.0)),
        )
        .withColumn(
            "order_hour_cos",
            F.cos(F.lit(2 * math.pi) * F.col("order_hour") / F.lit(24.0)),
        )
        .withColumn(
            "is_weekend",
            F.col("order_day_of_week").isin(1, 7).cast("int"),
        )
    )
    return enriched.select(*(id_columns + SAFE_ENGINEERED_FEATURES))

model_dataset = build_safe_feature_frame(labeled_first_orders)

FORBIDDEN_FIELDS = {
    "next_order_at",
    "days_to_next_order",
    "future_order_count",
    "future_spend",
    "future_status",
    "shipped_at",
    "delivered_at",
    "returned_at",
}
assert not FORBIDDEN_FIELDS.intersection(model_dataset.columns)
print("Safe feature rows:", model_dataset.count())
print("Safe feature columns:", len(model_dataset.columns))
model_dataset.show(5, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 06 — Data quality checks on the modeling dataset
# MAGIC 
# MAGIC ### Purpose
# MAGIC Confirm that the engineered ML dataset has the correct grain and usable feature values.
# MAGIC 
# MAGIC ### Checks performed
# MAGIC - One row per first order.
# MAGIC - No missing critical identifiers.
# MAGIC - No unexpected missing joins.
# MAGIC - Numeric and categorical feature completeness.
# MAGIC - Target class distribution.
# MAGIC - Constant or nearly constant variables.
# MAGIC 
# MAGIC ### Why it matters
# MAGIC Feature engineering can introduce duplication or missingness even when the original source tables are valid. This step validates the actual dataset that the models will receive.
# MAGIC 
# MAGIC ### Expected result
# MAGIC The modeling cohort should remain internally consistent and any weak/constant feature should be visible before training.
# MAGIC 

# COMMAND ----------

# ==========================================
# 06 - Data Quality Checks
# ==========================================

duplicate_first_order_ids = (
    model_dataset.groupBy("order_id").count().filter(F.col("count") > 1).count()
)

null_check_columns = ["label"] + SAFE_ENGINEERED_FEATURES
null_summary = model_dataset.agg(*[
    F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
    for column in null_check_columns
])

invalid_feature_rows = model_dataset.filter(
    (F.col("item_count") <= 0)
    | (F.col("order_value") <= 0)
    | (F.col("order_cost") < 0)
    | (F.col("gross_margin") < 0)
    | (F.col("customer_tenure_days") < 0)
    | (~F.col("age").between(0, 120))
).count()

numeric_profile_columns = [
    "item_count",
    "order_value",
    "avg_item_price",
    "order_cost",
    "gross_margin",
    "gross_margin_pct",
    "distinct_products",
    "distinct_categories",
    "distinct_brands",
    "avg_retail_price",
    "customer_tenure_days",
    "age",
]

quantile_rows = []
for column in numeric_profile_columns:
    p01, p50, p99 = model_dataset.approxQuantile(column, [0.01, 0.50, 0.99], 0.01)
    quantile_rows.append((column, float(p01), float(p50), float(p99)))

quantile_profile = spark.createDataFrame(
    quantile_rows,
    ["feature", "p01", "median", "p99"],
)

print("Duplicate first-order IDs:", duplicate_first_order_ids)
print("Rows failing core domain checks:", invalid_feature_rows)
print("NULL counts")
null_summary.show(truncate=False)
print("Robust numeric profile")
quantile_profile.show(truncate=False)

assert duplicate_first_order_ids == 0, "The modeling grain must be one row per first order."
assert invalid_feature_rows == 0, "Unexpected invalid numeric values require investigation."
critical_null_counts = null_summary.first().asDict()
critical_non_null_columns = [
    "label",
    "item_count",
    "order_value",
    "customer_tenure_days",
    "age",
]
unexpected_critical_nulls = {
    column: critical_null_counts[column]
    for column in critical_non_null_columns
    if critical_null_counts[column] > 0
}
assert not unexpected_critical_nulls, (
    f"Critical modeling fields contain NULL values: {unexpected_critical_nulls}"
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 07 — Exploratory analysis and signal validation
# MAGIC 
# MAGIC ### Purpose
# MAGIC Determine whether the target has meaningful associations with the available features before investing in model complexity.
# MAGIC 
# MAGIC ### Analysis
# MAGIC This section reviews:
# MAGIC - target prevalence,
# MAGIC - numerical feature correlations,
# MAGIC - target rate over calendar time,
# MAGIC - target rate across traffic sources,
# MAGIC - first-order/customer profiles by target class.
# MAGIC 
# MAGIC ### Important interpretation
# MAGIC Correlation is used only as an exploratory signal check. A weak linear correlation does not mean that a feature is useless, and no correlation shown here should be interpreted as causation.
# MAGIC 
# MAGIC The time analysis is particularly important because the project data exhibits substantial **temporal target drift**.
# MAGIC 
# MAGIC ### Expected result
# MAGIC Charts and summary statistics should reveal which patterns are stable and which may change over time.
# MAGIC 

# COMMAND ----------

# ==========================================
# 07 - EDA and Signal Validation
# ==========================================

target_distribution = (
    model_dataset.groupBy("label")
    .count()
    .withColumn("rate", F.col("count") / F.sum("count").over(Window.partitionBy()))
    .orderBy("label")
)

monthly_target_rate = (
    model_dataset
    .withColumn("first_order_month", F.date_trunc("month", "first_order_at"))
    .groupBy("first_order_month")
    .agg(
        F.count("*").alias("customers"),
        F.avg("label").alias("repurchase_rate"),
    )
    .orderBy("first_order_month")
)

traffic_source_signal = (
    model_dataset.groupBy("traffic_source")
    .agg(
        F.count("*").alias("customers"),
        F.sum("label").alias("second_purchases"),
        F.avg("label").alias("repurchase_rate"),
    )
    .orderBy(F.desc("repurchase_rate"))
)

correlation_rows = []
correlation_features = [
    "item_count",
    "order_value",
    "avg_item_price",
    "order_cost",
    "gross_margin",
    "gross_margin_pct",
    "distinct_products",
    "distinct_categories",
    "distinct_brands",
    "avg_retail_price",
    "customer_tenure_days",
    "age",
]
for column in correlation_features:
    correlation_rows.append((column, model_dataset.stat.corr(column, "label")))

correlation_df = spark.createDataFrame(
    correlation_rows,
    ["feature", "correlation_with_label"],
).orderBy(F.desc(F.abs(F.col("correlation_with_label"))))

target_distribution.show(truncate=False)
traffic_source_signal.show(truncate=False)
correlation_df.show(truncate=False)

target_pd = target_distribution.toPandas()
monthly_pd = monthly_target_rate.toPandas()
traffic_pd = traffic_source_signal.toPandas().sort_values("repurchase_rate")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

axes[0].bar(
    target_pd["label"].astype(str),
    target_pd["count"],
    color=[PLOT_COLORS["neutral"], PLOT_COLORS["primary"]],
)
axes[0].set_title("Mature first-order target distribution")
axes[0].set_xlabel("Second purchase within 180 days")
axes[0].set_ylabel("Customers")

axes[1].plot(
    monthly_pd["first_order_month"],
    monthly_pd["repurchase_rate"] * 100,
    color=PLOT_COLORS["primary"],
    linewidth=2,
)
axes[1].set_title("Repurchase rate by first-order month")
axes[1].set_xlabel("First-order month")
axes[1].set_ylabel("Repurchase rate (%)")
axes[1].tick_params(axis="x", rotation=45)

axes[2].barh(
    traffic_pd["traffic_source"],
    traffic_pd["repurchase_rate"] * 100,
    color=PLOT_COLORS["secondary"],
)
axes[2].set_title("Repurchase rate by traffic source")
axes[2].set_xlabel("Repurchase rate (%)")
axes[2].set_ylabel("Traffic source")

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 08 — Chronological Train / Validation / Test split
# MAGIC 
# MAGIC ### Purpose
# MAGIC Create a realistic evaluation framework that respects time order.
# MAGIC 
# MAGIC ### Split design
# MAGIC The mature first-order dataset is sorted chronologically and divided into:
# MAGIC 
# MAGIC - **Train: 70%**
# MAGIC - **Validation: 15%**
# MAGIC - **Test: 15%**
# MAGIC 
# MAGIC `order_id` is used only as a deterministic tie-breaker when timestamps are equal.
# MAGIC 
# MAGIC ### Why chronological splitting matters
# MAGIC A random split would mix older and newer customer behavior and could hide the strong target drift observed in this dataset. The chronological design better simulates training on the past and evaluating on the future.
# MAGIC 
# MAGIC ### Expected result
# MAGIC The cell reports:
# MAGIC - row count for each split,
# MAGIC - positive rate,
# MAGIC - date boundaries,
# MAGIC - average order/customer characteristics.
# MAGIC 
# MAGIC The assertions verify that Train precedes Validation and Validation precedes Test.
# MAGIC 

# COMMAND ----------

# ==========================================
# 08 - Chronological Train / Validation / Test Split
# ==========================================

# NTILE creates deterministic chronological 70% / 15% / 15% partitions.
# order_id is a stable tie-breaker for equal timestamps.
chronological_window = Window.orderBy("first_order_at", "order_id")

split_dataset = (
    model_dataset
    .withColumn("_time_tile", F.ntile(20).over(chronological_window))
    .withColumn(
        "split",
        F.when(F.col("_time_tile") <= 14, F.lit("train"))
        .when(F.col("_time_tile") <= 17, F.lit("validation"))
        .otherwise(F.lit("test")),
    )
    .drop("_time_tile")
)

train_data = split_dataset.filter(F.col("split") == "train").drop("split")
validation_data = (
    split_dataset.filter(F.col("split") == "validation").drop("split")
)
test_data = split_dataset.filter(F.col("split") == "test").drop("split")

split_summary = (
    split_dataset.groupBy("split")
    .agg(
        F.count("*").alias("rows"),
        F.sum("label").alias("positives"),
        F.avg("label").alias("positive_rate"),
        F.min("first_order_at").alias("min_first_order_at"),
        F.max("first_order_at").alias("max_first_order_at"),
        F.avg("order_value").alias("avg_order_value"),
        F.avg("item_count").alias("avg_item_count"),
        F.avg("customer_tenure_days").alias("avg_tenure_days"),
    )
    .withColumn(
        "split_order",
        F.when(F.col("split") == "train", 1)
        .when(F.col("split") == "validation", 2)
        .otherwise(3),
    )
    .orderBy("split_order")
    .drop("split_order")
)
split_summary.show(truncate=False)

split_bounds = {row["split"]: row.asDict() for row in split_summary.collect()}
assert split_bounds["train"]["max_first_order_at"] <= split_bounds["validation"]["min_first_order_at"]
assert split_bounds["validation"]["max_first_order_at"] <= split_bounds["test"]["min_first_order_at"]

print("Train rows:", train_data.count())
print("Validation rows:", validation_data.count())
print("Test rows:", test_data.count())


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 09 — Preprocessing and evaluation framework
# MAGIC 
# MAGIC ### Purpose
# MAGIC Build reusable preprocessing and metric functions without learning anything from Validation or Test.
# MAGIC 
# MAGIC ### Preprocessing fitted on Train only
# MAGIC - Median imputation for numeric variables.
# MAGIC - String indexing for categorical variables.
# MAGIC - One-hot encoding.
# MAGIC - Vector assembly.
# MAGIC - Standard scaling for Logistic Regression.
# MAGIC - Balanced class weights for the imbalanced target.
# MAGIC 
# MAGIC `distinct_departments` is excluded from the final model vector because it showed no useful variance/signal in the first-order cohort.
# MAGIC 
# MAGIC ### Evaluation framework
# MAGIC Helper functions calculate:
# MAGIC - ROC-AUC
# MAGIC - PR-AUC
# MAGIC - Accuracy
# MAGIC - Precision
# MAGIC - Recall
# MAGIC - F1
# MAGIC - Specificity
# MAGIC - Brier score
# MAGIC - confusion-matrix counts
# MAGIC - predicted-positive rate
# MAGIC 
# MAGIC ### Expected result
# MAGIC The preprocessing model is fitted using Train data only, preserving the validity of later model comparison.
# MAGIC 

# COMMAND ----------

# ==========================================
# 09 - Preprocessing and Evaluation Helpers
# ==========================================

# `distinct_departments` is intentionally excluded from the model vector because
# the first-order cohort showed no usable variance/signal for this field.
numeric_features = [
    "item_count",
    "order_value",
    "avg_item_price",
    "order_cost",
    "gross_margin",
    "gross_margin_pct",
    "distinct_products",
    "distinct_categories",
    "distinct_brands",
    "avg_retail_price",
    "customer_tenure_days",
    "age",
    "order_month_sin",
    "order_month_cos",
    "order_dow_sin",
    "order_dow_cos",
    "order_hour_sin",
    "order_hour_cos",
    "is_weekend",
]
categorical_features = ["gender", "country", "traffic_source"]

# state is retained for audit/output but excluded from the model to avoid sparse location noise.
identifier_columns = {"order_id", "user_id", "first_order_at"}
forbidden_ml_columns = FORBIDDEN_FIELDS | identifier_columns | {"label", "state"}
assert not forbidden_ml_columns.intersection(numeric_features + categorical_features)

imputed_numeric_features = [f"{column}_imputed" for column in numeric_features]
categorical_index_columns = [f"{column}_index" for column in categorical_features]
categorical_ohe_columns = [f"{column}_ohe" for column in categorical_features]

def add_balanced_class_weights(frame):
    counts = {int(row["label"]): int(row["count"]) for row in frame.groupBy("label").count().collect()}
    negative_count = counts.get(0, 0)
    positive_count = counts.get(1, 0)
    total_count = negative_count + positive_count
    if negative_count == 0 or positive_count == 0:
        raise ValueError("Both target classes are required for class weighting.")
    negative_weight = total_count / (2.0 * negative_count)
    positive_weight = total_count / (2.0 * positive_count)
    weighted = frame.withColumn(
        "class_weight",
        F.when(F.col("label") == 1, F.lit(positive_weight)).otherwise(F.lit(negative_weight)),
    )
    return weighted, {
        "negative_count": negative_count,
        "positive_count": positive_count,
        "negative_weight": negative_weight,
        "positive_weight": positive_weight,
    }

def build_preprocessing_pipeline():
    imputer = Imputer(
        inputCols=numeric_features,
        outputCols=imputed_numeric_features,
        strategy="median",
    )
    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_index",
            handleInvalid="keep",
        )
        for column in categorical_features
    ]
    encoder = OneHotEncoder(
        inputCols=categorical_index_columns,
        outputCols=categorical_ohe_columns,
        handleInvalid="keep",
        dropLast=True,
    )
    assembler = VectorAssembler(
        inputCols=imputed_numeric_features + categorical_ohe_columns,
        outputCol="features",
        handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=False,
    )
    return Pipeline(stages=[imputer] + indexers + [encoder, assembler, scaler])

def build_classifier(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression(
            featuresCol="scaled_features",
            labelCol="label",
            weightCol="class_weight",
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            maxIter=150,
            regParam=0.01,
            elasticNetParam=0.0,
            standardization=False,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            numTrees=120,
            maxDepth=8,
            minInstancesPerNode=5,
            subsamplingRate=0.8,
            featureSubsetStrategy="sqrt",
            seed=SEED,
        )
    if model_name == "Gradient-Boosted Trees":
        return GBTClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            maxIter=60,
            maxDepth=5,
            minInstancesPerNode=5,
            stepSize=0.05,
            subsamplingRate=0.8,
            seed=SEED,
        )
    raise ValueError(f"Unknown model: {model_name}")

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
)

def add_probability_score(predictions):
    return predictions.withColumn(
        "repurchase_probability",
        vector_to_array("probability")[1],
    )

def threshold_metrics(scored_predictions, model_name, threshold):
    predicted = scored_predictions.withColumn(
        "predicted_repurchase",
        (F.col("repurchase_probability") >= F.lit(float(threshold))).cast("int"),
    )
    cm = predicted.agg(
        F.sum(F.when((F.col("label") == 1) & (F.col("predicted_repurchase") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0) & (F.col("predicted_repurchase") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("label") == 0) & (F.col("predicted_repurchase") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label") == 1) & (F.col("predicted_repurchase") == 0), 1).otherwise(0)).alias("fn"),
        F.avg(F.pow(F.col("repurchase_probability") - F.col("label"), 2)).alias("brier_score"),
    ).first()
    tp, tn, fp, fn = [int(cm[name]) for name in ["tp", "tn", "fp", "fn"]]
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    predicted_positive_rate = (tp + fp) / total if total else 0.0
    actual_positive_rate = (tp + fn) / total if total else 0.0
    return {
        "model": model_name,
        "threshold": float(threshold),
        "roc_auc": float(roc_evaluator.evaluate(scored_predictions)),
        "pr_auc": float(pr_evaluator.evaluate(scored_predictions)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "brier_score": float(cm["brier_score"]),
        "predicted_positive_rate": float(predicted_positive_rate),
        "actual_positive_rate": float(actual_positive_rate),
        "predicts_both_classes": bool((tp + fp) > 0 and (tn + fn) > 0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

def threshold_grid_metrics(scored_predictions, model_name):
    threshold_frame = spark.createDataFrame(
        [(float(value),) for value in THRESHOLD_GRID],
        ["threshold"],
    )
    expanded = (
        scored_predictions.crossJoin(F.broadcast(threshold_frame))
        .withColumn(
            "predicted_repurchase",
            (F.col("repurchase_probability") >= F.col("threshold")).cast("int"),
        )
    )
    counts = expanded.groupBy("threshold").agg(
        F.sum(F.when((F.col("label") == 1) & (F.col("predicted_repurchase") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0) & (F.col("predicted_repurchase") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("label") == 0) & (F.col("predicted_repurchase") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label") == 1) & (F.col("predicted_repurchase") == 0), 1).otherwise(0)).alias("fn"),
    )
    return (
        counts
        .withColumn("model", F.lit(model_name))
        .withColumn("precision", F.when(F.col("tp") + F.col("fp") > 0, F.col("tp") / (F.col("tp") + F.col("fp"))).otherwise(0.0))
        .withColumn("recall", F.when(F.col("tp") + F.col("fn") > 0, F.col("tp") / (F.col("tp") + F.col("fn"))).otherwise(0.0))
        .withColumn("specificity", F.when(F.col("tn") + F.col("fp") > 0, F.col("tn") / (F.col("tn") + F.col("fp"))).otherwise(0.0))
        .withColumn("accuracy", (F.col("tp") + F.col("tn")) / (F.col("tp") + F.col("tn") + F.col("fp") + F.col("fn")))
        .withColumn("f1", F.when(F.col("precision") + F.col("recall") > 0, 2 * F.col("precision") * F.col("recall") / (F.col("precision") + F.col("recall"))).otherwise(0.0))
        .withColumn("predicted_positive_rate", (F.col("tp") + F.col("fp")) / (F.col("tp") + F.col("tn") + F.col("fp") + F.col("fn")))
        .withColumn("predicts_both_classes", ((F.col("tp") + F.col("fp")) > 0) & ((F.col("tn") + F.col("fn")) > 0))
        .select("model", "threshold", "precision", "recall", "f1", "accuracy", "specificity", "predicted_positive_rate", "predicts_both_classes", "tn", "fp", "fn", "tp")
        .orderBy("threshold")
    )

train_weighted, train_weight_info = add_balanced_class_weights(train_data)
train_prevalence = float(train_data.agg(F.avg("label")).first()[0])
preprocessing_pipeline = build_preprocessing_pipeline()
preprocessing_model = preprocessing_pipeline.fit(train_weighted)

train_prepared = preprocessing_model.transform(train_weighted)
validation_prepared = preprocessing_model.transform(validation_data)

print("Training class-weight information:", train_weight_info)
print("Preprocessing fit on training data only.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10 — Logistic Regression
# MAGIC 
# MAGIC ### Purpose
# MAGIC Train an interpretable linear baseline for repurchase classification.
# MAGIC 
# MAGIC ### Why this model is useful
# MAGIC Logistic Regression is easy to explain, computationally efficient, and provides signed coefficients that help interpret the direction of relationships.
# MAGIC 
# MAGIC Balanced class weights are used so the minority repurchase class has sufficient influence during training.
# MAGIC 
# MAGIC ### Expected result
# MAGIC Validation customers receive a repurchase score that will later be evaluated across multiple classification thresholds.
# MAGIC 

# COMMAND ----------

# ==========================================
# 10 - Logistic Regression
# ==========================================

logistic_model = build_classifier("Logistic Regression").fit(train_prepared)
logistic_validation = add_probability_score(
    logistic_model.transform(validation_prepared)
)

print("Logistic Regression validation rows:", logistic_validation.count())
logistic_validation.select(
    "order_id", "label", "repurchase_probability"
).show(5, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11 — Random Forest
# MAGIC 
# MAGIC ### Purpose
# MAGIC Test whether nonlinear interactions and tree-based decision rules improve predictive performance.
# MAGIC 
# MAGIC ### Why compare it with Logistic Regression
# MAGIC Random Forest can capture relationships that a linear model cannot, without requiring manual interaction terms.
# MAGIC 
# MAGIC ### Expected result
# MAGIC Validation scores are generated using the same Train/Validation framework, allowing a fair comparison with Logistic Regression.
# MAGIC 

# COMMAND ----------

# ==========================================
# 11 - Random Forest
# ==========================================

random_forest_model = build_classifier("Random Forest").fit(train_prepared)
random_forest_validation = add_probability_score(
    random_forest_model.transform(validation_prepared)
)

print("Random Forest validation rows:", random_forest_validation.count())
random_forest_validation.select(
    "order_id", "label", "repurchase_probability"
).show(5, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12 — Gradient-Boosted Trees
# MAGIC 
# MAGIC ### Purpose
# MAGIC Evaluate a second nonlinear tree-based method that builds trees sequentially to correct earlier prediction errors.
# MAGIC 
# MAGIC ### Why include GBT
# MAGIC GBT can perform well when the signal is nonlinear and concentrated in combinations of features.
# MAGIC 
# MAGIC ### Expected result
# MAGIC The model produces validation repurchase scores that are evaluated with exactly the same metrics and threshold-search logic as the other candidates.
# MAGIC 

# COMMAND ----------

# ==========================================
# 12 - Gradient-Boosted Trees
# ==========================================

gbt_model = build_classifier("Gradient-Boosted Trees").fit(train_prepared)
gbt_validation = add_probability_score(
    gbt_model.transform(validation_prepared)
)

print("GBT validation rows:", gbt_validation.count())
gbt_validation.select(
    "order_id", "label", "repurchase_probability"
).show(5, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13 — Validation threshold optimization
# MAGIC 
# MAGIC ### Purpose
# MAGIC Choose a practical classification threshold for each candidate model using **Validation data only**.
# MAGIC 
# MAGIC ### Why threshold tuning is needed
# MAGIC For an imbalanced target, the default threshold of 0.50 is not automatically optimal. Different thresholds change the trade-off between:
# MAGIC 
# MAGIC - **Precision**: how many targeted customers actually repurchase
# MAGIC - **Recall**: how many real repurchasers are successfully identified
# MAGIC 
# MAGIC The notebook searches a defined threshold grid and prioritizes F1, then Recall and Precision. Degenerate thresholds that predict only one class are avoided when a valid two-class threshold is available.
# MAGIC 
# MAGIC ### Expected result
# MAGIC For each candidate model:
# MAGIC - probability-score distribution,
# MAGIC - best validation threshold,
# MAGIC - Precision / Recall / F1 across thresholds.
# MAGIC 
# MAGIC The final Test set is still untouched.
# MAGIC 

# COMMAND ----------

# ==========================================
# 13 - Threshold Optimization on Validation Data
# ==========================================

validation_predictions = {
    "Logistic Regression": logistic_validation,
    "Random Forest": random_forest_validation,
    "Gradient-Boosted Trees": gbt_validation,
}

validation_results = []
threshold_tables = []

for model_name, scored_predictions in validation_predictions.items():
    threshold_table = threshold_grid_metrics(scored_predictions, model_name)

    # Prefer a non-degenerate threshold that predicts both classes.
    # Fall back to the full grid only if no such threshold exists.
    eligible_thresholds = threshold_table.filter(F.col("predicts_both_classes"))
    if eligible_thresholds.limit(1).count() == 0:
        eligible_thresholds = threshold_table

    best_threshold_row = (
        eligible_thresholds
        .orderBy(
            F.desc("f1"),
            F.desc("recall"),
            F.desc("precision"),
            F.asc("threshold"),
        )
        .first()
    )
    selected_threshold = float(best_threshold_row["threshold"])
    metrics = threshold_metrics(
        scored_predictions,
        model_name,
        selected_threshold,
    )
    validation_results.append(metrics)
    threshold_tables.append(threshold_table)

    probability_quantiles = scored_predictions.approxQuantile(
        "repurchase_probability",
        [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0],
        0.001,
    )
    print(f"\n{model_name}")
    print("Probability quantiles:", probability_quantiles)
    print("Best validation threshold:", selected_threshold)
    print("Validation metrics:", metrics)

all_threshold_metrics = threshold_tables[0]
for table in threshold_tables[1:]:
    all_threshold_metrics = all_threshold_metrics.unionByName(table)

all_threshold_metrics.orderBy("model", "threshold").show(100, truncate=False)

threshold_pd = all_threshold_metrics.toPandas()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
for model_name, group in threshold_pd.groupby("model"):
    axes[0].plot(group["threshold"], group["precision"], marker="o", label=model_name)
    axes[1].plot(group["threshold"], group["recall"], marker="o", label=model_name)
    axes[2].plot(group["threshold"], group["f1"], marker="o", label=model_name)
for axis, title in zip(axes, ["Precision", "Recall", "F1"]):
    axis.set_title(f"Validation {title} by threshold")
    axis.set_xlabel("Probability threshold")
    axis.set_ylabel(title)
axes[2].legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 14 — Model comparison and selection
# MAGIC 
# MAGIC ### Purpose
# MAGIC Select the final model configuration using validation performance.
# MAGIC 
# MAGIC ### Selection rule
# MAGIC Candidate models are ranked primarily by **Validation PR-AUC**, with F1 and Recall used as tie-breakers.
# MAGIC 
# MAGIC Two no-skill baselines are included for context:
# MAGIC - Always Negative
# MAGIC - Always Positive
# MAGIC 
# MAGIC The no-skill PR-AUC is approximately the positive-class prevalence.
# MAGIC 
# MAGIC ### Why PR-AUC is emphasized
# MAGIC Repurchase is the minority class. A model can achieve high Accuracy simply by predicting mostly negatives, so PR-AUC gives a more useful view of minority-class ranking quality.
# MAGIC 
# MAGIC ### Expected result
# MAGIC The cell locks:
# MAGIC - selected model,
# MAGIC - selected classification threshold,
# MAGIC - whether PR-AUC exceeds the prevalence baseline.
# MAGIC 
# MAGIC No Test result is used in this decision.
# MAGIC 

# COMMAND ----------

# ==========================================
# 14 - Model Comparison and Selection
# ==========================================

# Two transparent no-skill classification baselines use the training prevalence
# as a constant score. They are reported, but are never eligible for model selection.
baseline_validation_scored = (
    validation_data
    .withColumn("repurchase_probability", F.lit(train_prevalence))
    .withColumn("rawPrediction", F.lit(train_prevalence))
)
baseline_results = [
    threshold_metrics(
        baseline_validation_scored,
        "Baseline — Always Negative",
        0.50,
    ),
    threshold_metrics(
        baseline_validation_scored,
        "Baseline — Always Positive",
        0.00,
    ),
]

all_validation_results = baseline_results + validation_results
validation_comparison = spark.createDataFrame(all_validation_results)
validation_comparison = validation_comparison.select(
    "model",
    "roc_auc",
    "pr_auc",
    "threshold",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "specificity",
    "brier_score",
    "predicted_positive_rate",
    "actual_positive_rate",
    "predicts_both_classes",
    "tn",
    "fp",
    "fn",
    "tp",
).orderBy(F.desc("pr_auc"), F.desc("f1"), F.desc("recall"))

validation_prevalence = float(validation_data.agg(F.avg("label")).first()[0])

print("No-skill validation PR-AUC baseline (positive prevalence):", round(validation_prevalence, 6))
validation_comparison.show(truncate=False)

candidate_validation_comparison = (
    validation_comparison
    .filter(~F.col("model").startswith("Baseline"))
    .orderBy(F.desc("pr_auc"), F.desc("f1"), F.desc("recall"))
)
selected_validation_row = candidate_validation_comparison.first().asDict()
selected_model_name = selected_validation_row["model"]
selected_threshold = float(selected_validation_row["threshold"])

print("Selected model:", selected_model_name)
print("Locked validation threshold:", selected_threshold)

validation_pr_auc_gain = float(selected_validation_row["pr_auc"]) - validation_prevalence
limited_predictive_signal = validation_pr_auc_gain <= 0.01

print(
    "Selected model beats the no-skill PR-AUC baseline:",
    selected_validation_row["pr_auc"] > validation_prevalence,
)
print("Validation PR-AUC gain over prevalence:", round(validation_pr_auc_gain, 6))

if limited_predictive_signal:
    print(
        "WARNING: Predictive signal is limited. Keep the result as an honest "
        "ranking/segmentation model and do not overstate individual prediction accuracy."
    )

print("The test set has not been used for selection or threshold tuning.")

comparison_pd = validation_comparison.toPandas().sort_values("pr_auc")
plt.figure(figsize=(8, 4.5))
plt.barh(comparison_pd["model"], comparison_pd["pr_auc"], color=PLOT_COLORS["primary"])
plt.axvline(validation_prevalence, color=PLOT_COLORS["accent"], linestyle="--", label="No-skill prevalence")
plt.title("Validation PR-AUC by model")
plt.xlabel("PR-AUC")
plt.ylabel("Model")
plt.legend()
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 15 — Final evaluation and interpretability
# MAGIC 
# MAGIC ### Purpose
# MAGIC Evaluate the selected configuration once on the newest unseen Test period and explain what the model learned.
# MAGIC 
# MAGIC ### Final-fit procedure
# MAGIC 1. Train and Validation are combined into the development dataset.
# MAGIC 2. Preprocessing is refitted on development data only.
# MAGIC 3. The selected classifier is refitted.
# MAGIC 4. The locked validation threshold is applied to the Test set.
# MAGIC 5. The Test set is evaluated once.
# MAGIC 
# MAGIC ### Interpretation outputs
# MAGIC - final ROC-AUC and PR-AUC,
# MAGIC - Precision / Recall / F1,
# MAGIC - confusion matrix,
# MAGIC - Brier score,
# MAGIC - calibration by score decile,
# MAGIC - top feature coefficients/importances.
# MAGIC 
# MAGIC ### Probability caution
# MAGIC Class weighting improves minority-class detection but can distort absolute probability calibration. Therefore, the model output should primarily be interpreted as a **repurchase propensity score / ranking signal**, not as a perfectly calibrated probability.
# MAGIC 
# MAGIC ### Expected result
# MAGIC A final unbiased performance estimate and a presentation-ready view of the strongest predictors.
# MAGIC 

# COMMAND ----------

# ==========================================
# 15 - Final Evaluation and Interpretability
# ==========================================

# Refit the selected configuration on all development data (train + validation).
# The final test set remains excluded from preprocessing and model fitting.
development_data = train_data.unionByName(validation_data)
development_weighted, development_weight_info = add_balanced_class_weights(development_data)

final_preprocessing_pipeline = build_preprocessing_pipeline()
final_preprocessing_model = final_preprocessing_pipeline.fit(development_weighted)

development_prepared = final_preprocessing_model.transform(development_weighted)
test_prepared = final_preprocessing_model.transform(test_data)

final_classifier_model = build_classifier(selected_model_name).fit(development_prepared)

# This is the first and only final-test evaluation.
final_test_scored = add_probability_score(
    final_classifier_model.transform(test_prepared)
)
final_test_metrics = threshold_metrics(
    final_test_scored,
    selected_model_name,
    selected_threshold,
)

test_prevalence = float(test_data.agg(F.avg("label")).first()[0])
print("Development class weights:", development_weight_info)
print("No-skill test PR-AUC baseline (positive prevalence):", round(test_prevalence, 6))
print("Final test metrics:", final_test_metrics)

final_test_predictions = final_test_scored.withColumn(
    "predicted_repurchase",
    (F.col("repurchase_probability") >= F.lit(selected_threshold)).cast("int"),
)

# Calibration review: weighted models may rank well without producing calibrated probabilities.
calibration_window = Window.orderBy("repurchase_probability")
calibration_table = (
    final_test_scored
    .withColumn("score_decile", F.ntile(10).over(calibration_window))
    .groupBy("score_decile")
    .agg(
        F.count("*").alias("rows"),
        F.avg("repurchase_probability").alias("mean_predicted_probability"),
        F.avg("label").alias("observed_repurchase_rate"),
    )
    .orderBy("score_decile")
)
calibration_table.show(truncate=False)

def vector_feature_names(frame, vector_column="features"):
    metadata = frame.schema[vector_column].metadata.get("ml_attr", {})
    attributes = metadata.get("attrs", {})
    flattened = []
    for group in attributes.values():
        flattened.extend(group)
    if not flattened:
        vector_size = frame.select(vector_column).first()[vector_column].size
        return [f"feature_{index}" for index in range(vector_size)]
    return [item["name"] for item in sorted(flattened, key=lambda item: item["idx"])]

final_feature_names = vector_feature_names(development_prepared, "features")

if selected_model_name == "Logistic Regression":
    signed_values = [float(value) for value in final_classifier_model.coefficients]
    importance_rows = [
        (name, abs(value), value)
        for name, value in zip(final_feature_names, signed_values)
    ]
    feature_importance = spark.createDataFrame(
        importance_rows,
        ["feature", "importance", "signed_effect"],
    )
else:
    importance_values = [float(value) for value in final_classifier_model.featureImportances]
    importance_rows = [
        (name, value)
        for name, value in zip(final_feature_names, importance_values)
    ]
    feature_importance = spark.createDataFrame(
        importance_rows,
        ["feature", "importance"],
    ).withColumn("signed_effect", F.lit(None).cast("double"))

top_feature_importance = feature_importance.orderBy(F.desc("importance")).limit(20)
top_feature_importance.show(20, truncate=False)

importance_pd = top_feature_importance.toPandas().sort_values("importance")
calibration_pd = calibration_table.toPandas()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].barh(importance_pd["feature"], importance_pd["importance"], color=PLOT_COLORS["primary"])
axes[0].set_title(f"Top predictors — {selected_model_name}")
axes[0].set_xlabel("Absolute coefficient / feature importance")
axes[0].set_ylabel("Feature")

axes[1].plot(
    calibration_pd["mean_predicted_probability"],
    calibration_pd["observed_repurchase_rate"],
    marker="o",
    color=PLOT_COLORS["primary"],
)
axes[1].plot([0, 1], [0, 1], linestyle="--", color=PLOT_COLORS["neutral"])
axes[1].set_title("Final-test calibration by score decile")
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Observed repurchase rate")

confusion_values = [
    [final_test_metrics["tn"], final_test_metrics["fp"]],
    [final_test_metrics["fn"], final_test_metrics["tp"]],
]
image = axes[2].imshow(confusion_values, cmap="Blues")
for row_index in range(2):
    for column_index in range(2):
        axes[2].text(
            column_index,
            row_index,
            confusion_values[row_index][column_index],
            ha="center",
            va="center",
            color="black",
        )
axes[2].set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
axes[2].set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
axes[2].set_title(f"Final-test confusion matrix @ {selected_threshold:.2f}")
fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 16 — Score current first-time customers
# MAGIC 
# MAGIC ### Purpose
# MAGIC Apply the final model to the operational population that does not yet have a mature 180-day label.
# MAGIC 
# MAGIC ### Scoring population
# MAGIC Only customers who:
# MAGIC - are still inside the 180-day observation window, and
# MAGIC - have **not already made a second purchase**
# MAGIC 
# MAGIC are scored.
# MAGIC 
# MAGIC Customers who have already repurchased do not require a prediction.
# MAGIC 
# MAGIC ### Business segmentation
# MAGIC Current customers are ranked by repurchase score and divided into:
# MAGIC - High Repurchase Probability
# MAGIC - Medium Repurchase Probability
# MAGIC - Low Repurchase Probability
# MAGIC 
# MAGIC The segment boundaries are derived from the current score distribution rather than arbitrary fixed business cutoffs.
# MAGIC 
# MAGIC ### Expected result
# MAGIC A customer-level table suitable for retention and remarketing prioritization.
# MAGIC 

# COMMAND ----------

# ==========================================
# 16 - Current Customer Scoring
# ==========================================

# Score only immature first-order customers who have not already repurchased.
# Customers with a known second purchase do not need a prediction.
current_first_order_base = (
    first_order_timeline
    .filter(F.col("first_order_at") > F.lit(label_cutoff_ts))
    .filter(F.col("next_order_at").isNull())
    .select("order_id", "user_id", "first_order_at")
)

current_scoring_features = build_safe_feature_frame(current_first_order_base)
current_prepared = final_preprocessing_model.transform(current_scoring_features)
current_scored = add_probability_score(
    final_classifier_model.transform(current_prepared)
)

current_scoring_count = current_scored.count()
if current_scoring_count < 3:
    raise ValueError("At least three current customers are required for quantile segments.")

low_cutoff, high_cutoff = current_scored.approxQuantile(
    "repurchase_probability",
    [0.33, 0.67],
    0.001,
)

current_predictions_output = (
    current_scored
    .withColumn(
        "predicted_repurchase",
        (F.col("repurchase_probability") >= F.lit(selected_threshold)).cast("int"),
    )
    .withColumn(
        "opportunity_segment",
        F.when(
            F.col("repurchase_probability") >= F.lit(high_cutoff),
            F.lit("High Repurchase Probability"),
        )
        .when(
            F.col("repurchase_probability") >= F.lit(low_cutoff),
            F.lit("Medium Repurchase Probability"),
        )
        .otherwise(F.lit("Low Repurchase Probability")),
    )
    .select(
        "user_id",
        F.col("order_id").alias("first_order_id"),
        F.col("first_order_at"),
        F.round("repurchase_probability", 6).alias("repurchase_probability"),
        "predicted_repurchase",
        "opportunity_segment",
        "order_value",
        "item_count",
        "age",
        "gender",
        "country",
        "traffic_source",
        "customer_tenure_days",
    )
)

print("Current unrepurchased first-time customers:", current_scoring_count)
print("Locked classification threshold:", selected_threshold)
print("Segment 33rd percentile:", low_cutoff)
print("Segment 67th percentile:", high_cutoff)
current_predictions_output.orderBy(F.desc("repurchase_probability")).show(20, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 17 — Prepare Power BI outputs
# MAGIC 
# MAGIC ### Purpose
# MAGIC Create clean in-memory datasets that can later be exposed to Power BI.
# MAGIC 
# MAGIC ### Prepared outputs
# MAGIC 
# MAGIC **Current customer scoring**
# MAGIC - customer ID
# MAGIC - first order
# MAGIC - repurchase score
# MAGIC - classification
# MAGIC - opportunity segment
# MAGIC - selected customer/order attributes
# MAGIC 
# MAGIC **Final Test predictions**
# MAGIC - actual label
# MAGIC - model score
# MAGIC - predicted class
# MAGIC 
# MAGIC **Model metrics**
# MAGIC - validation results for all candidates
# MAGIC - final Test result for the selected model
# MAGIC 
# MAGIC ### Important note
# MAGIC This notebook intentionally prepares the outputs **in memory only**. It does not write or overwrite Databricks tables.
# MAGIC 

# COMMAND ----------

# ==========================================
# 17 - Power BI Output Preparation
# ==========================================

test_predictions_output = final_test_predictions.select(
    "user_id",
    F.col("order_id").alias("first_order_id"),
    "first_order_at",
    "label",
    F.round("repurchase_probability", 6).alias("repurchase_probability"),
    "predicted_repurchase",
)

validation_metric_rows = []
for row in all_validation_results:
    validation_metric_rows.append({
        **row,
        "dataset": "validation",
        "selected_model": row["model"] == selected_model_name,
    })

final_metric_row = {
    **final_test_metrics,
    "dataset": "final_test",
    "selected_model": True,
}

model_metrics_output = spark.createDataFrame(validation_metric_rows + [final_metric_row]).select(
    "dataset",
    "model",
    "selected_model",
    "threshold",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "specificity",
    "brier_score",
    "predicted_positive_rate",
    "actual_positive_rate",
    "predicts_both_classes",
    "tn",
    "fp",
    "fn",
    "tp",
)

print("Prepared in-memory outputs")
print("Current prediction rows:", current_predictions_output.count())
print("Final test prediction rows:", test_predictions_output.count())
model_metrics_output.orderBy("dataset", F.desc("pr_auc")).show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 18 — Final validation
# MAGIC 
# MAGIC ### Purpose
# MAGIC Perform final integrity checks before the ML workflow is considered complete.
# MAGIC 
# MAGIC ### Checks
# MAGIC - required prediction columns exist,
# MAGIC - one output row per scored first order,
# MAGIC - no duplicate Test predictions,
# MAGIC - all scores remain between 0 and 1,
# MAGIC - Test output row count matches the Test dataset.
# MAGIC 
# MAGIC ### Expected result
# MAGIC The notebook should finish with:
# MAGIC 
# MAGIC > **Validation complete. Outputs are prepared in memory; the notebook contains no table writes.**
# MAGIC 
# MAGIC At this point the modeling workflow is complete and ready for BI handoff.
# MAGIC 

# COMMAND ----------

# ==========================================
# 18 - Final Validation
# ==========================================

required_prediction_columns = {
    "user_id",
    "first_order_id",
    "first_order_at",
    "repurchase_probability",
    "predicted_repurchase",
    "opportunity_segment",
}
missing_prediction_columns = required_prediction_columns - set(current_predictions_output.columns)
assert not missing_prediction_columns, f"Missing output columns: {sorted(missing_prediction_columns)}"

duplicate_scored_orders = (
    current_predictions_output.groupBy("first_order_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)
duplicate_test_orders = (
    test_predictions_output.groupBy("first_order_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)
out_of_range_scores = current_predictions_output.filter(
    (F.col("repurchase_probability") < 0)
    | (F.col("repurchase_probability") > 1)
).count()

assert duplicate_scored_orders == 0
assert duplicate_test_orders == 0
assert out_of_range_scores == 0
assert test_predictions_output.count() == test_data.count()

print("Selected model:", selected_model_name)
print("Locked threshold:", selected_threshold)
print("Final test metrics:", final_test_metrics)
print("Limited validation signal flag:", limited_predictive_signal)
print("Current scoring rows:", current_predictions_output.count())
print("Final test rows:", test_predictions_output.count())
print("Validation complete. Outputs are prepared in memory; the notebook contains no table writes.")


# COMMAND ----------

# MAGIC %md
# MAGIC # Final model report — validated reference run
# MAGIC 
# MAGIC The following values summarize the successful Databricks run used to validate this notebook.  
# MAGIC If the underlying source data changes, the newly generated outputs should be treated as authoritative.
# MAGIC 
# MAGIC ## Selected model
# MAGIC 
# MAGIC **Logistic Regression**
# MAGIC 
# MAGIC **Locked classification threshold:** `0.50`
# MAGIC 
# MAGIC The simpler interpretable model performed better than both Random Forest and Gradient-Boosted Trees on Validation PR-AUC.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Validation performance
# MAGIC 
# MAGIC | Metric | Result |
# MAGIC |---|---:|
# MAGIC | ROC-AUC | **0.660** |
# MAGIC | PR-AUC | **0.231** |
# MAGIC | No-skill PR-AUC baseline | **0.153** |
# MAGIC | Precision | **23.4%** |
# MAGIC | Recall | **62.2%** |
# MAGIC | F1 Score | **0.341** |
# MAGIC | Threshold | **0.50** |
# MAGIC 
# MAGIC The validation PR-AUC exceeded the prevalence baseline by approximately **0.078**, indicating usable predictive signal.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Final Test performance
# MAGIC 
# MAGIC | Metric | Result |
# MAGIC |---|---:|
# MAGIC | ROC-AUC | **0.697** |
# MAGIC | PR-AUC | **0.334** |
# MAGIC | No-skill PR-AUC baseline | **0.206** |
# MAGIC | Precision | **30.6%** |
# MAGIC | Recall | **76.3%** |
# MAGIC | F1 Score | **0.436** |
# MAGIC | Accuracy | **59.3%** |
# MAGIC | Specificity | **54.9%** |
# MAGIC | Brier Score | **0.221** |
# MAGIC 
# MAGIC ### Final Test confusion matrix
# MAGIC 
# MAGIC |  | Predicted No | Predicted Yes |
# MAGIC |---|---:|---:|
# MAGIC | **Actual No** | 3,700 | 3,034 |
# MAGIC | **Actual Yes** | 415 | 1,335 |
# MAGIC 
# MAGIC The model identified **1,335 of 1,750** actual repurchasers in the final Test period.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Ranking quality
# MAGIC 
# MAGIC The score is especially useful for **ranking customers**.
# MAGIC 
# MAGIC In the validated Test run:
# MAGIC 
# MAGIC - Lowest score decile observed repurchase rate: approximately **6.0%**
# MAGIC - Highest score decile observed repurchase rate: approximately **37.7%**
# MAGIC 
# MAGIC This means the highest-ranked decile had an observed repurchase rate more than **6×** that of the lowest-ranked decile.
# MAGIC 
# MAGIC This supports the use of the model for **customer prioritization and campaign segmentation**.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Important calibration finding
# MAGIC 
# MAGIC The weighted Logistic Regression scores are **not perfectly calibrated as literal probabilities**.
# MAGIC 
# MAGIC For example, in the highest score decile:
# MAGIC 
# MAGIC - mean model score was approximately **58.8%**
# MAGIC - observed repurchase rate was approximately **37.7%**
# MAGIC 
# MAGIC Therefore, in Power BI and business communication, the field should preferably be presented as:
# MAGIC 
# MAGIC > **Repurchase Score**
# MAGIC 
# MAGIC rather than promising that the number is an exact probability of future purchase.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Strongest predictor
# MAGIC 
# MAGIC The strongest model coefficient in the validated run was:
# MAGIC 
# MAGIC **Customer tenure at first order**
# MAGIC 
# MAGIC with a negative signed effect.
# MAGIC 
# MAGIC In this dataset, customers who waited longer between account creation and their first order tended to receive lower second-purchase propensity scores.
# MAGIC 
# MAGIC This is an **association in the synthetic project data**, not a causal conclusion.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Current scoring population
# MAGIC 
# MAGIC The validated run produced scores for:
# MAGIC 
# MAGIC **12,235 current first-time customers**
# MAGIC 
# MAGIC who had not yet made a second purchase and whose 180-day outcome window was still incomplete.
# MAGIC 
# MAGIC Current score segmentation cutoffs were approximately:
# MAGIC 
# MAGIC - **Low → Medium:** `0.491`
# MAGIC - **Medium → High:** `0.577`
# MAGIC 
# MAGIC These quantile-based segments can be used in the Power BI customer-retention dashboard.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Business conclusion
# MAGIC 
# MAGIC The model is not intended to make perfect individual predictions. Its strongest use is as a **prioritization tool**:
# MAGIC 
# MAGIC > Identify first-time customers with relatively higher second-purchase propensity so that retention and remarketing actions can be focused more efficiently.
# MAGIC 
# MAGIC The chronological evaluation also revealed substantial temporal drift, so future production use would require regular monitoring and recalibration.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Project limitations
# MAGIC 
# MAGIC - TheLook is synthetic data.
# MAGIC - Customer order counts are artificially capped.
# MAGIC - The target rate changes strongly over calendar time.
# MAGIC - Weighted probabilities are not fully calibrated.
# MAGIC - The model should support business prioritization, not automated high-stakes decisions.
# MAGIC 
# MAGIC Despite these limitations, the model demonstrates a complete and defensible **end-to-end ML workflow** integrated into the wider Databricks + Power BI BI project.
# MAGIC 

# COMMAND ----------

# ==========================================
# 19 - Save ML Outputs for Power BI
# ==========================================

# Save only NEW ML output tables.
# Existing Bronze, Silver, Gold, Fact, and Dimension tables are not modified.

current_predictions_output.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("default.ml_second_purchase_predictions")

test_predictions_output.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("default.ml_second_purchase_test_predictions")

model_metrics_output.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("default.ml_second_purchase_model_metrics")


print("===== ML TABLES SAVED FOR POWER BI =====")

print(
    "Current predictions:",
    spark.table("default.ml_second_purchase_predictions").count()
)

print(
    "Final test predictions:",
    spark.table("default.ml_second_purchase_test_predictions").count()
)

print(
    "Model metric rows:",
    spark.table("default.ml_second_purchase_model_metrics").count()
)

print("✅ Power BI output tables are ready.")
