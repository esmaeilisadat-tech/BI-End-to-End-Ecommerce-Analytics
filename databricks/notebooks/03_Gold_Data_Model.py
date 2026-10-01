# Databricks notebook source
silver_users = spark.table("silver_users")
silver_orders = spark.table("silver_orders")
silver_order_items = spark.table("silver_order_items")
silver_products = spark.table("silver_products")

print("Silver tables loaded ✅")

# COMMAND ----------

dim_users = silver_users.select(
    "id",
    "first_name",
    "last_name",
    "email",
    "age",
    "gender",
    "state",
    "city",
    "country",
    "traffic_source",
    "created_at"
)

dim_products = silver_products.select(
    "id",
    "name",
    "brand",
    "category",
    "department",
    "cost",
    "retail_price",
    "sku",
    "distribution_center_id"
)

# COMMAND ----------

fact_orders = silver_orders.select(
    "order_id",
    "user_id",
    "status",
    "created_at",
    "shipped_at",
    "delivered_at",
    "returned_at",
    "num_of_item"
)

fact_order_items = silver_order_items.select(
    "id",
    "order_id",
    "user_id",
    "product_id",
    "inventory_item_id",
    "status",
    "created_at",
    "shipped_at",
    "delivered_at",
    "returned_at",
    "sale_price"
)

# COMMAND ----------

print("dim_users:", dim_users.count())
print("dim_products:", dim_products.count())
print("fact_orders:", fact_orders.count())
print("fact_order_items:", fact_order_items.count())

# COMMAND ----------

from pyspark.sql import functions as F

fact_order_items = (
    fact_order_items.alias("f")
    .join(
        silver_products
        .select(
            F.col("id").alias("product_id"),
            F.col("cost").alias("product_cost")
        )
        .alias("p"),
        "product_id",
        "left"
    )
    .withColumn("revenue", F.col("sale_price"))
    .withColumn("cost", F.col("product_cost"))
    .withColumn(
        "profit",
        F.col("sale_price") - F.col("product_cost")
    )
    .drop("product_cost")
)

display(fact_order_items.limit(10))

# COMMAND ----------

fact_order_items.select(
    F.count("*").alias("rows"),
    F.sum("revenue").alias("total_revenue"),
    F.sum("cost").alias("total_cost"),
    F.sum("profit").alias("total_profit")
).show()

# COMMAND ----------

min_date = (
    silver_orders
    .select(F.min(F.to_date("created_at")))
    .first()[0]
)

max_date = (
    silver_orders
    .select(F.max(F.to_date("created_at")))
    .first()[0]
)

dim_date = (
    spark.sql(
        f"""
        SELECT explode(
            sequence(
                to_date('{min_date}'),
                to_date('{max_date}'),
                interval 1 day
            )
        ) AS date
        """
    )
    .withColumn("year", F.year("date"))
    .withColumn("month", F.month("date"))
    .withColumn("month_name", F.date_format("date", "MMMM"))
    .withColumn("quarter", F.quarter("date"))
    .withColumn("day", F.dayofmonth("date"))
    .withColumn("day_name", F.date_format("date", "EEEE"))
    .withColumn("year_month", F.date_format("date", "yyyy-MM"))
)

# COMMAND ----------

print("Date rows:", dim_date.count())

dim_date.orderBy("date").show(5)

dim_date.orderBy(F.col("date").desc()).show(5)

# COMMAND ----------

dim_users.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("dim_users")

dim_products.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("dim_products")

dim_date.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("dim_date")

fact_orders.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("fact_orders")

fact_order_items.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("fact_order_items")

print("✅ All Gold tables saved successfully.")

# COMMAND ----------

print("===== GOLD VALIDATION =====")
print("dim_users:", spark.table("dim_users").count())
print("dim_products:", spark.table("dim_products").count())
print("dim_date:", spark.table("dim_date").count())
print("fact_orders:", spark.table("fact_orders").count())
print("fact_order_items:", spark.table("fact_order_items").count())
