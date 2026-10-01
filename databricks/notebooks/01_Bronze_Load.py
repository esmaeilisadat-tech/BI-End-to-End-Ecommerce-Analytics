# Databricks notebook source
users_path = "/Volumes/workspace/default/bi_final_raw/users_*.csv.csv"

df_users = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(users_path)
)

print("Users row count:", df_users.count())
display(df_users.limit(10))

# COMMAND ----------

df_users.printSchema()

# COMMAND ----------

print("Distinct user IDs:", df_users.select("id").distinct().count())

# COMMAND ----------

from pyspark.sql.functions import current_timestamp

bronze_users = (
    df_users
    .withColumn("_ingested_at", current_timestamp())
)

bronze_users.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.bronze_users")

print("bronze_users created successfully.")

# COMMAND ----------

df_check = spark.table("workspace.default.bronze_users")

print("Bronze users row count:", df_check.count())
display(df_check.limit(10))

# COMMAND ----------

orders_path = "/Volumes/workspace/default/bi_final_raw/orders_*.csv*"

df_orders = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(orders_path)
)

print("Orders row count:", df_orders.count())
print("Distinct order IDs:", df_orders.select("order_id").distinct().count())

display(df_orders.limit(10))

# COMMAND ----------

order_items_path = "/Volumes/workspace/default/bi_final_raw/order_items_*.csv*"

df_order_items = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(order_items_path)
)

print("Order items row count:", df_order_items.count())
print("Distinct item IDs:", df_order_items.select("id").distinct().count())

display(df_order_items.limit(10))

# COMMAND ----------

products_path = "/Volumes/workspace/default/bi_final_raw/products.csv*"

df_products = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(products_path)
)

print("Products row count:", df_products.count())
print("Distinct product IDs:", df_products.select("id").distinct().count())

display(df_products.limit(10))

# COMMAND ----------

display(dbutils.fs.ls("/Volumes/workspace/default/bi_final_raw/"))

# COMMAND ----------

import os

folder = "/Volumes/workspace/default/bi_final_raw"

for filename in os.listdir(folder):
    if filename.endswith(".csv.csv"):
        old_path = os.path.join(folder, filename)
        new_name = filename[:-4]   # حذف آخرین .csv
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
        print(f"{filename}  ->  {new_name}")

# COMMAND ----------

from pyspark.sql.functions import current_timestamp

# ORDER ITEMS
df_order_items = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("/Volumes/workspace/default/bi_final_raw/order_items_*.csv")
)

(
    df_order_items
    .withColumn("_ingested_at", current_timestamp())
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("workspace.default.bronze_order_items")
)


# PRODUCTS
df_products = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("/Volumes/workspace/default/bi_final_raw/products.csv")
)

(
    df_products
    .withColumn("_ingested_at", current_timestamp())
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("workspace.default.bronze_products")
)

print("bronze_order_items:", spark.table("workspace.default.bronze_order_items").count())
print("bronze_products:", spark.table("workspace.default.bronze_products").count())
