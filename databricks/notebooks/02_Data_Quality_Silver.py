# Databricks notebook source
users = spark.table("workspace.default.bronze_users")
orders = spark.table("workspace.default.bronze_orders")
order_items = spark.table("workspace.default.bronze_order_items")
products = spark.table("workspace.default.bronze_products")

print("Users:", users.count())
print("Orders:", orders.count())
print("Order Items:", order_items.count())
print("Products:", products.count())

# COMMAND ----------

print("Duplicate User IDs:",
      users.count() - users.select("id").distinct().count())

print("Duplicate Order IDs:",
      orders.count() - orders.select("order_id").distinct().count())

print("Duplicate Item IDs:",
      order_items.count() - order_items.select("id").distinct().count())

print("Duplicate Product IDs:",
      products.count() - products.select("id").distinct().count())

# COMMAND ----------

orders_without_user = (
    orders
    .join(users, orders["user_id"] == users["id"], "left_anti")
)

items_without_order = (
    order_items
    .join(orders, order_items["order_id"] == orders["order_id"], "left_anti")
)

items_without_product = (
    order_items
    .join(products, order_items["product_id"] == products["id"], "left_anti")
)

print("Orders without matching user:", orders_without_user.count())
print("Order items without matching order:", items_without_order.count())
print("Order items without matching product:", items_without_product.count())

# COMMAND ----------

print(
    "Orphan order item rows:",
    items_without_order.count()
)

print(
    "Distinct missing order IDs:",
    items_without_order.select("order_id").distinct().count()
)

display(
    items_without_order
    .select("id", "order_id", "user_id", "product_id", "status", "created_at")
    .orderBy("order_id")
)

# COMMAND ----------

display(
    items_without_order
    .select("order_id", "created_at")
    .distinct()
    .orderBy("order_id")
)

# COMMAND ----------

# 1) Duplicate ID check

from pyspark.sql import functions as F

duplicate_users = (
    bronze_users
    .groupBy("id")
    .count()
    .filter(F.col("count") > 1)
)

duplicate_orders = (
    bronze_orders
    .groupBy("order_id")
    .count()
    .filter(F.col("count") > 1)
)

duplicate_order_items = (
    bronze_order_items
    .groupBy("id")
    .count()
    .filter(F.col("count") > 1)
)

duplicate_products = (
    bronze_products
    .groupBy("id")
    .count()
    .filter(F.col("count") > 1)
)

print("Duplicate Users:", duplicate_users.count())
print("Duplicate Orders:", duplicate_orders.count())
print("Duplicate Order Items:", duplicate_order_items.count())
print("Duplicate Products:", duplicate_products.count())

# COMMAND ----------

from pyspark.sql import functions as F

def null_report(df, table_name):
    total_rows = df.count()

    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ])

    print(f"\n===== {table_name} =====")
    print("Total rows:", total_rows)
    display(null_counts)

null_report(bronze_users, "Users")
null_report(bronze_orders, "Orders")
null_report(bronze_order_items, "Order Items")
null_report(bronze_products, "Products")

# COMMAND ----------

silver_users = bronze_users

silver_orders = bronze_orders

silver_products = bronze_products

silver_order_items = (
    bronze_order_items.alias("oi")
    .join(
        bronze_orders.select("order_id").alias("o"),
        "order_id",
        "inner"
    )
)

# COMMAND ----------

print("Silver Users:", silver_users.count())
print("Silver Orders:", silver_orders.count())
print("Silver Order Items:", silver_order_items.count())
print("Silver Products:", silver_products.count())

# COMMAND ----------

silver_users.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_users")

silver_orders.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_orders")

silver_order_items.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_order_items")

silver_products.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_products")

print("✅ All Silver tables saved successfully.")

# COMMAND ----------

print("===== SILVER VALIDATION =====")

print("Users:", spark.table("silver_users").count())
print("Orders:", spark.table("silver_orders").count())
print("Order Items:", spark.table("silver_order_items").count())
print("Products:", spark.table("silver_products").count())
