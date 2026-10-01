# Data availability

The complete datasets are not committed. This project uses the Google BigQuery public **TheLook E-Commerce** dataset: `bigquery-public-data.thelook_ecommerce`.

Export `users`, `orders`, `order_items`, and `products` to CSV and upload copies to the Databricks staging Volume. The notebooks reproduce the Bronze ingestion tables, Silver validated tables, Gold dimensions/facts, and ML prediction/evaluation output tables.

Follow the root [README](../README.md#reproduce-the-project) and [reproduction guide](../docs/NvS_Reproducible_Project_Steps.pdf), including the documented fresh-workspace setup corrections. The source snapshot may change over time, so fresh exports can yield different row counts and model results.

Local datasets and generated table exports are excluded from Git to keep the repository focused on the source workflow and final project artifacts. Do not upload credentials or private connection configuration alongside data.
