# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = [
#   "cowsay",
# ]
# ///
# MAGIC %md 
# MAGIC ## Shared Compute for authoring context 

# COMMAND ----------



# COMMAND ----------

import cowsay 

# COMMAND ----------

print(cowsay.cow("Hello World"))

# COMMAND ----------



# COMMAND ----------

# MAGIC %sh
# MAGIC uv sync

# COMMAND ----------

# DBTITLE 1,[Design] Why not %pip install . for pyproject.toml
# MAGIC %pip install --verbose .

# COMMAND ----------

# MAGIC %pip install . 
