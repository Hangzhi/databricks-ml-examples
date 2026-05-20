# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "workspace-base-environments/dbe_65ad7304-6f87-4e25-b52c-f1cd39071e89"
# environment_version = "4"
# ///
# MAGIC %md
# MAGIC # Transformer Based Batch Inference
# MAGIC
# MAGIC This notebook aims to show how you can run batch inference using Spark's distributed capabilites, with a multi-machine multi-gpu setup. The code is designed to fit the model used into a single GPU, therefore some adjustment mightbe needed for using a larger model or a GPU with smaller memory.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.0 GPU ML Runtime
# MAGIC - Instance: `N24ads_A100_V4` on Azure. 3 A100 GPUs (1 Driver + 2 Worker)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install & Upgrade Libraries

# COMMAND ----------

!pip install -q --upgrade transformers
!pip install -q --upgrade accelerate
dbutils.library.restartPython()

test 

# COMMAND ----------

# MAGIC %md
# MAGIC ### GPU Stats

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parameters
# MAGIC
# MAGIC The model name & tokenizer name below should be adjustable to most of the other models existing in the hugging face world. The code is designed to fit the entire model into a single GPU, so futher tweaking might be required if a large model or a GPU with small memory is used.
# MAGIC
# MAGIC It would also make sense to change the prompt if there is a change to the model. This one is specifically designed for the LLAMA V2 model.

# COMMAND ----------

# Model Params
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MODEL_REVISION_ID = "08751db2aca9bf2f7f80d2e516117a53d7450235"
MAX_NEW_TOKENS = 300
MIN_LENGTH = 0
REPETITION_PENALTY = 1.2
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 50
DO_SAMPLE = True
USE_CACHE = True

# Tokenizer Params
TOKENIZER_NAME = MODEL_NAME
TOKENIZER_REVISION_ID = MODEL_REVISION_ID
MAX_TOKENS = 2048

# Run Params (How many articles to use)
MAX_EXAMPLES = 10000

# Storage Params
STORAGE_PATH = "/dbfs/llm-examples"

# Instruction
INSTRUCTION = """Please provide a concise summary for the following article: {text}"""

# Prompt
PROMPT_TEMPLATE = f"""<s>[INST]<<SYS>>
You are a direct and honest assistant. Please provide concise and factual answers and just the answers.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{INSTRUCTION}
[/INST]
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Storage Operations
# MAGIC
# MAGIC Makes sure that the directory is cleaned.

# COMMAND ----------

import shutil
import os

# Remove existing files
shutil.rmtree(STORAGE_PATH, ignore_errors=True)

# Build the new directory
os.makedirs(STORAGE_PATH, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Huggingface Login
# MAGIC
# MAGIC This step can be skipped if your model doesn't require a login. LLAMA V2 does.

# COMMAND ----------

# Login to hugging face
from huggingface_hub import notebook_login

# Login the huggingface
notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve Articles Data
# MAGIC
# MAGIC CNN & Daily Mail articles data set from Hugging Face Datasets is retrieved and combined to make a pyspark dataframe.

# COMMAND ----------

# Imports
from datasets.utils import logging as dataset_logging, disable_progress_bar
from pyspark.sql import functions as SF
from datasets import load_dataset

# Disable verbose loggers
dataset_logging.set_verbosity_error()
disable_progress_bar()

# Download dataset
dataset = load_dataset(
    path="cnn_dailymail", name="3.0.0", cache_dir=f"{STORAGE_PATH}/hf"
)

# Create spark dataframes
train_df = spark.createDataFrame(data=dataset["train"].to_pandas())
val_df = spark.createDataFrame(data=dataset["validation"].to_pandas())
test_df = spark.createDataFrame(data=dataset["test"].to_pandas())

# Union for all data
articles_df = train_df.union(val_df).union(test_df)
articles_df = articles_df.repartition(16)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Data
# MAGIC
# MAGIC Deterministic sampling through ID hexing for consistent results & benchmarking.

# COMMAND ----------

from pyspark.sql import functions as SF
import hashlib

# Build function for creating a random column
@SF.udf("string")
def generate_hex_from_string(input_string: str) -> str:
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode("utf-8"))
    return sha256.hexdigest()


# Generate random string
articles_df = articles_df.withColumn(
    "random_string", generate_hex_from_string(SF.col("id"))
)

# Order by randomness and limit dataframe size
articles_df = articles_df.orderBy(SF.col("random_string")).limit(MAX_EXAMPLES)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Prompts
# MAGIC
# MAGIC Prompts are applied with the article text to generate prepared instructions for the model.

# COMMAND ----------

from pyspark.sql import functions as SF

# Build function for generating instructions
@SF.udf("string")
def generate_instructions(article):
    return PROMPT_TEMPLATE.format(text=article)


# Generate instructions
articles_df = articles_df.withColumn(
    "instruction", generate_instructions(SF.col("article"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute Data Operations
# MAGIC

# COMMAND ----------

# Cache for performance
articles_df.cache()

# Trigger with action
articles_df.write.format("noop").mode("overwrite").save()

print(f"Number of Examples: {articles_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download Model & Tokenizer
# MAGIC
# MAGIC Downloading the model and the tokizer helps when it comes to loading the model faster during the multi machine inference step.
# MAGIC
# MAGIC If the model and the tokenizer are in the same repository, only one download will occur. In some cases, for example for Falcon-7B, they can be different. In that case, the code downloads both to the location specified in params.

# COMMAND ----------

# External Imports
from huggingface_hub.utils import (
    disable_progress_bars as hfhub_disable_progress_bar,
    logging as hf_logging,   
)
from huggingface_hub import snapshot_download
import os

# Turn Off Info Logging for Transfomers
hf_logging.set_verbosity_error()
hfhub_disable_progress_bar()

# Download the model 
local_model_path = f"{STORAGE_PATH}/model/"
os.makedirs(local_model_path, exist_ok=True)
model_download = snapshot_download(
    repo_id=MODEL_NAME,
    revision=MODEL_REVISION_ID,
    local_dir=local_model_path,
    local_dir_use_symlinks=False,
    ignore_patterns="*.safetensors*", # This argument is specific to LLAMA. Other models might not need it.
    max_workers=48 # Boosts download speed
)

# Download the tokenizer
if MODEL_NAME == TOKENIZER_NAME:
    local_tokenizer_path = local_model_path
else:
    local_tokenizer_path = f"{STORAGE_PATH}/tokenizer/"
    os.makedirs(local_tokenizer_path, exist_ok=True)
    tokenizer_download = snapshot_download(
        repo_id=TOKENIZER_NAME,
        local_dir=local_tokenizer_path,
        local_dir_use_symlinks=False,
        max_workers=48
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Model & Tokenizer
# MAGIC
# MAGIC Model and Tokenizer are loaded for the downloaded directory for testing.

# COMMAND ----------

# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Params
random_seed = 42

# Random seed set
torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    return_dict=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    pad_token_id=tokenizer.eos_token_id,
)

# Load model to GPU
model.to("cuda:0")

# Put model in eval mode
model.eval()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Run on Single Example
# MAGIC
# MAGIC Batch Generate functions takes a list of prompts, and returns a list of ouputs (generated_text). Generation parameters such a temperature and top_p are set within the function.
# MAGIC
# MAGIC Even though this example shows how to do a few examples, this function will be used during distributed inference.

# COMMAND ----------

# Imports
import torch

# Get sample data
sample_instructions = [x[0] for x in articles_df.select("instruction").limit(2).collect()]

# Define Inference Flow
@torch.inference_mode()
def batch_generate(batch_prompts, tokenizer=tokenizer, model=model):
    batch = tokenizer.batch_encode_plus(
        batch_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=MAX_TOKENS
    )
    batch = {k: v.to("cuda:0") for k, v in batch.items()}
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            min_length=MIN_LENGTH,
            use_cache=USE_CACHE,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Check out one example
print(batch_generate(sample_instructions[:1])[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch Test
# MAGIC
# MAGIC Optimal batch size changes depending on the GPU used. The GPU used in during this test has 80 GB of GPU memory, so going higher makes sense, however smaller machine like the A10s usually do better with smaller batch sizes.
# MAGIC
# MAGIC The code compares multiple batch sizes, and interation stops when out of memory error is raised. 

# COMMAND ----------

# Imports
import time

# Get sample data
sample_instructions = [x[0] for x in articles_df.select("instruction").limit(50).collect()]

def batch_size_optimiser():
    batch_sizes = [1, 2, 3, 4, 5, 7, 10, 12, 15, 17, 20, 25, 30]
    success = True
    for size in batch_sizes:
        start = time.perf_counter()
        try:
            batch_generate(batch_prompts=sample_instructions[:size])
        except torch.cuda.OutOfMemoryError:
            success = False
            break
        finally:
            elapsed = round(time.perf_counter() - start, 2)
            unit_time = round(elapsed/size, 2)
            yield {"batch_size": size, "elapsed_time": elapsed, "unit_time": unit_time, "success": success}

for result in batch_size_optimiser():
    print("- - " * 10)
    print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select Batch Size
# MAGIC
# MAGIC Doesn't necessarily has to be the largest batch size that succeeded without running into an OOM error. It is probably better to choose the 2nd or 3rd largest successful batch size so that OOM errors can be reduced during inference.

# COMMAND ----------

# Get the bactch size with minimum unit time
OPTIMAL_BATCH_SIZE = 15

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed Inference Logic
# MAGIC
# MAGIC All of the generation logic is carried into a Pandas UDF. This helps with the set up on the workers. An iterator to interator architecture is followed to handle batching processes. 
# MAGIC
# MAGIC In the case that the model runs into an OOM Error, the function handles the exception by returnin a OOM string as a result.

# COMMAND ----------

# External Imports
from pyspark.sql import functions as SF
import pandas as pd
from typing import Iterator

# Build Inference Function
@SF.pandas_udf("string", SF.PandasUDFType.SCALAR_ITER)
def run_distributed_inference(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:

    # External Imports
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import pandas as pd
    import torch
    import os

    # Params
    random_seed = 42

    # Random seed set
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        return_dict=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Load model to GPU
    model.to("cuda:0")
    
    # Put model in eval mode
    model.eval()

    for prompts in iterator:
        prompts = prompts.to_list()
        try:
            output = batch_generate(
                batch_prompts=prompts, 
                tokenizer=tokenizer, 
                model=model
            )
        except torch.cuda.OutOfMemoryError:
            # If out of memory, return a series of OOM strings that has the lenght of the input
            output = ["OOM"] * len(prompts)

        yield pd.Series(output)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference Configurations
# MAGIC
# MAGIC Automatically undertands how many workers are available in the cluster, and adjusts partitions accordingly. This means that the setup portion of the Pandas UDF which loads the model and tokenizer gets run only once during inference, and the data processing is handled with the iterator.
# MAGIC
# MAGIC Max Records Per Batch configuration controls how big the batch sizes are going to be. 

# COMMAND ----------

# Imports
from pyspark import SparkContext

# Auto get number of workers
sc = SparkContext.getOrCreate()

# Subtract 1 to exclude the driver
num_workers = len(sc._jsc.sc().statusTracker().getExecutorInfos()) - 1  

# Set the batch size for the Pandas UDF
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", OPTIMAL_BATCH_SIZE * 2)

# Repartition
articles_df = articles_df.repartition(num_workers)

# Cache DF
articles_df.cache()
articles_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Distributed Inference

# COMMAND ----------

import time

# Apply Inference UDF
articles_df = (
    articles_df
    .withColumn("llm_summary", run_distributed_inference(SF.col("instruction")))

)

# Materilize and Execute
inference_start_time = time.perf_counter()
articles_df.cache()
articles_df.write.format("noop").mode("overwrite").save()
inference_elapsed_time = round(time.perf_counter() - inference_start_time, 4)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean Summaries
# MAGIC
# MAGIC The summaries generated by the model also contains the prompt. This function cleanes the summary by removing the prompt, leaving only the generated text behind. The LLAMA V2 model uses a specific token to mark the ending of the instruction, which is used here to split the string. Updating the UDF while using a different model might be necessary.

# COMMAND ----------

# Imports
from pyspark.sql import functions as SF

# UDF Build
clean_llm_summary = SF.udf(lambda x: x.split("[/INST]")[-1].strip(), "string")

# Apply UDF
articles_df = (
    articles_df.withColumn(
        "cleaned_llm_summary", clean_llm_summary(SF.col("llm_summary"))
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate Token Counts

# COMMAND ----------

@SF.udf("int")
def calculate_n_tokens(target_text):
    return len(
        tokenizer.encode_plus(
            target_text,
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            add_special_tokens=False,
            return_attention_mask=True,
            max_length=MAX_TOKENS,
        )["input_ids"]
    )


# Calculate article tokens
articles_df = articles_df.withColumn(
    "article_token_count", calculate_n_tokens(SF.col("article"))
)

# Calculate instruction tokens
articles_df = articles_df.withColumn(
    "instruction_token_count", calculate_n_tokens(SF.col("instruction"))
)

# Calculate generated tokens
articles_df = articles_df.withColumn(
    "generated_token_count", calculate_n_tokens(SF.col("cleaned_llm_summary"))
)

# Cache DF
articles_df.cache()
articles_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display Stats

# COMMAND ----------

# Imports
from pyspark.sql import functions as SF
import datetime

text_stats = (
    articles_df.groupBy()
    .agg(
        SF.count(SF.col("id")).alias("articles_count"),
        SF.sum(SF.col("article_token_count")).alias("total_article_tokens"),
        SF.sum(SF.col("instruction_token_count")).alias("total_instruction_tokens"),
        SF.sum(SF.col("generated_token_count")).alias("total_generated_tokens"),
    )
    .first()
)

human_elapsed_time = str(datetime.timedelta(seconds=inference_elapsed_time))

print("-" * 3 + " Input " + "-" * 3)
print(f"Total Article Count: {text_stats['articles_count']}")
print(f"Articles Token Count: {text_stats['total_article_tokens']}")
print(f"With Instructions Token Count: {text_stats['total_instruction_tokens']}")

print("\n" + "-" * 3 + " Output " + "-" * 3)
print(f"Generated Tokens Count: {text_stats['total_generated_tokens']}")
print(f"Inference Elapsed Seconds: {inference_elapsed_time}")
print(f"Inference Elapsed Time: {human_elapsed_time}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Results
# MAGIC

# COMMAND ----------

# Save Table
articles_df.write.mode("overwrite").saveAsTable("llm_batch_inference_results")
