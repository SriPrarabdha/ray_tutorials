"""
High-throughput distributed query pipeline using Ray Data and vLLM.

What it does:
- Streams text files from S3 (input prefix)
- Applies text preprocessing
- Sends prompts to vLLM (Qwen-3) in batches for inference
- Writes generated output back to S3, keeping filename mapping

Adjust the VLLM_API_URL, S3 paths, batching sizes, and preprocessing as needed.
"""

import os
import json
import time
from typing import List, Dict

import boto3
import s3fs
import requests
import ray
import ray.data as rd
import pandas as pd

# ---------- CONFIG ----------
# S3 input and output prefixes
S3_BUCKET = "s3-bucket"
INPUT_PREFIX = "input_files/"
OUTPUT_PREFIX = "output_files/"

# vLLM server endpoint (assumed OpenAI-like /v1/generate or a simple POST endpoint)
# Example: "http://vllm-server-host:8080/v1/generate"
VLLM_API_URL = os.environ.get("VLLM_API_URL", "http://127.0.0.1:8080/v1/generate")

# Batch sizes
BATCH_SIZE_FILES = 16           # how many files to group into a batch for one map_batches call
PROMPTS_PER_BATCH = 8           # how many prompts to send in one model request (inside each batch)
MODEL_REQUEST_TIMEOUT = 60      # seconds

# Ray and S3 settings
RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "auto")  # use "auto" for connecting to existing cluster
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# ---------- HELPERS ----------
def s3_input_paths(bucket: str, prefix: str) -> List[str]:
    """List all s3:// paths under bucket/prefix."""
    fs = s3fs.S3FileSystem(anon=False)
    s3_path = f"{bucket}/{prefix}"
    # s3fs list returns relative paths inside bucket if given "bucket/prefix"
    found = fs.glob(f"{s3_path}*")
    # convert to s3:// format
    return [f"s3://{p}" if not p.startswith("s3://") else p for p in found]

def read_text_from_s3_path(s3_path: str) -> str:
    """Read a whole text file from S3 path."""
    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(s3_path, 'r') as f:
        return f.read()

def write_text_to_s3(s3_bucket: str, key: str, content: str):
    """Write text to S3 (overwrites). key is path under bucket (no leading slash)."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(Bucket=s3_bucket, Key=key, Body=content.encode("utf-8"))

# ---------- Preprocessing ----------
def preprocess_text(text: str) -> str:
    """Custom preprocessing for prompt creation.
       Example: simple normalization and join lines. Replace with your own logic.
    """
    # basic: strip, collapse spaces, remove extra blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    normalized = " ".join(lines)
    # Optionally build a prompt wrapper for the model:
    prompt = f"Perform the requested transformation on the following text and output the result clearly:\n\n{normalized}\n\nOutput:"
    return prompt

# ---------- vLLM model call (HTTP) ----------
def call_vllm_batch(prompts: List[str], api_url: str = VLLM_API_URL, timeout: int = MODEL_REQUEST_TIMEOUT) -> List[str]:
    """
    Sends a batch of prompts to the vLLM server and returns list of generated strings.
    This function assumes the vLLM server accepts POST JSON:
    {
      "inputs": ["prompt1", "prompt2", ...],
      "max_new_tokens": 256,
      "temperature": 0.0
    }
    and responds with:
    {"outputs": ["gen1", "gen2", ...]}
    Modify according to your vLLM server API.
    """
    payload = {
        "inputs": prompts,
        "max_new_tokens": 512,
        "temperature": 0.0,
        # Add other vLLM-specific kwargs if available
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # try a few common response formats:
    if "outputs" in data:
        return data["outputs"]
    if "generations" in data:  # sometimes called "generations"
        # could be list of lists
        gens = data["generations"]
        # if inner structure has 'text'
        out = []
        for g in gens:
            if isinstance(g, list):
                out.append(g[0].get("text") if isinstance(g[0], dict) else str(g[0]))
            elif isinstance(g, dict):
                out.append(g.get("text", ""))
            else:
                out.append(str(g))
        return out
    # Fallback: try to interpret 'choices' like OpenAI
    if "choices" in data:
        return [ch.get("text", "") for ch in data["choices"]]
    # If unknown format, try heuristics:
    # If top-level is a single string, replicate (not likely)
    if isinstance(data, str):
        return [data] * len(prompts)
    raise RuntimeError(f"Unexpected vLLM response format: {data}")

# ---------- Ray map_batches worker function ----------
def process_batch_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input `df` columns expected:
      - 'path' : original s3 path string, e.g. s3://bucket/input_files/file_1.txt
      - 'text' : full text content
    Returns a pandas DataFrame with columns:
      - 'out_path' : s3 output key under bucket (e.g. output_files/file_1.txt)
      - 'generated' : model output
    This will run on Ray workers.
    """
    results = []
    prompts = []
    rows = []

    # 1) preprocess each row into a prompt
    for _, row in df.iterrows():
        original_path = row["path"]
        text = row["text"]
        prompt = preprocess_text(text)
        prompts.append(prompt)
        rows.append(original_path)

    # 2) send prompts to vLLM in *chunks* to limit request size
    CHUNK = PROMPTS_PER_BATCH
    all_generated = []
    for i in range(0, len(prompts), CHUNK):
        chunk_prompts = prompts[i:i+CHUNK]
        try:
            generated_chunk = call_vllm_batch(chunk_prompts)
        except Exception as e:
            # basic retry logic on failure
            time.sleep(1.0)
            generated_chunk = call_vllm_batch(chunk_prompts)
        all_generated.extend(generated_chunk)

    # 3) map to out_path (mirror filename into OUTPUT_PREFIX)
    for orig_path, gen in zip(rows, all_generated):
        # orig_path like s3://bucket/input_files/file_1.txt
        # extract filename and build new key
        # If input path used a directory structure to preserve, you might adapt this logic.
        filename = orig_path.split("/")[-1]
        out_key = f"{OUTPUT_PREFIX}{filename}"
        results.append({"out_key": out_key, "generated": gen})

    out_df = pd.DataFrame(results)
    return out_df

def write_outputs_batch(df: pd.DataFrame):
    """
    Each row has: out_key, generated
    This function writes each generated string to s3://S3_BUCKET/out_key
    Runs on workers and uses boto3 (ensure credentials available to workers).
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)
    for _, row in df.iterrows():
        key = row["out_key"]
        content = row["generated"]
        # put_object
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=content.encode("utf-8"))

# ---------- Orchestrator (main) ----------
def run_pipeline():
    # 1) connect to Ray
    if RAY_ADDRESS == "auto":
        ray.init(address="auto")
    else:
        ray.init(address=RAY_ADDRESS)

    # 2) list all input files (S3 paths)
    s3_paths = s3_input_paths(S3_BUCKET, INPUT_PREFIX)
    if not s3_paths:
        raise SystemExit(f"No input files found under s3://{S3_BUCKET}/{INPUT_PREFIX}")

    # 3) create a Ray dataset of (path, text) by reading each file in parallel
    # We'll create a small pandas DataFrame for file paths and then map to read content so that read happens on workers
    df_files = pd.DataFrame({"path": s3_paths})

    ds = rd.from_pandas(df_files, parallelism= max(1, len(s3_paths)//2))

    # Map to read file contents on workers to avoid driver bottleneck
    def _read_file_rows(batch: pd.DataFrame) -> pd.DataFrame:
        contents = []
        for p in batch["path"].tolist():
            text = read_text_from_s3_path(p)
            contents.append({"path": p, "text": text})
        return pd.DataFrame(contents)

    ds = ds.map_batches(_read_file_rows, batch_size=1, batch_format="pandas")

    # 4) Process in batches: call model and produce out_key + generated text
    # Using map_batches with a larger batch size to increase throughput
    processed_ds = ds.map_batches(process_batch_pandas, batch_size=BATCH_SIZE_FILES, batch_format="pandas")

    # 5) Write outputs to S3 in parallel
    # Use map_batches to write; the write function uses boto3 to put objects
    processed_ds.map_batches(lambda pdf: (write_outputs_batch(pdf), pdf)[1], batch_size=64, batch_format="pandas").show(5)


if __name__ == "__main__":
    run_pipeline()
