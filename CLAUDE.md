# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project provides tools for filtering, deduplicating, clustering, and scoring the WildChat conversation dataset:

- **`process_wildchat.py`**: **Unified pipeline** that combines all phases into one script (recommended)
- **`filter_wildchat.py`**: Standalone filtering script for quality filters and deduplication
- **`cluster_wildchat.py`**: Standalone clustering script using embeddings and HDBSCAN
- **`score_wildchat.py`**: Standalone scoring script for evaluating conversations

## Environment Setup

Always activate the virtual environment before running commands:

```bash
source venv/bin/activate
# or use the convenience script
./activate_env.sh
```

Verify the environment is working:
```bash
# For filtering only
python -c "import datasketch, datasets, tqdm; print('Filtering environment OK')"

# For clustering (additional dependencies)
python -c "import transformers, torch, hdbscan; print('Clustering environment OK')"
```

## Running the Scripts

### Unified Pipeline (Recommended)

The `process_wildchat.py` script combines filtering, deduplication, embedding, clustering, sampling, and scoring into a single pipeline. It preserves **all input records** and adds columns for each phase.

**Output Schema (New Columns)**:
```
filter_passed: bool          # True if passed all quality + dedup checks
filter_reason: str | null    # Reason for filtering (null if passed)
embedding: list[float]       # Embedding vector (1536 dims for Azure)
cluster_id: int              # Cluster assignment (-1 for noise)
cluster_probability: float   # HDBSCAN probability
is_sampled: bool             # True if selected for scoring
score_difficulty: int | null # 1-5 scale (null for non-sampled)
score_creativity: int | null # 1-5 scale (null for non-sampled)
score_realism: int | null    # 1-5 scale (null for non-sampled)
score_reasoning: str | null  # LLM explanation (null for non-sampled)
```

```bash
# Basic usage (credentials from .env file)
source .env
python process_wildchat.py -i WildChat-1M/data -o processed.parquet

# Test on small subset first (recommended)
python process_wildchat.py -i input.parquet -o test.parquet --top-k 100 \
    --min-cluster-size 5 --min-samples 2 --samples-per-cluster 3

# Full pipeline with custom parameters
python process_wildchat.py -i WildChat-1M/data -o processed.parquet \
    --provider azure \
    --dedup-method minhash \
    --samples-per-cluster 10 \
    --parallelism 50

# Key command-line arguments:
#   --provider azure/local        Embedding provider (default: azure)
#   --dedup-method exact/minhash/simhash  Deduplication method (default: minhash)
#   --dedup-threshold 0.8         Similarity threshold for dedup
#   --min-cluster-size 50         HDBSCAN min cluster size
#   --min-samples 10              HDBSCAN min samples
#   --samples-per-cluster 5       Records to sample per cluster for scoring
#   --scoring-model gpt-5-nano    Model for scoring
#   --parallelism 30              Parallel API calls for scoring
#   --top-k N                     Process only first N records (testing)
```

### Filtering Script (Standalone)

```bash
# Test on small subset first (recommended for development)
python filter_wildchat.py --top-k 1000

# Full dataset processing with default MinHash deduplication
python filter_wildchat.py

# Use specific deduplication method
python filter_wildchat.py --dedup-method exact
python filter_wildchat.py --dedup-method minhash --dedup-threshold 0.9
python filter_wildchat.py --dedup-method simhash --dedup-threshold 5

# Custom input/output paths
python filter_wildchat.py -i /path/to/input -o output.parquet
```

### Clustering Script

**Recommended: Use Azure OpenAI** (fast, no GPU required)

```bash
# Basic usage with Azure OpenAI (credentials from .env file)
python cluster_wildchat.py -i wildchat_filtered.parquet -o wildchat_clustered.parquet --provider azure

# Test on small subset first (recommended)
python cluster_wildchat.py -i wildchat_filtered.parquet -o test_clustered.parquet --provider azure --top-k 1000

# Re-cluster with different parameters (reuses cached embeddings)
python cluster_wildchat.py -i wildchat_filtered.parquet -o clustered_v2.parquet \
    --provider azure --min-cluster-size 100 --use-cached-embeddings

# Full pipeline (filtering + clustering)
python filter_wildchat.py -i WildChat-1M/data -o filtered.parquet
python cluster_wildchat.py -i filtered.parquet -o clustered.parquet --provider azure

# With explicit Azure credentials
python cluster_wildchat.py -i filtered.parquet -o clustered.parquet --provider azure \
    --azure-endpoint https://your-endpoint.openai.azure.com/ \
    --azure-api-key YOUR_API_KEY
```

**Alternative: Local embeddings** (slow on CPU, requires GPU for reasonable speed)

```bash
# Local model (requires transformers and torch)
python cluster_wildchat.py -i filtered.parquet -o clustered.parquet --provider local

# Custom model and batch size (for GPU memory constraints)
python cluster_wildchat.py -i filtered.parquet -o clustered.parquet \
    --provider local --model Qwen/Qwen3-Embedding-0.6B --batch-size 16
```

## Architecture

### Two-Phase Filtering Pipeline

The script uses a sequential two-phase filtering approach:

**Phase 1: Quality Filtering** (`is_valid_conversation`)
- Filters are applied in sequence, with early termination on first match
- Order matters: empty checks before content analysis
- Each rejected record is tagged with a `filter_reason`
- Categories: empty, empty_user_input, too_short_user_input, toxic, spam_pattern

**Phase 2: Deduplication** (three methods available)
- Applied only to records that passed Phase 1
- All three methods use the same conversation comparison logic (see below)

### Conversation Comparison Logic

**Critical design decision:** Deduplication compares conversations based on everything EXCEPT the final assistant response. This treats the last response as the "evaluation target" and the rest as the "evaluation scenario."

Implementation in two functions:
- `get_conversation_fingerprint()` - Returns MD5 hash for exact matching
- `get_conversation_text_for_dedup()` - Returns raw text for similarity algorithms

Both functions:
1. Include ALL user messages
2. Include ALL assistant messages EXCEPT the last one
3. Preserve message order
4. Include conversation length

### Deduplication Methods

**Exact** (`deduplicate_exact`):
- MD5 hash-based matching using `get_conversation_fingerprint()`
- O(n) time complexity with hash set lookups
- Fastest, most memory-efficient

**MinHash LSH** (`deduplicate_minhash`):
- Uses datasketch library with LSH indexing
- Threshold: Jaccard similarity (0.0-1.0, default 0.8)
- Creates word tokens + 3-character n-grams for better similarity detection
- num_perm=128 (fixed)
- Approximate matching with O(n) average case via LSH

**SimHash** (`deduplicate_simhash`):
- Custom implementation using bit vectors
- Threshold: Hamming distance (integer, default 3)
- Same tokenization as MinHash (words + 3-grams)
- hash_bits=64 (fixed)
- O(n²) comparison in current implementation (no LSH optimization)

### Data Format Support

Input/Output format detection:
- Auto-detects from file extension (.json or .parquet) or directory structure
- Uses Hugging Face `datasets` library for Parquet I/O
- JSON uses standard library with custom encoder for datetime/bytes

### Output Structure

Main output:
- Filtered dataset in JSON or Parquet format

Filtered-out records:
- Separate JSON files per filter category: `{output_stem}_filtered_{reason}.json`
- Each record includes original data + `filter_reason` field
- Allows audit and analysis of filtering decisions

### Spam Pattern Detection

The `is_spam_pattern()` function uses five heuristics:
1. Character repetition (10+ consecutive identical characters)
2. Word repetition (single word >60% of content in texts >10 words)
3. Pure punctuation/symbols (no alphanumeric content, >10 chars)
4. Excessive punctuation strings (10+ consecutive punctuation marks)
5. Repetitive instruction lines (checks line similarity in long texts >2000 chars)

## Data Structure Assumptions

The script expects WildChat format:
```python
{
  "conversation": [
    {"role": "user", "content": "...", "toxic": false},
    {"role": "assistant", "content": "..."},
    # ... more turns
  ],
  # ... other fields
}
```

Key fields:
- `conversation`: List of message dictionaries
- `role`: "user" or "assistant"
- `content`: Message text
- `toxic`: Optional boolean flag (checked per message)

## Default Paths

- Default input: `/Users/qiushuo/wsp/datafiltering/WildChat-1M/data`
- Default output: `/Users/qiushuo/wsp/datafiltering/wildchat_filtered.parquet`

Update the defaults in `filter_wildchat.py:608-614` if working with different paths regularly.

## Performance Considerations

### Filtering Performance

- Use `--top-k` for development/testing to avoid processing full dataset
- MinHash LSH is the default for good balance of speed and accuracy
- Exact deduplication is fastest but misses near-duplicates
- SimHash has O(n²) complexity - slow on large datasets without LSH optimization
- Parquet format is more efficient than JSON for large datasets

### Clustering Performance

#### Azure OpenAI (Recommended)
- **Fast**: ~5-10 minutes for WildChat-1M (~850k records)
- **No GPU required**: Runs on any machine with internet connection
- **Cost**: ~$0.02 per 1M tokens (~$0.50 for WildChat-1M)
- **Embedding dimension**: 1536 (text-embedding-3-small)
- **Batch size**: 100 texts per API call (default)
- Use `--provider azure` and set credentials in `.env` file

#### Local Embeddings (Alternative)
- **GPU highly recommended**: Embedding generation is ~20-30x faster on GPU than CPU
- For WildChat-1M (~850k records after filtering):
  - **With GPU (RTX 3090)**: ~45 minutes total (30 min embedding + 15 min clustering)
  - **CPU only**: ~8-12 hours (very slow, not recommended)
  - **Azure OpenAI**: ~5-10 minutes (fastest, recommended)
- Memory requirements: ~8GB RAM + 4-6GB VRAM (local only)
- Use `--top-k 1000` for quick testing before running on full dataset
- Embedding cache speeds up parameter tuning (use `--use-cached-embeddings`)
- Reduce `--batch-size` if encountering GPU OOM errors (local only)

## Clustering Architecture

### Overview

`cluster_wildchat.py` is a standalone script that processes filtered datasets to group similar conversations into clusters using semantic embeddings and HDBSCAN clustering.

### Text for Clustering

Uses `get_conversation_text_for_dedup()` logic (copied from `filter_wildchat.py`):
- Includes ALL user messages
- Includes ALL assistant messages EXCEPT the last one
- Aligns with filtering philosophy: treats last response as "evaluation target", rest as "evaluation scenario"
- Clusters conversations by context rather than by final response

### Embedding Generation

Two providers supported:

#### Azure OpenAI (Recommended)
**Model**: text-embedding-3-small (default for Azure)
- Fast cloud-based API (no local compute required)
- Embedding dimension: 1536
- Max sequence length: 8191 tokens
- Batch processing: 100 texts per API call (default)
- Automatic retry with exponential backoff
- Credentials from environment variables or CLI args

**Setup**:
Create a `.env` file with Azure credentials:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key-here"
```

Then load it before running:
```bash
source .env
python cluster_wildchat.py -i input.parquet -o output.parquet --provider azure
```

#### Local Model (Alternative)
**Model**: Qwen/Qwen3-Embedding-0.6B (default for local)
- HuggingFace transformers-based model
- Max sequence length: 8192 tokens (model limit)
- Mean pooling over token embeddings (excluding padding)
- Batch processing for efficiency (default batch_size=32)
- Automatic GPU/CPU detection with performance warnings
- Requires: transformers, torch packages

**Caching** (Both Providers):
- Embeddings saved to `{output_stem}_embeddings.npy`
- Metadata saved to `{output_stem}_embeddings_metadata.json`
- Cache validated by input hash and model name
- Use `--use-cached-embeddings` to skip regeneration during parameter tuning

**Error Handling**:
- Automatic batch size reduction on GPU OOM
- Empty texts assigned "[empty]" placeholder
- Comprehensive progress bars via tqdm

### HDBSCAN Clustering

**Default Configuration**:
```python
min_cluster_size = 50      # Meaningful clusters for large datasets
min_samples = 10           # Conservative to reduce noise sensitivity
metric = 'euclidean'       # Standard for dense embeddings
cluster_selection_method = 'eom'  # Excess of Mass (default)
```

**Tuning Guidelines**:
- **Increase `min_cluster_size`** (e.g., 100-200) for fewer, larger clusters
- **Decrease `min_cluster_size`** (e.g., 20-30) for more granular clusters
- **Increase `min_samples`** to reduce noise points (more conservative)
- **Decrease `min_samples`** to allow more borderline points into clusters
- For smaller datasets (<10k records), scale down to `min_cluster_size=15-20`

**Validation Metrics**:
- DBCV (Density-Based Cluster Validation): Higher is better (>0.5 is good)
- Noise percentage: Lower is generally better (<20% is reasonable)
- Cluster size distribution: Should follow power law (few large, many small)

### Output Format

**Primary output**: Same format as input (Parquet/JSON) with added fields:
- `cluster_id`: Integer cluster ID (-1 for noise/outliers)
- `cluster_probability`: HDBSCAN probability score (0-1)
  - Higher values indicate stronger cluster membership
  - Low values suggest borderline membership

**Statistics file**: `{output_stem}_cluster_stats.json`
```json
{
  "clustering_parameters": {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "min_cluster_size": 50,
    "min_samples": 10,
    "metric": "euclidean",
    "cluster_selection_method": "eom"
  },
  "statistics": {
    "num_clusters": 156,
    "num_noise": 42891,
    "noise_percentage": 5.04,
    "dbcv_score": 0.6234
  },
  "cluster_size_distribution": { ... },
  "top_20_clusters": [ ... ]
}
```

### Command-Line Interface

**Required Arguments**:
- `-i, --input PATH`: Input parquet/json from filter_wildchat.py
- `-o, --output PATH`: Output file with cluster_id added

**Model Arguments**:
- `--model MODEL`: HuggingFace model (default: Qwen/Qwen3-Embedding-0.6B)
- `--batch-size N`: Batch size for embeddings (default: 32)
- `--max-length N`: Max sequence length (default: 8192)
- `--device DEVICE`: cuda/cpu/auto (default: auto)
- `--use-cached-embeddings`: Skip embedding generation if cached

**Clustering Arguments**:
- `--min-cluster-size N`: Minimum cluster size (default: 50)
- `--min-samples N`: Minimum samples (default: 10)
- `--metric METRIC`: Distance metric (default: euclidean)
- `--cluster-selection-method METHOD`: eom or leaf (default: eom)

**Other Arguments**:
- `--top-k N`: Test on first k records only
- `--seed N`: Random seed (default: 42)

## Workflow Examples

### Basic Filtering + Clustering Pipeline (Azure OpenAI)

```bash
# Step 0: Set up Azure credentials
source .env  # Contains AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY

# Step 1: Filter and deduplicate
python filter_wildchat.py -i WildChat-1M/data -o filtered.parquet

# Step 2: Cluster filtered data with Azure OpenAI
python cluster_wildchat.py -i filtered.parquet -o clustered.parquet --provider azure
```

### Development/Testing Workflow

```bash
# Set up credentials
source .env

# Quick test on small subset
python filter_wildchat.py --top-k 1000 -o test_filtered.parquet
python cluster_wildchat.py -i test_filtered.parquet -o test_clustered.parquet \
    --provider azure --top-k 100 --min-cluster-size 5

# Inspect results before full run
python -c "
from datasets import load_dataset
ds = load_dataset('parquet', data_files='test_clustered.parquet')['train']
print(f'Records: {len(ds)}')
print(f'Cluster IDs: {set(ds[\"cluster_id\"])}')
"
```

### Parameter Tuning Workflow

```bash
# Set up credentials
source .env

# Generate embeddings once with Azure
python cluster_wildchat.py -i filtered.parquet -o v1.parquet \
    --provider azure --min-cluster-size 50

# Try different clustering parameters (reuses embeddings)
python cluster_wildchat.py -i filtered.parquet -o v2.parquet \
    --provider azure --min-cluster-size 100 --use-cached-embeddings

python cluster_wildchat.py -i filtered.parquet -o v3.parquet \
    --provider azure --min-cluster-size 30 --min-samples 5 --use-cached-embeddings

# Compare cluster statistics
cat v1_cluster_stats.json
cat v2_cluster_stats.json
cat v3_cluster_stats.json
```

## Unified Pipeline Architecture

### Overview

`process_wildchat.py` is a unified script that runs all phases in sequence, preserving all input records.

### Phases

1. **Load data**: From JSON, Parquet file, or directory of Parquet files
2. **Quality filtering**: Adds `filter_passed` and `filter_reason` to ALL records (no removal)
3. **Deduplication**: Updates `filter_passed`/`filter_reason` for duplicates (no removal)
4. **Embedding generation**: Generates embeddings for ALL records, stores in `embedding` column
5. **HDBSCAN clustering**: Clusters ALL records, adds `cluster_id` and `cluster_probability`
6. **Sampling**: Selects N records per cluster from filtered records, adds `is_sampled`
7. **Scoring**: Scores only `is_sampled=True` records, adds `score_*` columns (null for others)
8. **Save**: Complete dataset with all original and new columns

### Key Design Decisions

- **Non-destructive**: Input row count = Output row count (records marked, not removed)
- **Embeddings for all**: ALL records get embeddings, enabling clustering of entire dataset
- **Selective scoring**: Only sampled records are scored (saves API costs)
- **Statistics file**: Generates `{output}_stats.json` with filtering/clustering/scoring metrics

### Output Files

- `{output}.parquet`: Complete dataset with all columns
- `{output}_stats.json`: Statistics summary for all phases

## Troubleshooting

### Clustering Issues

**Azure OpenAI Issues**:

**Credentials not found**:
- Ensure `.env` file exists with `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`
- Source the file: `source .env`
- Or pass explicitly: `--azure-endpoint URL --azure-api-key KEY`

**API rate limits**:
- Reduce `--batch-size` (default: 100 for Azure)
- Script has automatic retry with exponential backoff
- Check Azure portal for quota limits

**API timeout**:
- Check internet connection
- Verify endpoint URL is correct
- Check Azure service status

**Local Model Issues**:

**GPU Out of Memory**:
- Reduce `--batch-size` (try 16, 8, 4)
- Use smaller model (though Qwen3-0.6B is already compact)
- Or switch to `--provider azure` (no GPU needed)

**Clustering takes too long on CPU**:
- Use `--provider azure` (fastest, recommended)
- Or use GPU machine
- Or reduce dataset size with `--top-k`

**Too many noise points (>30%)**:
- Increase `--min-samples` (more conservative)
- Decrease `--min-cluster-size` (allow smaller clusters)
- Check DBCV score - low score suggests poor fit

**Too few/many clusters**:
- Adjust `--min-cluster-size`:
  - Increase for fewer, larger clusters
  - Decrease for more, smaller clusters
- Consider dataset size when setting parameters

**Cache issues**:
- Delete `*_embeddings.npy` and `*_embeddings_metadata.json` to regenerate
- Or ensure `--use-cached-embeddings` is set when reusing embeddings
