# WildChat Dataset Filter

A Python script for filtering and deduplicating the WildChat conversation dataset.

## Overview

`filter_wildchat.py` provides comprehensive filtering and deduplication for WildChat datasets:

- **Quality Filtering**: Removes empty, toxic, spam, and low-quality conversations
- **Deduplication**: Three methods available (exact, MinHash LSH, SimHash)
- **Format Support**: Works with JSON and Parquet formats
- **Detailed Statistics**: Reports filtering results with breakdown by category

## Quick Start

### 1. Activate the virtual environment

```bash
source venv/bin/activate
# or use the convenience script
./activate_env.sh
```

### 2. Run the script

```bash
# Process with default settings (MinHash deduplication)
python filter_wildchat.py

# Test with first 1000 records only
python filter_wildchat.py --top-k 1000

# Use exact deduplication
python filter_wildchat.py --dedup-method exact

# Use MinHash with custom threshold (0.0-1.0, higher = more similar required)
python filter_wildchat.py --dedup-method minhash --dedup-threshold 0.9

# Use SimHash with Hamming distance threshold
python filter_wildchat.py --dedup-method simhash --dedup-threshold 5

# Custom input/output paths
python filter_wildchat.py -i /path/to/input -o /path/to/output.parquet
```

### 3. Deactivate when done

```bash
deactivate
```

## Features

### Quality Filters
- Empty conversations
- Empty user input
- User input < 3 characters
- Toxic content (based on dataset flags)
- Spam patterns (repetitive text, excessive punctuation, etc.)

### Deduplication Methods
1. **Exact**: MD5 hash-based exact matching
2. **MinHash**: LSH-based approximate matching (default, threshold=0.8)
3. **SimHash**: Hamming distance-based similarity (threshold=3)

### Output
- Filtered dataset (JSON or Parquet)
- Separate files for each filter category
- Detailed statistics report

## Requirements

See `requirements.txt`:
- datasketch (>=1.6.0)
- datasets (>=2.14.0)
- tqdm (>=4.65.0)

## Project Structure

```
.
├── filter_wildchat.py         # Main filtering script
├── requirements.txt            # Python dependencies
├── venv/                       # Virtual environment
├── activate_env.sh             # Quick activation script
├── ENVIRONMENT_SETUP.md        # Detailed setup guide
└── WildChat-1M/                # Dataset directory (not tracked)
```

## Documentation

- See `ENVIRONMENT_SETUP.md` for detailed environment setup and usage instructions
- Run `python filter_wildchat.py --help` for all command-line options

## Notes

- Default input path: `WildChat-1M/data`
- Default output: `wildchat_filtered.parquet`
- Filtered records are saved separately with reasons for filtering
- Use `--top-k` parameter for testing on smaller subsets
# datafiltering_wildchat
