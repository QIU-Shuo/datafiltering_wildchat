"""Sample a subset of WildChat-1M for testing."""
import sys
from datasets import load_dataset

# Load only first 1000 records
print("Loading WildChat-1M dataset (first 1000 records)...")
dataset = load_dataset(
    'parquet',
    data_dir='/Users/qiushuo/wsp/datafiltering/WildChat-1M/data',
    split='train[:1000]'
)

print(f"Loaded {len(dataset)} records")

# Save to test directory
output_path = '/Users/qiushuo/wsp/datafiltering/test_data/wildchat_sample.parquet'
print(f"Saving to {output_path}...")
dataset.to_parquet(output_path)

print(f"✓ Done! Saved {len(dataset)} records")
