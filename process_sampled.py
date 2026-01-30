#!/usr/bin/env python3
"""
Process Pre-Sampled WildChat Data

Takes data from sample_data.py (with embeddings and scores already computed)
and applies filtering, deduplication, and clustering phases.

Input: Parquet file with 'embedding' and 'score_*' columns from sample_data.py
Output: Parquet file with additional columns for filtering and clustering

Output Schema (New Columns):
- filter_passed: bool        - True if passed all quality + dedup checks
- filter_reason: str | null  - Reason for filtering (null if passed)
- cluster_id: int            - Cluster assignment (-1 for noise)
- cluster_probability: float - HDBSCAN probability
- is_sampled: bool           - True if selected (1 per cluster + noise to fill quota)
"""

import json
import argparse
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from datasketch import MinHash, MinHashLSH
import hdbscan

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Text Extraction Functions
# =============================================================================

def get_user_input_text(conversation):
    """Extract all user input text from a conversation."""
    if not conversation:
        return ""

    text_parts = []
    for message in conversation:
        if isinstance(message, dict) and message.get('role') == 'user':
            content = message.get('content', '')
            if isinstance(content, str):
                text_parts.append(content)

    return " ".join(text_parts)


def get_conversation_text_for_dedup(record):
    """
    Extract text for embedding/deduplication.
    Includes all messages except the last assistant response.
    """
    conversation = record.get('conversation', [])

    messages = []
    for msg in conversation:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
            messages.append({'role': role, 'content': content})

    text_parts = []
    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content']

        if role == 'user':
            text_parts.append(content)
        elif role == 'assistant':
            is_last_message = (i == len(messages) - 1)
            if not is_last_message:
                text_parts.append(content)

    return " ".join(text_parts)


# =============================================================================
# Quality Filtering Functions (Non-destructive)
# =============================================================================

def is_spam_pattern(text):
    """Detect common spam patterns."""
    if not text:
        return False

    lower_text = text.lower().strip()

    # Pattern 1: Excessive repetition of same character
    if re.search(r'(.)\1{10,}', lower_text):
        return True

    # Pattern 2: Excessive repetition of same word
    words = lower_text.split()
    if len(words) > 10:
        word_counts = {}
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] / len(words) > 0.6:
                    return True

    # Pattern 3: Only special characters or numbers
    if re.match(r'^[^\w]+$', lower_text, re.UNICODE) and len(lower_text) > 10:
        return True

    # Pattern 4: Excessive use of same punctuation
    if re.search(r'[!?.,]{10,}', text):
        return True

    # Pattern 5: Very long repetitive instructions
    lines = text.split('\n')
    if len(text) > 2000 and len(lines) > 20:
        similar_lines = 0
        for i in range(len(lines) - 1):
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[i].strip() and len(lines[i]) > 20:
                    words_i = set(lines[i].lower().split())
                    words_j = set(lines[j].lower().split())
                    if len(words_i) > 0 and len(words_j) > 0:
                        similarity = len(words_i & words_j) / max(len(words_i), len(words_j))
                        if similarity > 0.7:
                            similar_lines += 1
        if similar_lines > len(lines) * 0.3:
            return True

    return False


def check_quality(record) -> Tuple[bool, Optional[str]]:
    """
    Check if a conversation record passes quality filters.

    Returns:
        Tuple of (passed, reason) where reason is None if passed
    """
    conversation = record.get('conversation', [])

    # Check if conversation is empty
    if not conversation:
        return False, "empty"

    # Check for toxic messages
    for message in conversation:
        if isinstance(message, dict) and message.get('toxic', False):
            return False, "toxic"

    # Get user input text
    user_text = get_user_input_text(conversation)

    # Check if user input is empty
    if not user_text or not user_text.strip():
        return False, "empty_user_input"

    # Check if user input is too short
    if len(user_text.strip()) < 3:
        return False, "too_short_user_input"

    # Check for spam patterns
    if is_spam_pattern(user_text):
        return False, "spam_pattern"

    return True, None


def apply_quality_filters(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply quality filters to all records (non-destructive).
    Adds filter_passed and filter_reason to each record.

    Returns:
        Same list with added fields (no records removed)
    """
    print("Phase 1: Applying quality filters...")

    stats = defaultdict(int)

    for record in tqdm(data, desc="Quality filtering"):
        passed, reason = check_quality(record)
        record['filter_passed'] = passed
        record['filter_reason'] = reason

        if passed:
            stats['passed'] += 1
        else:
            stats[reason] += 1

    # Print statistics
    print(f"  Passed quality filters: {stats['passed']:,}")
    print(f"  Failed:")
    for reason in ['empty', 'empty_user_input', 'too_short_user_input', 'toxic', 'spam_pattern']:
        if stats[reason] > 0:
            print(f"    - {reason}: {stats[reason]:,}")

    return data


# =============================================================================
# Deduplication Functions (Non-destructive)
# =============================================================================

def create_minhash(text, num_perm=128):
    """Create MinHash signature for text."""
    m = MinHash(num_perm=num_perm)

    words = text.lower().split()
    ngrams = []
    for word in words:
        if len(word) >= 3:
            for i in range(len(word) - 2):
                ngrams.append(word[i:i+3])

    tokens = words + ngrams
    for token in tokens:
        m.update(token.encode('utf-8'))

    return m


def apply_deduplication(
    data: List[Dict[str, Any]],
    threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Apply MinHash LSH deduplication to records that passed quality filters (non-destructive).
    Updates filter_passed and filter_reason for duplicates.

    Args:
        data: List of records with filter_passed field
        threshold: Jaccard similarity threshold (0.0-1.0, default 0.8)

    Returns:
        Same list with updated fields (no records removed)
    """
    print(f"Phase 2: Applying MinHash deduplication (threshold={threshold})...")

    duplicates = 0
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    for i, record in enumerate(tqdm(data, desc="MinHash dedup")):
        if not record['filter_passed']:
            continue

        text = get_conversation_text_for_dedup(record)
        if len(text.strip()) < 10:
            continue

        minhash = create_minhash(text, num_perm=128)
        key = f"record_{i}"
        similar = lsh.query(minhash)

        if similar:
            record['filter_passed'] = False
            record['filter_reason'] = 'duplicate_minhash'
            duplicates += 1
        else:
            lsh.insert(key, minhash)

    passed = sum(1 for r in data if r['filter_passed'])
    print(f"  Duplicates found: {duplicates:,}")
    print(f"  Records passing all filters: {passed:,}")

    return data


# =============================================================================
# Clustering Functions
# =============================================================================

def apply_clustering(
    data: List[Dict[str, Any]],
    min_cluster_size: int = 2,
    min_samples: int = 2,
    metric: str = 'euclidean',
    cluster_selection_method: str = 'eom'
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply HDBSCAN clustering to filtered records only using their embeddings.

    Args:
        data: List of records with embedding field
        min_cluster_size: Minimum cluster size (default: 2)
        min_samples: Minimum samples (default: 2)
        metric: Distance metric for HDBSCAN
        cluster_selection_method: 'eom' or 'leaf'

    Returns:
        Tuple of (data with cluster fields, clustering stats)
    """
    print("Phase 3: Clustering filtered records...")
    print(f"  min_cluster_size: {min_cluster_size}")
    print(f"  min_samples: {min_samples}")

    # Extract embeddings ONLY from filtered records
    filtered_indices = [i for i, r in enumerate(data) if r.get('filter_passed', False)]
    filtered_embeddings = np.array([data[i]['embedding'] for i in filtered_indices])

    print(f"  Clustering {len(filtered_indices):,} filtered records (excluding {len(data) - len(filtered_indices):,} filtered-out)")

    # Run HDBSCAN on filtered records only
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1
    )

    cluster_labels = clusterer.fit_predict(filtered_embeddings)
    probabilities = clusterer.probabilities_

    # Initialize ALL records with noise cluster
    for record in data:
        record['cluster_id'] = -1
        record['cluster_probability'] = 0.0

    # Assign cluster info ONLY to filtered records
    for idx, (label, prob) in enumerate(zip(cluster_labels, probabilities)):
        original_idx = filtered_indices[idx]
        data[original_idx]['cluster_id'] = int(label)
        data[original_idx]['cluster_probability'] = float(prob)

    # Compute statistics
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels[unique_labels >= 0])
    num_noise = int(np.sum(cluster_labels == -1))

    try:
        dbcv_score = float(clusterer.relative_validity_)
    except AttributeError:
        dbcv_score = None

    stats = {
        'num_clusters': num_clusters,
        'num_noise': num_noise,
        'noise_percentage': num_noise / len(cluster_labels) * 100,
        'dbcv_score': dbcv_score
    }

    print(f"  Clusters: {num_clusters}")
    print(f"  Noise points: {num_noise:,} ({stats['noise_percentage']:.2f}%)")
    if dbcv_score is not None:
        print(f"  DBCV score: {dbcv_score:.4f}")

    return data, stats


# =============================================================================
# Sampling Functions
# =============================================================================

def compute_combined_score(record: Dict[str, Any]) -> float:
    """Compute combined score from individual score dimensions."""
    scores = []
    for dim in ['difficulty', 'creativity', 'realism']:
        score = record.get(f'score_{dim}')
        if score is not None:
            scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def apply_sampling(
    data: List[Dict[str, Any]],
    total_samples: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply new sampling strategy after clustering:
    1. Pick 10% from noise points (best scored)
    2. Rank clusters by size (large to small), pick best from each until target reached
    3. If not enough after one pass, sample more from remaining noise
    4. If still not enough, repeat cluster sampling (2nd best, 3rd best, etc.)

    Only considers records that passed filtering (filter_passed=True).
    Filtered records (filter_passed=False) are always excluded.

    Args:
        data: List of records with cluster_id and filter_passed fields
        total_samples: Total number of samples to select (required)

    Returns:
        Tuple of (data with is_sampled field, sampling stats)
    """
    print("Phase 4: Sampling records...")
    print(f"  Target dataset size: {total_samples:,}")

    # Initialize is_sampled to False for all records
    for record in data:
        record['is_sampled'] = False

    # Group filtered records by cluster
    clusters = defaultdict(list)
    noise_records = []

    for i, record in enumerate(data):
        # Always exclude filtered-out records
        if not record.get('filter_passed', False):
            continue

        cluster_id = record.get('cluster_id', -1)
        if cluster_id == -1:
            noise_records.append(i)
        else:
            clusters[cluster_id].append(i)

    print(f"  Available: {len(clusters)} clusters, {len(noise_records):,} noise points")

    # Sort noise by score (descending)
    noise_records_sorted = sorted(
        noise_records,
        key=lambda i: compute_combined_score(data[i]),
        reverse=True
    )

    # Sort clusters by size (largest to smallest)
    clusters_sorted = sorted(
        clusters.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    # Pre-sort records within each cluster by score (descending)
    cluster_records_sorted = {}
    for cluster_id, indices in clusters_sorted:
        cluster_records_sorted[cluster_id] = sorted(
            indices,
            key=lambda i: compute_combined_score(data[i]),
            reverse=True
        )

    sampled_indices = set()
    noise_sampled = 0
    cluster_sampled = 0

    # Step 1: Pick 10% from noise (best scored)
    noise_quota = int(0.1 * total_samples)
    noise_quota = min(noise_quota, len(noise_records_sorted))  # Can't sample more than available

    for idx in noise_records_sorted[:noise_quota]:
        data[idx]['is_sampled'] = True
        sampled_indices.add(idx)
        noise_sampled += 1

    print(f"  Step 1: Selected {noise_sampled:,} from noise (10% quota)")

    # Remaining noise for later use
    remaining_noise = noise_records_sorted[noise_quota:]
    remaining_noise_idx = 0  # Track position in remaining noise

    # Step 2-4: Pick from clusters in rounds, with noise fill after first round
    round_num = 0
    max_cluster_size = max(len(records) for records in cluster_records_sorted.values()) if cluster_records_sorted else 0

    while len(sampled_indices) < total_samples and round_num < max_cluster_size:
        made_progress = False

        # Go through clusters from largest to smallest
        for cluster_id, _ in clusters_sorted:
            if len(sampled_indices) >= total_samples:
                break

            # Try to pick the round_num-th best record from this cluster
            cluster_records = cluster_records_sorted[cluster_id]
            if round_num < len(cluster_records):
                idx = cluster_records[round_num]
                if idx not in sampled_indices:
                    data[idx]['is_sampled'] = True
                    sampled_indices.add(idx)
                    cluster_sampled += 1
                    made_progress = True

        # After first complete pass through all clusters
        if round_num == 0 and len(sampled_indices) < total_samples:
            # Step 3: Sample more from remaining noise
            needed = min(total_samples - len(sampled_indices), len(remaining_noise) - remaining_noise_idx)
            noise_to_add = remaining_noise[remaining_noise_idx:remaining_noise_idx + needed]

            for idx in noise_to_add:
                if idx not in sampled_indices:
                    data[idx]['is_sampled'] = True
                    sampled_indices.add(idx)
                    noise_sampled += 1

            remaining_noise_idx += needed
            print(f"  Step 2: Selected {needed:,} additional from remaining noise")

        # Check if we reached target or can't make progress
        if len(sampled_indices) >= total_samples:
            break

        if not made_progress:
            break

        round_num += 1

    print(f"  Step 3: Selected {cluster_sampled:,} from clusters ({round_num + 1} rounds)")

    # Final summary
    total_sampled = len(sampled_indices)

    if total_sampled < total_samples:
        print(f"  Warning: Could not reach target size {total_samples:,}, sampled {total_sampled:,} records")

    stats = {
        'total_sampled': total_sampled,
        'from_clusters': cluster_sampled,
        'from_noise': noise_sampled,
        'num_clusters': len(clusters),
        'total_noise_available': len(noise_records_sorted),
        'sampling_rounds': round_num + 1,
        'target_size': total_samples
    }

    print(f"  Total sampled: {total_sampled:,} ({cluster_sampled:,} from clusters, {noise_sampled:,} from noise)")

    return data, stats


# =============================================================================
# I/O Functions
# =============================================================================

def load_data(input_file: str) -> List[Dict[str, Any]]:
    """Load data from Parquet file."""
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading data from {input_file}...")
    dataset = load_dataset('parquet', data_files=input_file)['train']
    data = [dict(record) for record in dataset]

    print(f"Loaded {len(data)} records")

    # Verify required columns exist
    if data:
        required_cols = ['embedding', 'score_difficulty', 'score_creativity', 'score_realism']
        missing = [col for col in required_cols if col not in data[0]]
        if missing:
            raise ValueError(f"Input file missing required columns: {missing}")

    return data


def save_data(data: List[Dict[str, Any]], output_file: str):
    """Save data to Parquet file."""
    output_path = Path(output_file)

    print(f"\nSaving data to {output_file}...")

    dataset = Dataset.from_list(data)
    dataset.to_parquet(output_file)

    print(f"Successfully saved {len(data):,} records to {output_file}")


def save_statistics(
    data: List[Dict[str, Any]],
    cluster_stats: Dict[str, Any],
    sampling_stats: Dict[str, Any],
    output_file: str
):
    """Save pipeline statistics to JSON file."""
    output_path = Path(output_file)
    stats_file = output_path.parent / f"{output_path.stem}_stats.json"

    # Compute filter statistics
    filter_stats = defaultdict(int)
    for record in data:
        if record['filter_passed']:
            filter_stats['passed'] += 1
        else:
            filter_stats[record['filter_reason']] += 1

    # Compute score statistics (scores already exist from sample_data.py)
    score_stats = {}
    for dim in ['difficulty', 'creativity', 'realism']:
        scores = [r[f'score_{dim}'] for r in data if r.get(f'score_{dim}') is not None]
        if scores:
            score_stats[dim] = {
                'average': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'distribution': {i: scores.count(i) for i in range(1, 6)}
            }

    stats = {
        'total_records': len(data),
        'filter_statistics': dict(filter_stats),
        'clustering_statistics': cluster_stats,
        'sampling_statistics': sampling_stats,
        'score_statistics': score_stats
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved statistics to {stats_file}")


# =============================================================================
# Main Pipeline Function
# =============================================================================

def process_sampled(
    input_file: str,
    output_file: str,
    total_samples: int,
    # Clustering options
    min_cluster_size: int = 2,
    min_samples: int = 2,
    # Filtering options
    dedup_threshold: float = 0.8,
    # Other options
    seed: int = 42
):
    """
    Run the processing pipeline on pre-sampled data.

    Phases:
    1. Load data (with embeddings and scores from sample_data.py)
    2. Quality filtering (adds filter_passed, filter_reason)
    3. MinHash deduplication (updates filter_passed, filter_reason for duplicates)
    4. HDBSCAN clustering (adds cluster_id, cluster_probability)
    5. Sampling with new strategy:
       - Pick 10% from noise (best scored)
       - Pick from clusters by size (largest first), until target reached
       - If not enough, sample more from remaining noise
       - If still not enough, repeat cluster sampling (2nd best, 3rd best, etc.)
    6. Save complete dataset
    """
    np.random.seed(seed)

    print("="*60)
    print("PROCESS PRE-SAMPLED DATA PIPELINE")
    print("="*60)
    print(f"Input: {input_file}")
    print(f"Total samples target: {total_samples:,}")
    print(f"Dedup threshold: {dedup_threshold}")
    print("="*60)

    # Phase 1: Load data
    data = load_data(input_file)
    initial_count = len(data)

    # Phase 2: Quality filtering
    data = apply_quality_filters(data)

    # Phase 3: MinHash Deduplication
    data = apply_deduplication(data, dedup_threshold)

    # Phase 4: Clustering (using existing embeddings)
    data, cluster_stats = apply_clustering(
        data=data,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    # Phase 5: Sampling (1 per cluster + noise to fill quota)
    data, sampling_stats = apply_sampling(data, total_samples=total_samples)

    # Phase 6: Save
    save_data(data, output_file)
    save_statistics(data, cluster_stats, sampling_stats, output_file)

    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total records: {len(data):,} (same as input: {initial_count:,})")
    print(f"Filter passed: {sum(1 for r in data if r['filter_passed']):,}")
    print(f"Clusters: {cluster_stats['num_clusters']}")
    print(f"Sampled: {sampling_stats['total_sampled']} ({sampling_stats['from_clusters']} from clusters, {sampling_stats['from_noise']} from noise)")
    print(f"Output: {output_file}")
    print("="*60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process pre-sampled WildChat data: filter, deduplicate, cluster, and sample',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with target dataset size
  python process_sampled.py -i 2000.parquet -o output.parquet --total-samples 500

  # Custom clustering parameters
  python process_sampled.py -i 2000.parquet -o output.parquet --total-samples 500 \\
      --min-cluster-size 10 --min-samples 5

  # Custom deduplication threshold
  python process_sampled.py -i 2000.parquet -o output.parquet --total-samples 500 \\
      --dedup-threshold 0.85

Sampling Strategy:
  1. Pick 10% of target from noise (best scored)
  2. Rank clusters by size, pick best from each (largest first) until target reached
  3. If not enough, sample more from remaining noise
  4. If still not enough, repeat cluster sampling (2nd best, 3rd best, etc.)
        """
    )

    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                        help='Input parquet file (from sample_data.py)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output parquet file')
    parser.add_argument('--total-samples', type=int, required=True,
                        help='Target dataset size (number of samples to select)')

    # Clustering arguments
    parser.add_argument('--min-cluster-size', type=int, default=2,
                        help='HDBSCAN min cluster size (default: 2)')
    parser.add_argument('--min-samples', type=int, default=2,
                        help='HDBSCAN min samples (default: 2)')

    # Filtering arguments
    parser.add_argument('--dedup-threshold', type=float, default=0.8,
                        help='MinHash deduplication threshold (Jaccard similarity 0.0-1.0, default: 0.8)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    process_sampled(
        input_file=args.input,
        output_file=args.output,
        total_samples=args.total_samples,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        dedup_threshold=args.dedup_threshold,
        seed=args.seed
    )
