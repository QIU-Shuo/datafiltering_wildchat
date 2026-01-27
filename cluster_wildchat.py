import json
import argparse
import hashlib
import os
import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import warnings
import time

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
import hdbscan
from sklearn.metrics import davies_bouldin_score

# Optional imports for different providers
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None

try:
    from openai import AzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Suppress some warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special types."""
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        return super().default(obj)


def get_conversation_text_for_dedup(record):
    """
    Extract text for embedding generation.
    Same logic as filter_wildchat.py: all messages except the last assistant response.

    This treats the conversation context as the feature space, excluding the final
    response which is the "evaluation target".
    """
    conversation = record.get('conversation', [])

    messages = []
    for msg in conversation:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
            messages.append({'role': role, 'content': content})

    # Collect text parts (same as fingerprint but return raw text)
    text_parts = []

    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content']

        # Always include user messages
        if role == 'user':
            text_parts.append(content)
        # Include assistant messages except the last one
        elif role == 'assistant':
            is_last_message = (i == len(messages) - 1)
            if not is_last_message:
                text_parts.append(content)

    return " ".join(text_parts)


def compute_input_hash(data: List[Dict[str, Any]], top_k: int = None) -> str:
    """Compute hash of input data for cache validation."""
    # Use first and last record IDs plus count for quick hash
    hash_input = f"{len(data)}_{top_k}"
    if data:
        # Add some content from first and last records
        hash_input += f"_{str(data[0])[:100]}_{str(data[-1])[:100]}"
    return hashlib.md5(hash_input.encode('utf-8')).hexdigest()


def load_data(input_file: str, top_k: int = None) -> List[Dict[str, Any]]:
    """Load data from JSON or Parquet file."""
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Detect format
    if input_path.suffix == '.json':
        print(f"Loading JSON data from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif input_path.suffix == '.parquet':
        print(f"Loading Parquet data from {input_file}...")
        dataset = load_dataset('parquet', data_files=input_file)['train']
        data = list(dataset)
    elif input_path.is_dir():
        print(f"Loading Parquet dataset from directory {input_file}...")
        dataset = load_dataset('parquet', data_dir=input_file)['train']
        data = list(dataset)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    print(f"Loaded {len(data)} records")

    # Apply top_k limit
    if top_k is not None and top_k > 0:
        original_count = len(data)
        data = data[:top_k]
        print(f"Limited to first {len(data)} records (from {original_count} total)")

    return data


def save_data(data: List[Dict[str, Any]], output_file: str):
    """Save data to JSON or Parquet file."""
    output_path = Path(output_file)
    output_format = 'json' if output_path.suffix == '.json' else 'parquet'

    print(f"Saving clustered data to {output_file} (format: {output_format})...")

    if output_format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        # Save as Parquet using Hugging Face datasets
        dataset = Dataset.from_list(data)
        dataset.to_parquet(output_file)

    print(f"✓ Successfully saved {len(data):,} records to {output_file}")


def load_embedding_model(model_name: str, device: str):
    """Load embedding model and tokenizer."""
    print(f"Loading embedding model: {model_name}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model, tokenizer


def generate_embeddings(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    batch_size: int = 32,
    max_length: int = 8192
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using batched processing.

    Args:
        texts: List of text strings
        model: Embedding model
        tokenizer: Tokenizer
        device: Device to use (cuda/cpu)
        batch_size: Batch size for processing
        max_length: Maximum sequence length

    Returns:
        numpy array of embeddings, shape (len(texts), embedding_dim)
    """
    embeddings = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]

        try:
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use mean pooling over tokens (excluding padding)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state

                # Mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

            embeddings.append(batch_embeddings.cpu().numpy())

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"\nWARNING: GPU OOM at batch size {batch_size}. Trying with smaller batches...")
                torch.cuda.empty_cache()
                # Retry with smaller batch size
                half_batch = max(1, batch_size // 2)
                for j in range(0, len(batch_texts), half_batch):
                    mini_batch = batch_texts[j:j + half_batch]
                    inputs = tokenizer(
                        mini_batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt'
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model(**inputs)
                        attention_mask = inputs['attention_mask']
                        token_embeddings = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        batch_embeddings = sum_embeddings / sum_mask

                    embeddings.append(batch_embeddings.cpu().numpy())
            else:
                raise e

    # Concatenate all batches
    embeddings = np.vstack(embeddings)
    return embeddings


def generate_embeddings_azure_openai(
    texts: List[str],
    azure_endpoint: str,
    azure_api_key: str,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> np.ndarray:
    """
    Generate embeddings using Azure OpenAI API.

    Args:
        texts: List of text strings
        azure_endpoint: Azure OpenAI endpoint URL
        azure_api_key: Azure OpenAI API key
        model: Model name (default: text-embedding-3-small)
        batch_size: Number of texts to embed in each API call
        max_retries: Maximum number of retries for failed API calls
        retry_delay: Delay between retries in seconds

    Returns:
        numpy array of embeddings, shape (len(texts), embedding_dim)
    """
    if not HAS_OPENAI:
        raise ImportError("openai package not installed. Install with: pip install openai")

    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2024-02-01"
    )

    embeddings = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings (Azure OpenAI)"):
        batch_texts = texts[i:i + batch_size]

        # Retry logic for API calls
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    input=batch_texts,
                    model=model
                )

                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                break  # Success, exit retry loop

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\nWarning: API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"\nError: API call failed after {max_retries} attempts")
                    raise e

    # Convert to numpy array
    embeddings = np.array(embeddings)
    return embeddings


def load_or_generate_embeddings(
    data: List[Dict[str, Any]],
    model,
    tokenizer,
    device: str,
    output_stem: str,
    use_cached: bool = False,
    batch_size: int = 32,
    max_length: int = 8192,
    model_name: str = None,
    top_k: int = None,
    provider: str = "local",
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None
) -> np.ndarray:
    """
    Load embeddings from cache or generate new ones.

    Args:
        provider: Embedding provider ('local' or 'azure')
        azure_endpoint: Azure OpenAI endpoint (required if provider='azure')
        azure_api_key: Azure OpenAI API key (required if provider='azure')

    Returns:
        numpy array of embeddings
    """
    cache_file = Path(f"{output_stem}_embeddings.npy")
    metadata_file = Path(f"{output_stem}_embeddings_metadata.json")

    # Check cache
    if use_cached and cache_file.exists() and metadata_file.exists():
        print("Checking cached embeddings...")

        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Validate cache
        current_hash = compute_input_hash(data, top_k)
        cache_valid = (
            metadata.get('input_hash') == current_hash and
            metadata.get('model_name') == model_name and
            metadata.get('provider') == provider
        )
        if cache_valid:
            print("✓ Using cached embeddings")
            embeddings = np.load(cache_file)
            print(f"  Loaded embeddings shape: {embeddings.shape}")
            return embeddings
        else:
            print("⚠ Cache invalid (input data, model, or provider changed), regenerating embeddings...")

    # Extract texts for embedding
    print("Extracting conversation texts...")
    texts = []
    for record in tqdm(data, desc="Extracting texts"):
        text = get_conversation_text_for_dedup(record)
        if not text.strip():
            # Empty text - use placeholder
            text = "[empty]"
        texts.append(text)

    # Generate embeddings based on provider
    print(f"Generating embeddings for {len(texts)} conversations using {provider}...")

    if provider == "azure":
        if not azure_endpoint or not azure_api_key:
            raise ValueError("Azure endpoint and API key required for Azure provider")
        embeddings = generate_embeddings_azure_openai(
            texts,
            azure_endpoint,
            azure_api_key,
            model=model_name if model_name else "text-embedding-3-small",
            batch_size=batch_size
        )
    elif provider == "local":
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers and torch not installed. Install with: pip install torch transformers")
        if model is None or tokenizer is None:
            raise ValueError("Model and tokenizer required for local provider")
        embeddings = generate_embeddings(texts, model, tokenizer, device, batch_size, max_length)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'local' or 'azure'")

    print(f"✓ Generated embeddings shape: {embeddings.shape}")

    # Save cache
    print(f"Saving embeddings cache to {cache_file}...")
    np.save(cache_file, embeddings)

    # Save metadata
    metadata = {
        'provider': provider,
        'model_name': model_name,
        'input_hash': compute_input_hash(data, top_k),
        'num_records': len(data),
        'embedding_dim': embeddings.shape[1],
        'max_length': max_length,
        'batch_size': batch_size,
        'timestamp': str(Path(cache_file).stat().st_mtime)
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved cache and metadata")

    return embeddings


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    metric: str = 'euclidean',
    cluster_selection_method: str = 'eom'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Run HDBSCAN clustering on embeddings.

    Returns:
        cluster_labels: Cluster IDs for each record (-1 for noise)
        probabilities: Cluster membership probabilities
        stats: Dictionary with clustering statistics
    """
    print(f"\nRunning HDBSCAN clustering...")
    print(f"  min_cluster_size: {min_cluster_size}")
    print(f"  min_samples: {min_samples}")
    print(f"  metric: {metric}")
    print(f"  cluster_selection_method: {cluster_selection_method}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1  # Use all CPU cores
    )

    cluster_labels = clusterer.fit_predict(embeddings)
    probabilities = clusterer.probabilities_

    # Compute statistics
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
    num_noise = np.sum(cluster_labels == -1)

    print(f"\n✓ Clustering complete")
    print(f"  Number of clusters: {num_clusters}")
    print(f"  Number of noise points: {num_noise} ({num_noise / len(cluster_labels) * 100:.2f}%)")

    # Cluster size distribution
    cluster_sizes = {}
    for label in unique_labels:
        if label >= 0:
            cluster_sizes[int(label)] = int(np.sum(cluster_labels == label))

    # Sort clusters by size
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

    # Compute DBCV score if available
    try:
        dbcv_score = clusterer.relative_validity_
        print(f"  DBCV score: {dbcv_score:.4f}")
    except AttributeError:
        dbcv_score = None
        print(f"  DBCV score: Not available")

    # Print top 10 clusters
    print(f"\nTop 10 largest clusters:")
    for i, (label, size) in enumerate(sorted_clusters[:10]):
        print(f"  {i+1}. Cluster {label}: {size:,} records")

    stats = {
        'num_clusters': num_clusters,
        'num_noise': num_noise,
        'noise_percentage': num_noise / len(cluster_labels) * 100,
        'dbcv_score': dbcv_score,
        'cluster_sizes': cluster_sizes,
        'top_clusters': sorted_clusters[:20]  # Top 20 for stats file
    }

    return cluster_labels, probabilities, stats


def save_cluster_stats(
    stats: Dict[str, Any],
    output_stem: str,
    model_name: str,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    provider: str = "local"
):
    """Save clustering statistics to JSON file."""
    stats_file = Path(f"{output_stem}_cluster_stats.json")

    full_stats = {
        'clustering_parameters': {
            'provider': provider,
            'model': model_name,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'metric': metric,
            'cluster_selection_method': cluster_selection_method
        },
        'statistics': {
            'num_clusters': int(stats['num_clusters']),
            'num_noise': int(stats['num_noise']),
            'noise_percentage': float(stats['noise_percentage']),
            'dbcv_score': float(stats['dbcv_score']) if stats['dbcv_score'] is not None else None
        },
        'cluster_size_distribution': stats['cluster_sizes'],
        'top_20_clusters': [
            {'cluster_id': int(label), 'size': int(size)}
            for label, size in stats['top_clusters']
        ]
    }

    with open(stats_file, 'w') as f:
        json.dump(full_stats, f, indent=2)

    print(f"\n✓ Saved cluster statistics to {stats_file}")


def save_clusters_to_json(
    data: List[Dict[str, Any]],
    output_stem: str,
    save_clusters: bool = True
):
    """
    Save each cluster's conversations to separate JSON files for debugging.

    Args:
        data: List of records with cluster_id field
        output_stem: Output file stem for naming
        save_clusters: Whether to save cluster JSON files
    """
    if not save_clusters:
        return

    # Create clusters directory
    clusters_dir = Path(f"{output_stem}_clusters")
    clusters_dir.mkdir(exist_ok=True)

    print(f"\nSaving individual cluster files to {clusters_dir}/...")

    # Group records by cluster_id
    clusters = {}
    for record in data:
        cluster_id = record.get('cluster_id', -1)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(record)

    # Save each cluster to a separate JSON file
    saved_count = 0
    for cluster_id in sorted(clusters.keys()):
        records = clusters[cluster_id]

        if cluster_id == -1:
            filename = clusters_dir / "noise.json"
        else:
            filename = clusters_dir / f"cluster_{cluster_id}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False, cls=JSONEncoder)

        saved_count += 1
        print(f"  ✓ Saved {len(records):,} records to {filename.name}")

    print(f"\n✓ Saved {saved_count} cluster files to {clusters_dir}/")


def cluster_wildchat(
    input_file: str,
    output_file: str,
    provider: str = "local",
    model_name: str = None,
    batch_size: int = 32,
    max_length: int = 8192,
    device: str = "auto",
    use_cached_embeddings: bool = False,
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    save_cluster_jsons: bool = True,
    top_k: int = None,
    seed: int = 42
):
    """
    Cluster WildChat conversations using embeddings and HDBSCAN.

    Args:
        input_file: Path to filtered parquet/json file
        output_file: Path to output file with cluster_id added
        provider: Embedding provider ('local' or 'azure')
        model_name: Model name (HuggingFace for local, Azure deployment for azure)
        batch_size: Batch size for embedding generation
        max_length: Maximum sequence length for tokenization (local only)
        device: Device to use (auto/cuda/cpu, local only)
        use_cached_embeddings: Whether to use cached embeddings if available
        azure_endpoint: Azure OpenAI endpoint URL (required if provider='azure')
        azure_api_key: Azure OpenAI API key (required if provider='azure')
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples parameter for HDBSCAN
        metric: Distance metric for HDBSCAN
        cluster_selection_method: Cluster selection method for HDBSCAN
        save_cluster_jsons: Whether to save individual cluster JSON files (default: True)
        top_k: Only process first k records (for testing)
        seed: Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)
    if HAS_TRANSFORMERS and torch is not None:
        torch.manual_seed(seed)

    # Set default model name based on provider
    if model_name is None:
        if provider == "azure":
            model_name = "text-embedding-3-small"
        else:
            model_name = "Qwen/Qwen3-Embedding-0.6B"

    # Load Azure credentials from environment if not provided
    if provider == "azure":
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_api_key is None:
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not azure_endpoint or not azure_api_key:
            raise ValueError(
                "Azure OpenAI credentials required. Provide via --azure-endpoint and --azure-api-key, "
                "or set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables."
            )

        print(f"Using Azure OpenAI embedding service")
        print(f"  Endpoint: {azure_endpoint}")
        print(f"  Model: {model_name}")

    # Detect device for local provider
    model = None
    tokenizer = None
    if provider == "local":
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers and torch not installed. Install with: pip install torch transformers\n"
                "Or use --provider azure for Azure OpenAI embeddings"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print("⚠ WARNING: No GPU detected. Embedding generation will be slow on CPU.")
                print("  Consider using Azure OpenAI (--provider azure) for faster embedding generation.")

        # Load embedding model
        model, tokenizer = load_embedding_model(model_name, device)

    # Load data
    data = load_data(input_file, top_k)

    if len(data) == 0:
        raise ValueError("No data loaded. Check input file.")

    # Get output stem for cache files
    output_path = Path(output_file)
    output_stem = output_path.parent / output_path.stem

    # Generate or load embeddings
    embeddings = load_or_generate_embeddings(
        data=data,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_stem=str(output_stem),
        use_cached=use_cached_embeddings,
        batch_size=batch_size,
        max_length=max_length,
        model_name=model_name,
        top_k=top_k,
        provider=provider,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key
    )

    # Free up GPU memory (local provider only)
    if provider == "local" and model is not None:
        del model
        del tokenizer
        if HAS_TRANSFORMERS and torch is not None and device == "cuda":
            torch.cuda.empty_cache()

    # Run HDBSCAN clustering
    cluster_labels, probabilities, stats = run_hdbscan(
        embeddings=embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method
    )

    # Add cluster information to records
    print("\nAdding cluster information to records...")
    for i, record in enumerate(data):
        record['cluster_id'] = int(cluster_labels[i])
        record['cluster_probability'] = float(probabilities[i])

    # Save clustered data
    save_data(data, output_file)

    # Save statistics
    save_cluster_stats(
        stats=stats,
        output_stem=str(output_stem),
        model_name=model_name,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        provider=provider
    )

    # Save individual cluster JSON files for debugging
    save_clusters_to_json(
        data=data,
        output_stem=str(output_stem),
        save_clusters=save_cluster_jsons
    )

    print("\n" + "="*60)
    print("CLUSTERING COMPLETE")
    print("="*60)
    print(f"Total records processed: {len(data):,}")
    print(f"Clusters identified: {stats['num_clusters']}")
    print(f"Noise points: {stats['num_noise']:,} ({stats['noise_percentage']:.2f}%)")
    if stats['dbcv_score'] is not None:
        print(f"DBCV score: {stats['dbcv_score']:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Cluster WildChat conversations using embeddings and HDBSCAN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Azure OpenAI (fast, recommended)
  python cluster_wildchat.py -i wildchat_filtered.parquet -o wildchat_clustered.parquet --provider azure

  # Basic usage with local model (slow on CPU)
  python cluster_wildchat.py -i wildchat_filtered.parquet -o wildchat_clustered.parquet --provider local

  # Test on small subset first
  python cluster_wildchat.py -i wildchat_filtered.parquet -o test.parquet --provider azure --top-k 1000

  # Re-cluster with different parameters (reuses cached embeddings)
  python cluster_wildchat.py -i wildchat_filtered.parquet -o clustered_v2.parquet \\
      --provider azure --min-cluster-size 100 --use-cached-embeddings

  # Full pipeline
  python filter_wildchat.py -i WildChat-1M/data -o filtered.parquet
  python cluster_wildchat.py -i filtered.parquet -o clustered.parquet --provider azure

  # With explicit Azure credentials
  python cluster_wildchat.py -i filtered.parquet -o clustered.parquet --provider azure \\
      --azure-endpoint https://your-endpoint.openai.azure.com/ --azure-api-key YOUR_KEY
        """
    )

    # Required arguments
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input parquet/json file from filter_wildchat.py'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with cluster_id added (.json or .parquet)'
    )

    # Provider arguments
    parser.add_argument(
        '--provider',
        choices=['local', 'azure'],
        default='azure',
        help='Embedding provider: local (slow on CPU) or azure (fast, requires credentials). Default: azure'
    )

    parser.add_argument(
        '--azure-endpoint',
        default=None,
        help='Azure OpenAI endpoint URL (reads from AZURE_OPENAI_ENDPOINT env var if not provided)'
    )

    parser.add_argument(
        '--azure-api-key',
        default=None,
        help='Azure OpenAI API key (reads from AZURE_OPENAI_API_KEY env var if not provided)'
    )

    # Model arguments
    parser.add_argument(
        '--model',
        default=None,
        help='Model name (HuggingFace for local, deployment name for Azure). Default: Qwen/Qwen3-Embedding-0.6B (local) or text-embedding-3-small (azure)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=8192,
        help='Maximum sequence length for tokenization (default: 8192)'
    )

    parser.add_argument(
        '--device',
        default='auto',
        help='Device to use: auto, cuda, or cpu (default: auto)'
    )

    parser.add_argument(
        '--use-cached-embeddings',
        action='store_true',
        help='Use cached embeddings if available (skip embedding generation)'
    )

    # Clustering arguments
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=50,
        help='Minimum cluster size for HDBSCAN (default: 50)'
    )

    parser.add_argument(
        '--min-samples',
        type=int,
        default=10,
        help='Minimum samples parameter for HDBSCAN (default: 10)'
    )

    parser.add_argument(
        '--metric',
        default='euclidean',
        help='Distance metric for HDBSCAN (default: euclidean)'
    )

    parser.add_argument(
        '--cluster-selection-method',
        default='eom',
        help='Cluster selection method for HDBSCAN: eom or leaf (default: eom)'
    )

    parser.add_argument(
        '--save-cluster-jsons',
        action='store_true',
        default=True,
        help='Save individual cluster JSON files for debugging (default: True)'
    )

    parser.add_argument(
        '--no-save-cluster-jsons',
        dest='save_cluster_jsons',
        action='store_false',
        help='Skip saving individual cluster JSON files'
    )

    # Other arguments
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Only process first k records (for testing, default: None)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    cluster_wildchat(
        input_file=args.input,
        output_file=args.output,
        provider=args.provider,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        use_cached_embeddings=args.use_cached_embeddings,
        azure_endpoint=args.azure_endpoint,
        azure_api_key=args.azure_api_key,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
        cluster_selection_method=args.cluster_selection_method,
        save_cluster_jsons=args.save_cluster_jsons,
        top_k=args.top_k,
        seed=args.seed
    )
