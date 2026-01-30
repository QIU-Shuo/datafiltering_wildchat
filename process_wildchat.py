#!/usr/bin/env python3
"""
Unified WildChat Pipeline Script

Combines filtering, deduplication, embedding, clustering, sampling, and scoring
into a single pipeline. Preserves ALL input records and adds columns for each phase.

Output Schema (New Columns):
- filter_passed: bool        - True if passed all quality + dedup checks
- filter_reason: str | null  - Reason for filtering (null if passed)
- embedding: list[float]     - Embedding vector (1536 dims for Azure)
- cluster_id: int            - Cluster assignment (-1 for noise)
- cluster_probability: float - HDBSCAN probability
- is_sampled: bool           - True if selected for scoring
- score_difficulty: int | null   - 1-5 scale (null for non-sampled)
- score_creativity: int | null   - 1-5 scale (null for non-sampled)
- score_realism: int | null      - 1-5 scale (null for non-sampled)
- score_reasoning: str | null    - LLM explanation (null for non-sampled)
"""

import json
import argparse
import os
import datetime
import asyncio
import re
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset, Dataset
from datasketch import MinHash, MinHashLSH
import hdbscan

# Optional imports
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None

try:
    from openai import AzureOpenAI, AsyncAzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special types."""
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


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


def get_conversation_text_for_scoring(record: Dict[str, Any]) -> str:
    """Extract text for scoring with role labels."""
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
            text_parts.append(f"User: {content}")
        elif role == 'assistant':
            is_last_message = (i == len(messages) - 1)
            if not is_last_message:
                text_parts.append(f"Assistant: {content}")

    return "\n\n".join(text_parts)


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
# Embedding Generation Functions
# =============================================================================

def generate_embeddings_azure(
    texts: List[str],
    azure_endpoint: str,
    azure_api_key: str,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> np.ndarray:
    """Generate embeddings using Azure OpenAI API."""
    if not HAS_OPENAI:
        raise ImportError("openai package not installed. Install with: pip install openai")

    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2024-02-01"
    )

    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings (Azure)"):
        batch_texts = texts[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    input=batch_texts,
                    model=model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\nWarning: API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise e

    return np.array(embeddings)


def generate_embeddings_local(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    batch_size: int = 32,
    max_length: int = 8192
) -> np.ndarray:
    """Generate embeddings using local model."""
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings (local)"):
        batch_texts = texts[i:i + batch_size]

        try:
            inputs = tokenizer(
                batch_texts,
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

        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
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

    return np.vstack(embeddings)


def apply_embeddings(
    data: List[Dict[str, Any]],
    provider: str,
    model_name: str,
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    batch_size: int = 100,
    device: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for ALL records and add as embedding field.

    Returns:
        Same list with embedding field added to each record
    """
    print("Phase 3: Generating embeddings for ALL records...")

    # Extract texts
    texts = []
    for record in tqdm(data, desc="Extracting texts"):
        text = get_conversation_text_for_dedup(record)
        if not text.strip():
            text = "[empty]"
        texts.append(text)

    # Generate embeddings
    if provider == "azure":
        embeddings = generate_embeddings_azure(
            texts,
            azure_endpoint,
            azure_api_key,
            model=model_name,
            batch_size=batch_size
        )
    else:
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers and torch not installed")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"  Loading local model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device)
        model.eval()

        embeddings = generate_embeddings_local(
            texts, model, tokenizer, device, batch_size=batch_size
        )

        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

    # Add embeddings to records
    print("  Adding embeddings to records...")
    for record, emb in zip(data, embeddings):
        record['embedding'] = emb.tolist()

    print(f"  Embedding dimension: {embeddings.shape[1]}")

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
    Apply HDBSCAN clustering to ALL records using their embeddings.

    Args:
        data: List of records with embedding field
        min_cluster_size: Minimum cluster size (default: 2)
        min_samples: Minimum samples (default: 2)
        metric: Distance metric for HDBSCAN
        cluster_selection_method: 'eom' or 'leaf'

    Returns:
        Tuple of (data with cluster fields, clustering stats)
    """
    print("Phase 4: Clustering ALL records...")
    print(f"  min_cluster_size: {min_cluster_size}")
    print(f"  min_samples: {min_samples}")

    # Extract embeddings from records
    embeddings = np.array([record['embedding'] for record in data])

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1
    )

    cluster_labels = clusterer.fit_predict(embeddings)
    probabilities = clusterer.probabilities_

    # Add cluster info to records
    for record, label, prob in zip(data, cluster_labels, probabilities):
        record['cluster_id'] = int(label)
        record['cluster_probability'] = float(prob)

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

def apply_sampling(
    data: List[Dict[str, Any]],
    samples_per_cluster: int = 5,
    include_noise: bool = True,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Sample records from each cluster (non-destructive).
    Adds is_sampled field to each record.

    Only samples from records that passed filters (filter_passed=True).

    Returns:
        Same list with is_sampled field added
    """
    print("Phase 5: Sampling from clusters...")

    np.random.seed(seed)

    # Initialize all records as not sampled
    for record in data:
        record['is_sampled'] = False

    # Group filtered records by cluster_id
    clusters = defaultdict(list)
    for i, record in enumerate(data):
        if record['filter_passed']:
            cluster_id = record['cluster_id']
            clusters[cluster_id].append(i)

    sampled_count = 0

    # Sample from each cluster
    for cluster_id in sorted(clusters.keys()):
        indices = clusters[cluster_id]

        # Skip noise if not included
        if cluster_id == -1 and not include_noise:
            continue

        n_samples = samples_per_cluster

        if len(indices) <= n_samples:
            # Take all if fewer than requested
            for idx in indices:
                data[idx]['is_sampled'] = True
                sampled_count += 1
        else:
            # Sort by cluster_probability and sample top
            records_with_prob = [
                (idx, data[idx].get('cluster_probability', 0) + np.random.uniform(0, 0.001))
                for idx in indices
            ]
            records_with_prob.sort(key=lambda x: x[1], reverse=True)
            for idx, _ in records_with_prob[:n_samples]:
                data[idx]['is_sampled'] = True
                sampled_count += 1

    num_clusters_with_samples = len([c for c in clusters.keys() if c != -1 or include_noise])
    print(f"  Sampled {sampled_count:,} records from {num_clusters_with_samples} clusters")
    print(f"  Samples per cluster: {samples_per_cluster}")

    return data


# =============================================================================
# Scoring Functions
# =============================================================================

SCORING_PROMPT = """You are evaluating a conversation between a user and an AI assistant.
Your task is to score the user's question/request on three dimensions.

## Conversation Context
{conversation}

## Scoring Criteria

**Difficulty (1-5)**: How challenging is this question/request to answer well?
- 1: Trivial, simple factual lookup or basic task
- 2: Easy, requires some knowledge but straightforward
- 3: Moderate, requires reasoning or domain expertise
- 4: Hard, requires deep expertise, multi-step reasoning, or complex analysis
- 5: Very hard, cutting-edge knowledge, expert-level reasoning, or novel problem-solving

**Creativity (1-5)**: How creative or original is the user's question/request?
- 1: Generic, common question asked frequently
- 2: Slightly varied, minor twist on common questions
- 3: Moderately creative, interesting angle or combination
- 4: Creative, unique approach or novel framing
- 5: Highly creative, original idea or innovative request

**Realism (1-5)**: How realistic and practical is this question/request?
- 1: Unrealistic, nonsensical, or impossible to answer meaningfully
- 2: Somewhat unrealistic, edge cases or hypotheticals with limited value
- 3: Moderately realistic, plausible but may have some impractical aspects
- 4: Realistic, represents genuine user needs or practical scenarios
- 5: Highly realistic, clearly represents real-world use cases

## Response Format
Respond with ONLY a JSON object in this exact format:
{{"difficulty": <1-5>, "creativity": <1-5>, "realism": <1-5>, "reasoning": "<brief explanation>"}}
"""


async def score_conversation(
    client: "AsyncAzureOpenAI",
    record: Dict[str, Any],
    index: int,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-5-nano",
    reasoning_effort: str = "medium",
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Tuple[int, Dict[str, Any]]:
    """Score a single conversation using Azure OpenAI (async)."""
    async with semaphore:
        conversation_text = get_conversation_text_for_scoring(record)

        if len(conversation_text) > 8000:
            conversation_text = conversation_text[:8000] + "\n\n[... truncated ...]"

        prompt = SCORING_PROMPT.format(conversation=conversation_text)

        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort=reasoning_effort,
                    max_completion_tokens=1000
                )

                content = response.choices[0].message.content.strip()

                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()

                scores = json.loads(content)

                for key in ['difficulty', 'creativity', 'realism']:
                    if key not in scores:
                        scores[key] = 3
                    else:
                        scores[key] = max(1, min(5, int(scores[key])))

                if 'reasoning' not in scores:
                    scores['reasoning'] = ""

                return (index, scores)

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                return (index, {
                    'difficulty': 3, 'creativity': 3, 'realism': 3,
                    'reasoning': f"Failed to parse response: {str(e)}"
                })

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                return (index, {
                    'difficulty': 3, 'creativity': 3, 'realism': 3,
                    'reasoning': f"API error: {str(e)}"
                })


async def score_sampled_records(
    data: List[Dict[str, Any]],
    azure_endpoint: str,
    azure_api_key: str,
    model: str = "gpt-5-nano",
    reasoning_effort: str = "medium",
    parallelism: int = 30
) -> List[Dict[str, Any]]:
    """
    Score only sampled records (is_sampled=True).
    Sets null for non-sampled records.
    """
    # Initialize all records with null scores
    for record in data:
        record['score_difficulty'] = None
        record['score_creativity'] = None
        record['score_realism'] = None
        record['score_reasoning'] = None

    # Get sampled records and their indices
    sampled_indices = [i for i, r in enumerate(data) if r.get('is_sampled', False)]

    if not sampled_indices:
        print("  No sampled records to score")
        return data

    print(f"  Scoring {len(sampled_indices)} sampled records...")

    client = AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2024-12-01-preview"
    )

    semaphore = asyncio.Semaphore(parallelism)

    # Create tasks for sampled records
    tasks = [
        score_conversation(
            client=client,
            record=data[idx],
            index=idx,
            semaphore=semaphore,
            model=model,
            reasoning_effort=reasoning_effort
        )
        for idx in sampled_indices
    ]

    # Run all tasks
    results = await tqdm_asyncio.gather(*tasks, desc="Scoring")

    # Apply scores to records
    for idx, scores in results:
        data[idx]['score_difficulty'] = scores['difficulty']
        data[idx]['score_creativity'] = scores['creativity']
        data[idx]['score_realism'] = scores['realism']
        data[idx]['score_reasoning'] = scores['reasoning']

    await client.close()

    return data


def apply_scoring(
    data: List[Dict[str, Any]],
    azure_endpoint: str,
    azure_api_key: str,
    model: str = "gpt-5-nano",
    reasoning_effort: str = "medium",
    parallelism: int = 30
) -> List[Dict[str, Any]]:
    """
    Apply scoring to sampled records only.
    """
    print("Phase 6: Scoring sampled records...")

    return asyncio.run(score_sampled_records(
        data=data,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        model=model,
        reasoning_effort=reasoning_effort,
        parallelism=parallelism
    ))


# =============================================================================
# I/O Functions
# =============================================================================

def load_data(input_file: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load data from JSON or Parquet file/directory."""
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

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

    if top_k is not None and top_k > 0:
        original_count = len(data)
        data = data[:top_k]
        print(f"Limited to first {len(data)} records (from {original_count} total)")

    return data


def save_data(data: List[Dict[str, Any]], output_file: str):
    """Save data to JSON or Parquet file."""
    output_path = Path(output_file)
    output_format = 'json' if output_path.suffix == '.json' else 'parquet'

    print(f"\nSaving data to {output_file} (format: {output_format})...")

    if output_format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=JSONEncoder)
    else:
        dataset = Dataset.from_list(data)
        dataset.to_parquet(output_file)

    print(f"Successfully saved {len(data):,} records to {output_file}")


def save_statistics(
    data: List[Dict[str, Any]],
    cluster_stats: Dict[str, Any],
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

    # Compute score statistics
    sampled = [r for r in data if r.get('is_sampled', False)]
    score_stats = {}
    if sampled:
        for dim in ['difficulty', 'creativity', 'realism']:
            scores = [r[f'score_{dim}'] for r in sampled if r[f'score_{dim}'] is not None]
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
        'sampling_statistics': {
            'sampled_count': sum(1 for r in data if r.get('is_sampled', False)),
            'not_sampled_count': sum(1 for r in data if not r.get('is_sampled', False))
        },
        'score_statistics': score_stats
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved statistics to {stats_file}")


# =============================================================================
# Main Pipeline Function
# =============================================================================

def process_wildchat(
    input_file: str,
    output_file: str,
    # Embedding/Clustering options
    provider: str = "azure",
    embedding_model: str = None,
    min_cluster_size: int = 2,
    min_samples: int = 2,
    # Filtering options
    dedup_threshold: float = 0.8,
    # Sampling options
    samples_per_cluster: int = 5,
    include_noise: bool = True,
    # Scoring options
    scoring_model: str = "gpt-5-nano",
    reasoning_effort: str = "medium",
    parallelism: int = 30,
    # Azure credentials
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    # Other options
    top_k: Optional[int] = None,
    seed: int = 42,
    batch_size: int = 100
):
    """
    Run the complete WildChat processing pipeline.

    Phases:
    1. Load data
    2. Quality filtering (adds filter_passed, filter_reason)
    3. MinHash deduplication (updates filter_passed, filter_reason for duplicates)
    4. Embedding generation (adds embedding to ALL records)
    5. HDBSCAN clustering (adds cluster_id, cluster_probability to ALL records)
    6. Sampling (adds is_sampled)
    7. Scoring (adds score_* fields, only for sampled records)
    8. Save complete dataset
    """
    np.random.seed(seed)

    print("="*60)
    print("UNIFIED WILDCHAT PIPELINE")
    print("="*60)

    # Load Azure credentials
    if azure_endpoint is None:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_api_key is None:
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if provider == "azure" and (not azure_endpoint or not azure_api_key):
        raise ValueError(
            "Azure credentials required. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
            "environment variables or use --azure-endpoint and --azure-api-key"
        )

    # Set default model names
    if embedding_model is None:
        embedding_model = "text-embedding-3-small" if provider == "azure" else "Qwen/Qwen3-Embedding-0.6B"

    print(f"Provider: {provider}")
    print(f"Embedding model: {embedding_model}")
    print(f"Scoring model: {scoring_model}")
    print(f"Dedup threshold: {dedup_threshold}")
    print("="*60)

    # Phase 1: Load data
    data = load_data(input_file, top_k)
    initial_count = len(data)

    # Phase 2: Quality filtering
    data = apply_quality_filters(data)

    # Phase 3: MinHash Deduplication
    data = apply_deduplication(data, dedup_threshold)

    # Phase 4: Embedding generation (ALL records)
    data = apply_embeddings(
        data=data,
        provider=provider,
        model_name=embedding_model,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        batch_size=batch_size
    )

    # Phase 5: Clustering (ALL records)
    data, cluster_stats = apply_clustering(
        data=data,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    # Phase 6: Sampling
    data = apply_sampling(
        data=data,
        samples_per_cluster=samples_per_cluster,
        include_noise=include_noise,
        seed=seed
    )

    # Phase 7: Scoring (only sampled records)
    data = apply_scoring(
        data=data,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        model=scoring_model,
        reasoning_effort=reasoning_effort,
        parallelism=parallelism
    )

    # Phase 8: Save
    save_data(data, output_file)
    save_statistics(data, cluster_stats, output_file)

    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total records: {len(data):,} (same as input: {initial_count:,})")
    print(f"Filter passed: {sum(1 for r in data if r['filter_passed']):,}")
    print(f"Clusters: {cluster_stats['num_clusters']}")
    print(f"Sampled for scoring: {sum(1 for r in data if r.get('is_sampled', False)):,}")
    print(f"Output: {output_file}")
    print("="*60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Unified WildChat Pipeline: filter, deduplicate, embed, cluster, sample, and score',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (credentials from .env file)
  source .env
  python process_wildchat.py -i WildChat-1M/data -o processed.parquet

  # Test on small subset
  python process_wildchat.py -i input.parquet -o test.parquet --top-k 100

  # Custom parameters
  python process_wildchat.py -i input.parquet -o output.parquet \\
      --provider azure \\
      --dedup-threshold 0.85 \\
      --samples-per-cluster 10 \\
      --parallelism 50
        """
    )

    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                        help='Input file/directory')
    parser.add_argument('-o', '--output', required=True,
                        help='Output parquet file')

    # Embedding/Clustering arguments
    parser.add_argument('--provider', choices=['azure', 'local'], default='azure',
                        help='Embedding provider (default: azure)')
    parser.add_argument('--embedding-model', default=None,
                        help='Embedding model name')
    parser.add_argument('--min-cluster-size', type=int, default=2,
                        help='HDBSCAN min cluster size (default: 2)')
    parser.add_argument('--min-samples', type=int, default=2,
                        help='HDBSCAN min samples (default: 2)')

    # Filtering arguments
    parser.add_argument('--dedup-threshold', type=float, default=0.8,
                        help='MinHash deduplication threshold (Jaccard similarity 0.0-1.0, default: 0.8)')

    # Sampling arguments
    parser.add_argument('--samples-per-cluster', type=int, default=5,
                        help='Samples per cluster (default: 5)')
    parser.add_argument('--include-noise', action='store_true', default=True,
                        help='Include noise in sampling (default: True)')
    parser.add_argument('--no-include-noise', dest='include_noise', action='store_false',
                        help='Exclude noise from sampling')

    # Scoring arguments
    parser.add_argument('--scoring-model', default='gpt-5-nano',
                        help='Model for scoring (default: gpt-5-nano)')
    parser.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'],
                        default='medium', help='Reasoning effort (default: medium)')
    parser.add_argument('--parallelism', type=int, default=30,
                        help='Parallel API calls for scoring (default: 30)')

    # Azure credentials
    parser.add_argument('--azure-endpoint', default=None,
                        help='Azure OpenAI endpoint')
    parser.add_argument('--azure-api-key', default=None,
                        help='Azure OpenAI API key')

    # Other arguments
    parser.add_argument('--top-k', type=int, default=None,
                        help='Process only first k records')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for embeddings (default: 100)')

    args = parser.parse_args()

    process_wildchat(
        input_file=args.input,
        output_file=args.output,
        provider=args.provider,
        embedding_model=args.embedding_model,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        dedup_threshold=args.dedup_threshold,
        samples_per_cluster=args.samples_per_cluster,
        include_noise=args.include_noise,
        scoring_model=args.scoring_model,
        reasoning_effort=args.reasoning_effort,
        parallelism=args.parallelism,
        azure_endpoint=args.azure_endpoint,
        azure_api_key=args.azure_api_key,
        top_k=args.top_k,
        seed=args.seed,
        batch_size=args.batch_size
    )
