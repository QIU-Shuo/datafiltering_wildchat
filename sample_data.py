"""Sample a subset of WildChat-1M with embeddings and quality scores."""
import os
import json
import asyncio
import argparse
from datasets import load_dataset, Dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from openai import AzureOpenAI, AsyncAzureOpenAI


# =============================================================================
# Text Extraction Functions (from process_wildchat.py)
# =============================================================================

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


def get_conversation_text_for_scoring(record):
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
# Embedding Generation (from process_wildchat.py)
# =============================================================================

def generate_embeddings_azure(
    texts,
    azure_endpoint,
    azure_api_key,
    model="text-embedding-3-small",
    batch_size=100,
    max_retries=3,
    retry_delay=1.0
):
    """Generate embeddings using Azure OpenAI API."""
    import time

    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2024-02-01"
    )

    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
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

    return embeddings


# =============================================================================
# Scoring Functions (from process_wildchat.py)
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
    client,
    record,
    index,
    semaphore,
    model="gpt-5-nano",
    reasoning_effort="low",
    max_retries=3,
    retry_delay=1.0
):
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


async def score_all_records(
    data,
    azure_endpoint,
    azure_api_key,
    model="gpt-5-nano",
    reasoning_effort="low",
    parallelism=30
):
    """Score all records asynchronously."""
    print(f"Scoring {len(data)} records...")

    client = AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2024-12-01-preview"
    )

    semaphore = asyncio.Semaphore(parallelism)

    tasks = [
        score_conversation(
            client=client,
            record=record,
            index=idx,
            semaphore=semaphore,
            model=model,
            reasoning_effort=reasoning_effort
        )
        for idx, record in enumerate(data)
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="Scoring")

    # Apply scores to records
    for idx, scores in results:
        data[idx]['score_difficulty'] = scores['difficulty']
        data[idx]['score_creativity'] = scores['creativity']
        data[idx]['score_realism'] = scores['realism']
        data[idx]['score_reasoning'] = scores['reasoning']

    await client.close()

    return data


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sample WildChat-1M with embeddings and quality scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults (1000 samples)
  source .env
  python sample_data.py

  # Custom sample size and output path
  python sample_data.py -n 500 -o output.parquet

  # Custom input directory
  python sample_data.py -i /path/to/data -n 2000 -o sampled.parquet
        """
    )

    parser.add_argument('-i', '--input', default='/Users/qiushuo/wsp/datafiltering/WildChat-1M/data',
                        help='Input data directory (default: WildChat-1M/data)')
    parser.add_argument('-o', '--output', default='/Users/qiushuo/wsp/datafiltering/test_data/wildchat_sample.parquet',
                        help='Output parquet file path')
    parser.add_argument('-n', '--num-samples', type=int, default=1000,
                        help='Number of samples to take (default: 1000)')
    parser.add_argument('--embedding-model', default='text-embedding-3-small',
                        help='Azure embedding model (default: text-embedding-3-small)')
    parser.add_argument('--scoring-model', default='gpt-5-nano',
                        help='Azure scoring model (default: gpt-5-nano)')
    parser.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'],
                        default='low', help='Reasoning effort for scoring (default: low)')
    parser.add_argument('--parallelism', type=int, default=50,
                        help='Parallel API calls for scoring (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for embeddings (default: 128)')

    args = parser.parse_args()

    # Load Azure credentials from environment
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not azure_endpoint or not azure_api_key:
        raise ValueError(
            "Azure credentials required. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
            "environment variables or source your .env file"
        )

    # Load records
    print(f"Loading WildChat-1M dataset (first {args.num_samples} records)...")
    dataset = load_dataset(
        'parquet',
        data_dir=args.input,
        split=f'train[:{args.num_samples}]'
    )

    print(f"Loaded {len(dataset)} records")

    # Convert to list of dicts for processing
    data = [dict(record) for record in dataset]

    # Phase 1: Generate embeddings
    print("\nPhase 1: Generating embeddings...")
    texts = []
    for record in tqdm(data, desc="Extracting texts"):
        text = get_conversation_text_for_dedup(record)
        if not text.strip():
            text = "[empty]"
        texts.append(text)

    embeddings = generate_embeddings_azure(
        texts,
        azure_endpoint,
        azure_api_key,
        model=args.embedding_model,
        batch_size=args.batch_size
    )

    for record, emb in zip(data, embeddings):
        record['embedding'] = emb

    print(f"Added embeddings (dimension: {len(embeddings[0])})")

    # Phase 2: Score all records
    print("\nPhase 2: Scoring records...")
    data = asyncio.run(score_all_records(
        data=data,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        model=args.scoring_model,
        reasoning_effort=args.reasoning_effort,
        parallelism=args.parallelism
    ))

    # Convert back to Dataset and save
    output_dataset = Dataset.from_list(data)
    print(f"\nSaving to {args.output}...")
    output_dataset.to_parquet(args.output)

    print(f"\nDone! Saved {len(data)} records with columns:")
    print("  - embedding: 1536-dimensional vector")
    print("  - score_difficulty: 1-5 scale")
    print("  - score_creativity: 1-5 scale")
    print("  - score_realism: 1-5 scale")
    print("  - score_reasoning: LLM explanation")


if __name__ == "__main__":
    main()
