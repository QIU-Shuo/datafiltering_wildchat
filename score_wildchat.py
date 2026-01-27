import json
import argparse
import os
import datetime
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from datasets import load_dataset, Dataset

try:
    from openai import AsyncAzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special types."""
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        return super().default(obj)


def get_conversation_text_for_scoring(record: Dict[str, Any]) -> str:
    """
    Extract text for scoring.
    Same logic as clustering: all messages except the last assistant response.
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
            text_parts.append(f"User: {content}")
        elif role == 'assistant':
            is_last_message = (i == len(messages) - 1)
            if not is_last_message:
                text_parts.append(f"Assistant: {content}")

    return "\n\n".join(text_parts)


def get_user_question(record: Dict[str, Any]) -> str:
    """Extract the first user message as the main question."""
    conversation = record.get('conversation', [])

    for msg in conversation:
        if isinstance(msg, dict) and msg.get('role') == 'user':
            return msg.get('content', '')

    return ""


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
) -> tuple[int, Dict[str, Any]]:
    """
    Score a single conversation using Azure OpenAI (async).

    Returns:
        Tuple of (index, scores dictionary)
    """
    async with semaphore:
        conversation_text = get_conversation_text_for_scoring(record)

        # Truncate if too long (keep first 8000 chars)
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

                # Parse JSON response
                # Handle potential markdown code blocks
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()

                scores = json.loads(content)

                # Validate scores
                for key in ['difficulty', 'creativity', 'realism']:
                    if key not in scores:
                        scores[key] = 3  # Default to middle score
                    else:
                        scores[key] = max(1, min(5, int(scores[key])))

                if 'reasoning' not in scores:
                    scores['reasoning'] = ""

                return (index, scores)

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                # Return default scores on parse failure
                return (index, {
                    'difficulty': 3,
                    'creativity': 3,
                    'realism': 3,
                    'reasoning': f"Failed to parse response: {str(e)}"
                })

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                # Return default scores on API failure
                return (index, {
                    'difficulty': 3,
                    'creativity': 3,
                    'realism': 3,
                    'reasoning': f"API error: {str(e)}"
                })


async def score_all_conversations(
    client: "AsyncAzureOpenAI",
    data: List[Dict[str, Any]],
    model: str = "gpt-5-nano",
    reasoning_effort: str = "medium",
    parallelism: int = 30
) -> List[Dict[str, Any]]:
    """
    Score all conversations in parallel.

    Args:
        client: AsyncAzureOpenAI client
        data: List of records to score
        model: Model name
        reasoning_effort: Reasoning effort level
        parallelism: Number of parallel requests (default: 30)

    Returns:
        List of score dictionaries in original order
    """
    semaphore = asyncio.Semaphore(parallelism)

    # Create tasks for all records
    tasks = [
        score_conversation(
            client=client,
            record=record,
            index=i,
            semaphore=semaphore,
            model=model,
            reasoning_effort=reasoning_effort
        )
        for i, record in enumerate(data)
    ]

    # Run all tasks with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="Scoring")

    # Sort results by index to maintain original order
    results.sort(key=lambda x: x[0])

    # Extract just the scores
    return [scores for _, scores in results]


def load_data(input_file: str) -> List[Dict[str, Any]]:
    """Load data from JSON or Parquet file."""
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
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    print(f"Loaded {len(data)} records")
    return data


def save_data(data: List[Dict[str, Any]], output_file: str):
    """Save data to JSON or Parquet file."""
    output_path = Path(output_file)
    output_format = 'json' if output_path.suffix == '.json' else 'parquet'

    print(f"Saving scored data to {output_file} (format: {output_format})...")

    if output_format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=JSONEncoder)
    else:
        dataset = Dataset.from_list(data)
        dataset.to_parquet(output_file)

    print(f"Successfully saved {len(data):,} records to {output_file}")


async def score_wildchat_async(
    input_file: str,
    output_file: str,
    model: str = "gpt-5-nano",
    reasoning_effort: str = "medium",
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    parallelism: int = 30,
    top_k: Optional[int] = None
):
    """
    Score WildChat conversations on difficulty, creativity, and realism (async version).
    """
    if not HAS_OPENAI:
        raise ImportError("openai package not installed. Install with: pip install openai")

    # Load credentials from environment if not provided
    if azure_endpoint is None:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_api_key is None:
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not azure_endpoint or not azure_api_key:
        raise ValueError(
            "Azure OpenAI credentials required. Provide via --azure-endpoint and --azure-api-key, "
            "or set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.\n"
            "You can also source your .env file: source .env"
        )

    print(f"Using Azure OpenAI")
    print(f"  Endpoint: {azure_endpoint}")
    print(f"  Model: {model}")
    print(f"  Reasoning effort: {reasoning_effort}")
    print(f"  Parallelism: {parallelism}")

    # Initialize async client
    client = AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2024-12-01-preview"
    )

    # Load data
    data = load_data(input_file)

    # Apply top_k limit
    if top_k is not None and top_k > 0:
        original_count = len(data)
        data = data[:top_k]
        print(f"Limited to first {len(data)} records (from {original_count} total)")

    # Score all conversations in parallel
    print(f"\nScoring {len(data)} conversations with {parallelism} parallel requests...")

    all_scores = await score_all_conversations(
        client=client,
        data=data,
        model=model,
        reasoning_effort=reasoning_effort,
        parallelism=parallelism
    )

    # Add scores to records and collect stats
    score_stats = {
        'difficulty': [],
        'creativity': [],
        'realism': []
    }

    for record, scores in zip(data, all_scores):
        record['score_difficulty'] = scores['difficulty']
        record['score_creativity'] = scores['creativity']
        record['score_realism'] = scores['realism']
        record['score_reasoning'] = scores['reasoning']

        score_stats['difficulty'].append(scores['difficulty'])
        score_stats['creativity'].append(scores['creativity'])
        score_stats['realism'].append(scores['realism'])

    # Calculate statistics
    print("\n" + "="*60)
    print("SCORING COMPLETE")
    print("="*60)
    print(f"Total records scored: {len(data):,}")
    print(f"\nScore Statistics:")
    for dim in ['difficulty', 'creativity', 'realism']:
        scores = score_stats[dim]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {dim.capitalize():12} - Avg: {avg:.2f}, Min: {min(scores)}, Max: {max(scores)}")

    # Distribution
    print(f"\nScore Distributions:")
    for dim in ['difficulty', 'creativity', 'realism']:
        scores = score_stats[dim]
        dist = {i: scores.count(i) for i in range(1, 6)}
        dist_str = " | ".join([f"{k}:{v}" for k, v in sorted(dist.items())])
        print(f"  {dim.capitalize():12} - {dist_str}")

    print("="*60)

    # Save scored data
    save_data(data, output_file)

    # Save statistics
    output_path = Path(output_file)
    stats_file = output_path.parent / f"{output_path.stem}_score_stats.json"

    stats = {
        'model': model,
        'reasoning_effort': reasoning_effort,
        'parallelism': parallelism,
        'total_records': len(data),
        'statistics': {
            dim: {
                'average': sum(score_stats[dim]) / len(score_stats[dim]) if score_stats[dim] else 0,
                'min': min(score_stats[dim]) if score_stats[dim] else 0,
                'max': max(score_stats[dim]) if score_stats[dim] else 0,
                'distribution': {i: score_stats[dim].count(i) for i in range(1, 6)}
            }
            for dim in ['difficulty', 'creativity', 'realism']
        }
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved statistics to {stats_file}")

    # Close client
    await client.close()


def score_wildchat(
    input_file: str,
    output_file: str,
    model: str = "gpt-5-nano",
    reasoning_effort: str = "medium",
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    parallelism: int = 30,
    top_k: Optional[int] = None
):
    """
    Score WildChat conversations on difficulty, creativity, and realism.

    Args:
        input_file: Path to input parquet/json file (from cluster_wildchat.py)
        output_file: Path to output file with scores added
        model: Azure OpenAI model name (default: gpt-5-nano)
        reasoning_effort: Reasoning effort level: low, medium, high (default: medium)
        azure_endpoint: Azure OpenAI endpoint URL
        azure_api_key: Azure OpenAI API key
        parallelism: Number of parallel API requests (default: 30)
        top_k: Only process first k records (for testing)
    """
    asyncio.run(score_wildchat_async(
        input_file=input_file,
        output_file=output_file,
        model=model,
        reasoning_effort=reasoning_effort,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        parallelism=parallelism,
        top_k=top_k
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score WildChat conversations on difficulty, creativity, and realism',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (credentials from .env file)
  source .env
  python score_wildchat.py -i test_dataset.parquet -o scored_dataset.parquet

  # Test on small subset first
  python score_wildchat.py -i test_dataset.parquet -o scored.parquet --top-k 10

  # Adjust parallelism (default: 30)
  python score_wildchat.py -i test_dataset.parquet -o scored.parquet --parallelism 50

  # Use different reasoning effort
  python score_wildchat.py -i test_dataset.parquet -o scored.parquet --reasoning-effort high

  # Full pipeline
  python filter_wildchat.py -i WildChat-1M/data -o filtered.parquet
  python cluster_wildchat.py -i filtered.parquet -o sampled.parquet --provider azure
  python score_wildchat.py -i sampled.parquet -o scored.parquet
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input parquet/json file (from cluster_wildchat.py)'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file with scores added (.json or .parquet)'
    )

    parser.add_argument(
        '--model',
        default='gpt-5-nano',
        help='Azure OpenAI model/deployment name (default: gpt-5-nano)'
    )

    parser.add_argument(
        '--reasoning-effort',
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Reasoning effort level for the model (default: medium)'
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

    parser.add_argument(
        '--parallelism',
        type=int,
        default=30,
        help='Number of parallel API requests (default: 30)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Only process first k records (for testing)'
    )

    args = parser.parse_args()

    score_wildchat(
        input_file=args.input,
        output_file=args.output,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        azure_endpoint=args.azure_endpoint,
        azure_api_key=args.azure_api_key,
        parallelism=args.parallelism,
        top_k=args.top_k
    )
