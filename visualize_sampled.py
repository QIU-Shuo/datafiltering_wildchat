#!/usr/bin/env python3
"""
Improved Visualization Tool for Pre-Sampled WildChat Data

Creates an interactive browser-based visualization for output from process_sampled.py.
Uses t-SNE for dimensionality reduction and Plotly for interactive visualization.

NEW FEATURES:
- Phase-based filtering (Phase 1-4)
- Support for new duplicate types (duplicate_user_input, duplicate_conversation)
- Enhanced filter controls with breakdown by filter reason
- Quick filter presets
"""

import argparse
import json
import webbrowser
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

import numpy as np
from datasets import load_dataset
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from tqdm import tqdm


def load_data(input_file: str) -> List[Dict[str, Any]]:
    """Load processed data from parquet file."""
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading data from {input_file}...")
    dataset = load_dataset('parquet', data_files=input_file)['train']
    data = list(dataset)
    print(f"Loaded {len(data)} records")

    return data


def reduce_dimensions(embeddings: np.ndarray, perplexity: int = 30, seed: int = 42) -> np.ndarray:
    """Reduce embeddings to 2D using t-SNE."""
    print(f"Running t-SNE (perplexity={perplexity})...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
        learning_rate='auto',
        init='pca'
    )

    coords_2d = tsne.fit_transform(embeddings)
    print(f"t-SNE complete. Output shape: {coords_2d.shape}")

    return coords_2d


def format_conversation_html(conversation: List[Dict[str, Any]]) -> str:
    """Format conversation as clean dialogue HTML (no truncation)."""
    if not conversation:
        return "<p><em>(empty conversation)</em></p>"

    html_parts = []
    for msg in conversation:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')

        # Clean role name
        role_display = 'User' if role == 'user' else 'Assistant' if role == 'assistant' else role.title()
        role_class = 'user' if role == 'user' else 'assistant'

        # Escape HTML but preserve newlines
        content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        content = content.replace('\n', '<br>')

        html_parts.append(f'''
            <div class="message {role_class}">
                <div class="role">{role_display}</div>
                <div class="content">{content}</div>
            </div>
        ''')

    return ''.join(html_parts)


def compute_avg_score(record: Dict[str, Any]) -> Optional[float]:
    """Compute average score for a record."""
    scores = []
    for key in ['score_difficulty', 'score_creativity', 'score_realism']:
        val = record.get(key)
        if val is not None:
            scores.append(val)

    if not scores:
        return None

    return sum(scores) / len(scores)


def get_cluster_colors(num_clusters: int) -> Dict[int, str]:
    """Generate distinct colors for clusters."""
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]

    cluster_colors = {}
    color_idx = 0

    for cluster_id in range(-1, num_clusters + 1):
        if cluster_id == -1:
            cluster_colors[-1] = '#808080'
        else:
            cluster_colors[cluster_id] = colors[color_idx % len(colors)]
            color_idx += 1

    return cluster_colors


def categorize_filter_reason(reason: Optional[str]) -> str:
    """Categorize filter reason by phase."""
    if reason is None:
        return 'passed'

    # Phase 1: Quality filters
    if reason in ['empty', 'empty_user_input', 'too_short_user_input', 'spam_pattern', 'toxic']:
        return 'phase1_quality'

    # Phase 2: Deduplication
    if reason in ['duplicate_user_input', 'duplicate_conversation', 'duplicate_minhash']:
        return 'phase2_dedup'

    return 'other'


def build_record_data(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Build data object for a record to be used in JavaScript."""
    avg_score = compute_avg_score(record)
    filter_reason = record.get('filter_reason')
    phase_category = categorize_filter_reason(filter_reason)

    return {
        'index': index,
        'cluster_id': record.get('cluster_id', -1),
        'filter_passed': record.get('filter_passed', True),
        'filter_reason': filter_reason,
        'phase_category': phase_category,
        'is_sampled': record.get('is_sampled', False),
        'score_difficulty': record.get('score_difficulty'),
        'score_creativity': record.get('score_creativity'),
        'score_realism': record.get('score_realism'),
        'score_reasoning': record.get('score_reasoning'),
        'avg_score': avg_score,
        'conversation_html': format_conversation_html(record.get('conversation', []))
    }


def compute_filter_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics about filtering."""
    total = len(data)
    passed = sum(1 for r in data if r.get('filter_passed', True))
    failed = total - passed

    # Count by filter reason
    filter_reasons = Counter(r.get('filter_reason') for r in data if not r.get('filter_passed', True))

    # Categorize by phase
    phase1_count = sum(1 for r in data if categorize_filter_reason(r.get('filter_reason')) == 'phase1_quality')
    phase2_count = sum(1 for r in data if categorize_filter_reason(r.get('filter_reason')) == 'phase2_dedup')

    # Clustering stats (Phase 3)
    cluster_ids = [r.get('cluster_id', -1) for r in data if r.get('filter_passed', True)]
    num_clusters = len(set(c for c in cluster_ids if c >= 0))
    noise_count = sum(1 for c in cluster_ids if c == -1)

    # Sampling stats (Phase 4)
    sampled_count = sum(1 for r in data if r.get('is_sampled', False))

    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'phase1_quality': phase1_count,
        'phase2_dedup': phase2_count,
        'filter_reasons': dict(filter_reasons),
        'num_clusters': num_clusters,
        'noise_count': noise_count,
        'sampled_count': sampled_count
    }


def create_visualization(
    data: List[Dict[str, Any]],
    coords_2d: np.ndarray,
    output_file: str
) -> str:
    """Create interactive Plotly visualization with enhanced filtering."""
    print("Creating visualization...")

    # Get unique cluster IDs
    cluster_ids = set(r.get('cluster_id', -1) for r in data)
    max_cluster = max((c for c in cluster_ids if c >= 0), default=0)
    cluster_colors = get_cluster_colors(max_cluster + 1)

    # Compute statistics
    stats = compute_filter_statistics(data)
    print(f"  Total: {stats['total']}")
    print(f"  Passed: {stats['passed']}")
    print(f"  Phase 1 (quality): {stats['phase1_quality']}")
    print(f"  Phase 2 (dedup): {stats['phase2_dedup']}")
    print(f"  Phase 3 (clusters): {stats['num_clusters']} clusters, {stats['noise_count']} noise")
    print(f"  Phase 4 (sampled): {stats['sampled_count']}")

    # Categorize records by filter reason
    filtered_by_reason = {}
    passed_not_sampled_indices = []
    sampled_indices = []

    for i, record in enumerate(data):
        if not record.get('filter_passed', True):
            reason = record.get('filter_reason', 'unknown')
            if reason not in filtered_by_reason:
                filtered_by_reason[reason] = []
            filtered_by_reason[reason].append(i)
        elif record.get('is_sampled', False):
            sampled_indices.append(i)
        else:
            passed_not_sampled_indices.append(i)

    # Build record data for JavaScript
    print("Building record data...")
    all_record_data = [build_record_data(record, i) for i, record in enumerate(tqdm(data, desc="Processing"))]

    # Create figure
    fig = go.Figure()

    # 1. Add filtered points (grey) - separate trace for each filter reason
    for reason, indices in sorted(filtered_by_reason.items()):
        if not indices:
            continue

        x = [coords_2d[i, 0] for i in indices]
        y = [coords_2d[i, 1] for i in indices]
        custom_data = indices

        # Create readable trace name
        trace_name = f"Filtered: {reason.replace('_', ' ').title()}"

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=6, color='#808080', opacity=0.5),
            name=trace_name,
            customdata=custom_data,
            hoverinfo='none',
            legendgroup=f'filtered_{reason}',
            showlegend=False  # Hide from legend, controlled by checkboxes
        ))

    # 2. Add passed but not sampled points (colored by cluster)
    if passed_not_sampled_indices:
        for cluster_id in sorted(cluster_ids):
            indices = [i for i in passed_not_sampled_indices if data[i].get('cluster_id', -1) == cluster_id]
            if not indices:
                continue

            x = [coords_2d[i, 0] for i in indices]
            y = [coords_2d[i, 1] for i in indices]
            custom_data = indices
            color = cluster_colors.get(cluster_id, '#808080')

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(size=7, color=color, opacity=0.7),
                name=f'Cluster {cluster_id}' if cluster_id >= 0 else 'Noise',
                customdata=custom_data,
                hoverinfo='none',
                legendgroup=f'cluster_{cluster_id}',
                showlegend=True
            ))

    # 3. Add sampled points (double circle with black border)
    if sampled_indices:
        for cluster_id in sorted(cluster_ids):
            indices = [i for i in sampled_indices if data[i].get('cluster_id', -1) == cluster_id]
            if not indices:
                continue

            x = [coords_2d[i, 0] for i in indices]
            y = [coords_2d[i, 1] for i in indices]
            custom_data = indices
            color = cluster_colors.get(cluster_id, '#808080')

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,
                    opacity=1.0,
                    line=dict(width=2, color='black')
                ),
                name=f'Cluster {cluster_id} (sampled)' if cluster_id >= 0 else 'Noise (sampled)',
                customdata=custom_data,
                hoverinfo='none',
                legendgroup=f'cluster_{cluster_id}_sampled',
                showlegend=True
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text='WildChat Data Visualization (t-SNE) - Click points to view details',
            font=dict(size=18)
        ),
        xaxis=dict(title='t-SNE Dimension 1', showgrid=True, zeroline=False),
        yaxis=dict(title='t-SNE Dimension 2', showgrid=True, zeroline=False),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title='Legend',
            itemsizing='constant',
            font=dict(size=10),
            x=0,
            y=1
        ),
        margin=dict(l=50, r=20, t=80, b=50),
        template='plotly_white',
        autosize=True
    )

    # Generate plot JSON
    plot_json = fig.to_json()

    # Build filter reasons for UI
    filter_reasons_json = json.dumps(stats['filter_reasons'])

    # Create HTML content with enhanced filtering
    html_content = create_html_template(all_record_data, plot_json, stats, filter_reasons_json)

    print(f"Saving visualization to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_file


def create_html_template(all_record_data, plot_json, stats, filter_reasons_json):
    """Create the HTML template with enhanced filtering UI."""
    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>WildChat Data Visualization (Enhanced)</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            flex-direction: row;
        }}
        #plotContainer {{
            flex: 1;
            min-width: 0;
            height: 100%;
            position: relative;
        }}
        #plotDiv {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
        }}
        #sidePanel {{
            width: 500px;
            flex-shrink: 0;
            height: 100%;
            border-left: 2px solid #ddd;
            background: #fafafa;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        #sidePanelHeader {{
            padding: 15px 20px;
            background: #f0f0f0;
            border-bottom: 1px solid #ddd;
            font-weight: bold;
            font-size: 16px;
        }}
        #filterControls {{
            padding: 12px 15px;
            background: #e8f4fc;
            border-bottom: 1px solid #ccc;
            max-height: 55vh;
            overflow-y: auto;
        }}
        #filterControls h4 {{
            margin: 0 0 10px 0;
            font-size: 13px;
            color: #333;
            font-weight: 700;
        }}
        .phase-section {{
            margin-bottom: 12px;
            padding: 10px;
            background: #f8f8f8;
            border-radius: 4px;
            border-left: 3px solid #4CAF50;
        }}
        .phase-section.phase1 {{ border-left-color: #FF9800; }}
        .phase-section.phase2 {{ border-left-color: #2196F3; }}
        .phase-section.phase3 {{ border-left-color: #9C27B0; }}
        .phase-section.phase4 {{ border-left-color: #4CAF50; }}
        .phase-section h5 {{
            margin: 0 0 8px 0;
            font-size: 11px;
            color: #555;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .filter-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 5px;
        }}
        .filter-row label {{
            display: flex;
            align-items: center;
            cursor: pointer;
            font-size: 11px;
            flex: 1;
        }}
        .filter-row input {{
            margin-right: 6px;
            cursor: pointer;
        }}
        .filter-count {{
            color: #666;
            font-size: 10px;
            margin-left: 8px;
            font-weight: 600;
        }}
        .quick-filters {{
            margin-bottom: 12px;
            padding: 8px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .quick-filter-btn {{
            padding: 5px 10px;
            margin: 3px 4px 3px 0;
            border: 1px solid #007bff;
            background: white;
            color: #007bff;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
            font-weight: 600;
        }}
        .quick-filter-btn:hover {{
            background: #007bff;
            color: white;
        }}
        #sidePanelContent {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }}
        .placeholder {{
            color: #888;
            text-align: center;
            padding: 40px 20px;
        }}
        .info-section {{
            margin-bottom: 20px;
        }}
        .info-section h3 {{
            margin: 0 0 10px 0;
            padding-bottom: 5px;
            border-bottom: 1px solid #ddd;
            color: #333;
            font-size: 14px;
        }}
        .info-row {{
            display: flex;
            margin-bottom: 8px;
        }}
        .info-label {{
            font-weight: 600;
            color: #555;
            width: 110px;
            flex-shrink: 0;
        }}
        .info-value {{
            color: #333;
        }}
        .status-passed {{
            color: #2ca02c;
            font-weight: 600;
        }}
        .status-filtered {{
            color: #d62728;
            font-weight: 600;
        }}
        .conversation-section {{
            margin-top: 20px;
        }}
        .message {{
            margin-bottom: 15px;
            padding: 12px;
            border-radius: 8px;
        }}
        .message.user {{
            background: #e3f2fd;
            border-left: 4px solid #1976d2;
        }}
        .message.assistant {{
            background: #f3e5f5;
            border-left: 4px solid #7b1fa2;
        }}
        .message .role {{
            font-weight: 700;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 8px;
            color: #555;
        }}
        .message .content {{
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .score-box {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            background: #e8e8e8;
            margin-right: 8px;
            font-size: 13px;
        }}
        .score-high {{
            background: #c8e6c9;
        }}
        .reasoning-box {{
            background: #fff3e0;
            padding: 12px;
            border-radius: 6px;
            margin-top: 10px;
            font-size: 13px;
            line-height: 1.5;
        }}
        .legend-hint {{
            font-size: 11px;
            color: #666;
            padding: 10px 15px;
            background: #f5f5f5;
            border-top: 1px solid #ddd;
        }}
        .stats-summary {{
            margin-bottom: 12px;
            padding: 10px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 11px;
        }}
        .stats-summary div {{
            margin-bottom: 4px;
        }}
    </style>
</head>
<body>
    <div id="plotContainer">
        <div id="plotDiv"></div>
    </div>
    <div id="sidePanel">
        <div id="sidePanelHeader">Record Details & Filters</div>
        <div id="filterControls">
            <div class="stats-summary">
                <div><strong>Total:</strong> {stats['total']:,} records</div>
                <div><strong>Passed:</strong> {stats['passed']:,} ({stats['passed']/stats['total']*100:.1f}%)</div>
                <div><strong>Sampled:</strong> {stats['sampled_count']:,}</div>
            </div>

            <div class="quick-filters">
                <strong style="font-size: 11px;">Quick Filters:</strong><br>
                <button class="quick-filter-btn" onclick="showAll()">Show All</button>
                <button class="quick-filter-btn" onclick="showOnlyPassed()">Only Passed</button>
                <button class="quick-filter-btn" onclick="showOnlySampled()">Only Sampled</button>
                <button class="quick-filter-btn" onclick="showOnlyFiltered()">Only Filtered</button>
            </div>

            <h4>Filter by Pipeline Phase</h4>

            <!-- Phase 1: Quality Filtering -->
            <div class="phase-section phase1">
                <h5>Phase 1: Quality Filtering ({stats['phase1_quality']} filtered)</h5>
                <div class="filter-row">
                    <label><input type="checkbox" class="filter-reason" value="empty_user_input" checked> Empty user input<span class="filter-count" id="count_empty_user_input"></span></label>
                </div>
                <div class="filter-row">
                    <label><input type="checkbox" class="filter-reason" value="too_short_user_input" checked> Too short user input<span class="filter-count" id="count_too_short_user_input"></span></label>
                </div>
                <div class="filter-row">
                    <label><input type="checkbox" class="filter-reason" value="spam_pattern" checked> Spam pattern<span class="filter-count" id="count_spam_pattern"></span></label>
                </div>
                <div class="filter-row">
                    <label><input type="checkbox" class="filter-reason" value="toxic" checked> Toxic<span class="filter-count" id="count_toxic"></span></label>
                </div>
            </div>

            <!-- Phase 2: Deduplication -->
            <div class="phase-section phase2">
                <h5>Phase 2: Deduplication ({stats['phase2_dedup']} filtered)</h5>
                <div class="filter-row">
                    <label><input type="checkbox" class="filter-reason" value="duplicate_user_input" checked> Duplicate user input<span class="filter-count" id="count_duplicate_user_input"></span></label>
                </div>
                <div class="filter-row">
                    <label><input type="checkbox" class="filter-reason" value="duplicate_conversation" checked> Duplicate conversation<span class="filter-count" id="count_duplicate_conversation"></span></label>
                </div>
            </div>

            <!-- Phase 3: Clustering -->
            <div class="phase-section phase3">
                <h5>Phase 3: Clustering ({stats['num_clusters']} clusters, {stats['noise_count']} noise)</h5>
                <div class="filter-row">
                    <label><input type="checkbox" id="showClustered" checked> Show clustered points<span class="filter-count">({stats['passed'] - stats['noise_count']})</span></label>
                </div>
                <div class="filter-row">
                    <label><input type="checkbox" id="showNoise" checked> Show noise points<span class="filter-count">({stats['noise_count']})</span></label>
                </div>
            </div>

            <!-- Phase 4: Sampling -->
            <div class="phase-section phase4">
                <h5>Phase 4: Sampling ({stats['sampled_count']} sampled)</h5>
                <div class="filter-row">
                    <label><input type="checkbox" id="showSampled" checked> Show sampled<span class="filter-count">({stats['sampled_count']})</span></label>
                </div>
                <div class="filter-row">
                    <label><input type="checkbox" id="showNotSampled" checked> Show not sampled<span class="filter-count">({stats['passed'] - stats['sampled_count']})</span></label>
                </div>
            </div>

            <!-- Show Passed -->
            <div class="phase-section">
                <h5>Overall View</h5>
                <div class="filter-row">
                    <label><input type="checkbox" id="showPassed" checked> Show all passed<span class="filter-count">({stats['passed']})</span></label>
                </div>
            </div>
        </div>
        <div id="sidePanelContent">
            <div class="placeholder">Click on a point to view details</div>
        </div>
        <div class="legend-hint">
            <strong>Legend:</strong> Grey = Filtered | Colored = Clusters | Black border = Sampled
        </div>
    </div>

    <script>
        // Record data embedded in page
        const recordData = {json.dumps(all_record_data)};

        // Plot data
        const plotData = {plot_json};

        // Filter reason counts
        const filterReasons = {filter_reasons_json};

        // Display filter reason counts
        for (const [reason, count] of Object.entries(filterReasons)) {{
            const elem = document.getElementById(`count_${{reason}}`);
            if (elem) {{
                elem.textContent = ` (${{count}})`;
            }}
        }}

        // Get elements
        const plotDiv = document.getElementById('plotDiv');
        const sidePanelContent = document.getElementById('sidePanelContent');

        // Render plot with responsive config
        Plotly.newPlot(plotDiv, plotData.data, plotData.layout, {{
            responsive: true,
            displayModeBar: true
        }});

        // Handle window resize
        window.addEventListener('resize', function() {{
            Plotly.Plots.resize(plotDiv);
        }});

        // Filter controls
        const filterReasonCheckboxes = document.querySelectorAll('.filter-reason');
        const showPassedCheckbox = document.getElementById('showPassed');
        const showSampledCheckbox = document.getElementById('showSampled');
        const showNotSampledCheckbox = document.getElementById('showNotSampled');
        const showClusteredCheckbox = document.getElementById('showClustered');
        const showNoiseCheckbox = document.getElementById('showNoise');

        // Quick filter functions
        function showAll() {{
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
            updateVisibility();
        }}

        function showOnlyPassed() {{
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
            showPassedCheckbox.checked = true;
            showSampledCheckbox.checked = true;
            showNotSampledCheckbox.checked = true;
            showClusteredCheckbox.checked = true;
            showNoiseCheckbox.checked = true;
            updateVisibility();
        }}

        function showOnlySampled() {{
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
            showPassedCheckbox.checked = true;
            showSampledCheckbox.checked = true;
            showClusteredCheckbox.checked = true;
            showNoiseCheckbox.checked = true;
            updateVisibility();
        }}

        function showOnlyFiltered() {{
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
            filterReasonCheckboxes.forEach(cb => cb.checked = true);
            updateVisibility();
        }}

        function updateVisibility() {{
            // Get selected filter reasons
            const selectedReasons = new Set();
            filterReasonCheckboxes.forEach(cb => {{
                if (cb.checked) {{
                    selectedReasons.add(cb.value);
                }}
            }});

            const showPassed = showPassedCheckbox.checked;
            const showSampled = showSampledCheckbox.checked;
            const showNotSampled = showNotSampledCheckbox.checked;
            const showClustered = showClusteredCheckbox.checked;
            const showNoise = showNoiseCheckbox.checked;

            // Determine which points to show
            const visibility = plotData.data.map(trace => {{
                const name = trace.name || '';

                // Filtered traces (one per filter reason)
                if (name.startsWith('Filtered:')) {{
                    // Extract reason from trace name "Filtered: Duplicate User Input" -> "duplicate_user_input"
                    const reasonPart = name.substring(10).toLowerCase().replace(/ /g, '_');

                    // Check if this specific reason is selected
                    return selectedReasons.has(reasonPart);
                }}

                // Sampled traces
                if (name.includes('(sampled)')) {{
                    if (!showSampled) return false;
                    if (!showPassed) return false;

                    // Check cluster vs noise
                    if (name.includes('Noise')) {{
                        return showNoise;
                    }} else {{
                        return showClustered;
                    }}
                }}

                // Regular cluster traces (passed but not sampled)
                if (!showNotSampled) return false;
                if (!showPassed) return false;

                // Check cluster vs noise
                if (name === 'Noise') {{
                    return showNoise;
                }} else {{
                    return showClustered;
                }}
            }});

            Plotly.restyle(plotDiv, {{ visible: visibility }});
        }}

        // Attach event listeners
        filterReasonCheckboxes.forEach(cb => cb.addEventListener('change', updateVisibility));
        showPassedCheckbox.addEventListener('change', updateVisibility);
        showSampledCheckbox.addEventListener('change', updateVisibility);
        showNotSampledCheckbox.addEventListener('change', updateVisibility);
        showClusteredCheckbox.addEventListener('change', updateVisibility);
        showNoiseCheckbox.addEventListener('change', updateVisibility);

        // Handle click events
        plotDiv.on('plotly_click', function(eventData) {{
            if (eventData.points && eventData.points.length > 0) {{
                const point = eventData.points[0];
                const recordIndex = point.customdata;

                if (recordIndex !== undefined && recordData[recordIndex]) {{
                    displayRecord(recordData[recordIndex]);
                }}
            }}
        }});

        function displayRecord(record) {{
            let html = '';

            // Basic info section
            html += '<div class="info-section">';
            html += '<h3>Basic Information</h3>';
            html += '<div class="info-row"><span class="info-label">Index:</span><span class="info-value">' + record.index + '</span></div>';
            html += '<div class="info-row"><span class="info-label">Cluster:</span><span class="info-value">' + (record.cluster_id === -1 ? 'Noise (-1)' : record.cluster_id) + '</span></div>';

            if (record.filter_passed) {{
                html += '<div class="info-row"><span class="info-label">Status:</span><span class="info-value status-passed">✓ Passed filters</span></div>';
            }} else {{
                html += '<div class="info-row"><span class="info-label">Status:</span><span class="info-value status-filtered">✗ Filtered</span></div>';
                html += '<div class="info-row"><span class="info-label">Reason:</span><span class="info-value">' + (record.filter_reason || 'unknown') + '</span></div>';
                html += '<div class="info-row"><span class="info-label">Phase:</span><span class="info-value">' + getPhaseName(record.phase_category) + '</span></div>';
            }}
            html += '<div class="info-row"><span class="info-label">Sampled:</span><span class="info-value ' + (record.is_sampled ? 'status-passed' : '') + '\">' + (record.is_sampled ? '✓ Yes' : '✗ No') + '</span></div>';
            html += '</div>';

            // Scores section
            html += '<div class="info-section">';
            html += '<h3>LLM Scores</h3>';

            const avgScore = record.avg_score ? record.avg_score.toFixed(1) : '-';
            const isHigh = record.avg_score && record.avg_score >= 3;

            html += '<div style="margin-bottom: 10px;">';
            html += '<span class="score-box">Difficulty: ' + (record.score_difficulty || '-') + '</span>';
            html += '<span class="score-box">Creativity: ' + (record.score_creativity || '-') + '</span>';
            html += '<span class="score-box">Realism: ' + (record.score_realism || '-') + '</span>';
            html += '<span class="score-box ' + (isHigh ? 'score-high' : '') + '">Avg: ' + avgScore + '</span>';
            html += '</div>';

            if (record.score_reasoning) {{
                html += '<div class="reasoning-box"><strong>Reasoning:</strong><br>' + escapeHtml(record.score_reasoning) + '</div>';
            }}
            html += '</div>';

            // Conversation section
            html += '<div class="info-section conversation-section">';
            html += '<h3>Conversation</h3>';
            html += record.conversation_html;
            html += '</div>';

            sidePanelContent.innerHTML = html;
        }}

        function getPhaseName(category) {{
            const names = {{
                'phase1_quality': 'Phase 1 (Quality)',
                'phase2_dedup': 'Phase 2 (Dedup)',
                'passed': 'Passed'
            }};
            return names[category] || category;
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
    </script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(
        description='Visualize processed WildChat data from process_sampled.py (Enhanced)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_sampled_improved.py -i test_improved_dedup.parquet
  python visualize_sampled_improved.py -i output.parquet -o custom_viz.html --perplexity 50
  python visualize_sampled_improved.py -i output.parquet --no-open
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Input parquet file from process_sampled.py')
    parser.add_argument('-o', '--output', default=None,
                        help='Output HTML file (default: {input_stem}_visualization.html)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter (default: 30)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for t-SNE (default: 42)')
    parser.add_argument('--no-open', action='store_true',
                        help="Don't automatically open browser")

    args = parser.parse_args()

    # Determine output file
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_visualization.html")

    print("=" * 60)
    print("WILDCHAT DATA VISUALIZATION (ENHANCED)")
    print("=" * 60)

    # Load data
    data = load_data(args.input)

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = np.array([record['embedding'] for record in tqdm(data, desc="Extracting")])
    print(f"Embedding shape: {embeddings.shape}")

    # Run t-SNE
    coords_2d = reduce_dimensions(embeddings, perplexity=args.perplexity, seed=args.seed)

    # Create visualization
    output_file = create_visualization(data, coords_2d, args.output)

    print("=" * 60)
    print("VISUALIZATION COMPLETE")
    print(f"Output: {output_file}")
    print("=" * 60)

    # Open in browser
    if not args.no_open:
        print("Opening in browser...")
        webbrowser.open(f"file://{Path(output_file).resolve()}")


if __name__ == "__main__":
    main()
