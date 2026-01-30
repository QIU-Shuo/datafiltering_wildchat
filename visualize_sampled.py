#!/usr/bin/env python3
"""
Visualization Tool for Pre-Sampled WildChat Data

Creates an interactive browser-based visualization for output from process_sampled.py.
Uses t-SNE for dimensionality reduction and Plotly for interactive visualization.

Input: Parquet file from process_sampled.py (all records have scores)

Features:
- 2D scatter plot with t-SNE reduced embeddings
- Color-coded by cluster_id (grey for filtered records)
- Double-circle markers for sampled records (is_sampled=True)
- Click on points to view full details in side panel (scrollable)
"""

import argparse
import json
import webbrowser
from pathlib import Path
from typing import List, Dict, Any, Optional

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


def build_record_data(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Build data object for a record to be used in JavaScript."""
    avg_score = compute_avg_score(record)

    return {
        'index': index,
        'cluster_id': record.get('cluster_id', -1),
        'filter_passed': record.get('filter_passed', True),
        'filter_reason': record.get('filter_reason'),
        'is_sampled': record.get('is_sampled', False),
        'score_difficulty': record.get('score_difficulty'),
        'score_creativity': record.get('score_creativity'),
        'score_realism': record.get('score_realism'),
        'score_reasoning': record.get('score_reasoning'),
        'avg_score': avg_score,
        'conversation_html': format_conversation_html(record.get('conversation', []))
    }


def create_visualization(
    data: List[Dict[str, Any]],
    coords_2d: np.ndarray,
    output_file: str
) -> str:
    """Create interactive Plotly visualization with side panel."""
    print("Creating visualization...")

    # Get unique cluster IDs
    cluster_ids = set(r.get('cluster_id', -1) for r in data)
    max_cluster = max((c for c in cluster_ids if c >= 0), default=0)
    cluster_colors = get_cluster_colors(max_cluster + 1)

    # Categorize records based on filter_passed and is_sampled
    filtered_indices = []
    passed_not_sampled_indices = []
    sampled_indices = []

    for i, record in enumerate(data):
        if not record.get('filter_passed', True):
            filtered_indices.append(i)
        elif record.get('is_sampled', False):
            sampled_indices.append(i)
        else:
            passed_not_sampled_indices.append(i)

    print(f"  Filtered (grey): {len(filtered_indices)}")
    print(f"  Passed, not sampled: {len(passed_not_sampled_indices)}")
    print(f"  Sampled (is_sampled=True): {len(sampled_indices)}")

    # Build record data for JavaScript
    print("Building record data...")
    all_record_data = [build_record_data(record, i) for i, record in enumerate(tqdm(data, desc="Processing"))]

    # Create figure
    fig = go.Figure()

    # 1. Add filtered points (grey)
    if filtered_indices:
        x = [coords_2d[i, 0] for i in filtered_indices]
        y = [coords_2d[i, 1] for i in filtered_indices]
        custom_data = [i for i in filtered_indices]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=6, color='#808080', opacity=0.5),
            name='Filtered',
            customdata=custom_data,
            hoverinfo='none'
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

            # Check if we already have a legend entry for this cluster
            has_legend = any(
                i in passed_not_sampled_indices and data[i].get('cluster_id', -1) == cluster_id
                for i in passed_not_sampled_indices
            )

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

    # Update layout - responsive sizing (no fixed width/height)
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

    # Generate the plot JSON data for manual rendering
    plot_json = fig.to_json()

    # Create full HTML with side panel
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>WildChat Data Visualization</title>
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
            width: 450px;
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
            width: 100px;
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
            font-size: 12px;
            color: #666;
            padding: 10px 20px;
            background: #f5f5f5;
            border-top: 1px solid #ddd;
        }}
        #filterControls {{
            padding: 12px 20px;
            background: #e8f4fc;
            border-bottom: 1px solid #ccc;
        }}
        #filterControls h4 {{
            margin: 0 0 10px 0;
            font-size: 13px;
            color: #333;
        }}
        .filter-row {{
            display: flex;
            align-items: center;
            margin-bottom: 6px;
        }}
        .filter-row label {{
            display: flex;
            align-items: center;
            cursor: pointer;
            font-size: 13px;
        }}
        .filter-row input {{
            margin-right: 8px;
            cursor: pointer;
        }}
        .filter-count {{
            color: #666;
            font-size: 11px;
            margin-left: 5px;
        }}
    </style>
</head>
<body>
    <div id="plotContainer">
        <div id="plotDiv"></div>
    </div>
    <div id="sidePanel">
        <div id="sidePanelHeader">Record Details</div>
        <div id="filterControls">
            <h4>Filter View</h4>
            <div class="filter-row">
                <label><input type="checkbox" id="showFiltered" checked> Show Filtered (grey)<span class="filter-count" id="filteredCount"></span></label>
            </div>
            <div class="filter-row">
                <label><input type="checkbox" id="showPassed" checked> Show Passed (not sampled)<span class="filter-count" id="passedCount"></span></label>
            </div>
            <div class="filter-row">
                <label><input type="checkbox" id="showSampled" checked> Show Sampled (black border)<span class="filter-count" id="sampledCount"></span></label>
            </div>
            <div class="filter-row" style="margin-top: 10px;">
                <label><input type="checkbox" id="onlySampled"> <strong>Only show sampled</strong></label>
            </div>
        </div>
        <div id="sidePanelContent">
            <div class="placeholder">Click on a point to view details</div>
        </div>
        <div class="legend-hint">
            Grey = Filtered | Colored = Clusters | Black border = Sampled (is_sampled=True)
        </div>
    </div>

    <script>
        // Record data embedded in page
        const recordData = {json.dumps(all_record_data)};

        // Plot data
        const plotData = {plot_json};

        // Count records by category
        const filteredCount = recordData.filter(r => !r.filter_passed).length;
        const sampledCount = recordData.filter(r => r.filter_passed && r.is_sampled).length;
        const passedCount = recordData.filter(r => r.filter_passed && !r.is_sampled).length;

        // Display counts
        document.getElementById('filteredCount').textContent = ` (${{filteredCount}})`;
        document.getElementById('passedCount').textContent = ` (${{passedCount}})`;
        document.getElementById('sampledCount').textContent = ` (${{sampledCount}})`;

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
        const showFilteredCheckbox = document.getElementById('showFiltered');
        const showPassedCheckbox = document.getElementById('showPassed');
        const showSampledCheckbox = document.getElementById('showSampled');
        const onlySampledCheckbox = document.getElementById('onlySampled');

        function updateVisibility() {{
            const showFiltered = showFilteredCheckbox.checked;
            const showPassed = showPassedCheckbox.checked;
            const showSampled = showSampledCheckbox.checked;
            const onlySampled = onlySampledCheckbox.checked;

            // If "only sampled" is checked, override other settings
            if (onlySampled) {{
                showFilteredCheckbox.checked = false;
                showPassedCheckbox.checked = false;
                showSampledCheckbox.checked = true;
                showFilteredCheckbox.disabled = true;
                showPassedCheckbox.disabled = true;
                showSampledCheckbox.disabled = true;
            }} else {{
                showFilteredCheckbox.disabled = false;
                showPassedCheckbox.disabled = false;
                showSampledCheckbox.disabled = false;
            }}

            const visibility = plotData.data.map(trace => {{
                const name = trace.name || '';
                // Filtered trace
                if (name === 'Filtered') {{
                    return onlySampled ? false : showFiltered;
                }}
                // Sampled traces (have "sampled" in name or legendgroup ends with _sampled)
                if (name.includes('(sampled)')) {{
                    return onlySampled ? true : showSampled;
                }}
                // Regular cluster traces (passed but not sampled)
                return onlySampled ? false : showPassed;
            }});

            Plotly.restyle(plotDiv, {{ visible: visibility }});
        }}

        showFilteredCheckbox.addEventListener('change', updateVisibility);
        showPassedCheckbox.addEventListener('change', updateVisibility);
        showSampledCheckbox.addEventListener('change', updateVisibility);
        onlySampledCheckbox.addEventListener('change', updateVisibility);

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
                html += '<div class="info-row"><span class="info-label">Status:</span><span class="info-value status-passed">Passed filters</span></div>';
            }} else {{
                html += '<div class="info-row"><span class="info-label">Status:</span><span class="info-value status-filtered">Filtered</span></div>';
                html += '<div class="info-row"><span class="info-label">Reason:</span><span class="info-value">' + (record.filter_reason || 'unknown') + '</span></div>';
            }}
            html += '<div class="info-row"><span class="info-label">Sampled:</span><span class="info-value ' + (record.is_sampled ? 'status-passed' : '') + '">' + (record.is_sampled ? 'Yes' : 'No') + '</span></div>';
            html += '</div>';

            // Scores section (all records have scores in process_sampled.py output)
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

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
    </script>
</body>
</html>
'''

    print(f"Saving visualization to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Visualize processed WildChat data from process_sampled.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_sampled.py -i 2000_processed.parquet
  python visualize_sampled.py -i 2000_processed.parquet -o custom_viz.html --perplexity 50
  python visualize_sampled.py -i 2000_processed.parquet --no-open
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
    print("WILDCHAT DATA VISUALIZATION (for process_sampled.py output)")
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
