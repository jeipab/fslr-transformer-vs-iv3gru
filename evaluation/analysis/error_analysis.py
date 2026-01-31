#!/usr/bin/env python3
"""
CTC Error Analysis Script for Continuous Sign Language Recognition

Performs detailed error analysis for CTC models on continuous signing datasets.
Evaluates error breakdown, temporal accuracy, context-based trends, and signer/strategy patterns.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter


def load_json_files(input_dir):
    """Load all JSON files from a directory and normalize structure."""
    data = []
    for f in os.listdir(input_dir):
        if f.endswith(".json"):
            with open(os.path.join(input_dir, f), 'r', encoding='utf-8') as file:
                sample = json.load(file)

                # Normalize structure
                if "segments" in sample:
                    # Extract gloss list
                    sample["glosses"] = [seg["gloss_label"] for seg in sample["segments"]]

                    # Rename timestamp fields
                    for seg in sample["segments"]:
                        seg["start"] = seg.pop("timestamp_start_ms", 0)
                        seg["end"] = seg.pop("timestamp_end_ms", 0)

                data.append(sample)
    return data


def compute_ctc_error_types(pred_sequence, gt_sequence):
    """Compute CTC-style insertion, deletion, substitution errors."""
    insertions = deletions = substitutions = 0
    min_len = min(len(pred_sequence), len(gt_sequence))

    for i in range(min_len):
        p, g = pred_sequence[i], gt_sequence[i]
        if p == g:
            continue
        elif p not in gt_sequence:
            insertions += 1
        elif g not in pred_sequence:
            deletions += 1
        else:
            substitutions += 1

    # Handle length mismatch
    if len(pred_sequence) > len(gt_sequence):
        insertions += len(pred_sequence) - len(gt_sequence)
    elif len(gt_sequence) > len(pred_sequence):
        deletions += len(gt_sequence) - len(pred_sequence)

    return {"insertions": insertions, "deletions": deletions, "substitutions": substitutions}


def temporal_error_analysis(pred_segments, gt_segments):
    """Compute boundary and duration errors."""
    boundary_errors = []
    duration_errors = []

    min_len = min(len(pred_segments), len(gt_segments))
    for i in range(min_len):
        pred = pred_segments[i]
        gt = gt_segments[i]
        start_diff = abs(pred["start"] - gt["start"])
        end_diff = abs(pred["end"] - gt["end"])
        duration_diff = abs((pred["end"] - pred["start"]) - (gt["end"] - gt["start"]))

        boundary_errors.append((start_diff + end_diff) / 2)
        duration_errors.append(duration_diff)

    return {
        "boundary_error_mean": np.mean(boundary_errors) if boundary_errors else 0.0,
        "duration_error_mean": np.mean(duration_errors) if duration_errors else 0.0
    }


def analyze_errors_by_position(pred_sequences, gt_sequences):
    """Analyze errors at different positions in sequences (start/middle/end)."""
    position_errors = {
        'start': {'errors': 0, 'total': 0},      # First 33% of sequence
        'middle': {'errors': 0, 'total': 0},     # Middle 33% of sequence  
        'end': {'errors': 0, 'total': 0}         # Last 33% of sequence
    }
    
    for pred_seq, gt_seq in zip(pred_sequences, gt_sequences):
        seq_len = max(len(pred_seq), len(gt_seq))
        if seq_len == 0:
            continue
            
        # Define position boundaries
        start_end = seq_len // 3
        middle_start = start_end
        middle_end = 2 * start_end
        
        for i in range(seq_len):
            pred_token = pred_seq[i] if i < len(pred_seq) else None
            gt_token = gt_seq[i] if i < len(gt_seq) else None
            
            # Determine position
            if i < start_end:
                position = 'start'
            elif i < middle_end:
                position = 'middle'
            else:
                position = 'end'
            
            position_errors[position]['total'] += 1
            if pred_token != gt_token:
                position_errors[position]['errors'] += 1
    
    # Calculate error rates
    for position in position_errors:
        total = position_errors[position]['total']
        errors = position_errors[position]['errors']
        position_errors[position]['error_rate'] = errors / total if total > 0 else 0
    
    return position_errors


def analyze_errors_after_transitions(pred_sequences, gt_sequences):
    """Analyze errors after specific gloss transitions."""
    transition_errors = {}
    
    for pred_seq, gt_seq in zip(pred_sequences, gt_sequences):
        # Pad sequences to same length
        max_len = max(len(pred_seq), len(gt_seq))
        pred_padded = pred_seq + [None] * (max_len - len(pred_seq))
        gt_padded = gt_seq + [None] * (max_len - len(gt_seq))
        
        for i in range(1, max_len):
            prev_gt = gt_padded[i-1]
            curr_gt = gt_padded[i]
            curr_pred = pred_padded[i]
            
            if prev_gt is None or curr_gt is None:
                continue
                
            transition = f"{prev_gt} → {curr_gt}"
            if transition not in transition_errors:
                transition_errors[transition] = {'errors': 0, 'total': 0}
            
            transition_errors[transition]['total'] += 1
            if curr_pred != curr_gt:
                transition_errors[transition]['errors'] += 1
    
    # Calculate error rates
    for transition in transition_errors:
        total = transition_errors[transition]['total']
        errors = transition_errors[transition]['errors']
        transition_errors[transition]['error_rate'] = errors / total if total > 0 else 0
    
    # Sort by error rate
    return dict(sorted(transition_errors.items(), key=lambda x: x[1]['error_rate'], reverse=True))


def analyze_strategy_errors(error_log):
    """Compare error patterns between strategy 1 and 2."""
    df = pd.DataFrame(error_log)
    
    if 'strategy' not in df.columns:
        return None
    
    strategy_stats = {}
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        strategy_stats[f'strategy_{strategy}'] = {
            'avg_insertions': strategy_df['insertions'].mean(),
            'avg_deletions': strategy_df['deletions'].mean(),
            'avg_substitutions': strategy_df['substitutions'].mean(),
            'avg_boundary_error': strategy_df['boundary_error_mean'].mean(),
            'avg_duration_error': strategy_df['duration_error_mean'].mean(),
            'total_sequences': len(strategy_df)
        }
    
    return strategy_stats


def generate_detailed_breakdown(error_log, labels_df=None):
    """Generate comprehensive error breakdown by various factors."""
    df = pd.DataFrame(error_log)
    
    breakdown = {
        'by_category': {},
        'by_gloss_frequency': {},
        'by_sequence_length': {},
        'by_error_type': {}
    }
    
    # Error breakdown by category (if available)
    if 'category' in df.columns and labels_df is not None:
        category_stats = df.groupby('category').agg({
            'insertions': 'mean',
            'deletions': 'mean', 
            'substitutions': 'mean',
            'boundary_error_mean': 'mean',
            'duration_error_mean': 'mean'
        }).round(3)
        breakdown['by_category'] = category_stats.to_dict('index')
    
    # Error breakdown by sequence length
    if 'num_segments' in df.columns:
        df['length_group'] = pd.cut(df['num_segments'], bins=[0, 3, 5, 10, float('inf')], 
                                   labels=['short (1-3)', 'medium (4-5)', 'long (6-10)', 'very_long (10+)'])
        length_stats = df.groupby('length_group').agg({
            'insertions': 'mean',
            'deletions': 'mean',
            'substitutions': 'mean'
        }).round(3)
        breakdown['by_sequence_length'] = length_stats.to_dict('index')
    
    # Error type distribution
    total_insertions = df['insertions'].sum()
    total_deletions = df['deletions'].sum()
    total_substitutions = df['substitutions'].sum()
    total_errors = total_insertions + total_deletions + total_substitutions
    
    if total_errors > 0:
        breakdown['by_error_type'] = {
            'insertions': {'count': int(total_insertions), 'percentage': total_insertions / total_errors},
            'deletions': {'count': int(total_deletions), 'percentage': total_deletions / total_errors},
            'substitutions': {'count': int(total_substitutions), 'percentage': total_substitutions / total_errors}
        }
    
    return breakdown


def plot_temporal_timeline(error_log, output_dir):
    """Generate timeline showing when errors occur in sequences."""
    df = pd.DataFrame(error_log)
    
    if 'num_segments' not in df.columns:
        return
    
    # Create timeline data
    timeline_data = []
    for _, row in df.iterrows():
        seq_length = row['num_segments']
        total_errors = row['insertions'] + row['deletions'] + row['substitutions']
        
        # Distribute errors across timeline (simplified model)
        for i in range(seq_length):
            timeline_data.append({
                'position': i / seq_length,  # Relative position (0-1)
                'error_density': total_errors / seq_length,
                'signer': row.get('signer', 'unknown'),
                'strategy': row.get('strategy', 'unknown')
            })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    # Create timeline plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Temporal Error Analysis', fontsize=16)
    
    # Error density over time
    axes[0,0].hist(timeline_df['position'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Error Distribution Over Sequence Position')
    axes[0,0].set_xlabel('Relative Position in Sequence')
    axes[0,0].set_ylabel('Frequency')
    
    # Error density by signer
    if 'signer' in timeline_df.columns:
        signer_timeline = timeline_df.groupby(['signer', 'position'])['error_density'].mean().unstack()
        signer_timeline.plot(kind='line', ax=axes[0,1], marker='o')
        axes[0,1].set_title('Error Density by Signer Over Time')
        axes[0,1].set_xlabel('Relative Position in Sequence')
        axes[0,1].set_ylabel('Average Error Density')
        axes[0,1].legend(title='Signer')
    
    # Error density by strategy
    if 'strategy' in timeline_df.columns:
        strategy_timeline = timeline_df.groupby(['strategy', 'position'])['error_density'].mean().unstack()
        strategy_timeline.plot(kind='line', ax=axes[1,0], marker='s')
        axes[1,0].set_title('Error Density by Strategy Over Time')
        axes[1,0].set_xlabel('Relative Position in Sequence')
        axes[1,0].set_ylabel('Average Error Density')
        axes[1,0].legend(title='Strategy')
    
    # Box plot of error density by position quartiles
    timeline_df['position_quartile'] = pd.cut(timeline_df['position'], bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    timeline_df.boxplot(column='error_density', by='position_quartile', ax=axes[1,1])
    axes[1,1].set_title('Error Density Distribution by Position Quartiles')
    axes[1,1].set_xlabel('Position Quartile')
    axes[1,1].set_ylabel('Error Density')
    
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_error_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Temporal error timeline saved to: {output_dir / 'temporal_error_timeline.png'}")


def plot_advanced_error_patterns(error_log, output_dir):
    """Generate advanced error pattern visualizations."""
    df = pd.DataFrame(error_log)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Advanced Error Pattern Analysis', fontsize=16)
    
    # Error type distribution pie chart
    error_types = ['Insertions', 'Deletions', 'Substitutions']
    error_counts = [df['insertions'].sum(), df['deletions'].sum(), df['substitutions'].sum()]
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    axes[0,0].pie(error_counts, labels=error_types, autopct='%1.1f%%', colors=colors)
    axes[0,0].set_title('Distribution of Error Types')
    
    # Error rate vs sequence length
    if 'num_segments' in df.columns:
        df['total_errors'] = df['insertions'] + df['deletions'] + df['substitutions']
        df['error_rate'] = df['total_errors'] / df['num_segments']
        
        axes[0,1].scatter(df['num_segments'], df['error_rate'], alpha=0.6, color='purple')
        axes[0,1].set_title('Error Rate vs Sequence Length')
        axes[0,1].set_xlabel('Sequence Length (segments)')
        axes[0,1].set_ylabel('Error Rate')
        
        # Add trend line
        z = np.polyfit(df['num_segments'], df['error_rate'], 1)
        p = np.poly1d(z)
        axes[0,1].plot(df['num_segments'], p(df['num_segments']), "r--", alpha=0.8)
    
    # Temporal error comparison
    axes[1,0].scatter(df['boundary_error_mean'], df['duration_error_mean'], alpha=0.6, color='orange')
    axes[1,0].set_title('Boundary Error vs Duration Error')
    axes[1,0].set_xlabel('Boundary Error (ms)')
    axes[1,0].set_ylabel('Duration Error (ms)')
    
    # Signer performance comparison
    if 'signer' in df.columns:
        signer_stats = df.groupby('signer').agg({
            'insertions': 'mean',
            'deletions': 'mean',
            'substitutions': 'mean'
        })
        
        x = np.arange(len(signer_stats))
        width = 0.25
        
        axes[1,1].bar(x - width, signer_stats['insertions'], width, label='Insertions', color='lightcoral')
        axes[1,1].bar(x, signer_stats['deletions'], width, label='Deletions', color='lightblue')
        axes[1,1].bar(x + width, signer_stats['substitutions'], width, label='Substitutions', color='lightgreen')
        
        axes[1,1].set_title('Average Errors by Signer')
        axes[1,1].set_xlabel('Signer')
        axes[1,1].set_ylabel('Average Error Count')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(signer_stats.index)
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "advanced_error_patterns.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Advanced error patterns saved to: {output_dir / 'advanced_error_patterns.png'}")


def generate_error_report(error_summary, output_dir):
    """Save summary report as JSON and PDF."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"error_report_{timestamp}.json")
    pdf_path = os.path.join(output_dir, f"error_report_{timestamp}.pdf")

    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=4)

    # Simple PDF summary
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table_data = [[k, v] for k, v in error_summary.items()]
    ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()


def visualize_error_patterns(df, output_dir):
    """Generate per-signer heatmap."""
    if "category" not in df.columns:
        df["category"] = "ALL"

    pivot = df.pivot_table(values='error_count', index='signer', columns='category', fill_value=0)
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, cmap="Reds", fmt=".2f")
    plt.title("Error Distribution Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_heatmap.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="CTC Error Analysis for Continuous Signing")
    parser.add_argument("--input", required=True, help="Directory with prediction JSON files")
    parser.add_argument("--ground-truth-dir", required=True, help="Directory with ground truth JSON files")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports")
    parser.add_argument("--labels-ref", type=str, default='data/labels_reference.csv', help='Path to labels reference CSV')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("CTC ERROR ANALYSIS FOR CONTINUOUS SIGNING")
    print("=" * 80)

    # Load data
    print("\nLoading prediction and ground truth data...")
    predictions = load_json_files(args.input)
    ground_truths = load_json_files(args.ground_truth_dir)
    print(f"   Loaded {len(predictions)} prediction files")
    print(f"   Loaded {len(ground_truths)} ground truth files")

    # Load labels reference if available
    labels_df = None
    try:
        labels_df = pd.read_csv(args.labels_ref)
        print(f"   Loaded labels reference: {len(labels_df)} entries")
    except Exception as e:
        print(f"   Warning: Could not load labels reference: {e}")

    # Basic error analysis
    print("\nComputing basic CTC error analysis...")
    all_errors = []
    pred_sequences = []
    gt_sequences = []

    for pred, gt in zip(predictions, ground_truths):
        ctc_errors = compute_ctc_error_types(pred["glosses"], gt["glosses"])
        temporal_errors = temporal_error_analysis(pred["segments"], gt["segments"])

        total_errors = sum(ctc_errors.values())
        error_entry = {
            "file": pred.get("file_name", "unknown"),
            "signer": pred.get("signer", "unknown"),
            "strategy": pred.get("strategy", "unknown"),
            "num_segments": pred.get("num_segments", len(pred["glosses"])),
            "error_count": total_errors,
            **ctc_errors,
            **temporal_errors
        }
        
        # Add category if available
        if labels_df is not None and 'gloss' in pred:
            try:
                category = labels_df[labels_df['gloss_id'] == pred['gloss']]['category'].iloc[0]
                error_entry['category'] = category
            except:
                pass
        
        all_errors.append(error_entry)
        pred_sequences.append(pred["glosses"])
        gt_sequences.append(gt["glosses"])

    df = pd.DataFrame(all_errors)
    
    # Generate comprehensive summary
    print("\nGenerating comprehensive error summary...")
    summary = {
        "avg_insertions": df["insertions"].mean(),
        "avg_deletions": df["deletions"].mean(),
        "avg_substitutions": df["substitutions"].mean(),
        "avg_boundary_error": df["boundary_error_mean"].mean(),
        "avg_duration_error": df["duration_error_mean"].mean(),
        "per_signer": df.groupby("signer")["error_count"].mean().to_dict(),
        "total_sequences": len(df),
        "total_errors": df["error_count"].sum()
    }

    # Advanced analysis
    print("\nPerforming advanced error analysis...")
    
    # Position-based analysis
    position_analysis = analyze_errors_by_position(pred_sequences, gt_sequences)
    summary["position_analysis"] = position_analysis
    print("   Position-based error analysis completed")
    
    # Transition-based analysis
    transition_analysis = analyze_errors_after_transitions(pred_sequences, gt_sequences)
    summary["transition_analysis"] = dict(list(transition_analysis.items())[:20])  # Top 20 transitions
    print("   Transition-based error analysis completed")
    
    # Strategy comparison
    strategy_analysis = analyze_strategy_errors(all_errors)
    if strategy_analysis:
        summary["strategy_analysis"] = strategy_analysis
        print("   Strategy comparison analysis completed")
    
    # Detailed breakdown
    detailed_breakdown = generate_detailed_breakdown(all_errors, labels_df)
    summary["detailed_breakdown"] = detailed_breakdown
    print("   Detailed error breakdown completed")

    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Basic error patterns
    visualize_error_patterns(df, args.output_dir)
    print("   Basic error patterns visualization completed")
    
    # Advanced error patterns
    plot_advanced_error_patterns(all_errors, args.output_dir)
    print("   Advanced error patterns visualization completed")
    
    # Temporal timeline
    plot_temporal_timeline(all_errors, args.output_dir)
    print("   Temporal error timeline visualization completed")

    # Generate comprehensive report
    print("\nGenerating comprehensive error report...")
    generate_error_report(summary, args.output_dir)
    
    # Save detailed analysis as JSON
    detailed_analysis_path = os.path.join(args.output_dir, "detailed_error_analysis.json")
    with open(detailed_analysis_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "position_analysis": position_analysis,
            "transition_analysis": transition_analysis,
            "strategy_analysis": strategy_analysis,
            "detailed_breakdown": detailed_breakdown,
            "raw_errors": all_errors
        }, f, indent=2, ensure_ascii=False)
    
    print(f"   Detailed analysis saved to: {detailed_analysis_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total Sequences Analyzed: {summary['total_sequences']}")
    print(f"Total Errors: {summary['total_errors']}")
    print(f"Average Errors per Sequence: {summary['total_errors'] / summary['total_sequences']:.2f}")
    print(f"\nError Type Breakdown:")
    print(f"   Insertions: {summary['avg_insertions']:.2f} avg per sequence")
    print(f"   Deletions: {summary['avg_deletions']:.2f} avg per sequence")
    print(f"   Substitutions: {summary['avg_substitutions']:.2f} avg per sequence")
    print(f"\nTemporal Errors:")
    print(f"   Boundary Error: {summary['avg_boundary_error']:.1f} ms avg")
    print(f"   Duration Error: {summary['avg_duration_error']:.1f} ms avg")
    
    if position_analysis:
        print(f"\nPosition-Based Error Rates:")
        for position, stats in position_analysis.items():
            print(f"   {position.capitalize()}: {stats['error_rate']:.3f} ({stats['errors']}/{stats['total']})")
    
    print(f"\nAll reports and visualizations saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
