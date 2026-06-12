"""
Generate publication-ready figures and tables from evaluation JSON files.
Outputs:
  - figures/ directory with PNG plots
  - report_tables.txt with markdown tables
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

# Use relative path from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SETS_DIR = os.path.join(SCRIPT_DIR, "Evaluation_sets")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "report_outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

# Friendly names for documents
DOC_NAMES = {
    "Therapeutic_exercise_Foundations_and_techniques_by_Colby_Lynn_Allen-572-650.pdf": 
        "Therapeutic exercise (Colby et al.)",
    "Copy of Skulder og skulderbue.pdf": 
        "Skulder og skulderbue",
    "Adhesive_capsulitis_JOSPT.pdf": 
        "Adhesive capsulitis JOSPT",
    "Shoulderdoc - Shoulder Rehab Book.pdf": 
        "Shoulderdoc – Shoulder Rehab",
    "Rotator cuff tendinopathy CPG.pdf":
        "Rotator cuff CPG",
    "Subacromial pain syndrome.pdf":
        "Subacromial pain syndrome"
}

# Setup styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]


def setup_output_dirs():
    """Create output directories."""
    print(f"   Creating: {os.path.abspath(FIGURES_DIR)}")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"   Creating: {os.path.abspath(TABLES_DIR)}")
    os.makedirs(TABLES_DIR, exist_ok=True)
    print(f"📁 ✅ Output directories ready")


def short_model_name(model_name: str) -> str:
    """Return a short human-friendly model label (e.g. Qwen-14B, Llama-8B)."""
    name = str(model_name)
    # If path-like, take last segment
    name = name.split('/')[-1]
    # Remove trailing run ids separated by underscore
    name = name.split('_')[0]
    # Simplify known prefixes
    name = name.replace('Qwen3', 'Qwen')
    name = name.replace('Llama-3.1', 'Llama')
    name = name.replace('meta-llama', 'Llama')
    name = name.replace('Mistral-7B-Instruct-v0.3', 'Mistral-7B')
    # Remove common suffixes
    for suf in ['-Instruct', '-instruct', '-v0.3']:
        name = name.replace(suf, '')
    # Try to capture prefix and size
    import re
    m = re.search(r'([A-Za-z]+)[^0-9]*-?([0-9]+B)', name)
    if m:
        prefix = m.group(1).capitalize()
        size = m.group(2)
        return f"{prefix}-{size}"
    # Fallback: truncate
    return name[:15]


def load_evaluation_files() -> Dict[str, dict]:
    """Load all evaluation JSON files from Evaluation_sets folder."""
    print(f"🔍 Looking for eval files in: {EVAL_SETS_DIR}")
    print(f"   Directory exists: {os.path.isdir(EVAL_SETS_DIR)}")

    eval_files = glob.glob(os.path.join(EVAL_SETS_DIR, "eval_*.json"))
    print(f"   Found {len(eval_files)} files")

    if not eval_files:
        # Try alternative search
        print("   Trying alternative glob pattern...")
        eval_files = glob.glob(os.path.join(EVAL_SETS_DIR, "*.json"))
        print(f"   Found {len(eval_files)} JSON files total")

    evaluations = {}
    for file_path in sorted(eval_files):
        print(f"   Processing: {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract model name from filename: eval_MODEL_TIMESTAMP.json
                filename = os.path.basename(file_path)
                model_name = filename.replace("eval_", "").rsplit("_", 1)[0]
                evaluations[model_name] = data
                print(f"✅ Loaded: {model_name}")
        except Exception as e:
            print(f"⚠️  Failed to load {file_path}: {e}")
    
    return evaluations


def create_figure_1_overall_metrics(evaluations: Dict[str, dict]):
    """
    Figure 1: Bar chart with overall metrics across all models.
    Shows: retrieval_accuracy, answer_letter_accuracy, answer_text_accuracy, fully_correct
    """
    models = []
    metrics_data = {
        "Retrieval Accuracy": [],
        "Answer-Letter Accuracy": [],        "Answer-Text Accuracy": [],
        "Fully-Correct Rate": []
    }
    
    for model_name, eval_data in sorted(evaluations.items()):
        models.append(short_model_name(model_name))
        summary = eval_data.get("summary", {})
        metrics_data["Retrieval Accuracy"].append(summary.get("retrieval_accuracy_pct", 0))
        metrics_data["Answer-Letter Accuracy"].append(summary.get("answer_letter_accuracy_pct", 0))
        metrics_data["Answer-Text Accuracy"].append(summary.get("answer_text_accuracy_pct", 0))
        metrics_data["Fully-Correct Rate"].append(summary.get("fully_correct_pct", 0))
    
    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax.bar(x + i*width, values, width, label=metric, color=COLORS[i])
    
    # Styling for readability
    ax.set_xlabel("Model", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
    ax.set_title("PhysioRAG Evaluation Metrics Across Models", fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim([0, 105])
    # Faint horizontal grid lines to help comparison
    ax.grid(axis='y', alpha=0.15, linestyle='--')

    # Add value labels on bars
    for i, (metric, values) in enumerate(metrics_data.items()):
        for xi, v in zip(x + i*width, values):
            ax.text(xi, v + 1.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "01_overall_metrics.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved Figure 1: {fig_path}")
    plt.close()
    # Save caption for figure 1
    try:
        caption_path = os.path.join(OUTPUT_DIR, 'figure_captions.md')
        total_questions = sum([evaluations[m].get('summary', {}).get('total_questions', 0) for m in evaluations])
        with open(caption_path, 'a', encoding='utf-8') as cf:
            cf.write(f"### Figure: Overall metrics across models\n")
            cf.write(f"MCQ gold-standard set; total questions across included runs: {total_questions}.\n")
            cf.write("Metrics plotted: Retrieval = retrieval_accuracy_pct; Answer-letter = answer_letter_accuracy_pct; Answer-text = answer_text_accuracy_pct; Fully-correct = fully_correct_pct.\n")
            cf.write("Values read from the 'summary' block in each evaluation JSON (Section 3.4). Error bars omitted due to small sample size.\n\n")
    except Exception:
        pass


def create_figure_2_per_document(evaluations: Dict[str, dict]):
    """
    Figure 2: Heatmap of metrics broken down by document for each model.
    Shows document performance across all models.
    """
    # Prepare data for heatmap
    models = sorted(evaluations.keys())
    # Maintain consistent document order defined in DOC_NAMES
    orig_doc_keys = list(DOC_NAMES.keys())
    documents = [DOC_NAMES[k] for k in orig_doc_keys]

    # Create separate heatmaps for each metric
    metrics_to_plot = [
        ("retrieval_accuracy", "Retrieval Accuracy (%)"),
        ("answer_text_accuracy", "Answer-Text Accuracy (%)"),
        ("fully_correct", "Fully-Correct Rate (%)")
    ]
    
    for metric_key, metric_label in metrics_to_plot:
        heatmap_data = []
        # track sample sizes per cell for caption/table
        sample_counts = []

        for model_name in models:
            eval_data = evaluations[model_name]
            row = []
            for orig_doc_name, friendly_name in DOC_NAMES.items():
                doc_results = eval_data.get("per_document_breakdown", {}).get(orig_doc_name, {})
                total = doc_results.get("total", 0)
                
                if total > 0:
                    if metric_key == "retrieval_accuracy":
                        correct = doc_results.get("retrieval_correct", 0)
                    elif metric_key == "answer_text_accuracy":
                        correct = doc_results.get("answer_text_correct", 0)
                    elif metric_key == "fully_correct":
                        correct = doc_results.get("fully_correct", 0)
                    
                    pct = (correct / total) * 100
                else:
                    pct = 0
                row.append(pct)
                sample_counts.append(total)
            heatmap_data.append(row)
        # sample_counts length = models * documents; reshape
        sample_matrix = [sample_counts[i*len(orig_doc_keys):(i+1)*len(orig_doc_keys)] for i in range(len(models))]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 5))
        df = pd.DataFrame(
            heatmap_data,
            index=[short_model_name(m) for m in models],
            columns=[d for d in documents]
        )
        
        sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                    cbar_kws={'label': 'Accuracy (%)'}, ax=ax, linewidths=0.5,
                    annot_kws={"size": 11})
        ax.set_title(f"PhysioRAG: {metric_label} by Document", fontsize=13, fontweight='bold')
        ax.set_xlabel("Document", fontsize=11, fontweight='bold')
        ax.set_ylabel("Model", fontsize=11, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        
        fig_path = os.path.join(FIGURES_DIR, f"02_{metric_key}_heatmap.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved heatmap: {fig_path}")
        plt.close()
        # Save sample size table for this metric
        try:
            sizes_path = os.path.join(OUTPUT_DIR, f"samples_{metric_key}.md")
            with open(sizes_path, 'w', encoding='utf-8') as sf:
                sf.write(f"# Sample sizes for {metric_label}\n\n")
                # header
                sf.write("| Document |")
                for m in models:
                    sf.write(f" {short_model_name(m)} |")
                sf.write("\n")
                sf.write("|" + "---|" * (len(models) + 1) + "\n")
                for j, orig in enumerate(orig_doc_keys):
                    sf.write(f"| {DOC_NAMES[orig]} |")
                    for i in range(len(models)):
                        cnt = sample_matrix[i][j]
                        sf.write(f" {cnt} |")
                    sf.write("\n")
            print(f"📊 Saved sample sizes: {sizes_path}")
        except Exception:
            pass


def create_figure_3_model_comparison_line(evaluations: Dict[str, dict]):
    """
    Figure 3: Line plot showing all 4 metrics for easier comparison.
    """
    models = sorted(evaluations.keys())
    model_labels = [short_model_name(m) for m in models]

    metrics_data = {
        "Retrieval": [],
        "Answer-Letter": [],
        "Answer-Text": [],
        "Fully-Correct": []
    }
    
    for model_name in models:
        eval_data = evaluations[model_name]
        summary = eval_data.get("summary", {})
        metrics_data["Retrieval"].append(summary.get("retrieval_accuracy_pct", 0))
        metrics_data["Answer-Letter"].append(summary.get("answer_letter_accuracy_pct", 0))
        metrics_data["Answer-Text"].append(summary.get("answer_text_accuracy_pct", 0))
        metrics_data["Fully-Correct"].append(summary.get("fully_correct_pct", 0))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_labels))
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax.plot(x, values, marker='o', linewidth=2.5, markersize=8, 
                label=metric, color=COLORS[i])
        # Add value labels
        for xi, v in zip(x, values):
            ax.text(xi, v + 1.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Model", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
    ax.set_title("PhysioRAG Model Performance Comparison (Appendix)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha='right', fontsize=11)
    ax.legend(loc='lower left', fontsize=11)
    ax.set_ylim([70, 105])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "03_model_comparison_line_appendix.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved Figure 3 (appendix): {fig_path}")
    plt.close()


def create_table_1_per_document(evaluations: Dict[str, dict]) -> str:
    """
    Table 1: Metrics per document (using the best model or first model).
    Returns markdown table as string.
    """
    # Use the first model (or best one)
    model_name = sorted(evaluations.keys())[0]
    eval_data = evaluations[model_name]
    per_doc = eval_data.get("per_document_breakdown", {})
    
    rows = []
    rows.append("| Document | Total | Retrieval Correct | Answer-Text Correct | Fully-Correct |")
    rows.append("|----------|-------|------------------|---------------------|---------------|")
    
    for orig_name, friendly_name in DOC_NAMES.items():
        doc_data = per_doc.get(orig_name, {})
        total = doc_data.get("total", 0)
        retrieval = doc_data.get("retrieval_correct", 0)
        answer_text = doc_data.get("answer_text_correct", 0)
        fully = doc_data.get("fully_correct", 0)
        
        if total > 0:
            rows.append(
                f"| {friendly_name} | {total} | {retrieval} | {answer_text} | {fully} |"
            )
    
    return "\n".join(rows)


def create_table_2_model_summary(evaluations: Dict[str, dict]) -> str:
    """
    Table 2: Summary metrics per model.
    Returns markdown table as string.
    """
    rows = []
    rows.append("| Model | Total Qs | Retrieval % | Answer-Letter % | Answer-Text % | Fully-Correct % |")
    rows.append("|-------|----------|-----------|-----------------|---------------|-----------------|")
    
    for model_name in sorted(evaluations.keys()):
        eval_data = evaluations[model_name]
        summary = eval_data.get("summary", {})
        
        total = summary.get("total_questions", 0)
        retrieval = summary.get("retrieval_accuracy_pct", 0)
        answer_letter = summary.get("answer_letter_accuracy_pct", 0)
        answer_text = summary.get("answer_text_accuracy_pct", 0)
        fully = summary.get("fully_correct_pct", 0)
        
        model_display = model_name.replace("-v0.3", "").replace("-Instruct", "")
        rows.append(
            f"| {model_display} | {total} | {retrieval:.2f} | {answer_letter:.2f} | {answer_text:.2f} | {fully:.2f} |"
        )
    
    return "\n".join(rows)


def create_table_3_model_per_document(evaluations: Dict[str, dict]) -> str:
    """
    Table 3: Multi-model breakdown by document (fully_correct metric).
    Returns markdown table as string.
    """
    models = sorted(evaluations.keys())
    
    # Create header
    header = "| Document |"
    for model_name in models:
        model_short = model_name.replace("-v0.3", "").replace("-Instruct", "")[:15]
        header += f" {model_short} |"
    rows = [header]
    
    # Create separator
    sep = "|" + "|".join(["---"] * (len(models) + 1)) + "|"
    rows.append(sep)
    
    # Add data rows
    for orig_name, friendly_name in DOC_NAMES.items():
        row = f"| {friendly_name} |"
        for model_name in models:
            eval_data = evaluations[model_name]
            doc_data = eval_data.get("per_document_breakdown", {}).get(orig_name, {})
            total = doc_data.get("total", 0)
            fully = doc_data.get("fully_correct", 0)
            
            if total > 0:
                pct = (fully / total) * 100
                row += f" {pct:.1f}% ({fully}/{total}) |"
            else:
                row += " - |"
        
        rows.append(row)
    
    return "\n".join(rows)


def save_all_tables(evaluations: Dict[str, dict]):
    """Save all tables to a markdown file."""
    output_file = os.path.join(TABLES_DIR, "report_tables.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# PhysioRAG Evaluation Report - Tables\n\n")
        
        f.write("## Table 1: Per-Document Performance (Best Model)\n")
        f.write(f"*Model: {sorted(evaluations.keys())[0]}*\n\n")
        f.write(create_table_1_per_document(evaluations))
        f.write("\n\n")
        
        f.write("## Table 2: Model Summary Metrics\n\n")
        f.write(create_table_2_model_summary(evaluations))
        f.write("\n\n")
        
        f.write("## Table 3: Fully-Correct Rate by Document and Model\n\n")
        f.write(create_table_3_model_per_document(evaluations))
        f.write("\n\n")
    
    print(f"📋 Saved all tables: {output_file}")


def main():
    """Main execution."""
    print("=" * 70)
    print("  PhysioRAG Report Generation: Figures & Tables")
    print("=" * 70)
    
    # Setup
    try:
        setup_output_dirs()
    except Exception as e:
        print(f"❌ Error creating output directories: {e}")
        return

    # Load all evaluation files
    print(f"\n📂 Loading evaluation files from: {EVAL_SETS_DIR}")
    try:
        evaluations = load_evaluation_files()
    except Exception as e:
        print(f"❌ Error loading evaluation files: {e}")
        import traceback
        traceback.print_exc()
        return

    if not evaluations:
        print("❌ No evaluation files found!")
        print(f"   Checked directory: {EVAL_SETS_DIR}")
        print(f"   Directory exists: {os.path.isdir(EVAL_SETS_DIR)}")
        if os.path.isdir(EVAL_SETS_DIR):
            print(f"   Contents: {os.listdir(EVAL_SETS_DIR)}")
        return
    
    print(f"\n✅ Loaded {len(evaluations)} evaluation(s)\n")
    
    # Generate figures
    print("🎨 Generating figures...")
    try:
        create_figure_1_overall_metrics(evaluations)
        create_figure_2_per_document(evaluations)
        create_figure_3_model_comparison_line(evaluations)
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()

    # Generate tables
    print("\n📋 Generating tables...")
    try:
        save_all_tables(evaluations)
    except Exception as e:
        print(f"❌ Error generating tables: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "=" * 70)
    print("✅ Report generation complete!")
    print(f"📁 Figures saved to: {os.path.abspath(FIGURES_DIR)}")
    print(f"📁 Tables saved to: {os.path.abspath(TABLES_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

