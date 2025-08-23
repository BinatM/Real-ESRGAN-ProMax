import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: str) -> Dict[str, Any]:
    """Load evaluation results from a JSON file."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, 'r') as f:
        return json.load(f)


def compare_summaries(baseline_summary: Dict[str, Any], new_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two summary results and calculate improvements."""
    comparison = {}

    # Calculate PSNR improvement (higher is better)
    comparison['psnr_improvement'] = new_summary['average_psnr'] - baseline_summary['average_psnr']
    comparison['psnr_improvement_percent'] = (comparison['psnr_improvement'] / baseline_summary['average_psnr']) * 100

    # Calculate SSIM improvement (higher is better)
    comparison['ssim_improvement'] = new_summary['average_ssim'] - baseline_summary['average_ssim']
    comparison['ssim_improvement_percent'] = (comparison['ssim_improvement'] / baseline_summary['average_ssim']) * 100

    # Calculate processing time improvement (lower is better)
    comparison['time_improvement'] = baseline_summary['average_processing_time'] - new_summary['average_processing_time']
    comparison['time_improvement_percent'] = (comparison['time_improvement'] / baseline_summary['average_processing_time']) * 100

    # Calculate LPIPS improvement if available (lower is better)
    if 'average_lpips' in baseline_summary and 'average_lpips' in new_summary:
        comparison['lpips_improvement'] = baseline_summary['average_lpips'] - new_summary['average_lpips']
        comparison['lpips_improvement_percent'] = (comparison['lpips_improvement'] / baseline_summary['average_lpips']) * 100

    return comparison


def plot_comparison(baseline_summary: Dict[str, Any], new_summary: Dict[str, Any], comparison: Dict[str, Any], output_path: str):
    """Create comparison plots between baseline and new model."""
    # Setup
    plt.figure(figsize=(15, 12))

    # Model names
    baseline_name = baseline_summary.get('model', 'Baseline')
    new_name = new_summary.get('model', 'New Model')

    # PSNR comparison
    plt.subplot(2, 2, 1)
    values = [baseline_summary['average_psnr'], new_summary['average_psnr']]
    bars = plt.bar(['Baseline', 'New Model'], values, color=['blue', 'green'])
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR Comparison: {comparison["psnr_improvement"]:.2f}dB improvement ({comparison["psnr_improvement_percent"]:.2f}%)')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    # SSIM comparison
    plt.subplot(2, 2, 2)
    values = [baseline_summary['average_ssim'], new_summary['average_ssim']]
    bars = plt.bar(['Baseline', 'New Model'], values, color=['blue', 'green'])
    plt.ylabel('SSIM')
    plt.title(f'SSIM Comparison: {comparison["ssim_improvement"]:.4f} improvement ({comparison["ssim_improvement_percent"]:.2f}%)')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    # Time comparison
    plt.subplot(2, 2, 3)
    values = [baseline_summary['average_processing_time'], new_summary['average_processing_time']]
    bars = plt.bar(['Baseline', 'New Model'], values, color=['blue', 'green'])
    plt.ylabel('Time (s)')
    plt.title(f'Processing Time: {comparison["time_improvement"]:.2f}s improvement ({comparison["time_improvement_percent"]:.2f}%)')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}s',
                 ha='center', va='bottom')

    # LPIPS comparison (if available)
    if 'average_lpips' in baseline_summary and 'average_lpips' in new_summary:
        plt.subplot(2, 2, 4)
        values = [baseline_summary['average_lpips'], new_summary['average_lpips']]
        bars = plt.bar(['Baseline', 'New Model'], values, color=['blue', 'green'])
        plt.ylabel('LPIPS (lower is better)')
        plt.title(f'LPIPS: {comparison["lpips_improvement"]:.4f} improvement ({comparison["lpips_improvement_percent"]:.2f}%)')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}',
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")


def plot_detailed_comparison(baseline_results: List[Dict[str, Any]], new_results: List[Dict[str, Any]], output_path: str):
    """Create detailed per-image comparison plots."""
    # Prepare data - need to match images between baseline and new results
    baseline_dict = {os.path.basename(result['image_path']): result for result in baseline_results}
    new_dict = {os.path.basename(result['image_path']): result for result in new_results}

    # Find common images
    common_images = sorted(set(baseline_dict.keys()).intersection(set(new_dict.keys())))

    if not common_images:
        print("No common images found between baseline and new results!")
        return

    # Prepare data for plotting
    image_names = common_images
    psnr_improvements = []
    ssim_improvements = []
    time_improvements = []
    lpips_improvements = []

    for img in common_images:
        baseline = baseline_dict[img]
        new = new_dict[img]

        # Calculate improvements
        psnr_improvements.append(new['psnr_rgb'] - baseline['psnr_rgb'])
        ssim_improvements.append(new['ssim_rgb'] - baseline['ssim_rgb'])
        time_improvements.append(baseline['processing_time'] - new['processing_time'])

        if 'lpips' in baseline and 'lpips' in new and baseline['lpips'] is not None and new['lpips'] is not None:
            lpips_improvements.append(baseline['lpips'] - new['lpips'])
        else:
            lpips_improvements.append(0)

    # Create figure
    plt.figure(figsize=(20, 15))

    # PSNR improvements
    plt.subplot(2, 2, 1)
    plt.bar(range(len(image_names)), psnr_improvements, color=['green' if x > 0 else 'red' for x in psnr_improvements])
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axhline(y=np.mean(psnr_improvements), color='blue', linestyle='--',
                label=f'Avg: {np.mean(psnr_improvements):.2f} dB')
    plt.xlabel('Images')
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('PSNR Improvements (positive is better)')
    plt.xticks([])
    plt.legend()

    # SSIM improvements
    plt.subplot(2, 2, 2)
    plt.bar(range(len(image_names)), ssim_improvements, color=['green' if x > 0 else 'red' for x in ssim_improvements])
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axhline(y=np.mean(ssim_improvements), color='blue', linestyle='--',
                label=f'Avg: {np.mean(ssim_improvements):.4f}')
    plt.xlabel('Images')
    plt.ylabel('SSIM Improvement')
    plt.title('SSIM Improvements (positive is better)')
    plt.xticks([])
    plt.legend()

    # Processing time improvements
    plt.subplot(2, 2, 3)
    plt.bar(range(len(image_names)), time_improvements, color=['green' if x > 0 else 'red' for x in time_improvements])
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axhline(y=np.mean(time_improvements), color='blue', linestyle='--',
                label=f'Avg: {np.mean(time_improvements):.2f}s')
    plt.xlabel('Images')
    plt.ylabel('Time Improvement (s)')
    plt.title('Processing Time Improvements (positive is better)')
    plt.xticks([])
    plt.legend()

    # LPIPS improvements
    if any(lpips_improvements):
        plt.subplot(2, 2, 4)
        plt.bar(range(len(image_names)), lpips_improvements, color=['green' if x > 0 else 'red' for x in lpips_improvements])
        plt.axhline(y=0, color='black', linestyle='-')
        plt.axhline(y=np.mean(lpips_improvements), color='blue', linestyle='--',
                    label=f'Avg: {np.mean(lpips_improvements):.4f}')
        plt.xlabel('Images')
        plt.ylabel('LPIPS Improvement')
        plt.title('LPIPS Improvements (positive is better)')
        plt.xticks([])
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Detailed comparison plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline and new model evaluation results')
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline results directory')
    parser.add_argument('--new', type=str, required=True, help='Path to new model results directory')
    parser.add_argument('--output', type=str, default='comparison_results', help='Output directory for comparison results')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load summary results
    baseline_summary = load_results(os.path.join(args.baseline, 'summary.json'))
    new_summary = load_results(os.path.join(args.new, 'summary.json'))

    # Load detailed results
    baseline_results = load_results(os.path.join(args.baseline, 'results.json'))
    new_results = load_results(os.path.join(args.new, 'results.json'))

    # Compare summaries
    comparison = compare_summaries(baseline_summary, new_summary)

    # Save comparison results
    with open(os.path.join(args.output, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=4)

    # Create comparison plots
    plot_comparison(baseline_summary, new_summary, comparison, os.path.join(args.output, 'summary_comparison.png'))
    plot_detailed_comparison(baseline_results, new_results, os.path.join(args.output, 'detailed_comparison.png'))

    # Print summary
    print("\nComparison Summary:")
    print(f"PSNR: {comparison['psnr_improvement']:.2f} dB improvement ({comparison['psnr_improvement_percent']:.2f}%)")
    print(f"SSIM: {comparison['ssim_improvement']:.4f} improvement ({comparison['ssim_improvement_percent']:.2f}%)")
    print(f"Processing Time: {comparison['time_improvement']:.2f}s improvement ({comparison['time_improvement_percent']:.2f}%)")

    if 'lpips_improvement' in comparison:
        print(f"LPIPS: {comparison['lpips_improvement']:.4f} improvement ({comparison['lpips_improvement_percent']:.2f}%)")


if __name__ == "__main__":
    main()