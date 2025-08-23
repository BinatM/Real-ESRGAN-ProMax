#!/usr/bin/env python3
import argparse
import glob
import json
import os
import subprocess
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def create_dirs(base_dir):
    """Create the necessary directories for the evaluation."""
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'original'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'downscaled'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'upscaled'), exist_ok=True)

def downscale_image(input_path, output_path, scale_factor=0.25):
    """Downscale an image by the given scale factor."""
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return False

    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    downscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, downscaled)
    return True

def parse_metrics(output):
    """Parse the metrics from the eval_metrics.py output."""
    lines = output.split('\n')
    metrics = {}

    for line in lines:
        if 'PSNR_RGB:' in line:
            metrics['psnr_rgb'] = float(line.split(':')[1].strip().split()[0])
        elif 'SSIM_RGB:' in line:
            metrics['ssim_rgb'] = float(line.split(':')[1].strip())
        elif 'PSNR_Y:' in line:
            metrics['psnr_y'] = float(line.split(':')[1].strip().split()[0])
        elif 'SSIM_Y:' in line:
            metrics['ssim_y'] = float(line.split(':')[1].strip())
        elif 'LPIPS (alex):' in line:
            metrics['lpips'] = float(line.split(':')[1].strip())

    return metrics

def create_plots(results, output_dir):
    """Create plots for the evaluation metrics."""
    # Extract metrics
    image_names = [os.path.basename(r['image_path']).split('.')[0] for r in results]
    psnr_values = [r['psnr_rgb'] for r in results]
    ssim_values = [r['ssim_rgb'] for r in results]
    psnr_y_values = [r.get('psnr_y', 0) for r in results]
    ssim_y_values = [r.get('ssim_y', 0) for r in results]
    lpips_values = [r.get('lpips', 0) for r in results if 'lpips' in r]
    proc_times = [r['processing_time'] for r in results]

    # Calculate averages
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_psnr_y = np.mean(psnr_y_values)
    avg_ssim_y = np.mean(ssim_y_values)
    avg_time = np.mean(proc_times)
    avg_lpips = np.mean(lpips_values) if lpips_values else 0

    # Create figure with subplots
    plt.figure(figsize=(20, 15))

    # PSNR plot
    plt.subplot(3, 2, 1)
    plt.bar(range(len(image_names)), psnr_values, color='blue')
    plt.axhline(y=avg_psnr, color='r', linestyle='-', label=f'Avg: {avg_psnr:.2f} dB')
    plt.xlabel('Images')
    plt.ylabel('PSNR RGB (dB)')
    plt.title('PSNR RGB Values')
    plt.xticks([])
    plt.legend()

    # SSIM plot
    plt.subplot(3, 2, 2)
    plt.bar(range(len(image_names)), ssim_values, color='green')
    plt.axhline(y=avg_ssim, color='r', linestyle='-', label=f'Avg: {avg_ssim:.4f}')
    plt.xlabel('Images')
    plt.ylabel('SSIM RGB')
    plt.title('SSIM RGB Values')
    plt.xticks([])
    plt.legend()

    # PSNR Y plot
    plt.subplot(3, 2, 3)
    plt.bar(range(len(image_names)), psnr_y_values, color='cyan')
    plt.axhline(y=avg_psnr_y, color='r', linestyle='-', label=f'Avg: {avg_psnr_y:.2f} dB')
    plt.xlabel('Images')
    plt.ylabel('PSNR Y (dB)')
    plt.title('PSNR Y Values')
    plt.xticks([])
    plt.legend()

    # SSIM Y plot
    plt.subplot(3, 2, 4)
    plt.bar(range(len(image_names)), ssim_y_values, color='magenta')
    plt.axhline(y=avg_ssim_y, color='r', linestyle='-', label=f'Avg: {avg_ssim_y:.4f}')
    plt.xlabel('Images')
    plt.ylabel('SSIM Y')
    plt.title('SSIM Y Values')
    plt.xticks([])
    plt.legend()

    # LPIPS plot (if available)
    if lpips_values:
        plt.subplot(3, 2, 5)
        plt.bar(range(len(image_names)), lpips_values, color='purple')
        plt.axhline(y=avg_lpips, color='r', linestyle='-', label=f'Avg: {avg_lpips:.4f}')
        plt.xlabel('Images')
        plt.ylabel('LPIPS (lower is better)')
        plt.title('LPIPS Values')
        plt.xticks([])
        plt.legend()

    # Processing time plot
    plt.subplot(3, 2, 6)
    plt.bar(range(len(image_names)), proc_times, color='orange')
    plt.axhline(y=avg_time, color='r', linestyle='-', label=f'Avg: {avg_time:.2f}s')
    plt.xlabel('Images')
    plt.ylabel('Processing Time (s)')
    plt.title('Processing Times')
    plt.xticks([])
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_plots.png'))

    # Create a summary plot
    plt.figure(figsize=(12, 8))
    labels = ['PSNR RGB (dB)', 'PSNR Y (dB)', 'SSIM RGB', 'SSIM Y', 'LPIPS', 'Time (s)']
    values = [avg_psnr, avg_psnr_y, avg_ssim, avg_ssim_y, avg_lpips, avg_time]

    # Normalize values for better visualization (since they have different scales)
    plt.bar(labels[:4], values[:4], color=['blue', 'cyan', 'green', 'magenta'])

    # Add LPIPS and time on a separate axis if they exist
    if lpips_values:
        ax2 = plt.twinx()
        ax2.bar(labels[4:], values[4:], color=['purple', 'orange'], alpha=0.7)
        ax2.set_ylabel('LPIPS / Time (s)')

    plt.title('Average Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_plot.png'))

def main():
    parser = argparse.ArgumentParser(description='Baseline evaluation for Real-ESRGAN')
    parser.add_argument('--input', type=str, default='/Users/chenzamostiano/Downloads/DIV2K_train_HR',
                        help='Input folder containing original DIV2K images')
    parser.add_argument('--output', type=str, default='baseline_eval',
                        help='Output folder for evaluation results')
    parser.add_argument('--model', type=str, default='RealESRGAN_x4plus',
                        help='Model name for Real-ESRGAN')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to process (default: 100)')
    parser.add_argument('--scale_factor', type=float, default=0.25,
                        help='Scale factor for downscaling (default: 0.25)')
    parser.add_argument('--start_idx', type=int, default=701,
                        help='Starting image index in DIV2K (default: 701)')
    args = parser.parse_args()

    # Create directories
    create_dirs(args.output)

    # Find input images
    image_pattern = os.path.join(args.input, "*.png")
    all_images = sorted(glob.glob(image_pattern))

    if not all_images:
        print(f"Error: No images found in {args.input}")
        return

    # Select images to process
    images_to_process = all_images[:args.num_images]
    print(f"Found {len(all_images)} images, processing {len(images_to_process)}")

    results = []

    # Process each image
    for idx, img_path in enumerate(tqdm(images_to_process, desc="Processing images")):
        img_name = os.path.basename(img_path).split('.')[0]
        print(f"\nProcessing image {idx+1}/{len(images_to_process)}: {img_name}")

        # Copy original image
        original_path = os.path.join(args.output, 'original', f"{img_name}.png")
        os.system(f"cp '{img_path}' '{original_path}'")

        # Downscale image
        downscaled_path = os.path.join(args.output, 'downscaled', f"{img_name}_down.png")
        if not downscale_image(img_path, downscaled_path, args.scale_factor):
            continue

        # Upscale with Real-ESRGAN
        start_time = time.time()
        cmd = f"python inference_realesrgan.py -i '{downscaled_path}' -o '{os.path.join(args.output, 'upscaled')}' -n {args.model} --fp32"
        upscale_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        proc_time = time.time() - start_time

        if upscale_result.returncode != 0:
            print(f"Error upscaling image {img_name}:")
            print(upscale_result.stderr)
            continue

        # Run evaluation
        upscaled_path = os.path.join(args.output, 'upscaled', f"{img_name}_down_out.png")
        eval_cmd = f"python scripts/eval_metrics.py --gt '{original_path}' --restored '{upscaled_path}' --y"
        eval_result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)

        if eval_result.returncode != 0:
            print(f"Error evaluating image {img_name}:")
            print(eval_result.stderr)
            continue

        # Parse metrics
        metrics = parse_metrics(eval_result.stdout)
        metrics['image_path'] = img_path
        metrics['processing_time'] = proc_time
        results.append(metrics)

        print(f"Results for {img_name}:")
        print(f"  PSNR RGB: {metrics.get('psnr_rgb', 'N/A'):.2f} dB")
        print(f"  SSIM RGB: {metrics.get('ssim_rgb', 'N/A'):.4f}")
        print(f"  PSNR Y: {metrics.get('psnr_y', 'N/A'):.2f} dB")
        print(f"  SSIM Y: {metrics.get('ssim_y', 'N/A'):.4f}")
        print(f"  LPIPS: {metrics.get('lpips', 'N/A'):.4f}")
        print(f"  Processing time: {proc_time:.2f}s")

    if not results:
        print("No results to save.")
        return

    # Save detailed results
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Calculate averages
    avg_psnr_rgb = np.mean([r['psnr_rgb'] for r in results if 'psnr_rgb' in r])
    avg_ssim_rgb = np.mean([r['ssim_rgb'] for r in results if 'ssim_rgb' in r])
    avg_psnr_y = np.mean([r['psnr_y'] for r in results if 'psnr_y' in r])
    avg_ssim_y = np.mean([r['ssim_y'] for r in results if 'ssim_y' in r])
    avg_lpips = np.mean([r['lpips'] for r in results if 'lpips' in r])
    avg_time = np.mean([r['processing_time'] for r in results])

    # Save summary
    summary = {
        'model': args.model,
        'num_images': len(results),
        'average_psnr_rgb': float(avg_psnr_rgb),
        'average_ssim_rgb': float(avg_ssim_rgb),
        'average_psnr_y': float(avg_psnr_y),
        'average_ssim_y': float(avg_ssim_y),
        'average_lpips': float(avg_lpips),
        'average_processing_time': float(avg_time)
    }

    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Create plots
    create_plots(results, args.output)

    print("\nEvaluation complete!")
    print(f"Processed {len(results)} images")
    print(f"Average PSNR RGB: {avg_psnr_rgb:.2f} dB")
    print(f"Average SSIM RGB: {avg_ssim_rgb:.4f}")
    print(f"Average PSNR Y: {avg_psnr_y:.2f} dB")
    print(f"Average SSIM Y: {avg_ssim_y:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"Average processing time: {avg_time:.2f}s")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()