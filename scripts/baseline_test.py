import argparse
import glob
import json
import os
import subprocess

# Set up paths to make imports work
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_image_bgr(path: str) -> np.ndarray:
    """Load an image in BGR format."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def downscale_image(img: np.ndarray, scale_factor: float = 0.25) -> np.ndarray:
    """Downscale an image by the given scale factor."""
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Simple SSIM calculation."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Compute means
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    # Compute variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    # SSIM formula
    ssim_num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim = ssim_num / ssim_den

    return np.mean(ssim)

def main():
    parser = argparse.ArgumentParser(description='Test Real-ESRGAN on a few images')
    parser.add_argument('--input', type=str, default='/Users/chenzamostiano/Downloads/DIV2K_train_HR', help='Input folder containing original images')
    parser.add_argument('--output', type=str, default='test_results', help='Output folder for results')
    parser.add_argument('--model', type=str, default='RealESRGAN_x4plus', help='Model name')
    parser.add_argument('--max_images', type=int, default=5, help='Maximum number of images to process')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'downscaled'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'upscaled'), exist_ok=True)

    # Get input images
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        image_paths = []
        for ext in ['png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'webp']:
            image_paths.extend(glob.glob(os.path.join(args.input, f'*.{ext}')))
        image_paths = sorted(image_paths)[:args.max_images]

    if not image_paths:
        print(f"No images found in {args.input}")
        return

    results = []

    print(f"Processing {len(image_paths)} images...")
    for idx, path in enumerate(image_paths):
        img_name = os.path.splitext(os.path.basename(path))[0]
        print(f"Processing image {idx+1}/{len(image_paths)}: {img_name}")

        # Load original image
        try:
            original_img = load_image_bgr(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        # Downscale image
        downscaled_img = downscale_image(original_img, 0.25)
        down_path = os.path.join(args.output, 'downscaled', f"{img_name}_down.png")
        cv2.imwrite(down_path, downscaled_img)

        # Use the existing inference script to upscale the image
        up_path = os.path.join(args.output, 'upscaled', f"{img_name}_up.png")

        start_time = time.time()

        # Call the inference script
        cmd = f"python inference_realesrgan.py -i {down_path} -o {os.path.join(args.output, 'upscaled')} -n {args.model} --suffix up"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

        proc_time = time.time() - start_time

        # Load the upscaled image
        upscaled_img = load_image_bgr(up_path)

        # Resize original to match upscaled if needed
        if original_img.shape != upscaled_img.shape:
            original_img = cv2.resize(original_img, (upscaled_img.shape[1], upscaled_img.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)

        # Compute metrics
        psnr_val = calculate_psnr(original_img, upscaled_img)
        ssim_val = calculate_ssim(original_img, upscaled_img)

        # Store results
        result = {
            'image_path': path,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'processing_time': proc_time,
        }
        results.append(result)

        print(f"Results: PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}, Time={proc_time:.2f}s")

    # Save results to JSON
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Generate summary
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    avg_time = np.mean([r['processing_time'] for r in results])

    summary = {
        'model': args.model,
        'num_images': len(results),
        'average_psnr': avg_psnr,
        'average_ssim': avg_ssim,
        'average_processing_time': avg_time,
    }

    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # Create simple bar plot
    plt.figure(figsize=(15, 10))

    # PSNR plot
    plt.subplot(2, 2, 1)
    plt.bar(range(len(image_paths)), [r['psnr'] for r in results], color='blue')
    plt.axhline(y=avg_psnr, color='r', linestyle='-', label=f'Avg: {avg_psnr:.2f}dB')
    plt.title('PSNR Values')
    plt.ylabel('PSNR (dB)')
    plt.xticks([])
    plt.legend()

    # SSIM plot
    plt.subplot(2, 2, 2)
    plt.bar(range(len(image_paths)), [r['ssim'] for r in results], color='green')
    plt.axhline(y=avg_ssim, color='r', linestyle='-', label=f'Avg: {avg_ssim:.4f}')
    plt.title('SSIM Values')
    plt.ylabel('SSIM')
    plt.xticks([])
    plt.legend()

    # Time plot
    plt.subplot(2, 2, 3)
    plt.bar(range(len(image_paths)), [r['processing_time'] for r in results], color='orange')
    plt.axhline(y=avg_time, color='r', linestyle='-', label=f'Avg: {avg_time:.2f}s')
    plt.title('Processing Times')
    plt.ylabel('Time (s)')
    plt.xticks([])
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'metrics_plots.png'))

    print("\nEvaluation Complete!")
    print(f"Average PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}, Processing Time: {avg_time:.2f}s")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()