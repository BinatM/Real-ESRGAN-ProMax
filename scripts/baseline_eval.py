import argparse
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import from Real-ESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet

# Import evaluation metrics
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False


def load_model(model_name: str, device: torch.device) -> Tuple[Any, int]:
    """Load a Real-ESRGAN model based on the model name."""
    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    # determine model paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(ROOT_DIR, 'weights', f'{model_name}.pth')

    # Check if model exists, otherwise download it
    if not os.path.isfile(model_path):
        from basicsr.utils.download_util import load_file_from_url
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    return model, netscale, model_path


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


def compute_lpips(img_bgr_a: np.ndarray, img_bgr_b: np.ndarray) -> Optional[float]:
    """Compute LPIPS distance between two BGR images."""
    if not _HAS_LPIPS:
        return None
    # convert to RGB and normalize to [-1, 1]
    img_rgb_a = cv2.cvtColor(img_bgr_a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb_b = cv2.cvtColor(img_bgr_b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    ten_a = torch.from_numpy(img_rgb_a).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    ten_b = torch.from_numpy(img_rgb_b).permute(2, 0, 1).unsqueeze(0) * 2 - 1

    # Move tensors to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = lpips.LPIPS(net='alex').to(device)
    ten_a = ten_a.to(device)
    ten_b = ten_b.to(device)

    with torch.no_grad():
        dist = loss_fn(ten_a, ten_b).item()
    return float(dist)


def create_plots(results: List[Dict[str, Any]], output_dir: str):
    """Create plots for the evaluation metrics."""
    # Extract metrics
    image_names = [os.path.basename(r['image_path']) for r in results]
    psnr_values = [r['psnr_rgb'] for r in results]
    ssim_values = [r['ssim_rgb'] for r in results]
    lpips_values = [r.get('lpips', 0) for r in results]
    proc_times = [r['processing_time'] for r in results]

    # Calculate averages
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean([v for v in lpips_values if v is not None])
    avg_time = np.mean(proc_times)

    # Create figure with subplots
    plt.figure(figsize=(20, 15))

    # PSNR plot
    plt.subplot(2, 2, 1)
    plt.bar(range(len(image_names)), psnr_values, color='blue')
    plt.axhline(y=avg_psnr, color='r', linestyle='-', label=f'Avg: {avg_psnr:.2f} dB')
    plt.xlabel('Images')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Values')
    plt.xticks([])
    plt.legend()

    # SSIM plot
    plt.subplot(2, 2, 2)
    plt.bar(range(len(image_names)), ssim_values, color='green')
    plt.axhline(y=avg_ssim, color='r', linestyle='-', label=f'Avg: {avg_ssim:.4f}')
    plt.xlabel('Images')
    plt.ylabel('SSIM')
    plt.title('SSIM Values')
    plt.xticks([])
    plt.legend()

    # LPIPS plot (if available)
    if any(lpips_values):
        plt.subplot(2, 2, 3)
        plt.bar(range(len(image_names)), lpips_values, color='purple')
        plt.axhline(y=avg_lpips, color='r', linestyle='-', label=f'Avg: {avg_lpips:.4f}')
        plt.xlabel('Images')
        plt.ylabel('LPIPS (lower is better)')
        plt.title('LPIPS Values')
        plt.xticks([])
        plt.legend()

    # Processing time plot
    plt.subplot(2, 2, 4)
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
    plt.figure(figsize=(10, 6))
    labels = ['PSNR (dB)', 'SSIM', 'LPIPS', 'Time (s)']
    values = [avg_psnr, avg_ssim, avg_lpips, avg_time]

    plt.bar(labels, values)
    plt.title('Average Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_plot.png'))


def main():
    parser = argparse.ArgumentParser(description='Baseline evaluation for Real-ESRGAN')
    parser.add_argument('--input', type=str, default='inputs', help='Input folder containing original images')
    parser.add_argument('--output', type=str, default='baseline_results', help='Output folder for results')
    parser.add_argument('--model', type=str, default='RealESRGAN_x4plus',
                        help='Model name: RealESRGAN_x4plus, RealESRNet_x4plus, RealESRGAN_x4plus_anime_6B, or realesr-general-x4v3')
    parser.add_argument('--max_images', type=int, default=100, help='Maximum number of images to process')
    parser.add_argument('--downscale_factor', type=float, default=0.25, help='Factor to downscale the original images')
    parser.add_argument('--tile', type=int, default=0, help='Tile size for large images, 0 for no tiling')
    parser.add_argument('--half', action='store_true', help='Use half precision for inference')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'downscaled'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'upscaled'), exist_ok=True)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, netscale, model_path = load_model(args.model, device)

    # Initialize upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=10,
        pre_pad=0,
        half=args.half,
        device=device
    )

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
    for idx, path in enumerate(tqdm(image_paths)):
        img_name = os.path.splitext(os.path.basename(path))[0]

        # Load original image
        try:
            original_img = load_image_bgr(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        # Downscale image
        downscaled_img = downscale_image(original_img, args.downscale_factor)
        down_path = os.path.join(args.output, 'downscaled', f"{img_name}_down.png")
        cv2.imwrite(down_path, downscaled_img)

        # Upscale with Real-ESRGAN
        start_time = time.time()
        try:
            upscaled_img, _ = upsampler.enhance(downscaled_img, outscale=netscale)
        except Exception as e:
            print(f"Error upscaling {path}: {e}")
            continue
        proc_time = time.time() - start_time

        # Save upscaled image
        up_path = os.path.join(args.output, 'upscaled', f"{img_name}_up.png")
        cv2.imwrite(up_path, upscaled_img)

        # Ensure same size for comparison (original might be slightly different than upscaled)
        if original_img.shape != upscaled_img.shape:
            # Resize original to match upscaled dimensions
            original_img = cv2.resize(original_img, (upscaled_img.shape[1], upscaled_img.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)

        # Compute metrics
        psnr_rgb = calculate_psnr(original_img * 1.0, upscaled_img * 1.0, crop_border=0, input_order='HWC')
        ssim_rgb = calculate_ssim(original_img * 1.0, upscaled_img * 1.0, crop_border=0, input_order='HWC')

        # Compute LPIPS if available
        lpips_val = None
        if _HAS_LPIPS:
            lpips_val = compute_lpips(original_img, upscaled_img)

        # Store results
        result = {
            'image_path': path,
            'psnr_rgb': psnr_rgb,
            'ssim_rgb': ssim_rgb,
            'lpips': lpips_val,
            'processing_time': proc_time,
        }
        results.append(result)

        print(f"Image {idx+1}/{len(image_paths)}: PSNR={psnr_rgb:.2f}dB, SSIM={ssim_rgb:.4f}, "
              f"LPIPS={lpips_val:.4f if lpips_val else 'N/A'}, Time={proc_time:.2f}s")

    # Save results to JSON
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Generate summary
    avg_psnr = np.mean([r['psnr_rgb'] for r in results])
    avg_ssim = np.mean([r['ssim_rgb'] for r in results])
    avg_time = np.mean([r['processing_time'] for r in results])

    if _HAS_LPIPS:
        avg_lpips = np.mean([r['lpips'] for r in results if r['lpips'] is not None])
        lpips_str = f", LPIPS: {avg_lpips:.4f}"
    else:
        lpips_str = ", LPIPS: Not available"

    summary = {
        'model': args.model,
        'num_images': len(results),
        'average_psnr': avg_psnr,
        'average_ssim': avg_ssim,
        'average_processing_time': avg_time,
    }

    if _HAS_LPIPS:
        summary['average_lpips'] = avg_lpips

    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print("\nEvaluation Complete!")
    print(f"Average PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}{lpips_str}, Processing Time: {avg_time:.2f}s")
    print(f"Results saved to {args.output}")

    # Create plots
    create_plots(results, args.output)


if __name__ == "__main__":
    main()