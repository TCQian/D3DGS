#
# Basketball Gaussian Tracking Utility
# Identifies and visualizes Gaussians that contribute to basketball object
#

import os
from typing import Dict, Optional, Tuple

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from tqdm import tqdm

from gaussian_renderer import render
from utils.sh_utils import eval_sh

PANOPTIC_FAR = 100.0


def get_basketball_mask_from_image(rendered_image, method='manual', mask_path=None):
    """
    Get mask for basketball region.

    Args:
        rendered_image: Rendered image tensor [3, H, W]
        method: 'manual' (requires user input), 'color' (color-based segmentation), or 'load' (load from file)
        mask_path: Path to saved mask file (for 'load' method)

    Returns:
        mask: Binary mask [H, W] where True indicates basketball pixels
    """
    img_np = rendered_image.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)
    H, W = img_np.shape[:2]

    if method == 'load' and mask_path and os.path.exists(mask_path):
        # Load saved mask
        mask = np.load(mask_path)
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        return mask.astype(bool)

    elif method == 'manual':
        # Use OpenCV to let user select region
        print("Click and drag to select basketball region")
        print("Press SPACE or ENTER to confirm, ESC to cancel")
        r = cv2.selectROI("Select Basketball Region", img_np, showCrosshair=True)
        cv2.destroyAllWindows()
        mask = np.zeros((H, W), dtype=bool)
        if len(r) == 4 and r[2] > 0 and r[3] > 0:  # Valid selection
            mask[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])] = True
            print(f"Selected region: x={r[0]}, y={r[1]}, w={r[2]}, h={r[3]}")
        else:
            print("No region selected or selection cancelled")
        return mask

    elif method == 'color':
        # Color-based segmentation (adjust thresholds for basketball color)
        # Basketball color range from sample:
        # rgba(93, 39, 39), rgba(181, 63, 62), rgba(106, 44, 44), rgba(177, 58, 60)
        # These convert to HSV approximately:
        # - H: 0-5 (red range)
        # - S: 0.58-0.67 (moderate to high saturation, 148-171 in 0-255)
        # - V: 0.36-0.71 (moderate brightness, 92-181 in 0-255)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Strict range based on actual basketball colors
        # H: 0-8 (tight red range, avoiding orange)
        # S: 140-180 (moderate to high saturation - excludes low-saturation skin tones)
        # V: 80-200 (moderate brightness - excludes very dark shadows and very bright highlights)
        lower = np.array([0, 140, 80])  # Lower bound: red hue, high saturation, moderate brightness
        upper = np.array([8, 180, 200])  # Upper bound: tight red range, high saturation, moderate-high brightness

        # Also check for red wrap-around (hue near 180) with same strict criteria
        lower_wrap = np.array([175, 140, 80])
        upper_wrap = np.array([179, 180, 200])

        mask1 = cv2.inRange(hsv, lower, upper)
        mask2 = cv2.inRange(hsv, lower_wrap, upper_wrap)
        mask = (mask1 | mask2) > 0

        # Morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)  # Smaller kernel for more precise mask
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Remove small connected components (noise and small detections)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        min_area = 100  # Increased minimum area to filter out small detections (like hands/fingers)
        mask_cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask_cleaned[labels == i] = True
        mask = mask_cleaned

        # Additional filtering: Remove very elongated regions (likely arms/hands, not basketball)
        # Basketball should be roughly circular, so filter by aspect ratio
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    # Basketball should be roughly circular (aspect ratio close to 1)
                    # Filter out elongated regions (aspect ratio > 2.5 suggests arm/hand)
                    if aspect_ratio > 2.5:
                        mask_cleaned[labels == i] = False

        mask = mask_cleaned

        if mask.sum() == 0:
            print("Warning: Color-based segmentation found no pixels. Try:")
            print("  1. Using --basketball_mask_method manual for interactive selection")
            print("  2. Adjusting HSV thresholds in utils/basketball_tracker.py")
            print("  3. Checking if the basketball is visible in the rendered frame")

        return mask.astype(bool)

    else:
        raise ValueError(f"Unknown method: {method}")


def plot_gaussian_visibility_statistics(
    gaussians,
    camera,
    basketball_gaussian_mask,
    output_dir,
    image_size=None,
    opacity_threshold=0.01,
    max_samples_for_hist=50000,
    max_samples_for_all=10000,
):
    """
    Plot comprehensive statistics about Gaussian visibility and opacity.
    For large datasets, only processes basketball Gaussians fully and samples a subset for "all Gaussians" comparison.

    Args:
        gaussians: GaussianModel instance
        camera: Camera viewpoint
        basketball_gaussian_mask: Boolean mask [N] indicating basketball Gaussians
        output_dir: Directory to save plots
        image_size: Optional tuple of (H, W) for frustum culling
        opacity_threshold: Opacity threshold for transparency check
        max_samples_for_hist: Maximum number of samples to use for histograms (for memory efficiency)
        max_samples_for_all: Maximum number of samples from all Gaussians to process (for memory efficiency)
    """
    print("Analyzing Gaussian visibility and opacity statistics...")

    # Get statistics for basketball Gaussians
    basketball_mask_np = basketball_gaussian_mask.cpu().numpy()
    num_basketball = basketball_mask_np.sum()
    num_total = len(basketball_mask_np)

    if num_basketball == 0:
        print("Warning: No basketball Gaussians found, skipping statistics plots")
        return

    # For large datasets, only process basketball Gaussians + a sample of all Gaussians
    print(f"  Processing {num_basketball} basketball Gaussians...")

    # Get basketball Gaussian indices
    basketball_indices = np.where(basketball_mask_np)[0]

    # Sample indices for "all Gaussians" comparison if dataset is too large
    if num_total > max_samples_for_all:
        print(f"  Sampling {max_samples_for_all} Gaussians for comparison (dataset has {num_total} total)...")
        all_sample_indices = np.random.choice(num_total, max_samples_for_all, replace=False)
        # Ensure we include all basketball Gaussians in the sample
        all_sample_indices = np.unique(np.concatenate([all_sample_indices, basketball_indices]))
    else:
        all_sample_indices = np.arange(num_total)

    # Create a temporary mask for sampled Gaussians
    sample_mask = torch.zeros(num_total, dtype=torch.bool, device=basketball_gaussian_mask.device)
    sample_mask[all_sample_indices] = True

    # Process only the sampled subset for "all Gaussians" statistics
    print("  Computing visibility for sampled subset...")

    # Get visibility info for sampled subset only
    # We need to create a temporary GaussianModel with only sampled points, but that's complex
    # Instead, let's process basketball Gaussians directly and sample others separately

    # Process basketball Gaussians directly
    basketball_indices_torch = torch.from_numpy(basketball_indices).to(basketball_gaussian_mask.device)

    # Get basketball Gaussian data
    xyz = gaussians.get_xyz  # [N, 3]
    xyz_basketball = xyz[basketball_indices_torch]  # [num_basketball, 3]

    # Get camera matrices
    if hasattr(camera, 'world_view_transform'):
        w2c = camera.world_view_transform.cuda()
        proj = camera.full_proj_transform.cuda()
    elif isinstance(camera, dict) and 'camera' in camera:
        raster_settings = camera['camera']
        w2c = raster_settings.viewmatrix
        proj = raster_settings.projmatrix
        if w2c.dim() == 3:
            w2c = w2c.squeeze(0)
        if proj.dim() == 3:
            proj = proj.squeeze(0)
        w2c = w2c.T  # CMU viewmatrix is W2C^T; need W2C for p_cam = W2C @ p
    else:
        raise ValueError("Unknown camera format")

    # Transform basketball Gaussians to camera space
    N_basketball = xyz_basketball.shape[0]
    xyz_h_basketball = torch.cat([xyz_basketball, torch.ones(N_basketball, 1, device=xyz_basketball.device)], dim=1)
    xyz_cam_basketball = (w2c @ xyz_h_basketball.T).T
    z_depth_basketball_torch = xyz_cam_basketball[:, 2]
    behind_camera_basketball_torch = z_depth_basketball_torch < 0

    # Get opacity for basketball Gaussians
    opacities = gaussians.get_opacity
    opacity_values_basketball_torch = opacities[basketball_indices_torch]
    if opacity_values_basketball_torch.dim() > 1:
        opacity_values_basketball_torch = opacity_values_basketball_torch.squeeze(-1)
    transparent_basketball_torch = opacity_values_basketball_torch < opacity_threshold

    # Convert to numpy and ensure 1D arrays
    behind_camera_basketball = behind_camera_basketball_torch.cpu().numpy().flatten()
    transparent_basketball = transparent_basketball_torch.cpu().numpy().flatten()
    z_depth_basketball = z_depth_basketball_torch.cpu().numpy().flatten()
    opacity_basketball = opacity_values_basketball_torch.cpu().numpy().flatten()

    # For "all Gaussians" statistics, use a smaller sample
    print("  Sampling subset for 'all Gaussians' comparison...")
    sample_size = min(max_samples_for_all, num_total)
    sample_indices = np.random.choice(num_total, sample_size, replace=False)
    sample_indices_torch = torch.from_numpy(sample_indices).to(basketball_gaussian_mask.device)

    xyz_sample = xyz[sample_indices_torch]
    N_sample = xyz_sample.shape[0]
    xyz_h_sample = torch.cat([xyz_sample, torch.ones(N_sample, 1, device=xyz_sample.device)], dim=1)
    xyz_cam_sample = (w2c @ xyz_h_sample.T).T
    z_depth_sample_torch = xyz_cam_sample[:, 2]
    behind_camera_sample_torch = z_depth_sample_torch < 0

    opacity_sample_torch = opacities[sample_indices_torch]
    if opacity_sample_torch.dim() > 1:
        opacity_sample_torch = opacity_sample_torch.squeeze(-1)
    transparent_sample_torch = opacity_sample_torch < opacity_threshold

    # Convert sample to numpy and ensure 1D arrays
    behind_camera_all = behind_camera_sample_torch.cpu().numpy().flatten()
    transparent_all = transparent_sample_torch.cpu().numpy().flatten()
    z_depth_all = z_depth_sample_torch.cpu().numpy().flatten()
    opacity_all = opacity_sample_torch.cpu().numpy().flatten()

    # Free GPU memory
    del xyz_basketball, xyz_cam_basketball, z_depth_basketball_torch, opacity_values_basketball_torch
    del xyz_sample, xyz_cam_sample, z_depth_sample_torch, opacity_sample_torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # For frustum checking, we'll skip it for the sample to save memory
    outside_frustum_all = None
    outside_frustum_basketball = None

    # Sample data for histograms if dataset is too large
    def sample_if_needed(data, max_samples):
        if len(data) > max_samples:
            indices = np.random.choice(len(data), max_samples, replace=False)
            return data[indices]
        return data

    print("  Creating visualizations...")

    # Ensure all arrays are 1D (defensive - in case flatten didn't work)
    if z_depth_basketball.ndim > 1:
        z_depth_basketball = z_depth_basketball.flatten()
    if opacity_basketball.ndim > 1:
        opacity_basketball = opacity_basketball.flatten()
    if behind_camera_basketball.ndim > 1:
        behind_camera_basketball = behind_camera_basketball.flatten()
    if transparent_basketball.ndim > 1:
        transparent_basketball = transparent_basketball.flatten()

    # Compute valid masks
    valid_basketball_mask = ~(behind_camera_basketball | transparent_basketball)
    valid_all_mask = ~(behind_camera_all | transparent_all)

    # Create comprehensive statistics plot
    fig = plt.figure(figsize=(16, 12))

    # 1. Summary statistics bar chart
    ax1 = plt.subplot(3, 3, 1)
    categories = ['Total (sampled)', 'Basketball']
    # Scale counts to estimate full dataset (for sampled data)
    sample_scale = num_total / len(behind_camera_all) if len(behind_camera_all) > 0 else 1.0
    valid_counts = [int(valid_all_mask.sum() * sample_scale), valid_basketball_mask.sum()]
    behind_counts = [int(behind_camera_all.sum() * sample_scale), behind_camera_basketball.sum()]
    transparent_counts = [int(transparent_all.sum() * sample_scale), transparent_basketball.sum()]
    if outside_frustum_all is not None:
        outside_counts = [int(outside_frustum_all.sum() * sample_scale), outside_frustum_basketball.sum()]
    else:
        outside_counts = [0, 0]

    x = np.arange(len(categories))
    width = 0.2
    ax1.bar(x - 1.5 * width, valid_counts, width, label='Valid', color='green', alpha=0.7)
    ax1.bar(x - 0.5 * width, behind_counts, width, label='Behind Camera', color='red', alpha=0.7)
    if outside_frustum_all is not None:
        ax1.bar(x + 0.5 * width, outside_counts, width, label='Outside Frustum', color='orange', alpha=0.7)
    ax1.bar(x + 1.5 * width, transparent_counts, width, label='Transparent', color='purple', alpha=0.7)
    ax1.set_xlabel('Gaussian Group')
    ax1.set_ylabel('Count')
    ax1.set_title('Visibility Statistics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Percentage comparison
    ax2 = plt.subplot(3, 3, 2)
    total_percentages = [
        valid_all_mask.sum() / len(behind_camera_all) * 100,
        behind_camera_all.sum() / len(behind_camera_all) * 100,
        transparent_all.sum() / len(transparent_all) * 100,
    ]
    if outside_frustum_all is not None:
        total_percentages.append(outside_frustum_all.sum() / len(outside_frustum_all) * 100)

    basketball_percentages = [
        valid_basketball_mask.sum() / num_basketball * 100,
        behind_camera_basketball.sum() / num_basketball * 100,
        transparent_basketball.sum() / num_basketball * 100,
    ]
    if outside_frustum_all is not None:
        basketball_percentages.append(outside_frustum_basketball.sum() / num_basketball * 100)

    labels = ['Valid', 'Behind Camera', 'Transparent']
    if outside_frustum_all is not None:
        labels.append('Outside Frustum')

    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width / 2, total_percentages, width, label='All Gaussians', alpha=0.7)
    ax2.bar(x + width / 2, basketball_percentages, width, label='Basketball Gaussians', alpha=0.7)
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Visibility Percentages')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Opacity distribution histogram
    ax3 = plt.subplot(3, 3, 3)
    opacity_all_sampled = sample_if_needed(opacity_all, max_samples_for_hist)
    opacity_basketball_sampled = sample_if_needed(opacity_basketball, max_samples_for_hist)
    ax3.hist(opacity_all_sampled, bins=50, alpha=0.5, label='All Gaussians', color='blue', density=True)
    ax3.hist(opacity_basketball_sampled, bins=50, alpha=0.5, label='Basketball', color='red', density=True)
    ax3.axvline(opacity_threshold, color='black', linestyle='--', label=f'Threshold ({opacity_threshold})')
    ax3.set_xlabel('Opacity')
    ax3.set_ylabel('Density')
    ax3.set_title('Opacity Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Depth (Z) distribution histogram
    ax4 = plt.subplot(3, 3, 4)
    # Only plot positive depths (in front of camera)
    z_positive_all = z_depth_all[z_depth_all > 0]
    z_positive_basketball = z_depth_basketball[z_depth_basketball > 0]
    z_positive_all_sampled = sample_if_needed(z_positive_all, max_samples_for_hist)
    z_positive_basketball_sampled = sample_if_needed(z_positive_basketball, max_samples_for_hist)
    if len(z_positive_all_sampled) > 0:
        ax4.hist(z_positive_all_sampled, bins=50, alpha=0.5, label='All Gaussians', color='blue', density=True)
    if len(z_positive_basketball_sampled) > 0:
        ax4.hist(z_positive_basketball_sampled, bins=50, alpha=0.5, label='Basketball', color='red', density=True)
    ax4.set_xlabel('Depth (Z in camera space)')
    ax4.set_ylabel('Density')
    ax4.set_title('Depth Distribution (Front of Camera)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Free memory after creating histograms - keep sampled versions for CDF plot
    del z_positive_all, z_positive_all_sampled, z_positive_basketball_sampled
    # Now we can delete the large arrays (but keep opacity_all_sampled for CDF)
    del z_depth_all
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Opacity vs Depth scatter (for basketball Gaussians)
    ax5 = plt.subplot(3, 3, 5)
    # Filter to valid basketball Gaussians
    if valid_basketball_mask.sum() > 0:
        valid_z = z_depth_basketball[valid_basketball_mask]
        valid_opacity = opacity_basketball[valid_basketball_mask]
        # Sample if too many points for scatter plot
        if len(valid_z) > max_samples_for_hist:
            indices = np.random.choice(len(valid_z), max_samples_for_hist, replace=False)
            valid_z = valid_z[indices]
            valid_opacity = valid_opacity[indices]
        scatter = ax5.scatter(valid_z, valid_opacity, alpha=0.5, s=10, c=valid_opacity, cmap='viridis')
        ax5.set_xlabel('Depth (Z)')
        ax5.set_ylabel('Opacity')
        ax5.set_title('Basketball Gaussians: Opacity vs Depth')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Opacity')

    # 6. Visibility status pie chart (all Gaussians - sampled)
    ax6 = plt.subplot(3, 3, 6)
    valid_count = valid_all_mask.sum()
    invalid_count = len(valid_all_mask) - valid_count
    sizes = [valid_count, invalid_count]
    labels_pie = ['Valid', 'Invalid']
    colors_pie = ['green', 'red']
    ax6.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax6.set_title('All Gaussians (sampled): Valid vs Invalid')

    # 7. Visibility status pie chart (basketball Gaussians)
    ax7 = plt.subplot(3, 3, 7)
    valid_basketball_count = valid_basketball_mask.sum()
    invalid_basketball_count = num_basketball - valid_basketball_count
    sizes = [valid_basketball_count, invalid_basketball_count]
    ax7.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax7.set_title('Basketball Gaussians: Valid vs Invalid')

    # 8. Cumulative opacity distribution
    ax8 = plt.subplot(3, 3, 8)
    # Use already sampled data for cumulative distribution
    opacity_basketball_sorted = np.sort(opacity_basketball_sampled)
    opacity_all_sorted = np.sort(opacity_all_sampled)
    y_all = np.arange(1, len(opacity_all_sorted) + 1) / len(opacity_all_sorted)
    y_basketball = np.arange(1, len(opacity_basketball_sorted) + 1) / len(opacity_basketball_sorted)
    ax8.plot(opacity_all_sorted, y_all, label='All Gaussians', alpha=0.7)
    ax8.plot(opacity_basketball_sorted, y_basketball, label='Basketball', alpha=0.7)
    ax8.axvline(opacity_threshold, color='black', linestyle='--', label=f'Threshold ({opacity_threshold})')
    ax8.set_xlabel('Opacity')
    ax8.set_ylabel('Cumulative Probability')
    ax8.set_title('Cumulative Opacity Distribution')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Final cleanup - delete all large arrays
    del opacity_all, opacity_all_sampled, opacity_all_sorted, opacity_basketball_sorted
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 9. Statistics text summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Compute depth stats for basketball (positive depths only)
    z_positive_basketball = z_depth_basketball[z_depth_basketball > 0]

    # Compute outside frustum stats for basketball
    outside_basketball_count = outside_frustum_basketball.sum() if outside_frustum_basketball is not None else 0
    outside_basketball_pct = (
        outside_frustum_basketball.sum() / num_basketball * 100 if outside_frustum_basketball is not None else 0.0
    )

    # Compute depth stats
    z_mean_str = f"{z_positive_basketball.mean():.4f}" if len(z_positive_basketball) > 0 else "N/A"
    z_median_str = f"{np.median(z_positive_basketball):.4f}" if len(z_positive_basketball) > 0 else "N/A"
    z_min_str = f"{z_positive_basketball.min():.4f}" if len(z_positive_basketball) > 0 else "N/A"
    z_max_str = f"{z_positive_basketball.max():.4f}" if len(z_positive_basketball) > 0 else "N/A"

    # Scale counts for display (estimated from sample)
    valid_count_scaled = int(valid_all_mask.sum() * sample_scale)
    behind_count_scaled = int(behind_camera_all.sum() * sample_scale)
    transparent_count_scaled = int(transparent_all.sum() * sample_scale)
    outside_count_scaled = int(outside_frustum_all.sum() * sample_scale) if outside_frustum_all is not None else 0

    stats_text = f"""
    STATISTICS SUMMARY
    ===================
    
    Total Gaussians: {num_total:,}
    Basketball Gaussians: {num_basketball:,} ({num_basketball/num_total*100:.2f}%)
    Sample size for "All": {len(behind_camera_all):,}
    
    ALL GAUSSIANS (estimated from sample):
    - Valid: ~{valid_count_scaled:,} ({valid_all_mask.sum()/len(behind_camera_all)*100:.2f}%)
    - Behind Camera: ~{behind_count_scaled:,} ({behind_camera_all.sum()/len(behind_camera_all)*100:.2f}%)
    - Transparent: ~{transparent_count_scaled:,} ({transparent_all.sum()/len(transparent_all)*100:.2f}%)
    - Outside Frustum: ~{outside_count_scaled:,} (N/A - not computed for sample)
    
    BASKETBALL GAUSSIANS:
    - Valid: {valid_basketball_mask.sum():,} ({valid_basketball_mask.sum()/num_basketball*100:.2f}%)
    - Behind Camera: {behind_camera_basketball.sum():,} ({behind_camera_basketball.sum()/num_basketball*100:.2f}%)
    - Transparent: {transparent_basketball.sum():,} ({transparent_basketball.sum()/num_basketball*100:.2f}%)
    - Outside Frustum: {outside_basketball_count:,} ({outside_basketball_pct:.2f}% if checked)
    
    OPACITY STATS (Basketball):
    - Mean: {opacity_basketball.mean():.4f}
    - Median: {np.median(opacity_basketball):.4f}
    - Min: {opacity_basketball.min():.4f}
    - Max: {opacity_basketball.max():.4f}
    - Std: {opacity_basketball.std():.4f}
    
    DEPTH STATS (Basketball, front only):
    - Mean: {z_mean_str}
    - Median: {z_median_str}
    - Min: {z_min_str}
    - Max: {z_max_str}
    """
    ax9.text(0.1, 0.5, stats_text, fontsize=9, family='monospace', verticalalignment='center')

    plt.tight_layout()
    stats_plot_path = os.path.join(output_dir, "gaussian_visibility_statistics.png")
    plt.savefig(stats_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visibility statistics plot to {stats_plot_path}")

    # Save detailed statistics to file
    stats_file_path = os.path.join(output_dir, "gaussian_visibility_statistics.txt")
    with open(stats_file_path, 'w') as f:
        f.write(stats_text)
    print(f"Saved detailed statistics to {stats_file_path}")


def project_gaussians_to_2d(gaussians, camera, image_size):
    """
    Project 3D Gaussian centers to 2D screen space.

    Args:
        gaussians: GaussianModel instance
        camera: Camera viewpoint
        image_size: Tuple of (H, W)

    Returns:
        means2D: [N, 2] projected positions in pixel coordinates
        radii: [N] approximate radii in pixels
    """
    # Get camera matrices
    if hasattr(camera, 'world_view_transform'):
        w2c = camera.world_view_transform.cuda()
        proj = camera.full_proj_transform.cuda()
    elif isinstance(camera, dict) and 'camera' in camera:
        # PanopticSports/CMU: setup_camera stores viewmatrix=W2C^T, projmatrix=(Opengl@W2C)^T
        # (transposed for the C++ rasterizer). For p_clip=(Opengl@W2C)@p use proj.T; for
        # p_cam=W2C@p use viewmatrix.T (setup_camera passes viewmatrix=w2c.transpose(1,2)).
        raster_settings = camera['camera']
        w2c = raster_settings.viewmatrix
        proj = raster_settings.projmatrix
        if w2c.dim() == 3:
            w2c = w2c.squeeze(0)
        if proj.dim() == 3:
            proj = proj.squeeze(0)
        w2c = w2c.T  # viewmatrix is W2C^T; need W2C for xyz_cam = W2C @ p
        proj = proj.T  # projmatrix is (Opengl@W2C)^T; use proj.T so proj@p gives correct clip
    else:
        raise ValueError("Unknown camera format")

    # Get 3D positions
    xyz = gaussians.get_xyz  # [N, 3]
    N = xyz.shape[0]

    # Homogeneous coordinates
    xyz_h = torch.cat([xyz, torch.ones(N, 1, device=xyz.device)], dim=1)  # [N, 4]

    # Transform to camera space (for radii and depth)
    xyz_cam = (w2c @ xyz_h.T).T  # [N, 4]

    # Project to NDC/clip: proj is world-to-clip (same as rasterizer's projmatrix).
    # Must apply to world xyz_h, not camera; rasterizer does p_hom = projmatrix @ p_orig (world).
    xyz_screen = (proj @ xyz_h.T).T  # [N, 4]

    # Perspective divide
    xyz_screen = xyz_screen / (xyz_screen[:, 3:4] + 1e-7)

    # Convert to pixel coordinates
    H, W = image_size
    means2D = torch.zeros(N, 2, device=xyz.device)
    # Extract x and y coordinates, ensuring they're 1D tensors
    x_coord = xyz_screen[:, 0].contiguous().view(-1)
    y_coord = xyz_screen[:, 1].contiguous().view(-1)
    means2D[:, 0] = (x_coord + 1) * W / 2
    means2D[:, 1] = (y_coord + 1) * H / 2

    # Compute exact radii using the same method as CUDA rasterizer
    # This matches forward.cu:preprocessCUDA() lines 215-232
    try:
        # Get camera intrinsics
        if hasattr(camera, 'FoVx') and hasattr(camera, 'FoVy'):
            import math

            tan_fovx = math.tan(camera.FoVx * 0.5)
            tan_fovy = math.tan(camera.FoVy * 0.5)
            focal_x = W / (2 * tan_fovx)
            focal_y = H / (2 * tan_fovy)
        elif isinstance(camera, dict) and 'camera' in camera:
            raster_settings = camera['camera']
            tan_fovx = raster_settings.tanfovx
            tan_fovy = raster_settings.tanfovy
            focal_x = W / (2 * tan_fovx)
            focal_y = H / (2 * tan_fovy)
        else:
            raise ValueError("Cannot determine camera intrinsics")

        # Get 3D covariance matrices (uses get_covariance which matches CUDA)
        cov3D = gaussians.get_covariance(1.0)  # [N, 6] format: [xx, xy, xz, yy, yz, zz]

        # Transform points to camera space
        xyz_cam_3d = xyz_cam[:, :3]  # [N, 3]

        # Extract viewmatrix rotation part (first 3x3)
        w2c_rot = w2c[:3, :3]  # [3, 3]

        # Vectorized computation of 2D covariance and radii
        t = xyz_cam_3d  # [N, 3]

        # Clamp to frustum limits (matches CUDA lines 82-87)
        limx = 1.3 * tan_fovx
        limy = 1.3 * tan_fovy
        txtz = t[:, 0] / (t[:, 2] + 1e-7)
        tytz = t[:, 1] / (t[:, 2] + 1e-7)
        t_clamped = torch.stack(
            [torch.clamp(txtz, -limx, limx) * t[:, 2], torch.clamp(tytz, -limy, limy) * t[:, 2], t[:, 2]], dim=1
        )  # [N, 3]

        # Jacobian matrices J for all Gaussians (CUDA lines 89-92)
        tz = t_clamped[:, 2:3]  # [N, 1]
        J = torch.zeros(N, 3, 3, device=xyz.device)
        J[:, 0, 0] = focal_x / tz.squeeze()
        J[:, 0, 2] = -(focal_x * t_clamped[:, 0]) / (tz.squeeze() * tz.squeeze())
        J[:, 1, 1] = focal_y / tz.squeeze()
        J[:, 1, 2] = -(focal_y * t_clamped[:, 1]) / (tz.squeeze() * tz.squeeze())

        # Transformation matrices T = W * J (CUDA lines 94-99)
        T = w2c_rot.unsqueeze(0) @ J  # [N, 3, 3]

        # Reconstruct 3x3 covariance matrices from 6 values (CUDA lines 101-104)
        Vrk = torch.zeros(N, 3, 3, device=xyz.device)
        Vrk[:, 0, 0] = cov3D[:, 0]  # xx
        Vrk[:, 0, 1] = cov3D[:, 1]  # xy
        Vrk[:, 0, 2] = cov3D[:, 2]  # xz
        Vrk[:, 1, 0] = cov3D[:, 1]  # xy (symmetric)
        Vrk[:, 1, 1] = cov3D[:, 3]  # yy
        Vrk[:, 1, 2] = cov3D[:, 4]  # yz
        Vrk[:, 2, 0] = cov3D[:, 2]  # xz (symmetric)
        Vrk[:, 2, 1] = cov3D[:, 4]  # yz (symmetric)
        Vrk[:, 2, 2] = cov3D[:, 5]  # zz

        # 2D covariance: cov = T^T * Vrk^T * T (CUDA line 106)
        cov2D = T.transpose(1, 2) @ Vrk.transpose(1, 2) @ T  # [N, 3, 3]

        # Apply low-pass filter (CUDA lines 110-111)
        cov2D[:, 0, 0] += 0.3
        cov2D[:, 1, 1] += 0.3

        # Extract 2D covariance values (only need 2x2 upper left)
        cov_xx = cov2D[:, 0, 0]  # [N]
        cov_xy = cov2D[:, 0, 1]  # [N]
        cov_yy = cov2D[:, 1, 1]  # [N]

        # Compute eigenvalues (CUDA lines 229-231)
        det = cov_xx * cov_yy - cov_xy * cov_xy  # [N]
        mid = 0.5 * (cov_xx + cov_yy)  # [N]
        lambda1 = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))  # [N]
        lambda2 = mid - torch.sqrt(torch.clamp(mid * mid - det, min=0.1))  # [N]

        # Compute radius (CUDA line 232): ceil(3.0 * sqrt(max(lambda1, lambda2)))
        radii = torch.ceil(3.0 * torch.sqrt(torch.maximum(lambda1, lambda2)))  # [N]

    except Exception as e:
        # Fallback to approximate method if exact computation fails
        print(f"Warning: Exact radius computation failed ({e}), using approximation")
        scales = gaussians.get_scaling  # [N, 3]
        max_scale = scales.max(dim=1)[0]  # [N]
        distances = xyz_cam[:, 2].abs().contiguous().view(-1)  # [N]
        radii = (max_scale / (distances + 1e-7)) * W * 0.5

    return means2D, radii


def identify_basketball_gaussians(gaussians, samples, threshold_radius=1.0, min_contributions=5):
    """
    Identify Gaussians that contribute to basketball pixels.

    Supports one or more (camera, mask) samples. When multiple samples are given,
    only Gaussians that meet min_contributions in **every** sample are kept
    (overlap/intersection across views/frames). With one sample, the single-sample
    qualified set is used.

    Args:
        gaussians: GaussianModel instance
        samples: List of (camera, basketball_mask) tuples. Each basketball_mask is [H, W].
        threshold_radius: Maximum distance from pixel center to consider contribution
        min_contributions: Minimum number of pixels a Gaussian must contribute to (per sample)

    Returns:
        basketball_gaussian_mask: Boolean mask [N] indicating basketball Gaussians
        contribution_map: Dict mapping gaussian_idx -> list of (pixel_y, pixel_x) from all samples
    """
    if not samples:
        return torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda"), {}

    per_sample_qualified = []  # list of sets: g_indices that meet min_contributions in each sample
    contribution_map = {}
    any_basketball_pixels = False

    for s, (camera, basketball_mask) in enumerate(samples):
        H, W = basketball_mask.shape

        # Set timestamp for dynamic scenes (uses this camera's time)
        t = getattr(camera, "timestamp", 0)
        if hasattr(gaussians, "set_timestamp"):
            gaussians.set_timestamp(t, training=False)

        means2D, radii = project_gaussians_to_2d(gaussians, camera, (H, W))

        # Get pixel coordinates where basketball is
        pixel_y, pixel_x = np.where(basketball_mask)

        if len(pixel_y) == 0:
            continue
        any_basketball_pixels = True

        # Per-sample contribution counts
        gaussian_contributions = {}
        contribution_map_this = {}

        means2D_np = means2D.detach().cpu().numpy()
        radii_np = radii.detach().cpu().numpy()

        n_samples = len(samples)
        desc = f"Processing pixels (sample {s + 1}/{n_samples})" if n_samples > 1 else "Processing pixels"
        print(f"Analyzing {len(pixel_y)} basketball pixels in sample {s + 1}/{n_samples}...")
        for py, px in tqdm(zip(pixel_y, pixel_x), total=len(pixel_y), desc=desc):
            distances = np.sqrt((means2D_np[:, 0] - px) ** 2 + (means2D_np[:, 1] - py) ** 2)
            nearby_mask = distances < (radii_np + threshold_radius)
            nearby_indices = np.where(nearby_mask)[0]

            for g_idx in nearby_indices:
                if g_idx not in gaussian_contributions:
                    gaussian_contributions[g_idx] = 0
                    contribution_map_this[g_idx] = []
                gaussian_contributions[g_idx] += 1
                contribution_map_this[g_idx].append((py, px))

        # Qualified in this sample: count >= min_contributions
        qualified_this = {g_idx for g_idx, count in gaussian_contributions.items() if count >= min_contributions}
        per_sample_qualified.append(qualified_this)

        for g_idx, pixels in contribution_map_this.items():
            contribution_map.setdefault(g_idx, []).extend(pixels)

    if not any_basketball_pixels:
        print("Warning: No pixels in basketball mask in any sample!")
        print("This means the color-based segmentation didn't find any basketball pixels.")
        print("Suggestions:")
        print("  1. Try --basketball_mask_method manual for interactive selection")
        print("  2. Check if the basketball is visible in the rendered frame(s)")
        print("  3. The rendered image might have different colors than expected")
        print("  4. Adjust HSV thresholds in utils/basketball_tracker.py if needed")
        return torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda"), {}

    # Overlap: keep only Gaussians qualified in every sample that had basketball pixels
    gaussian_qualified = per_sample_qualified[0]
    for q in per_sample_qualified[1:]:
        gaussian_qualified = gaussian_qualified & q

    basketball_gaussian_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    for g_idx in gaussian_qualified:
        basketball_gaussian_mask[g_idx] = True

    print(
        f"Identified {basketball_gaussian_mask.sum().item()} basketball Gaussians "
        f"out of {len(basketball_gaussian_mask)} total"
    )

    return basketball_gaussian_mask, contribution_map


def color_basketball_gaussians(gaussians, basketball_mask, color=[200 / 255, 87 / 255, 83 / 255], restore_colors=None):
    """
    Temporarily override colors of basketball Gaussians for visualization.

    Args:
        gaussians: GaussianModel instance
        basketball_mask: Boolean mask [N] indicating basketball Gaussians
        color: RGB color [3] to apply (default: red)
        restore_colors: If provided, restore these colors instead of modifying

    Returns:
        original_colors: Original color values (if not restoring)
    """
    # Use _fwd_features_dc if available (used during rendering), otherwise fall back to _features_dc
    if hasattr(gaussians, '_fwd_features_dc') and gaussians._fwd_features_dc is not None:
        features_tensor = gaussians._fwd_features_dc
    else:
        features_tensor = gaussians._features_dc

    if restore_colors is not None:
        # Restore original colors - modify the tensor that is used during rendering
        features_tensor.data = restore_colors
        return None

    # Store original colors from the tensor that will be used during rendering
    original_colors = features_tensor.clone()

    # Set basketball Gaussians to specified color
    # Convert RGB to SH DC coefficient
    # SH DC = RGB - 0.5 (approximate conversion)
    # features_tensor has shape [N, 1, 3] (transposed from [N, 3, 1])
    sh_color = torch.tensor(color, device=features_tensor.device).view(1, 1, 3) - 0.5
    num_basketball = basketball_mask.sum().item()
    if num_basketball > 0:
        features_tensor[basketball_mask] = sh_color.repeat(num_basketball, 1, 1)

    return original_colors


def plot_world_space_positions(trajectories, output_dir, max_points=10000):
    """
    Plot basketball Gaussians in world space (3D positions over time).

    Args:
        trajectories: Array of [T, N_basketball, 3] positions over time
        output_dir: Directory to save plots
        max_points: Maximum number of points to plot (for performance)
    """
    if trajectories.shape[1] == 0:
        print("Warning: No basketball Gaussians to plot in world space")
        return

    print("Plotting world space positions...")

    # Flatten trajectories: [T, N_basketball, 3] -> [T*N_basketball, 3]
    T, N_basketball, _ = trajectories.shape
    all_positions = trajectories.reshape(-1, 3)  # [T*N_basketball, 3]

    # Create time indices for coloring (before sampling)
    time_indices = np.repeat(np.arange(T), N_basketball)

    # Sample if too many points
    if len(all_positions) > max_points:
        indices = np.random.choice(len(all_positions), max_points, replace=False)
        all_positions = all_positions[indices]
        time_indices = time_indices[indices]
        print(f"  Sampling {max_points} points from {T*N_basketball} total positions")

    # Create 3D scatter plot colored by time
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        all_positions[:, 0],
        all_positions[:, 1],
        all_positions[:, 2],
        c=time_indices,
        cmap='viridis',
        alpha=0.6,
        s=10,
        edgecolors='none',
    )

    ax.set_xlabel('X (World Space)')
    ax.set_ylabel('Y (World Space)')
    ax.set_zlabel('Z (World Space)')
    ax.set_title('Basketball Gaussians in World Space (Colored by Time)')
    plt.colorbar(scatter, ax=ax, label='Frame Index')

    plt.tight_layout()
    world_space_path = os.path.join(output_dir, "world_space_positions.png")
    plt.savefig(world_space_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved world space plot to {world_space_path}")


def plot_camera_space_positions(
    gaussians,
    cameras,
    basketball_mask,
    trajectories,
    output_dir,
    max_frames=None,
    frame_interval=1,
    far_depth=None,
):
    """
    Plot basketball Gaussians in camera space for each frame.

    Args:
        gaussians: GaussianModel instance
        cameras: List of cameras
        basketball_mask: Boolean mask [N] indicating basketball Gaussians
        trajectories: Array of [T, N_basketball, 3] world space positions
        output_dir: Directory to save plots
        max_frames: Maximum number of frames to plot (None = all)
        frame_interval: Plot every Nth frame
        far_depth: Max depth (Z in camera space); points with z >= far_depth are
            out of view. Defaults to scene.cmu_dataset.PANOPTIC_FAR when
            PanopticDataset is used.
    """
    if far_depth is None:
        far_depth = PANOPTIC_FAR
    if trajectories.shape[1] == 0:
        print("Warning: No basketball Gaussians to plot in camera space")
        return

    if cameras is None or len(cameras) == 0:
        print("Warning: No cameras available for camera space plotting")
        return

    print("Plotting camera space positions for each frame...")

    T = trajectories.shape[0]
    if max_frames is not None:
        T = min(T, max_frames)

    basketball_indices = np.where(basketball_mask.cpu().numpy())[0]
    if len(basketball_indices) == 0:
        print("Warning: No basketball Gaussian indices found")
        return

    basketball_indices_torch = torch.from_numpy(basketball_indices).to(basketball_mask.device)

    # Create output directory for camera space plots
    camera_space_dir = os.path.join(output_dir, "camera_space")
    os.makedirs(camera_space_dir, exist_ok=True)

    for t in tqdm(range(0, T, frame_interval), desc="Plotting camera space"):
        try:
            # Set timestamp
            gaussians.set_timestamp(t, training=False)

            # Get camera
            camera = cameras[t] if t < len(cameras) else cameras[0]

            # Get camera matrices
            if hasattr(camera, 'world_view_transform'):
                w2c = camera.world_view_transform
                if not isinstance(w2c, torch.Tensor):
                    w2c = torch.tensor(w2c, device=basketball_mask.device)
                if w2c.device != basketball_mask.device:
                    w2c = w2c.to(basketball_mask.device)
                # Get projection matrix and image size
                proj = camera.full_proj_transform
                if not isinstance(proj, torch.Tensor):
                    proj = torch.tensor(proj, device=basketball_mask.device)
                if proj.device != basketball_mask.device:
                    proj = proj.to(basketball_mask.device)
                H = camera.image_height if hasattr(camera, 'image_height') else None
                W = camera.image_width if hasattr(camera, 'image_width') else None
            elif isinstance(camera, dict) and 'camera' in camera:
                raster_settings = camera['camera']
                w2c = raster_settings.viewmatrix
                proj = raster_settings.projmatrix
                H = raster_settings.image_height if hasattr(raster_settings, 'image_height') else None
                W = raster_settings.image_width if hasattr(raster_settings, 'image_width') else None
            else:
                print(f"Warning: Unknown camera format at frame {t}, skipping")
                continue

            # Ensure w2c and proj are 2D [4, 4] (PanopticSports/CMU use batched [1, 4, 4])
            if w2c.dim() == 3:
                w2c = w2c.squeeze(0)
            if proj.dim() == 3:
                proj = proj.squeeze(0)
            # CMU setup_camera stores viewmatrix=W2C^T; use W2C for p_cam = W2C @ p
            if isinstance(camera, dict) and 'camera' in camera:
                w2c = w2c.T

            # Get current basketball positions in world space
            xyz = gaussians.get_xyz  # [N, 3]
            xyz_basketball = xyz[basketball_indices_torch]  # [N_basketball, 3]

            # Transform to camera space
            N_basketball = xyz_basketball.shape[0]
            xyz_h_basketball = torch.cat(
                [xyz_basketball, torch.ones(N_basketball, 1, device=xyz_basketball.device)], dim=1
            )
            xyz_cam = (w2c @ xyz_h_basketball.T).T  # [N_basketball, 4]
            xyz_cam_3d = xyz_cam[:, :3].detach().cpu().numpy()  # [N_basketball, 3]

            # Create 3D plot in camera space
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Color by depth (Z in camera space)
            z_values = xyz_cam_3d[:, 2]
            scatter = ax.scatter(
                xyz_cam_3d[:, 0],
                xyz_cam_3d[:, 1],
                xyz_cam_3d[:, 2],
                c=z_values,
                cmap='viridis',
                alpha=0.6,
                s=20,
                edgecolors='none',
            )

            ax.set_xlabel('X (Camera Space)')
            ax.set_ylabel('Y (Camera Space)')
            ax.set_zlabel('Z (Camera Space - Depth)')
            ax.set_title(f'Basketball Gaussians in Camera Space - Frame {t}')
            plt.colorbar(scatter, ax=ax, label='Depth (Z)')

            # Add camera origin
            ax.scatter([0], [0], [0], c='red', s=100, marker='x', label='Camera Origin')
            ax.legend()

            plt.tight_layout()
            camera_space_path = os.path.join(camera_space_dir, f"camera_space_frame_{t:05d}.png")
            plt.savefig(camera_space_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Also create 2D projection (X-Y plane, colored by depth)
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                xyz_cam_3d[:, 0], xyz_cam_3d[:, 1], c=z_values, cmap='viridis', alpha=0.6, s=20, edgecolors='none'
            )
            ax.set_xlabel('X (Camera Space)')
            ax.set_ylabel('Y (Camera Space)')
            ax.set_title(f'Basketball Gaussians 2D Projection (X-Y) - Frame {t}')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Depth (Z)')
            plt.tight_layout()
            camera_space_2d_path = os.path.join(camera_space_dir, f"camera_space_2d_frame_{t:05d}.png")
            plt.savefig(camera_space_2d_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Project to image pixel coordinates. Use project_gaussians_to_2d (same as
            # identify_basketball_gaussians) so projected centers align with the mask's
            # ball-shaped region. [H,W] and (x,y) convention match basketball_mask.
            if H is not None and W is not None:
                means2D, radii_approx = project_gaussians_to_2d(gaussians, camera, (H, W))
                means2D_b = means2D[basketball_indices_torch].detach().cpu().numpy()
                radii_b = radii_approx[basketball_indices_torch].detach().cpu().numpy()
                pixel_x = means2D_b[:, 0]  # col
                pixel_y = means2D_b[:, 1]  # row

                # Frustum culling: match CUDA in_frustum() operation
                # CUDA checks: p_view.z > 0.2f (near plane culling only)
                # This matches the check in forward.cu:in_frustum() -> auxiliary.h:in_frustum()

                # Opacity check: match CUDA renderCUDA() operation
                # CUDA checks: alpha < 1.0f / 255.0f (approximately 0.00392) during rendering
                # We check raw opacity here (before exponential falloff), using a slightly higher threshold
                opacities = gaussians.get_opacity
                op = opacities[basketball_indices_torch]
                if op.dim() > 1:
                    op = op.squeeze(-1)
                opacity_np = op.detach().cpu().numpy()
                _opacity_threshold = 1.0 / 255.0  # Match CUDA's alpha threshold (0.00392)

                valid_mask = (xyz_cam_3d[:, 2] > 0.2) & (  # Frustum culling (matches in_frustum)
                    opacity_np >= _opacity_threshold
                )  # Opacity check (matches renderCUDA)
                valid_pixel_x = pixel_x[valid_mask]
                valid_pixel_y = pixel_y[valid_mask]
                valid_z = z_values[valid_mask]
                valid_radii = radii_b[valid_mask]  # Screen-space radii in pixels
                valid_opacities = opacity_np[valid_mask]  # Base opacity values (con_o.w in CUDA)

                # Compute rendered colors for basketball Gaussians (matching render() function)
                # Get camera center
                if hasattr(camera, 'camera_center'):
                    cam_center = camera.camera_center
                    if not isinstance(cam_center, torch.Tensor):
                        cam_center = torch.tensor(cam_center, device=basketball_mask.device)
                    if cam_center.device != basketball_mask.device:
                        cam_center = cam_center.to(basketball_mask.device)
                elif isinstance(camera, dict) and 'camera' in camera:
                    # For PanopticSports/CMU, use campos from rasterization settings
                    raster_settings = camera['camera']
                    if hasattr(raster_settings, 'campos'):
                        cam_center = raster_settings.campos
                        if not isinstance(cam_center, torch.Tensor):
                            cam_center = torch.tensor(cam_center, device=basketball_mask.device)
                        if cam_center.device != basketball_mask.device:
                            cam_center = cam_center.to(basketball_mask.device)
                    else:
                        # Fallback: extract camera center from viewmatrix
                        # w2c is already in original format (untransposed) after line 875
                        # Match helpers.py: cam_center = torch.inverse(w2c)[:3, 3]
                        w2c_inv = torch.inverse(w2c)
                        cam_center = w2c_inv[:3, 3]
                else:
                    cam_center = None

                # Compute rendered colors if camera center is available
                rendered_colors = None
                if cam_center is not None:
                    # Get SH features for basketball Gaussians
                    # get_features returns [N, (deg+1)^2, 3] (concatenated features_dc and features_rest)
                    sh_features = gaussians.get_features  # [N, (deg+1)^2, 3]
                    sh_basketball = sh_features[basketball_indices_torch]  # [N_basketball, (deg+1)^2, 3]

                    # Compute direction from camera center to each Gaussian
                    dir_pp = xyz_basketball - cam_center.unsqueeze(0)  # [N_basketball, 3]
                    dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-7)

                    # Reshape SH features to match render() function: [N, (deg+1)^2, 3] -> [N, 3, (deg+1)^2]
                    sh_basketball_view = sh_basketball.transpose(1, 2)  # [N_basketball, 3, (deg+1)^2]

                    # Evaluate SH to get RGB colors
                    sh2rgb = eval_sh(
                        gaussians.active_sh_degree, sh_basketball_view, dir_pp_normalized
                    )  # [N_basketball, 3]

                    # Apply same transformation as in render(): clamp_min(sh2rgb + 0.5, 0.0)
                    # For matplotlib visualization, also clamp to [0, 1] range
                    rendered_colors_torch = torch.clamp(sh2rgb + 0.5, 0.0, 1.0)  # [N_basketball, 3]
                    rendered_colors = rendered_colors_torch.detach().cpu().numpy()  # [N_basketball, 3]
                    valid_colors = rendered_colors[valid_mask]  # [N_valid, 3]

                fig, ax = plt.subplots(figsize=(12, 10))
                if len(valid_pixel_x) > 0:
                    # Scale marker size based on radius (radius is in pixels, s is in points^2)
                    # Use radius directly scaled to marker size: larger radius = larger marker
                    # Clamp radii to reasonable range for visualization
                    marker_sizes = np.clip(valid_radii * 2, 5, 200)  # Scale factor and limits

                    if rendered_colors is not None:
                        # Use rendered colors, size by radius, alpha by opacity
                        # Note: In actual rendering, alpha = opacity * exp(power) where power
                        # decreases exponentially with distance from center. Here we use base opacity
                        # for visualization transparency.
                        scatter = ax.scatter(
                            valid_pixel_x,
                            valid_pixel_y,
                            c=valid_colors,
                            s=marker_sizes,
                            alpha=valid_opacities,  # Use actual opacity values (varies per Gaussian)
                            edgecolors="black",
                            linewidths=0.5,
                        )
                        # Also draw circles showing actual radius for a few samples
                        # Sample every Nth Gaussian to avoid clutter
                        sample_indices = np.arange(0, len(valid_pixel_x), max(1, len(valid_pixel_x) // 20))
                        for idx in sample_indices:
                            circle = Circle(
                                (valid_pixel_x[idx], valid_pixel_y[idx]),
                                valid_radii[idx],
                                fill=False,
                                edgecolor='gray',
                                linewidth=0.5,
                                alpha=0.3,
                                linestyle='--',
                            )
                            ax.add_patch(circle)
                    else:
                        # Fallback to depth coloring if colors unavailable
                        print("Fallback to depth coloring")
                        scatter = ax.scatter(
                            valid_pixel_x,
                            valid_pixel_y,
                            c=valid_z,
                            cmap="viridis",
                            s=marker_sizes,
                            alpha=valid_opacities,  # Use actual opacity values (varies per Gaussian)
                            edgecolors="black",
                            linewidths=0.5,
                        )
                        # Also draw circles showing actual radius for a few samples
                        sample_indices = np.arange(0, len(valid_pixel_x), max(1, len(valid_pixel_x) // 20))
                        for idx in sample_indices:
                            circle = plt.Circle(
                                (valid_pixel_x[idx], valid_pixel_y[idx]),
                                valid_radii[idx],
                                fill=False,
                                edgecolor='gray',
                                linewidth=0.5,
                                alpha=0.3,
                                linestyle='--',
                            )
                            ax.add_patch(circle)
                    ax.set_xlim(0, W)
                    ax.set_ylim(H, 0)
                    ax.set_xlabel("Pixel X (col, 0=left)")
                    ax.set_ylabel("Pixel Y (row; row 0 = viewport bottom in rasterizer)")
                    if rendered_colors is not None:
                        ax.set_title(
                            f"Basketball Gaussians Projected to Image Pixel Coordinates (Rendered Colors + Radius + Opacity) - Frame {t}\n"
                            f"({len(valid_pixel_x)}/{N_basketball} visible; size ∝ radius; transparency ∝ base opacity; "
                            f"dashed circles show 3σ radius; actual alpha = opacity × exp(power) per pixel)"
                        )
                    else:
                        ax.set_title(
                            f"Basketball Gaussians Projected to Image Pixel Coordinates (Radius + Opacity) - Frame {t}\n"
                            f"({len(valid_pixel_x)}/{N_basketball} visible; size ∝ radius; transparency ∝ base opacity; "
                            f"dashed circles show 3σ radius; actual alpha = opacity × exp(power) per pixel)"
                        )
                    ax.grid(True, alpha=0.3)
                    if rendered_colors is None:
                        # Only show colorbar for depth coloring
                        plt.colorbar(scatter, ax=ax, label="Depth (Z in Camera Space)")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No basketball Gaussians visible in image\nfor frame {t}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=14,
                    )
                    ax.set_xlim(0, W)
                    ax.set_ylim(H, 0)
                    ax.set_xlabel("Pixel X (col, 0=left)")
                    ax.set_ylabel("Pixel Y (row; row 0 = viewport bottom in rasterizer)")
                    ax.set_title(f"Basketball Gaussians Projected to Image Pixel Coordinates - Frame {t}")

                plt.tight_layout()
                pixel_coords_path = os.path.join(camera_space_dir, f"pixel_coordinates_frame_{t:05d}.png")
                plt.savefig(pixel_coords_path, dpi=150, bbox_inches="tight")
                plt.close()
        except Exception as e:
            print(f"Error plotting camera space for frame {t}: {e}")
            continue

    print(f"Saved camera space plots to {camera_space_dir}/")


def track_basketball_gaussians(
    gaussians,
    scene,
    basketball_mask,
    output_dir="./basketball_tracking",
    cameras=None,
    cam_type=None,
    pipeline=None,
    background=None,
    save_frames=True,
    frame_interval=1,
):
    """
    Track basketball Gaussians across time and visualize their movement.

    Args:
        gaussians: GaussianModel instance
        scene: Scene instance
        basketball_mask: Boolean mask [N] indicating basketball Gaussians
        output_dir: Directory to save visualizations
        cameras: List of cameras to use (if None, uses video cameras)
        cam_type: Camera type string
        pipeline: Pipeline parameters
        background: Background color tensor
        save_frames: Whether to save rendered frames
        frame_interval: Save every Nth frame

    Returns:
        trajectories: Array of [T, N_basketball, 3] positions over time
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Get cameras
    if cameras is None:
        video_cameras = scene.getVideoCameras()
        if video_cameras is None or len(video_cameras) == 0:
            video_cameras = scene.getTestCameras()
            if video_cameras is None or len(video_cameras) == 0:
                video_cameras = scene.getTrainCameras()
        cameras = video_cameras

    if cameras is None or len(cameras) == 0:
        raise ValueError("No cameras available for tracking. Check if video, test, or train cameras exist.")

    max_frames = gaussians.max_frames if hasattr(gaussians, 'max_frames') and gaussians.max_frames else len(cameras)
    max_frames = min(max_frames, len(cameras)) if cameras is not None else 0

    # Store trajectories
    trajectories = []  # List of [N_basketball, 3] positions over time

    print(f"Tracking basketball Gaussians across {max_frames} frames...")
    for t in tqdm(range(max_frames), desc="Tracking"):
        # Set timestamp
        gaussians.set_timestamp(t, training=False)

        # Get current positions
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        basketball_xyz = xyz[basketball_mask.cpu().numpy()]
        trajectories.append(basketball_xyz)

        # Render with colored basketball
        if save_frames and t % frame_interval == 0:
            camera = cameras[t] if t < len(cameras) else cameras[0]

            # Temporarily color basketball Gaussians
            original_colors = color_basketball_gaussians(gaussians, basketball_mask, color=[1.0, 0.0, 0.0])  # Full red

            render_pkg = render(camera, gaussians, pipeline, background, cam_type=cam_type)
            rendered = render_pkg["render"]

            # Restore colors
            color_basketball_gaussians(gaussians, basketball_mask, restore_colors=original_colors)

            # Save rendered image
            img_np = rendered.detach().cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            plt.imsave(f"{output_dir}/frame_{t:05d}.png", img_np)

    # Convert to numpy array
    trajectories = np.array(trajectories)  # [T, N_basketball, 3]

    if trajectories.shape[1] == 0:
        print("Warning: No basketball Gaussians found!")
        return trajectories

    # Visualize trajectories
    print("Creating trajectory visualizations...")

    # Plot trajectory paths
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot a subset of trajectories to avoid clutter
    num_trajectories_to_plot = min(100, trajectories.shape[1])
    if num_trajectories_to_plot > 0:
        indices_to_plot = np.linspace(0, trajectories.shape[1] - 1, num_trajectories_to_plot, dtype=int)

        for i in indices_to_plot:
            traj = trajectories[:, i, :]  # [T, 3]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.3, linewidth=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Basketball Gaussian Trajectories')
    plt.savefig(f"{output_dir}/trajectories_3d.png", dpi=150)
    plt.close()

    # Plot center of mass trajectory
    center_of_mass = trajectories.mean(axis=1)  # [T, 3]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for dim, ax in enumerate(axes):
        ax.plot(center_of_mass[:, dim])
        ax.set_ylabel(['X', 'Y', 'Z'][dim])
        ax.set_title(f'Basketball Center of Mass - {["X", "Y", "Z"][dim]}')
        ax.grid(True)
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/center_of_mass.png", dpi=150)
    plt.close()

    # Save trajectory data
    np.save(f"{output_dir}/trajectories.npy", trajectories)
    np.save(f"{output_dir}/center_of_mass.npy", center_of_mass)

    # Plot world space positions
    plot_world_space_positions(trajectories, output_dir)

    # Plot camera space positions for each frame
    plot_camera_space_positions(
        gaussians,
        cameras,
        basketball_mask,
        trajectories,
        output_dir,
        frame_interval=frame_interval,
    )

    print(f"Visualizations saved to {output_dir}/")
    return trajectories


def identify_and_visualize_basketball(
    gaussians,
    scene,
    pipeline,
    background,
    mask_method='manual',
    mask_path=None,
    output_dir="./basketball_analysis",
    threshold_radius=3.0,
    min_contributions=5,
    basketball_color=[200 / 255, 87 / 255, 83 / 255],
    cam_type=None,
    sample_frame=0,
    sample_frame_2=None,
):
    """
    Complete pipeline to identify and visualize basketball Gaussians.

    Args:
        gaussians: GaussianModel instance
        scene: Scene instance
        pipeline: Pipeline parameters
        background: Background color tensor
        mask_method: Method to get mask ('manual', 'color', 'load')
        mask_path: Path to saved mask (for 'load' method)
        output_dir: Output directory
        threshold_radius: Radius threshold for Gaussian identification
        min_contributions: Minimum pixel contributions
        basketball_color: RGB color for basketball Gaussians
        cam_type: Camera type
        sample_frame: Frame index for first mask sample
        sample_frame_2: If set, second frame index for two-sample identification (ignored when mask_method='load')

    Returns:
        basketball_gaussian_mask: Boolean mask [N]
        trajectories: Trajectory array [T, N_basketball, 3]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get basketball mask(s) from rendered frame(s)
    test_cameras = scene.getTestCameras()
    print(f"Number of test cameras: {len(test_cameras)}")
    if len(test_cameras) == 0:
        test_cameras = scene.getTrainCameras()
        print(f"Number of train cameras: {len(test_cameras)}")

    if len(test_cameras) == 0:
        raise ValueError("No cameras available for mask selection")

    use_two_samples = sample_frame_2 is not None and mask_method != "load"
    if use_two_samples:
        indices = [
            min(sample_frame, len(test_cameras) - 1),
            min(sample_frame_2, len(test_cameras) - 1),
        ]
    else:
        indices = [min(sample_frame, len(test_cameras) - 1)]

    samples = []
    rendered_image = None
    sample_camera = None

    for i, idx in enumerate(indices):
        cam = test_cameras[idx]
        t = getattr(cam, "timestamp", idx)
        if hasattr(gaussians, "set_timestamp"):
            gaussians.set_timestamp(t, training=False)

        suffix = f"_{i}" if use_two_samples else ""
        print(f"Rendering sample {i + 1}/{len(indices)} for mask selection...")
        render_pkg = render(cam, gaussians, pipeline, background, cam_type=cam_type)
        img = render_pkg["render"]
        if rendered_image is None:
            rendered_image = img
            sample_camera = cam

        # Save rendered image
        mask_img_path = os.path.join(output_dir, f"basketball_mask_image{suffix}.png")
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        imageio.imwrite(mask_img_path, (img_np * 255).astype(np.uint8))
        print(f"Saved rendered image to {mask_img_path}")

        # Get mask (for 'load', only first sample uses mask_path)
        mpath = mask_path if (i == 0 and mask_method == "load") else None
        mask_2d = get_basketball_mask_from_image(img, method=mask_method, mask_path=mpath)

        mask_save_path = os.path.join(output_dir, f"basketball_mask{suffix}.npy")
        np.save(mask_save_path, mask_2d)
        print(f"Saved mask to {mask_save_path}")

        mask_bw_path = os.path.join(output_dir, f"basketball_mask{suffix}.png")
        imageio.imwrite(mask_bw_path, mask_2d.astype(np.uint8) * 255)
        print(f"Saved mask image to {mask_bw_path}")

        samples.append((cam, mask_2d))

    # Step 2: Identify basketball Gaussians (from one or two samples)
    print("Identifying basketball Gaussians...")
    basketball_gaussian_mask, contribution_map = identify_basketball_gaussians(
        gaussians,
        samples,
        threshold_radius=threshold_radius,
        min_contributions=min_contributions,
    )

    # Save Gaussian mask
    np.save(os.path.join(output_dir, "basketball_gaussian_mask.npy"), basketball_gaussian_mask.cpu().numpy())

    # Step 2.5: Plot visibility and opacity statistics
    print("Generating visibility and opacity statistics...")
    H, W = rendered_image.shape[1], rendered_image.shape[2]
    plot_gaussian_visibility_statistics(
        gaussians,
        sample_camera,
        basketball_gaussian_mask,
        output_dir,
        image_size=(H, W),
        opacity_threshold=0.01,
    )

    # Step 3: Track and visualize
    print("Tracking basketball Gaussians over time...")
    trajectories = track_basketball_gaussians(
        gaussians,
        scene,
        basketball_gaussian_mask,
        output_dir=output_dir,
        cam_type=cam_type,
        pipeline=pipeline,
        background=background,
    )

    return basketball_gaussian_mask, trajectories
