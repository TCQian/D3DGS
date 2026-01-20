#
# Basketball Gaussian Tracking Main Script
# Standalone script to identify and visualize basketball Gaussians
#

import os
import sys
from argparse import ArgumentParser

import taichi as ti

ti.init(arch=ti.cuda, offline_cache=False)

import torch

from arguments import FlowParams, ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.basketball_tracker import color_basketball_gaussians, identify_and_visualize_basketball
from utils.general_utils import safe_state


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Basketball Gaussian Tracking Script")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    f = FlowParams(parser)

    # Standard arguments
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ckpt_name", type=str, default="chkpnt30000.pth", help="Checkpoint file name")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, help="Path to config file")

    # Basketball tracking arguments
    parser.add_argument(
        "--basketball_mask_method",
        type=str,
        default="manual",
        choices=['manual', 'color', 'load'],
        help="Method to get basketball mask: 'manual' (interactive selection), 'color' (color-based), 'load' (load from file)",
    )
    parser.add_argument(
        "--basketball_mask_path",
        type=str,
        default=None,
        help="Path to saved basketball mask file (for 'load' method)",
    )
    parser.add_argument(
        "--basketball_output_dir",
        type=str,
        default=None,
        help="Output directory for basketball tracking results (default: model_path/basketball_tracking)",
    )
    parser.add_argument(
        "--basketball_threshold_radius",
        type=float,
        default=3.0,
        help="Radius threshold for identifying contributing Gaussians (pixels)",
    )
    parser.add_argument(
        "--basketball_min_contributions",
        type=int,
        default=5,
        help="Minimum number of pixels a Gaussian must contribute to be considered part of basketball",
    )
    parser.add_argument(
        "--basketball_color",
        type=float,
        nargs=3,
        default=[200 / 255, 87 / 255, 83 / 255],
        help="RGB color for basketball Gaussians (default: rgba(200, 87, 83) normalized to [0,1])",
    )
    parser.add_argument(
        "--sample_frame",
        type=int,
        default=0,
        help="Frame index to use for mask selection (default: 0)",
    )
    parser.add_argument(
        "--sample_frame_2",
        type=int,
        default=None,
        help="Optional second frame index for two-sample basketball Gaussian identification (improves coverage across views/frames); ignored when --basketball_mask_method=load",
    )
    parser.add_argument(
        "--render_with_colors",
        action="store_true",
        help="Render output frames with basketball Gaussians colored",
    )
    parser.add_argument(
        "--render_output_dir",
        type=str,
        default=None,
        help="Directory to save rendered frames with colored basketball (default: basketball_output_dir/renders)",
    )
    parser.add_argument(
        "--render_frame_interval",
        type=int,
        default=10,
        help="Save every Nth frame when rendering (default: 10)",
    )

    args = get_combined_args(parser)
    print("Basketball Tracking for ", args.model_path)

    # Handle config file if provided
    if args.configs:
        import mmcv

        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Extract parameters
    dataset = model.extract(args)
    opt = op.extract(args)
    flow_args = f.extract(args)
    pipe = pipeline.extract(args)

    # Set up output directory - use getattr with defaults for arguments that might not be in config
    basketball_output_dir = getattr(args, 'basketball_output_dir', None)
    if basketball_output_dir is None:
        basketball_output_dir = os.path.join(dataset.model_path, "basketball_tracking")

    basketball_mask_method = getattr(args, 'basketball_mask_method', 'manual')
    basketball_mask_path = getattr(args, 'basketball_mask_path', None)
    basketball_threshold_radius = getattr(args, 'basketball_threshold_radius', 3.0)
    basketball_min_contributions = getattr(args, 'basketball_min_contributions', 5)
    basketball_color = getattr(args, 'basketball_color', [200 / 255, 87 / 255, 83 / 255])
    sample_frame = getattr(args, 'sample_frame', 0)
    sample_frame_2 = getattr(args, 'sample_frame_2', None)
    render_with_colors = getattr(args, 'render_with_colors', False)
    render_output_dir = getattr(args, 'render_output_dir', None)
    render_frame_interval = getattr(args, 'render_frame_interval', 10)

    print("\n" + "=" * 60)
    print("BASKETBALL TRACKING")
    print("=" * 60)
    print(f"Model path: {dataset.model_path}")
    print(f"Checkpoint: {args.ckpt_name}")
    print(f"Output directory: {basketball_output_dir}")
    print(f"Mask method: {basketball_mask_method}")
    if basketball_mask_path:
        print(f"Mask path: {basketball_mask_path}")
    print(f"Threshold radius: {basketball_threshold_radius}")
    print(f"Min contributions: {basketball_min_contributions}")
    print("=" * 60 + "\n")

    # Initialize Gaussian model
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.sh_degree,
            max_steps=opt.iterations + 1,
            xyz_traj_feat_dim=flow_args.xyz_traj_feat_dim,
            xyz_trajectory_type=flow_args.xyz_trajectory_type,
            rot_traj_feat_dim=flow_args.rot_traj_feat_dim,
            rot_trajectory_type=flow_args.rot_trajectory_type,
            scale_traj_feat_dim=flow_args.scale_traj_feat_dim,
            scale_trajectory_type=flow_args.scale_trajectory_type,
            opc_traj_feat_dim=flow_args.opc_traj_feat_dim,
            opc_trajectory_type=flow_args.opc_trajectory_type,
            feature_traj_feat_dim=flow_args.feature_traj_feat_dim,
            feature_trajectory_type=flow_args.feature_trajectory_type,
            feature_dc_trajectory_type=flow_args.feature_dc_trajectory_type,
            traj_init=flow_args.traj_init,
            poly_base_factor=flow_args.poly_base_factor,
            Hz_base_factor=flow_args.Hz_base_factor,
            normliaze=flow_args.normliaze,
            factor_t=opt.factor_t,
            factor_t_value=opt.factor_t_value,
            offset_t=opt.offset_t,
            offset_t_value=opt.offset_t_value,
        )

        # Load scene
        scene = Scene(dataset, gaussians, shuffle=False, load_img_factor=pipe.load_img_factor)
        cam_type = "PanopticSports" if scene.is_panoptic else None

        # Set up training parameters
        gaussians.training_setup(opt, flow_args)

        # Load checkpoint
        checkpoint_path = os.path.join(dataset.model_path, args.ckpt_name)
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)

        print(f"Loading checkpoint from {checkpoint_path}...")
        (model_params, first_iter) = torch.load(checkpoint_path, weights_only=False)
        gaussians.restore(model_params, opt, flow_args)
        print(f"Checkpoint loaded from iteration {first_iter}")

        # Set background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Run basketball tracking
        print("\nStarting basketball tracking...")
        basketball_mask, trajectories = identify_and_visualize_basketball(
            gaussians,
            scene,
            pipe,
            background,
            mask_method=basketball_mask_method,
            mask_path=basketball_mask_path,
            output_dir=basketball_output_dir,
            threshold_radius=basketball_threshold_radius,
            min_contributions=basketball_min_contributions,
            basketball_color=basketball_color,
            cam_type=cam_type,
            sample_frame=sample_frame,
            sample_frame_2=sample_frame_2,
        )

        print(f"\n✓ Basketball tracking complete!")
        print(f"  - Identified {basketball_mask.sum().item()} basketball Gaussians")
        print(f"  - Tracked {trajectories.shape[0]} frames")
        print(f"  - Results saved to: {basketball_output_dir}")

        # Optional: Render frames with colored basketball
        if render_with_colors:
            print("\nRendering frames with colored basketball Gaussians...")
            if render_output_dir is None:
                render_output_dir = os.path.join(basketball_output_dir, "renders")
            os.makedirs(render_output_dir, exist_ok=True)

            # Get cameras
            video_cameras = scene.getVideoCameras()
            if video_cameras is None or len(video_cameras) == 0:
                video_cameras = scene.getTestCameras()
                if video_cameras is None or len(video_cameras) == 0:
                    video_cameras = scene.getTrainCameras()

            if video_cameras is None or len(video_cameras) == 0:
                print("Warning: No cameras available for rendering. Skipping frame rendering.")
            else:
                import imageio
                import numpy as np

                max_frames = (
                    gaussians.max_frames
                    if hasattr(gaussians, 'max_frames') and gaussians.max_frames
                    else len(video_cameras)
                )
                max_frames = min(max_frames, len(video_cameras))

                for t in range(0, max_frames, render_frame_interval):
                    gaussians.set_timestamp(t, training=False)
                    camera = video_cameras[t] if t < len(video_cameras) else video_cameras[0]

                    # Color basketball Gaussians
                    original_colors = color_basketball_gaussians(gaussians, basketball_mask, color=[1.0, 0.0, 0.0])

                    # Render
                    render_pkg = render(camera, gaussians, pipe, background, cam_type=cam_type)
                    rendered = render_pkg["render"]

                    # Restore colors
                    color_basketball_gaussians(gaussians, basketball_mask, restore_colors=original_colors)

                    # Save image
                    img_np = rendered.detach().cpu().numpy().transpose(1, 2, 0)
                    img_np = np.clip(img_np, 0, 1)
                    imageio.imwrite(
                        os.path.join(render_output_dir, f"frame_{t:05d}.png"),
                        (img_np * 255).astype(np.uint8),
                    )

                print(f"  - Rendered frames saved to: {render_output_dir}")

        print("\n" + "=" * 60)
        print("BASKETBALL TRACKING COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
