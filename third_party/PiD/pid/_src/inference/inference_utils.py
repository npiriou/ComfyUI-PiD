# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared utilities for FlashVSR inference scripts.

This module provides common functions for:
- IO operations (video/image loading, saving)
- Data processing (tensor conversion, prompt embedding)
- Tag generation from checkpoint paths
- Distributed processing utilities
- S3 upload utilities (optional)
"""

import io
import os
import pickle
import re
from configparser import ConfigParser
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import imageio.v3 as iio
import numpy as np
import torch as th
from einops import rearrange
from PIL import Image

# =============================================================================
# S3 Upload Configuration
# =============================================================================

# AWS Profile name for S3 access
S3_PROFILE_NAME = "pdx-yiflu"

# Default S3 bucket name
S3_BUCKET_NAME = "pid"

# S3 root prefix (folder structure: <ROOT_PREFIX>/<group_name>/<experiment_name>/*.mp4)
S3_ROOT_PREFIX = "streamlit_assets"

# Default group name
S3_DEFAULT_GROUP_NAME = "pid_inference"


class InputType(Enum):
    """Enum for input types"""

    VIDEO_FILE = "video_file"
    VIDEO_FOLDER = "video_folder"
    IMAGE_FOLDER = "image_folder"


# =============================================================================
# IO Related Functions
# =============================================================================


def is_video(path: str) -> bool:
    """Check if path is a video file"""
    return os.path.isfile(path) and path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))


def is_image(path: str) -> bool:
    """Check if path is an image file"""
    return os.path.isfile(path) and path.lower().endswith((".png", ".jpg", ".jpeg"))


def natural_key(name: str):
    """Natural sort key for filenames (e.g., img_1.png, img_2.png, ..., img_10.png)"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", os.path.basename(name))]


def list_images_natural(folder: str) -> List[str]:
    """List images in folder with natural sorting"""
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def list_videos_in_directory(directory: str) -> List[str]:
    """List all video files in a directory (excluding files with 'hq' in name)"""
    if not os.path.isdir(directory):
        raise ValueError(f"Not a directory: {directory}")

    video_files = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if is_video(filepath) and "hq" not in filepath.lower():
            video_files.append(filepath)

    return video_files


def list_files_in_directory(
    directory: str,
    rank: int = 0,
    world_size: int = 1,
    include_images: bool = False,
) -> List[str]:
    """
    List all input files in a directory, with optional data parallel sharding.

    Args:
        directory: Path to directory containing videos/images
        rank: Current process rank (0-indexed)
        world_size: Total number of processes
        include_images: If True, also look for image sequences in subdirectories

    Returns:
        List of file/folder paths assigned to this rank
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Not a directory: {directory}")

    files = []

    for entry in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, entry)

        # Check if it's a video file
        if is_video(filepath) and "hq" not in filepath.lower():
            files.append(filepath)
        # Check if it's an image sequence folder
        elif include_images and os.path.isdir(filepath):
            images = list_images_natural(filepath)
            if images:
                files.append(filepath)

    # Data parallel sharding: each rank processes a subset
    if world_size > 1:
        total_files = len(files)
        files = files[rank::world_size]  # Interleaved sharding
        print(f"[Rank {rank}/{world_size}] Assigned {len(files)}/{total_files} files")

    return files


def detect_input_type(path: str) -> InputType:
    """
    Detect input type from path.

    Args:
        path: Input path (file or directory)

    Returns:
        InputType enum value
    """
    if is_video(path):
        return InputType.VIDEO_FILE

    if os.path.isdir(path):
        # Check if it contains videos
        videos = list_videos_in_directory(path) if os.path.isdir(path) else []
        if videos:
            return InputType.VIDEO_FOLDER

        # Check if it contains images (image sequence)
        images = list_images_natural(path)
        if images:
            return InputType.IMAGE_FOLDER

        raise ValueError(f"Directory {path} contains neither videos nor images")

    raise ValueError(f"Unsupported input path: {path}")


def tensor2video(frames: th.Tensor) -> List[Image.Image]:
    """
    Convert tensor (C, T, H, W) in [-1, 1] range to list of PIL images.

    Args:
        frames: Tensor of shape (C, T, H, W) in range [-1, 1]

    Returns:
        List of PIL Image objects
    """
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return [Image.fromarray(frame) for frame in frames]


def save_video(
    frames: List[Image.Image],
    save_path: str,
    fps: int,
    quality: int = 6,
) -> None:
    """
    Save frames as video file.

    Args:
        frames: List of PIL Image objects
        save_path: Output path for video file
        fps: Frame rate
        quality: Video quality (1-10)
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    video_array = np.stack([np.array(f) for f in frames], axis=0)
    iio.imwrite(save_path, video_array, fps=fps, quality=quality)


# =============================================================================
# Data Processing Functions
# =============================================================================


def load_prompt_embedding(
    prompt_path: str,
    dtype=th.bfloat16,
    device: str = "cuda",
) -> th.Tensor:
    """
    Load prompt embedding from file (.pkl or .pth).

    Args:
        prompt_path: Path to embedding file
        dtype: Target dtype
        device: Target device

    Returns:
        Embedding tensor of shape (1, seq_len, dim)
    """
    if prompt_path.endswith(".pkl"):
        with open(prompt_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "embedding" in data:
            embedding = data["embedding"]
        else:
            embedding = data
    else:
        embedding = th.load(prompt_path, map_location=device)

    if isinstance(embedding, np.ndarray):
        embedding = th.from_numpy(embedding)

    embedding = embedding.to(dtype=dtype, device=device)

    # Ensure proper shape (1, seq_len, dim)
    if embedding.dim() == 2:
        embedding = embedding.unsqueeze(0)

    return embedding


def pil_to_tensor(img: Image.Image, dtype=th.bfloat16, device: str = "cuda") -> th.Tensor:
    """
    Convert PIL image to tensor in [-1, 1] range.

    Args:
        img: PIL Image object
        dtype: Target dtype
        device: Target device

    Returns:
        Tensor of shape (C, H, W) in range [-1, 1]
    """
    t = th.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=th.float32)
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


# =============================================================================
# Fix batch PNG encoding/decoding helpers
# =============================================================================
# fix_batch .pt files store images as PNG-encoded bytes to save disk space (~10x
# smaller than raw float32 tensors). These helpers encode tensors to PNG bytes
# for saving, and decode PNG bytes back to tensors for inference/training.


def encode_tensor_as_png(tensor: th.Tensor) -> bytes:
    """Encode [C, H, W] or [1, C, H, W] tensor in [-1, 1] to PNG bytes.

    Returns raw PNG bytes that can be stored in a .pt file.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]  # [1, C, H, W] -> [C, H, W]
    arr = ((tensor.float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def decode_image_bytes_to_tensor(png_bytes: bytes, device: str = "cpu") -> th.Tensor:
    """Decode image bytes (PNG/JPG/etc) to [1, C, H, W] float32 tensor in [-1, 1].

    Inverse of encode_tensor_as_png(). Also works with JPEG and other PIL-supported formats.
    """
    pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(pil_img, dtype=np.uint8)
    t = th.from_numpy(arr).float().permute(2, 0, 1) / 127.5 - 1.0  # [C, H, W] in [-1, 1]
    return t.unsqueeze(0).to(device)  # [1, C, H, W]


def load_fix_batch(pt_path: str, device: str = "cpu") -> dict:
    """Load a fix_batch .pt file, auto-detecting PNG-encoded vs raw tensor format.

    Handles both new format ("HQ_video_or_image") and legacy format ("image").
    Always returns both "HQ_video_or_image" and "image" pointing to the same tensor
    so callers can use either key.

    Returns dict with tensors in [-1, 1] float32:
        "HQ_video_or_image" / "image": [1, 3, H, W] or None
        "LQ_video_or_image": [1, 3, H_lq, W_lq]
        "LQ_latent": [1, C, H, W] (optional, pre-computed VAE latent — skips VAE encode)
        "caption": list[str]
    """
    data = th.load(pt_path, map_location="cpu", weights_only=False)

    for key in ["HQ_video_or_image", "LQ_video_or_image"]:
        if key not in data:
            continue
        val = data[key]
        if isinstance(val, bytes):
            # Image bytes (PNG/JPG) -> decode to tensor
            data[key] = decode_image_bytes_to_tensor(val, device=device)
        elif isinstance(val, th.Tensor):
            # Raw tensor (legacy format)
            if val.dtype == th.uint8:
                data[key] = val.float() / 127.5 - 1.0
            data[key] = data[key].to(device)

    # Move LQ_latent to target device if present
    if "LQ_latent" in data and isinstance(data["LQ_latent"], th.Tensor):
        data["LQ_latent"] = data["LQ_latent"].to(device).unsqueeze(0)

    # Provide "image" alias for model compatibility (input_data_key="image")
    if "HQ_video_or_image" in data:
        data["image"] = data["HQ_video_or_image"]

    return data


def largest_8n_minus_3(n: int) -> int:
    """
    Find largest number of form 8k-3 that is <= n.
    This is used to satisfy the constraint: num_frames = 8k - 3 for some integer k >= 1.
    The sequence is: 5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, ...
    """
    if n < 5:
        return 0
    # 8k - 3 <= n => k <= (n + 3) / 8
    k = (n + 3) // 8
    return 8 * k - 3


def video_file_to_tensor(
    video_path: str,
    num_frames: int,
    dtype=th.bfloat16,
    device: str = "cuda",
) -> Tuple[th.Tensor, int]:
    """
    Load video file and convert to tensor.

    Args:
        video_path: Path to video file
        num_frames: Maximum number of frames to load
        dtype: Target dtype
        device: Target device

    Returns:
        Tuple of:
        - Tensor of shape (1, C, T, H, W) in range [-1, 1]
        - FPS of the video
    """
    import decord

    decord.bridge.set_bridge("torch")

    video_reader = decord.VideoReader(video_path)
    total_frames = len(video_reader)
    num_frames = min(num_frames, total_frames)

    # Adjust to satisfy num_frames = 8n - 3
    num_frames = largest_8n_minus_3(num_frames)
    if num_frames == 0:
        raise RuntimeError(f"Not enough frames in {video_path}, need at least 5 frames")

    print(f"Loading {num_frames} frames from {video_path}")

    frames = video_reader.get_batch(range(num_frames))  # [T, H, W, C]
    frames = frames.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
    frames = frames.to(device=device) / 127.5 - 1.0
    video_tensor = frames.to(dtype=dtype)

    # Extract FPS
    fps = 24
    try:
        fps_val = video_reader.get_avg_fps()
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) and fps_val > 0 else 24
    except Exception:
        pass

    return video_tensor, fps


def image_folder_to_tensor(
    folder: str,
    num_frames: int,
    scale: int = 4,
    dtype=th.bfloat16,
    device: str = "cuda",
) -> Tuple[th.Tensor, int, int]:
    """
    Load image sequence from folder and convert to tensor.

    Args:
        folder: Path to folder containing images
        num_frames: Maximum number of frames to load
        scale: Upscaling factor for target resolution
        dtype: Target dtype
        device: Target device

    Returns:
        Tuple of:
        - Tensor of shape (1, C, T, tH, tW) in range [-1, 1], upscaled to target resolution
        - Target height
        - Target width
    """
    image_paths = list_images_natural(folder)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {folder}")

    # Get original dimensions
    with Image.open(image_paths[0]) as img:
        w0, h0 = img.size

    total_images = len(image_paths)
    num_frames = min(num_frames, total_images)

    # Adjust to satisfy num_frames = 8n - 3
    num_frames = largest_8n_minus_3(num_frames)
    if num_frames == 0:
        raise RuntimeError(f"Not enough images in {folder}, need at least 5 images")

    image_paths = image_paths[:num_frames]

    print(f"[{os.path.basename(folder)}] Loading {num_frames} images, original size: {w0}x{h0}")

    # Compute target dimensions (must be multiple of 128)
    tW = max(128, ((w0 * scale) // 128) * 128)
    tH = max(128, ((h0 * scale) // 128) * 128)
    print(f"[{os.path.basename(folder)}] Target size: {tW}x{tH}")

    frames = []
    for p in image_paths:
        with Image.open(p).convert("RGB") as img:
            # Resize to target dimensions
            img_resized = img.resize((tW, tH), Image.BICUBIC)
        frames.append(pil_to_tensor(img_resized, dtype, device))

    video_tensor = th.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)

    return video_tensor, tH, tW


# =============================================================================
# Tag Generation Functions
# =============================================================================


def generate_tag_from_checkpoint(
    checkpoint_path: str,
    extra_params: Optional[dict] = None,
    load_ema: bool = False,
) -> str:
    """
    Generate tag from checkpoint path and parameters.

    Examples:
        checkpoint_path = ".../flashvsr_0119_stage2_xxx_cp1/checkpoints/iter_000009790"
        -> base_tag = "flashvsr_0119_stage2_xxx_cp1_iter_000009790"

    Args:
        checkpoint_path: Path to model checkpoint
        extra_params: Dictionary of extra parameters to append to tag
        load_ema: Whether EMA weights are loaded

    Returns:
        Generated tag string
    """
    # Normalize path
    path = checkpoint_path.rstrip("/")

    # Extract iter name (last component)
    iter_name = os.path.basename(path)

    # Extract experiment name (parent of checkpoints dir)
    parent = os.path.dirname(path)
    if os.path.basename(parent) == "checkpoints":
        experiment_name = os.path.basename(os.path.dirname(parent))
    else:
        # Fallback: use parent directory name
        experiment_name = os.path.basename(parent)

    # Build base tag
    tag = f"{experiment_name}_{iter_name}"

    # Append extra parameters
    if extra_params:
        for key, value in extra_params.items():
            if value is not None:
                tag += f"_{key}{value}"

    # Append EMA/reg suffix
    tag += "_ema" if load_ema else "_reg"

    return tag


# =============================================================================
# Distributed Processing Functions
# =============================================================================


def get_rank_and_world_size() -> Tuple[int, int]:
    """
    Get rank and world_size from environment (set by torchrun).

    Returns:
        Tuple of (rank, world_size)
    """
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def init_data_parallel() -> Tuple[int, int]:
    """
    Initialize data parallel environment.

    Returns:
        Tuple of (rank, world_size)
    """
    rank, world_size = get_rank_and_world_size()

    if world_size > 1:
        th.cuda.set_device(rank)
        print(f"[Rank {rank}/{world_size}] Using GPU {rank}")

    return rank, world_size


def create_data_batch(
    lq_video: th.Tensor,
    vsr_embedding: th.Tensor,
    scale: int = 4,
    dtype=th.bfloat16,
    device: str = "cuda",
) -> dict:
    """
    Create a data batch dictionary for inference.

    Args:
        lq_video: Low-quality video tensor (1, C, T, H, W)
        vsr_embedding: VSR prompt embedding tensor
        scale: Upscaling factor
        dtype: Target dtype
        device: Target device

    Returns:
        Data batch dictionary
    """
    T, H, W = lq_video.shape[2:]

    data_batch = {
        "dataset_name": "video_data",
        "LQ_video_or_image": lq_video,
        "video": th.zeros((1, 3, T, H * scale, W * scale), dtype=dtype, device=device),
        "t5_text_embeddings": th.randn(1, 512, 4096, dtype=dtype, device=device),
        "vsr_predefined_embedding": vsr_embedding,
        "fps": th.tensor([24], dtype=th.int64, device=device),
        "padding_mask": th.zeros(1, 1, T, H * scale, W * scale, dtype=dtype, device=device),
        "LQ_video_or_image_vae_latent": th.zeros(1, device=device),
        "LQ_video_or_image_upscaled_vae_latent": th.zeros(1, device=device),
        "LQ_video_or_image_vae_latent_upscaled": th.zeros(1, device=device),
        "is_preprocessed": True,
    }

    return data_batch


def generate_output_path(
    input_path: str,
    output_dir: str,
    tag: str,
    suffix: str = ".mp4",
) -> str:
    """
    Generate output path for a single input file/folder.

    Args:
        input_path: Input file or folder path
        output_dir: Base output directory
        tag: Tag string for subdirectory
        suffix: Output file suffix

    Returns:
        Full output path
    """
    # Get input name without extension
    basename = os.path.basename(input_path.rstrip("/"))
    name = os.path.splitext(basename)[0] if os.path.isfile(input_path) else basename

    # Create output directory with tag
    tagged_output_dir = os.path.join(output_dir, tag)
    os.makedirs(tagged_output_dir, exist_ok=True)

    return os.path.join(tagged_output_dir, f"{name}{suffix}")


# =============================================================================
# S3 Upload Functions
# =============================================================================


def _parse_aws_credentials(
    cred_path_or_profile: Union[str, Path, None] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse AWS credentials from file, AWS profile, or environment variables.

    Args:
        cred_path_or_profile: Path to credentials file, AWS profile name, or None

    Returns:
        Tuple of (endpoint_url, access_key, secret_key, region)
    """
    if cred_path_or_profile:
        cred_path = Path(cred_path_or_profile)

        # Check if it's a file path
        if cred_path.exists() and cred_path.is_file():
            import json

            credentials = json.load(open(cred_path))
            endpoint = credentials.get("endpoint_url")
            access_key = credentials.get("aws_access_key_id")
            secret_key = credentials.get("aws_secret_access_key")
            region = credentials.get("region_name", None)
        else:
            # Treat as AWS profile name
            profile = str(cred_path_or_profile)

            credentials_file = Path.home() / ".aws" / "credentials"
            config_file = Path.home() / ".aws" / "config"

            # Parse credentials file
            credentials_parser = ConfigParser()
            if credentials_file.exists():
                credentials_parser.read(credentials_file)
            else:
                raise FileNotFoundError(f"AWS credentials file not found: {credentials_file}")

            # Parse config file
            config_parser = ConfigParser()
            if config_file.exists():
                config_parser.read(config_file)

            # Get credentials from credentials file
            if profile in credentials_parser:
                access_key = credentials_parser[profile].get("aws_access_key_id")
                secret_key = credentials_parser[profile].get("aws_secret_access_key")
                region = credentials_parser[profile].get("region")
                endpoint = credentials_parser[profile].get("endpoint_url")
            else:
                access_key = None
                secret_key = None
                region = None
                endpoint = None

            # If not found in credentials file, try config file
            if not region or not endpoint:
                config_section = f"profile {profile}" if profile != "default" else profile

                if config_section in config_parser:
                    if not region:
                        region = config_parser[config_section].get("region")
                    if not endpoint:
                        endpoint = config_parser[config_section].get("endpoint_url")
    else:
        # Load from environment variables
        endpoint = os.getenv("AWS_ENDPOINT_URL")
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION")

    # If no endpoint specified, use AWS S3 default endpoint
    if not endpoint:
        endpoint = "https://s3.amazonaws.com"

    return endpoint, access_key, secret_key, region


def get_s3_client(profile_name: Optional[str] = None):
    """
    Get a boto3 S3 client.

    Args:
        profile_name: AWS profile name (defaults to S3_PROFILE_NAME)

    Returns:
        boto3 S3 client
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for S3 upload. Install with: pip install boto3")

    if profile_name is None:
        profile_name = S3_PROFILE_NAME

    endpoint, access_key, secret_key, region = _parse_aws_credentials(profile_name)

    kwargs = {"endpoint_url": endpoint}
    if access_key:
        kwargs["aws_access_key_id"] = access_key
    if secret_key:
        kwargs["aws_secret_access_key"] = secret_key
    if region:
        kwargs["region_name"] = region

    return boto3.client("s3", **kwargs)


def upload_file_to_s3(
    local_path: str,
    s3_key: str,
    bucket_name: Optional[str] = None,
    s3_client=None,
) -> bool:
    """
    Upload a single file to S3.

    Args:
        local_path: Path to local file
        s3_key: S3 key (path in bucket)
        bucket_name: S3 bucket name (defaults to S3_BUCKET_NAME)
        s3_client: boto3 S3 client (will create one if not provided)

    Returns:
        True if upload succeeded, False otherwise
    """
    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME

    if s3_client is None:
        s3_client = get_s3_client()

    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        return True
    except Exception as e:
        print(f"Failed to upload {local_path} to s3://{bucket_name}/{s3_key}: {e}")
        return False


def download_file_from_s3(
    s3_key: str,
    local_path: str,
    bucket_name: Optional[str] = None,
    s3_client=None,
) -> bool:
    """
    Download a single file from S3.

    Args:
        s3_key: S3 key (path in bucket)
        local_path: Path to save the downloaded file
        bucket_name: S3 bucket name (defaults to S3_BUCKET_NAME)
        s3_client: boto3 S3 client (will create one if not provided)

    Returns:
        True if download succeeded, False otherwise
    """
    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME

    if s3_client is None:
        s3_client = get_s3_client()

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True
    except Exception as e:
        print(f"Failed to download s3://{bucket_name}/{s3_key} to {local_path}: {e}")
        return False


def upload_video_to_s3(
    local_path: str,
    group_name: str,
    experiment_name: str,
    bucket_name: Optional[str] = None,
    s3_client=None,
) -> bool:
    """
    Upload a video file to S3 with standard path structure.

    The file will be uploaded to: s3://<bucket>/<ROOT_PREFIX>/<group_name>/<experiment_name>/<filename>

    Args:
        local_path: Path to local video file
        group_name: Group name (e.g., "large_motion_lq")
        experiment_name: Experiment name (typically the tag)
        bucket_name: S3 bucket name (defaults to S3_BUCKET_NAME)
        s3_client: boto3 S3 client (will create one if not provided)

    Returns:
        True if upload succeeded, False otherwise
    """
    filename = os.path.basename(local_path)
    s3_key = f"{S3_ROOT_PREFIX}/{group_name}/{experiment_name}/{filename}"

    success = upload_file_to_s3(local_path, s3_key, bucket_name, s3_client)

    if success:
        print(f"Uploaded to s3://{bucket_name or S3_BUCKET_NAME}/{s3_key}")

    return success


def upload_directory_to_s3(
    local_dir: str,
    group_name: str,
    experiment_name: str,
    bucket_name: Optional[str] = None,
    file_extensions: Tuple[str, ...] = (".mp4", ".png", ".jpg", ".jpeg"),
) -> Tuple[int, int]:
    """
    Upload all matching files in a directory to S3.

    Args:
        local_dir: Path to local directory
        group_name: Group name
        experiment_name: Experiment name (typically the tag)
        bucket_name: S3 bucket name (defaults to S3_BUCKET_NAME)
        file_extensions: Tuple of file extensions to upload

    Returns:
        Tuple of (success_count, total_count)
    """
    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME

    s3_client = get_s3_client()

    success_count = 0
    total_count = 0

    for filename in os.listdir(local_dir):
        if filename.lower().endswith(file_extensions):
            total_count += 1
            local_path = os.path.join(local_dir, filename)

            if upload_video_to_s3(local_path, group_name, experiment_name, bucket_name, s3_client):
                success_count += 1

    return success_count, total_count


def upload_directory_to_s3_parallel(
    local_dir: str,
    group_name: str,
    experiment_name: str,
    bucket_name: Optional[str] = None,
    file_extensions: Tuple[str, ...] = (".mp4", ".png", ".jpg", ".jpeg"),
    max_workers: int = 16,
) -> Tuple[int, int]:
    """
    Upload all matching files in a directory to S3 using parallel threads.

    Args:
        local_dir: Path to local directory
        group_name: Group name
        experiment_name: Experiment name (typically the tag)
        bucket_name: S3 bucket name (defaults to S3_BUCKET_NAME)
        file_extensions: Tuple of file extensions to upload
        max_workers: Maximum number of parallel upload threads

    Returns:
        Tuple of (success_count, total_count)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME

    # Collect files to upload
    files_to_upload = [
        os.path.join(local_dir, filename)
        for filename in os.listdir(local_dir)
        if filename.lower().endswith(file_extensions)
    ]
    total_count = len(files_to_upload)

    if total_count == 0:
        return 0, 0

    def upload_single_file(local_path):
        """Worker function for uploading a single file."""
        s3_client = get_s3_client()
        return upload_video_to_s3(local_path, group_name, experiment_name, bucket_name, s3_client)

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(upload_single_file, f): f for f in files_to_upload}
        for future in as_completed(futures):
            if future.result():
                success_count += 1

    return success_count, total_count


def maybe_upload_video(
    local_path: str,
    tag: str,
    upload: bool,
    group_name: Optional[str] = None,
) -> bool:
    """
    Optionally upload a single video to S3 immediately after generation.

    Args:
        local_path: Path to the local video file
        tag: Experiment tag (used as experiment_name in S3)
        upload: Whether to upload
        group_name: S3 group name (defaults to S3_DEFAULT_GROUP_NAME)

    Returns:
        True if upload succeeded or was skipped, False if upload failed
    """
    if not upload:
        return True

    if group_name is None:
        group_name = S3_DEFAULT_GROUP_NAME

    if not os.path.isfile(local_path):
        print(f"Video file not found: {local_path}")
        return False

    try:
        s3_client = get_s3_client()
        success = upload_video_to_s3(local_path, group_name, tag, s3_client=s3_client)
        return success
    except Exception as e:
        print(f"Upload failed for {local_path}: {e}")
        return False
