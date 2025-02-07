import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import cv2
import io
import os
from PIL import Image
from multiprocessing import Pool, cpu_count
import subprocess
from pathlib import Path

# Parameters
deadzone_limit = 0.3
rim_limit = 0.9
interp_factor = 10                          # Interpolated points between frames
bins = 300                                  # Bins used for heatmap
sigma = 1                                   # Smoothing factor for the heatmap
fps = 60                                    # Frames per second for video
image_filename = "heatmap.png"              # Name of image filename
video_filename = "heatmap_timelapse.mp4"    # Name of video filename
num_processes = max(1, cpu_count() - 1)     # CPU cores used

# Highlight parameters
recent_frames_to_highlight = 25
highlight_chunk_size = 5
base_multiplier = 20.0
decay_factor = 0.8

def generate_heatmap(data, highlight_frames):
    """Generates a heatmap with an emphasized motion trail and returns it as an image array."""
    x_coords = np.array([point["x"] for point in data])
    y_coords = np.array([point["y"] for point in data])

    # Interpolation
    frames = np.arange(len(x_coords))
    new_frames = np.linspace(frames.min(), frames.max(), len(frames) * interp_factor)
    interp_x = interp1d(frames, x_coords, kind="linear")(new_frames)
    interp_y = interp1d(frames, y_coords, kind="linear")(new_frames)

    # Set base weights
    magnitude = np.sqrt(interp_x**2 + interp_y**2)
    weights = np.ones_like(magnitude)
    weights[magnitude <= deadzone_limit] = magnitude[magnitude <= deadzone_limit] / deadzone_limit
    weights[magnitude > rim_limit] = (1 - magnitude[magnitude > rim_limit]) / (1 - rim_limit)
    weights = np.clip(weights, 0, 1)

    # Apply trail effect for the most recent frames
    recent_highlights = highlight_frames[-recent_frames_to_highlight:]
    for idx, frame_index in enumerate(reversed(recent_highlights)):
        if 0 <= frame_index < len(weights):
            chunk_index = idx // highlight_chunk_size
            multiplier = base_multiplier * (decay_factor ** chunk_index)
            weights[frame_index] *= multiplier

    # Generate heatmap
    heatmap, _, _ = np.histogram2d(
        interp_x, interp_y, 
        bins=bins, range=[[-1, 1], [-1, 1]],
        density=True, weights=weights
    )
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap.T, cmap="magma", cbar=False, square=True,
        norm=plt.Normalize(vmin=0, vmax=np.percentile(heatmap, 98)),
        xticklabels=[], yticklabels=[], ax=ax
    )
    ax.invert_yaxis()
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf))
    buf.close()
    
    return img

def process_frames(args):
    data, start_idx, end_idx, output_filename = args
    height, width, _ = generate_heatmap(data[:1], []).shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    highlight_frames = []

    # Process frames from start_idx to end_idx
    for i in range(max(start_idx, 1), end_idx + 1):
        highlight_frames.append((i - 1) * interp_factor)
        while len(highlight_frames) > recent_frames_to_highlight:
            highlight_frames.pop(0)
        frame = generate_heatmap(data[:i], highlight_frames)
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    return output_filename

def create_video_parallel(data, output_filename=video_filename, num_processes=num_processes):
    """Splits the frame processing across multiple processes and merges the results."""
    total_frames = len(data)
    chunk_size = total_frames // num_processes
    temp_files = []

    os.makedirs("temp_videos", exist_ok=True)

    # Split into chunks for multiprocessing
    tasks = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_processes - 1 else total_frames
        temp_filename = f"temp_videos/part_{i}.mp4"
        temp_files.append(temp_filename)
        tasks.append((data, start_idx, end_idx, temp_filename))

    # Run multiprocessing pool
    with Pool(num_processes) as pool:
        pool.map(process_frames, tasks)

    # Concatenate the temporary videos
    concat_list = "temp_videos/concat_list.txt"
    with open(concat_list, "w") as f:
        for temp_file in temp_files:
            full_path = os.path.abspath(temp_file)
            f.write(f"file '{full_path}'\n")

    final_command = f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c copy {Path(output_filename, video_filename)}"
    subprocess.run(final_command, shell=True)

    # Cleanup temporary files
    for temp_file in temp_files:
        os.remove(temp_file)
    os.remove(concat_list)
    os.rmdir("temp_videos")

    print(f"Final video saved as {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate a heatmap animation from a data file")
    parser.add_argument("data", type=str, help="Path to the data file")
    parser.add_argument("--mode", type=str, choices=["image", "video"], default="image", help="Select output mode (default: image)")
    parser.add_argument("--output", type=str, help="Set output path")
    parser.add_argument("--workers", type=int, default=num_processes, help="Number of parallel workers (default: max CPU cores - 1)")
    args = parser.parse_args()

    with open(args.data, 'r') as f:
        data = json.load(f)

    if args.mode == "video":
        create_video_parallel(data, args.output, args.workers)
    elif args.mode == "image":
        img = generate_heatmap(data, [])
        Image.fromarray(img).save(Path(args.output, image_filename))
        print(f"Image saved as {args.output}")

if __name__ == "__main__":
    main()