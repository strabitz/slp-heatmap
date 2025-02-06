import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


deadzone_limit = 0.3
rim_limit = .9
interp_factor = 10
bins = 300


def generate_heatmap(data):
    # Extract x and y coordinates
    x_coords = np.array([point["x"] for point in data])
    y_coords = np.array([point["y"] for point in data])

    # Interpolation function 
    frames = np.arange(len(x_coords))
    new_frames = np.linspace(frames.min(), frames.max(), len(frames) * interp_factor)
    interp_x = interp1d(frames, x_coords, kind="linear")(new_frames)
    interp_y = interp1d(frames, y_coords, kind="linear")(new_frames)

    # Set weights
    magnitude = np.sqrt(interp_x**2 + interp_y**2)
    weights = np.ones_like(magnitude)
    weights[magnitude <= deadzone_limit] = magnitude[magnitude <= deadzone_limit] / deadzone_limit
    weights[magnitude > rim_limit] = (1 - magnitude[magnitude > rim_limit]) / (1 - rim_limit)
    weights = np.clip(weights, 0, 1)

    # Create plot
    heatmap, x_edges, y_edges = np.histogram2d(interp_x, interp_y, bins=bins, density=True, weights=weights)
    heatmap = gaussian_filter(heatmap, sigma=1)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(heatmap.T, cmap="magma", cbar=False, square=True,
                    norm=plt.Normalize(vmin=0, vmax=np.percentile(heatmap, 98)),
                    xticklabels=[], yticklabels=[])
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.savefig(f"output.png", dpi=1000, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate analog stick heatmaps from a data file")
    parser.add_argument("data", type=str, help="Path to the data file")
    args = parser.parse_args()
    with open(args.data, 'r') as f:
        data = json.load(f)

    generate_heatmap(data)
    

if __name__ == "__main__":
    main()
