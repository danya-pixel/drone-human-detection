import os
from typing import Tuple

import matplotlib.pyplot as plt
from PIL import Image


def get_image_resolution(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def width_height_stats(dataset_folder: str) -> Tuple[list, list]:
    resolutions = []
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                image_path = os.path.join(root, file)
                width, height = get_image_resolution(image_path)
                resolutions.append((width, height))

    widths = [res[0] for res in resolutions]
    heights = [res[1] for res in resolutions]

    plt.hist(widths, bins=50, alpha=0.5, color="blue", label="Width")
    plt.hist(heights, bins=50, alpha=0.5, color="red", label="Height")
    plt.title("Distribution of Image Resolutions")
    plt.xlabel("Resolution (pixels)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    return widths, heights
