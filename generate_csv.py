from sklearn.preprocessing import scale
from PIL import Image
import numpy as np
import json
import csv


def generate_csv(image_data, source_dir, dest):
    with open(dest, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(list(range(4096)) + ["target"])
        for index, subject in enumerate(image_data):
            for image_name in subject:
                image = Image.open(source_dir + "\\" + image_name)
                arr = np.asarray(image.getdata(), dtype=np.float32)
                arr = arr.flatten()
                arr = scale(arr)
                arr = np.append(arr, index)
                writer.writerow(arr)


if __name__ == "__main__":
    # load image data from file
    with open("image_data.json") as f:
        image_data = json.load(f)

    processed_dir = "processed_images"

    generate_csv(image_data, processed_dir, "face_data.csv")
