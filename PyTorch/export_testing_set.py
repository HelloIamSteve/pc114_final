import torch
import torchvision
import torchvision.transforms as transforms
import os
import csv

import config

if __name__ == "__main__":
    output_dir = '../testing_set'
    image_dir = os.path.join(output_dir, 'images')
    labels_csv_path = os.path.join(output_dir, 'labels.csv')
    labels_txt_path = os.path.join(output_dir, 'labels.txt')

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # dataset
    dataset_test = torchvision.datasets.MNIST(root=config.dataset_root_dir, train=False, download=True)

    with open(labels_csv_path, "w", newline="", encoding="utf-8") as csv_file, \
         open(labels_txt_path, "w", encoding="utf-8") as txt_file:

        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["filename", "label"])

        for i, (image, label) in enumerate(dataset_test):
            filename = f"{i:05d}.jpg"
            image_path = os.path.join(image_dir, filename)

            # Save raw grayscale MNIST image, size = 28x28
            image.save(image_path)

            csv_writer.writerow([filename, label])
            txt_file.write(f"{filename} {label}\n")

    print("Done.")
    print(f"Saved images to: {image_dir}")
    print(f"Saved labels to: {labels_csv_path}")
    print(f"Total images: {len(dataset_test)}")