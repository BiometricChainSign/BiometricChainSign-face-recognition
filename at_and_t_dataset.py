import os
import cv2
import numpy as np
import shutil
import random


# Function to load images from a specific folder
def load_images_from_folder(folder: str):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(
                {
                    "id": f"{filename.split('.')[0]}",
                    "img": img,
                }
            )
    return images


# Function to apply distortions to the image
def apply_distortion(image: cv2.typing.MatLike):
    # Randomly choose between rotation and mirroring
    distortion_type = np.random.choice(["rotation", "mirroring"], p=[0.5, 0.5])

    if distortion_type == "rotation":
        # Random rotation
        rows, cols = image.shape
        rotation_matrix = cv2.getRotationMatrix2D(
            (cols / 2, rows / 2), np.random.uniform(-10, 10), 1
        )
        distorted_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    else:
        # Random mirroring
        distorted_image = cv2.flip(
            image, 1
        )  # 1 for horizontal flip, 0 for vertical flip

    return distorted_image


# Function to apply random blur to the image
def apply_blur(image: cv2.typing.MatLike):
    # Example: random blur
    blur_value = (
        np.random.randint(1, 6) * 2 + 1
    )  # Random odd blur kernel size between 1 and 5
    blurred_image = cv2.GaussianBlur(image, (blur_value, blur_value), 0)
    return blurred_image


if __name__ == "__main__":
    # Dataset directory
    dataset_dir = "dataset/AT&T Database of Faces"

    # Output directory for training and testing data
    output_dir = f"{dataset_dir}/output"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create directories for training and testing
    training_dir = os.path.join(output_dir, "training-data")
    testing_dir = os.path.join(output_dir, "test-data")

    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(testing_dir, exist_ok=True)

    # Loop through folders
    num_classes = len(os.listdir(dataset_dir)) - 1  # ignore README file
    for i in range(1, num_classes):
        subject_folder = os.path.join(dataset_dir, f"s{i}")

        # Load images from the current folder
        images = load_images_from_folder(subject_folder)

        # Shuffle the images
        random.shuffle(images)

        # Calculate the number of additional images to generate
        total_images = len(images)
        target_images = (3000 + num_classes - 1) / (
            num_classes - 1
        )  # Desired total + num_classes
        additional_images = int((target_images - total_images))

        # Create subdirectories for each class
        train_subdir = os.path.join(training_dir, f"s{i}")
        test_subdir = os.path.join(testing_dir, f"s{i}")
        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(test_subdir, exist_ok=True)

        # Split the images into training and testing sets
        train_size = int(0.7 * total_images)
        train_images = images[:train_size]
        test_images = images[train_size:]

        # Save original test images
        for j, image in enumerate(test_images, start=1):
            cv2.imwrite(
                os.path.join(test_subdir, f"{j}_{image['id']}.pgm"), image["img"]
            )

        # Save original training images
        for j, image in enumerate(train_images, start=1):
            cv2.imwrite(
                os.path.join(train_subdir, f"{j}_{image['id']}.pgm"), image["img"]
            )

        # Save additional distorted images for testing
        len_test_images = len(test_images)
        for j in range(
            len_test_images + 1, len_test_images + int(additional_images * 0.3) + 1
        ):
            original_image = random.choice(test_images)
            distorted_image = apply_distortion(original_image["img"])
            blurred_image = apply_blur(distorted_image)
            cv2.imwrite(
                os.path.join(
                    test_subdir, f"{j}_{original_image['id']}_augmentation.pgm"
                ),
                blurred_image,
            )

        # Save additional distorted images for training
        len_train_images = len(train_images)
        for j in range(
            len_train_images + 1, len_train_images + int(additional_images * 0.70) + 1
        ):
            original_image = random.choice(train_images)
            distorted_image = apply_distortion(original_image["img"])
            blurred_image = apply_blur(distorted_image)
            cv2.imwrite(
                os.path.join(
                    train_subdir, f"{j}_{original_image['id']}_augmentation.pgm"
                ),
                blurred_image,
            )

    # Calculate the total number of images left
    training_imgs = sum(
        [len(files) for r, d, files in os.walk(f"{output_dir}/training-data")]
    )
    test_imgs = sum([len(files) for r, d, files in os.walk(f"{output_dir}/test-data")])

    print(
        f"Total training data: {training_imgs}\nTotal test data: {test_imgs}\nTotal dataset data: {training_imgs+test_imgs}"
    )
