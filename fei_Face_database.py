import os
import random
import shutil
import zipfile


def unzip_file(zip_file_path: str, extract_to_path: str):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)


def unzip_dataset(source_path: str):
    paths = os.listdir(source_path)

    for path in paths:
        if path.endswith(".zip"):
            folder_name = path.split(".")[0]
            dst_path = os.path.join(source_path, folder_name)

            if os.path.exists(dst_path):
                pass
            else:
                os.makedirs(dst_path, exist_ok=True)
                unzip_file(os.path.join(source_path, path), dst_path)
        else:
            pass


def copy_random_images(source_path: str, destination_path: str, test_count: int = 4):
    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)

    for path in os.listdir(source_path):
        if path.startswith(".") or path.endswith(".zip"):
            continue
        images = [
            f for f in os.listdir(os.path.join(source_path, path)) if f.endswith(".jpg")
        ]

        class_images = {}

        for img in images:
            class_name = img.split("-")[0]
            if class_name not in class_images:
                class_images[class_name] = []
            class_images[class_name].append(img)

        train_destination = os.path.join(destination_path, "training-data")
        test_destination = os.path.join(destination_path, "test-data")

        os.makedirs(train_destination, exist_ok=True)
        os.makedirs(test_destination, exist_ok=True)

        for class_name, images in class_images.items():
            random.shuffle(images)

            train_images = images[test_count:]
            test_images = images[:test_count]

            os.makedirs(
                os.path.join(train_destination, f"s{class_name}"), exist_ok=True
            )
            os.makedirs(os.path.join(test_destination, f"s{class_name}"), exist_ok=True)

            for img in train_images:
                src_path = os.path.join(source_path, path, img)
                dst_path = os.path.join(train_destination, f"s{class_name}", img)
                shutil.copy(src_path, dst_path)

            for img in test_images:
                src_path = os.path.join(source_path, path, img)
                dst_path = os.path.join(test_destination, f"s{class_name}", img)
                shutil.copy(src_path, dst_path)


if __name__ == "__main__":
    source_path = "dataset/FEI_Face_Database"

    destination_path = "dataset/FEI_Face_Database/output"

    unzip_dataset(source_path)

    test_count = 4

    unzip_dataset(source_path)
    copy_random_images(source_path, destination_path, test_count)
