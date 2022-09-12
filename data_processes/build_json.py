import json
import argparse
from functions import build_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Builds a JSON file with train, validation, and test splits"
    )

    parser.add_argument("--images", help="Path to the folder containing image-files")
    parser.add_argument("--captions", help="Path to the file containing captions")
    parser.add_argument(
        "--train_size", type=float, default=0.7, help="Size of train split"
    )
    parser.add_argument(
        "--save_path",
        default="RAW Data/captions.json",
        help="path to save the JSON file",
    )
    args = parser.parse_args()

    images_file = args.images
    captions_file = args.captions
    train_size = args.train_size
    save_path = args.save_path

    # load images and captions
    with open(images_file, "r") as f:
        images = f.readlines()
    with open(captions_file, "r") as f:
        captions = f.readlines()

    images = [name.strip() for name in images]
    captions = [cap.lower().strip() for cap in captions]

    # split captions to image_name and text
    captions = [tuple(cap.split(" - ", 1)) for cap in captions]
    # remove any caption that does not have both image_name and text
    good_caps = [cap for cap in captions if len(cap) == 2]

    # build json file for images and captions
    file = build_json(images, captions, train_size)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(file, f, ensure_ascii=False, indent=2)
