import contractions
import re
from random import seed, shuffle
from tqdm import tqdm


def split_data(dataset, size):
    """
    Splits dataset to train, validation, and test sets.

    :param dataset: a complete list of images in the dataset
    :param size: float representing the size of train set
    """
    assert type(size) == float

    total = len(dataset)
    train_size = int(total * size)
    remain = total - train_size
    val_size = int(remain * 0.7)

    seed(123)
    shuffle(dataset)
    train = dataset[:train_size]
    val = dataset[train_size : train_size + val_size]
    test = dataset[train_size + val_size :]

    return train, val, test


def caption_processing(caption, cap_id, img_id):
    """
    Preprocesses caption text and creates a dictionary for the caption

    :param caption: caption to be processed
    :param cap_id: caption id
    :param img_id: image id
    """

    cap_dict = {}
    caption = contractions.fix(caption)
    #     caption = re.sub(r"[^\w\s]{2}", '', caption)
    cap_dict["imgid"] = img_id
    cap_dict["raw"] = caption
    cap_dict["tokens"] = re.findall(r"\w+", caption)
    cap_dict["sentid"] = cap_id

    return cap_dict


def build_json(img_files, captions, train_size=0.7):
    """
    Builds a JSON file with training, validation, and test splits, similar to that
    constructed by Andrej Karpathy for the MSCOCO, Flicker30k, and Flicker8k datasets.

    :param img_files: a complete list of images in the dataset
    :param captions: list of tuples containing (img_file, caption)
    :param train_size: float representing the size of train set
    """

    trainset, valset, _ = split_data(img_files, train_size)

    images_data = []
    curr_id = 0

    for i, img in enumerate(tqdm(img_files)):
        img_data = {}
        img_name = img.replace("-", " ")
        img_name, _ = img_name.split(".")

        img_data["imgid"] = i
        img_data["filename"] = img
        img_data["sentids"] = []
        img_data["sentences"] = []

        if img in trainset:
            img_data["split"] = "train"
        elif img in valset:
            img_data["split"] = "val"
        else:
            img_data["split"] = "test"

        for cap_info in captions:
            try:
                cap_name, cap = cap_info
            except ValueError:
                captions.remove(cap_info)
                continue
            if cap_name == img_name:
                cap_dict = caption_processing(cap, curr_id, i)
                img_data["sentences"].append(cap_dict)
                img_data["sentids"].append(curr_id)

                curr_id += 1

        images_data.append(img_data)

    return images_data
