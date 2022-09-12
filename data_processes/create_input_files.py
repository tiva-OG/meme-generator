from utils import create_input_files

if __name__ == "__main__":

    # Create input files (along with word map)
    create_input_files(
        json_path="RAW Data/captions.json",
        image_folder="RAW Data/memes/",
        captions_per_image=140,
        min_word_freq=2,
        output_folder="INPUT Files/",
        max_len=35,
    )
