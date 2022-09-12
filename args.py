from argparse import ArgumentParser

"""
ARGS:
"""


def train_args():

    parser = ArgumentParser()

    parser.add_argument(
        "--data_path",
        "-p",
        type=str,
        default=False,
        help="Path to train and test source directories",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=20,
        help="Number of epochs to train for; **default=20**",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=64,
        help="Batch size **default=64**",
    )
    parser.add_argument(
        "--encoder_lr",
        "-elr",
        type=float,
        default=1e-4,
        help="Encoder learning rate; **default=1e-4**",
    )
    parser.add_argument(
        "--decoder_lr",
        "-dlr",
        type=float,
        default=4e-4,
        help="Decoder learning rate; **default=4e-4**",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
        help="Gradient threshold value; **default=5.0**",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to train network on; **default=`cuda`**",
    )
    parser.add_argument(
        "--attention",
        "-a",
        type=bool,
        default=False,
        help="Use attention or not",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout value; **default=0.5**",
    )
    parser.add_argument(
        "--finetune",
        "-f",
        type=bool,
        default=False,
        help="Finetune encoder or not",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of sub-processes to use for loading data; **default=-1**",
    )
    parser.add_argument(
        "--alpha_c",
        type=float,
        default=1.0,
        help="regularization parameter for `doubly stochastic attention`; **default=-1**",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=200,
        help="Iteration interval to print out information; **default=200**",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Checkpoint to resume training",
    )

    return parser.parse_args()


def predict_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--img_path",
        "-p",
        type=str,
        default="./RAW Data/memes/african-boy-checka.jpg",
        help="Path to image file",
    )
    parser.add_argument(
        "--weights_file",
        "-w",
        type=str,
        default=r".\Naive-Checkpoints\BEST_checkpoint_140_cap_per_img_2_min_word_freq.pth.tar",
        help="Path to pre-trained weights",
    )
    parser.add_argument(
        "--beam_size",
        "-b",
        type=int,
        default=3,
        help="Beam size to use for the beam search; **default=3**",
    )
    parser.add_argument(
        "--attention",
        "-a",
        type=bool,
        default=False,
        help="Use attention or not",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="C:/Users/tiva/Desktop/MEME-PROJECT/MY SH$T/Captioned",
        help="Path to save the captioned-image",
    )

    return parser.parse_args()
