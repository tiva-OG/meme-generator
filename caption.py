import os
import json
import textwrap
from PIL import Image, ImageDraw, ImageFont
from string import ascii_letters

# import argparse
# import torch.nn as nn
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from cv2 import imread, resize
from torchvision import transforms

from args import predict_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_image(encoder, img_path):

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([normalize])

    img = imread(img_path)
    if img.ndim == 2:
        img = img[..., np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = resize(img, (256, 256))
    img = np.transpose(img, (2, 0, 1))
    img = img / 255.0
    img = torch.FloatTensor(img).to(device)
    img = transform(img)
    img = torch.unsqueeze(img, 0)

    return encoder(img)


def beam_search(model, img_path, beam_size, attention=False):

    encoder, decoder = model

    feature_map = encode_image(encoder, img_path)
    feature_map_size = feature_map.size(1)
    encoder_dim = feature_map.size(3)

    feature_map = feature_map.view(1, -1, encoder_dim)
    num_pixels = feature_map.size(1)
    feature_map = feature_map.expand(beam_size, num_pixels, encoder_dim)

    k_prev_words = torch.LongTensor([[word_map["<start>"]]] * beam_size).to(device)
    seqs = k_prev_words

    topk_scores = torch.zeros(beam_size, 1)
    seqs_alpha = torch.zeros(beam_size, 1, feature_map_size, feature_map_size)

    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []

    step = 1
    h, c = decoder.init_hidden_state(feature_map)

    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)

        if attention:
            awe, alpha = decoder.attention(feature_map, h)
            alpha = alpha.view(-1, feature_map_size, feature_map_size)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        else:
            h, c = decoder.decode_step(embeddings, (h, c))

        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)
        scores = topk_scores.expand_as(scores) + scores

        if step == 1:
            topk_scores, topk_words = scores[0].topk(beam_size, 0, True, True)
        else:
            topk_scores, topk_words = scores.view(-1).topk(beam_size, 0, True, True)

        prev_word_inds = topk_words // vocab_size
        next_word_inds = topk_words % vocab_size

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

        if attention:
            seqs_alpha = torch.cat(
                [seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1
            )

        incomplete_inds = [
            i for i, word in enumerate(topk_words) if word != word_map["<end>"]
        ]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(list(seqs[complete_inds]))
            complete_seqs_scores.extend(topk_scores[complete_inds])
            if attention:
                complete_seqs_alpha.extend(list((seqs_alpha[complete_inds])))

        beam_size -= len(complete_inds)

        if beam_size == 0:
            break

        if attention:
            seqs_alpha = seqs_alpha[incomplete_inds]

        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        # topk_words = topk_words[incomplete_inds].squeeze(1)
        topk_scores = topk_scores[incomplete_inds].unsqueeze(1)
        feature_map = feature_map[prev_word_inds[incomplete_inds]]
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50:
            break

        step += 1

    max_score_idx = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[max_score_idx]

    if attention:
        alpha = complete_seqs_alpha[max_score_idx]

        return seq, alpha

    return seq


def decode_sequence(seq):
    seq = [
        i.item()
        for i in seq
        if i.item() not in (word_map["<start>"], word_map["<end>"])
    ]
    caption = " ".join([word_map_rev[ind] for ind in seq])

    return caption


def visualize_caption(img_path, caption, save_path):
    image = Image.open(img_path)
    font = ImageFont.truetype(r".\fonts\kultros-regular.ttf", 20)
    # eatday
    # Kultros-Regular
    # Milky Quaker

    avg_char_width = sum(font.getsize(char)[0] for char in ascii_letters) / len(
        ascii_letters
    )
    max_char_count = int(image.size[0] * 0.65 / avg_char_width)

    text = textwrap.fill(caption, width=max_char_count)

    image_editable = ImageDraw.Draw(image)
    image_editable.text((5, 10), text, "#ffffff", font=font)
    image.save(save_path)
    image.show()


if __name__ == "__main__":

    args = predict_args()
    img_name, ext = os.path.splitext(os.path.basename(args.img_path))
    img_name = img_name + f"_beam_size_{args.beam_size}" + ext
    save_path = os.path.join(args.save_path, img_name)
    word_map_file = r".\INPUT Files\WORDMAP_140_cap_per_img_2_min_word_freq.json"

    # load word_map
    with open(word_map_file, "r") as j:
        word_map = json.load(j)
    vocab_size = len(word_map)
    word_map_rev = {v: k for k, v in word_map.items()}

    # load pretrained_weights
    pretrained_weights = torch.load(args.weights_file, map_location=device)
    encoder = pretrained_weights["encoder"]
    encoder.eval()
    decoder = pretrained_weights["decoder"]
    decoder.eval()
    model = (encoder, decoder)

    seq = beam_search((encoder, decoder), args.img_path, args.beam_size)
    caption = decode_sequence(seq)

    visualize_caption(args.img_path, caption, save_path)
