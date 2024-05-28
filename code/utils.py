# -*- coding:utf-8 -*-
import copy
import csv
import json
import logging
import os
import pprint
import random
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import torch
from jsonlines import jsonlines
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize


class ConfigParser(object):
    """
    A simple configuration parser that sets attributes based on a provided options dictionary.
    """

    def __init__(self, options):
        """
        Initializes the ConfigParser with the provided options.

        Args:
            options (dict): A dictionary of options to set as attributes.
        """
        for i, j in options.items():
            if isinstance(j, dict):
                for k, v in j.items():
                    setattr(self, k, v)
            else:
                setattr(self, i, j)

    def update(self, args):
        """
        Updates the ConfigParser attributes with the provided args.

        Args:
            args (object): An object with attributes to set on the ConfigParser.
        """
        for args_k in args.__dict__:
            if getattr(args, args_k) is not None:
                setattr(self, args_k, getattr(args, args_k))

    def save(self, save_dir):
        """
        Saves the ConfigParser attributes to a JSON file.

        Args:
            save_dir (str): The directory to save the JSON file to.
        """
        dic = self.__dict__
        dic["device"] = "cuda" if dic["device"] == torch.device(
            "cuda") else "cpu"
        js = json.dumps(dic)
        with open(save_dir, "w") as f:
            f.write(js)


def init_method(PREFIX, args):
    """
    Initializes the directories for image files, log files, model files, and result files.
    Also sets up the logger for the application.

    Args:
        PREFIX (str): The base directory for the application.
        args (object): An object with attributes to set on the ConfigParser.

    Returns:
        tuple: A tuple containing the directories for image files, log files, model files, and result files.
    """
    img_dir = os.path.join(PREFIX, "img_file")
    save_dir = os.path.join(PREFIX, "log_file")
    model_save_dir = os.path.join(PREFIX, "model_file")
    results_save_dir = os.path.join(PREFIX, "results_file")
    train_dir = os.path.join(results_save_dir, "train")
    eval_dir = os.path.join(results_save_dir, "eval")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.isdir(results_save_dir):
        os.makedirs(results_save_dir)
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    try:
        hyper = copy.deepcopy(args.__dict__)
        pprint.pprint(hyper)
        hyper["device"] = "cuda"
        json_str = json.dumps(hyper, indent=4)
        with open(os.path.join(save_dir, "hyper.json"), "w") as json_file:
            json_file.write(json_str)
    except:
        print("can't process args")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel("INFO")
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel("WARNING")
    fhlr = logging.FileHandler(os.path.join(save_dir, "logger.log"))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return img_dir, save_dir, model_save_dir, results_save_dir


def setup_seed(seed):
    """
    Sets the seed for all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed to set.
    """
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_jsonlines(file):
    """
    Loads a JSONLines file and returns a list of its objects.

    Args:
        file (str): The path to the JSONLines file.

    Returns:
        list: A list of objects from the JSONLines file.
    """
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def match(prediction, ground_truth):
    for gt in ground_truth:
        if normalize_answer(gt) in normalize_answer(prediction):
            return 1
    return 0


def plot_detection_results(
        input_ids,
        rep_reader_scores_dict,
        THRESHOLD,
        img_dir=None,
        chosen_idx=0,
        start_answer_token=":",
):
    """
    Plots the detection results.

    Args:
        input_ids (list): List of tokenized input strings.
        rep_reader_scores_dict (dict): Dictionary of reputation reader scores.
        THRESHOLD (float): Threshold value for score normalization.
        img_dir (str, optional): Directory to save the plot image. If None, the plot is displayed. Defaults to None.
        chosen_idx (int, optional): Index of the chosen data point. Defaults to 0.
        start_answer_token (str, optional): Token to start the answer. Defaults to ":".
    """
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["r", (255 / 255, 255 / 255, 224 / 255), "g"], N=256
    )
    colormap = cmap

    # Define words and their colors
    words = [token.replace("▁", " ") for token in input_ids]

    # Create a new figure
    fig, ax = plt.subplots(figsize=(12.8, 10), dpi=200)

    # Set limits for the x and y axes
    xlim = 1000
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 10)

    # Remove ticks and labels from the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Starting position of the words in the plot
    x_start, y_start = 1, 8
    y_pad = 0.3
    # Initialize positions and maximum line width
    x, y = x_start, y_start
    max_line_width = xlim

    y_pad = 0.3
    word_width = 0

    iter = 0

    selected_concepts = ["honesty"]
    norm_style = ["mean"]
    selection_style = ["neg"]

    for rep, s_style, n_style in zip(selected_concepts, selection_style, norm_style):
        rep_scores = np.array(rep_reader_scores_dict[rep])
        mean, std = np.median(rep_scores), rep_scores.std()
        rep_scores[
            (rep_scores > mean + 5 * std) | (rep_scores < mean - 5 * std)
        ] = mean  # get rid of outliers

        mag = max(0.3, np.abs(rep_scores).std() / 10)
        min_val, max_val = -mag, mag
        norm = Normalize(
            vmin=min_val, vmax=max_val
        )

        if "mean" in n_style:
            rep_scores = rep_scores - THRESHOLD
            rep_scores = rep_scores / np.std(
                rep_scores[5:]
            )
            rep_scores = np.clip(rep_scores, -mag, mag)
        if "flip" in n_style:
            rep_scores = -rep_scores

        rep_scores[np.abs(rep_scores) < 0.0] = 0

        if s_style == "neg":
            rep_scores = np.clip(rep_scores, -np.inf, 0)
            rep_scores[rep_scores == 0] = mag
        elif s_style == "pos":
            rep_scores = np.clip(rep_scores, 0, np.inf)

        # Initialize positions and maximum line width
        x, y = x_start, y_start
        max_line_width = xlim
        started = False

        for word, score in zip(words[5:], rep_scores[5:]):
            if start_answer_token in word:
                started = True
                continue
            if not started:
                continue

            color = colormap(norm(score))

            # Check if the current word would exceed the maximum line width
            if x + word_width > max_line_width:
                # Move to next line
                x = x_start
                y -= 3

            # Compute the width of the current word
            text = ax.text(x, y, word, fontsize=13)
            word_width = (
                text.get_window_extent(fig.canvas.get_renderer())
                .transformed(ax.transData.inverted())
                .width
            )
            word_height = (
                text.get_window_extent(fig.canvas.get_renderer())
                .transformed(ax.transData.inverted())
                .height
            )

            # Remove the previous text
            if iter:
                text.remove()

            # Add the text with background color
            text = ax.text(
                x,
                y + y_pad * (iter + 1),
                word,
                color="white",
                alpha=0,
                bbox=dict(
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.8,
                    boxstyle=f"round,pad=0",
                    linewidth=0,
                ),
                fontsize=13,
            )

            # Update the x position for the next word
            x += word_width + 0.1

        iter += 1
        if img_dir is not None:
            plt.savefig(
                os.path.join(img_dir, str(chosen_idx) +
                             "_detection_results.png")
            )
            plt.close()

            pairs = list(zip(words, norm(rep_scores)))

            with open(
                    os.path.join(img_dir, str(chosen_idx) +
                                 "_detection_results.csv"),
                    "w",
                    newline="", encoding='utf-8'
            ) as file:
                writer = csv.writer(file)
                writer.writerow(["word", "norm_score"])
                writer.writerows(pairs)
        else:
            plt.show()


def plot_result_given_q(
        test_input,
        user_tag,
        assistant_tag,
        honesty_rep_reader,
        hidden_layers,
        layers,
        rep_reading_pipeline,
        tokenizer,
        model,
        # chosen_idx=0,
        THRESHOLD=0,
        max_new_tokens=100,
        img_dir=None,
):
    """
    Plots the result given a question.

    Args:
        test_input (list): List of test input strings.
        user_tag (str): User tag string.
        assistant_tag (str): Assistant tag string.
        honesty_rep_reader (object): Honesty reputation reader object.
        hidden_layers (list): List of hidden layers.
        layers (list): List of layers.
        rep_reading_pipeline (object): Reputation reading pipeline object.
        tokenizer (object): Tokenizer object.
        model (object): Model object.
        THRESHOLD (float, optional): Threshold value for score normalization. Defaults to 0.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        img_dir (str, optional): Directory to save the plot image. If None, the plot is displayed. Defaults to None.
    """
    rep_reader_scores_dict = {}
    rep_reader_scores_mean_dict = {}
    template_str = "{user_tag} {scenario} {assistant_tag}"
    test_input = [
        template_str.format(scenario=s, user_tag=user_tag,
                            assistant_tag=assistant_tag)
        for s in test_input
    ]

    test_data = []
    for t in test_input:
        with torch.no_grad():
            output = model.generate(
                **tokenizer(t, return_tensors="pt").to(model.device),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0], skip_special_tokens=True)
        # logging.info(completion)
        test_data.append(completion)

    for chosen_idx in range(len(test_data)):
        chosen_str = test_data[chosen_idx]
        input_ids = tokenizer.tokenize(chosen_str)

        results = []

        for ice_pos in range(len(input_ids)):
            ice_pos = -len(input_ids) + ice_pos
            H_tests = rep_reading_pipeline(
                [chosen_str],
                rep_reader=honesty_rep_reader,
                rep_token=ice_pos,
                hidden_layers=hidden_layers,
            )
            results.append(H_tests)

        honesty_scores = []
        honesty_scores_means = []
        for pos in range(len(results)):
            tmp_scores = []
            tmp_scores_all = []
            for layer in hidden_layers:
                tmp_scores_all.append(
                    results[pos][0][layer][0]
                    * honesty_rep_reader.direction_signs[layer][0]
                )
                if layer in layers:
                    tmp_scores.append(
                        results[pos][0][layer][0]
                        * honesty_rep_reader.direction_signs[layer][0]
                    )
            honesty_scores.append(tmp_scores_all)
            honesty_scores_means.append(np.mean(tmp_scores))

        rep_reader_scores_dict["honesty"] = honesty_scores
        rep_reader_scores_mean_dict["honesty"] = honesty_scores_means
        plot_detection_results(
            input_ids,
            rep_reader_scores_mean_dict,
            THRESHOLD,
            img_dir=img_dir,
            chosen_idx=chosen_idx,
            start_answer_token=tokenizer.tokenize(assistant_tag)[-1],
        )
