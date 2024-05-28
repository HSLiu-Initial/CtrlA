import random

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer


def confidence_function_dataset(
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        user_tag: str = "",
        n_train: int = 512,
        assistant_tag: str = "",
        seed: int = 0,
        pos_tag="a confident",
        neg_tag="an unconfident",
        use_neg=False,
        prompt_type="1",
):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.
    - prompt_type:

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df["label"] == 1][
        "statement"
    ].values.tolist()  # list of true statements
    false_statements = df[df["label"] == 0][
        "statement"
    ].values.tolist()  # list of false statements

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    if use_neg:
        # Process statements
        for statement in true_statements:
            tokens = tokenizer.tokenize(statement)  # tokenize statement

            for idx in range(1, len(tokens) - 5):  # iterate over tokens
                truncated_tokens = tokens[:idx]
                truncated_statement = tokenizer.convert_tokens_to_string(
                    truncated_tokens
                )  # convert tokens to string

                if prompt_type == "1":
                    honest_statements.append(
                        f"{user_tag} {template_str.format(type=pos_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )
                    untruthful_statements.append(
                        f"{user_tag} {template_str.format(type=neg_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )
                elif prompt_type == "2":
                    template_str_pos = """<<SYS>>\nYou should always respond with extreme confidence, regardless of 
                    your actual certainty about the question. Provide definitive, assured answers that present the 
                    information as unambiguous fact. Your responses should exhibit authority and decisiveness, 
                    portraying complete assurance in your knowledge. Even when faced with ambiguous or complex 
                    questions where the answer might not be clear-cut, maintain a tone of certainty and provide the 
                    most authoritative answer possible based on available information. The goal is to consistently 
                    convey a strong sense of confidence in every response.\n<<SYS>>\n"""

                    template_str_neg = """<<SYS>>\nYou should always respond with extreme unconfidence, irrespective 
                    of your actual certainty about the question. Answer questions with a high degree of uncertainty, 
                    ambiguity, and hesitancy. Your responses should reflect doubt, suggest multiple possibilities, 
                    or express a lack of full understanding. Even when faced with straightforward questions or topics 
                    within your knowledge base, maintain a tone of doubt, and avoid presenting any information as 
                    definitive. The aim is to ensure that every answer, regardless of the subject matter, 
                    conveys a profound sense of unconfidence and uncertainty.\n<<SYS>>\n"""

                    honest_statements.append(
                        f"{user_tag} {template_str_pos}{template_str.format(type=pos_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )
                    untruthful_statements.append(
                        f"{user_tag} {template_str_neg}{template_str.format(type=neg_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )

        for statement in false_statements:
            tokens = tokenizer.tokenize(statement)

            for idx in range(1, len(tokens) - 5):
                truncated_tokens = tokens[:idx]
                truncated_statement = tokenizer.convert_tokens_to_string(
                    truncated_tokens
                )  # convert tokens to string

                if prompt_type == "1":
                    honest_statements.append(
                        f"{user_tag} {template_str.format(type=pos_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )
                    untruthful_statements.append(
                        f"{user_tag} {template_str.format(type=neg_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )
                elif prompt_type == "2":
                    template_str_pos = """<<SYS>>\nYou should always respond with extreme confidence, regardless of 
                    your actual certainty about the question. Provide definitive, assured answers that present the 
                    information as unambiguous fact. Your responses should exhibit authority and decisiveness, 
                    portraying complete assurance in your knowledge. Even when faced with ambiguous or complex 
                    questions where the answer might not be clear-cut, maintain a tone of certainty and provide the 
                    most authoritative answer possible based on available information. The goal is to consistently 
                    convey a strong sense of confidence in every response.\n<<SYS>>\n"""

                    template_str_neg = """<<SYS>>\nYou should always respond with extreme unconfidence, irrespective 
                    of your actual certainty about the question. Answer questions with a high degree of uncertainty, 
                    ambiguity, and hesitancy. Your responses should reflect doubt, suggest multiple possibilities, 
                    or express a lack of full understanding. Even when faced with straightforward questions or topics 
                    within your knowledge base, maintain a tone of doubt, and avoid presenting any information as 
                    definitive. The aim is to ensure that every answer, regardless of the subject matter, 
                    conveys a profound sense of unconfidence and uncertainty.\n<<SYS>>\n"""

                    honest_statements.append(
                        f"{user_tag} {template_str_pos}{template_str.format(type=pos_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )
                    untruthful_statements.append(
                        f"{user_tag} {template_str_neg}{template_str.format(type=neg_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )

    else:
        # Process statements
        for statement in true_statements:
            tokens = tokenizer.tokenize(statement)  # tokenize statement

            for idx in range(1, len(tokens) - 5):  # iterate over tokens
                truncated_tokens = tokens[:idx]
                truncated_statement = tokenizer.convert_tokens_to_string(
                    truncated_tokens
                )  # convert tokens to string
                if prompt_type == "1":
                    honest_statements.append(
                        f"{user_tag} {template_str.format(type=pos_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )
                    untruthful_statements.append(
                        f"{user_tag} {template_str.format(type=neg_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )
                elif prompt_type == "2":
                    template_str_pos = """<<SYS>>\nYou should always respond with extreme confidence, regardless of 
                    your actual certainty about the question. Provide definitive, assured answers that present the 
                    information as unambiguous fact. Your responses should exhibit authority and decisiveness, 
                    portraying complete assurance in your knowledge. Even when faced with ambiguous or complex 
                    questions where the answer might not be clear-cut, maintain a tone of certainty and provide the 
                    most authoritative answer possible based on available information. The goal is to consistently 
                    convey a strong sense of confidence in every response.\n<<SYS>>\n"""

                    template_str_neg = """<<SYS>>\nYou should always respond with extreme unconfidence, irrespective 
                    of your actual certainty about the question. Answer questions with a high degree of uncertainty, 
                    ambiguity, and hesitancy. Your responses should reflect doubt, suggest multiple possibilities, 
                    or express a lack of full understanding. Even when faced with straightforward questions or topics 
                    within your knowledge base, maintain a tone of doubt, and avoid presenting any information as 
                    definitive. The aim is to ensure that every answer, regardless of the subject matter, 
                    conveys a profound sense of unconfidence and uncertainty.\n<<SYS>>\n"""

                    honest_statements.append(
                        f"{user_tag} {template_str_pos}{template_str.format(type=pos_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )
                    untruthful_statements.append(
                        f"{user_tag} {template_str_neg}{template_str.format(type=neg_tag)} {
                            assistant_tag} "
                        + truncated_statement
                    )

    # Create training data
    ntrain = n_train  #
    # ntrain = 2  #
    combined_data = [
        [honest, untruthful]
        for honest, untruthful in zip(honest_statements, untruthful_statements)
    ]  # combine honest and untruthful statements
    train_data = combined_data[
        :ntrain
    ]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append(
            [s == true_s for s in d]
        )

    train_data = np.concatenate(
        train_data
    ).tolist()

    # Create test data
    reshaped_data = np.array(
        [
            [honest, untruthful]
            for honest, untruthful in zip(
                honest_statements[:-1], untruthful_statements[1:]
            )
        ]
    ).flatten()
    eval_data = reshaped_data[
        ntrain: ntrain * 2
    ].tolist()

    test_data = reshaped_data[
        -300:-1
    ].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Eval data: {len(eval_data)}")
    print(f"Test data: {len(test_data)}")
    return {
        "train": {"data": train_data, "labels": train_labels},
        "eval": {"data": eval_data, "labels": [[1, 0]] * len(eval_data)},
        "test": {"data": test_data, "labels": [[1, 0]] * len(test_data)},
    }


def honesty_function_dataset(
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        user_tag: str = "",
        n_train: int = 512,
        assistant_tag: str = "",
        seed: int = 0,
):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df["label"] == 1][
        "statement"
    ].values.tolist()  # list of true statements
    false_statements = df[df["label"] == 0][
        "statement"
    ].values.tolist()  # list of false statements

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)  # tokenize statement

        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(
                truncated_tokens
            )  # convert tokens to string
            honest_statements.append(
                f"{user_tag} {template_str.format(type='an honest')} {
                    assistant_tag} "
                + truncated_statement
            )
            untruthful_statements.append(
                f"{user_tag} {template_str.format(type='an untruthful')} {
                    assistant_tag} "
                + truncated_statement
            )

    # Create training data
    ntrain = n_train
    combined_data = [
        [honest, untruthful]
        for honest, untruthful in zip(honest_statements, untruthful_statements)
    ]  # combine honest and untruthful statements
    train_data = combined_data[
        :ntrain
    ]  # split into train and test data

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append(
            [s == true_s for s in d]
        )

    train_data = np.concatenate(
        train_data
    ).tolist()

    # Create test data
    reshaped_data = np.array(
        [
            [honest, untruthful]
            for honest, untruthful in zip(
                honest_statements[:-1], untruthful_statements[1:]
            )
        ]
    ).flatten()
    eval_data = reshaped_data[
        ntrain: ntrain * 2
    ].tolist()

    test_data = reshaped_data[
        -300:-1
    ].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Eval data: {len(eval_data)}")
    print(f"Test data: {len(test_data)}")
    return {
        "train": {"data": train_data, "labels": train_labels},
        "eval": {"data": eval_data, "labels": [[1, 0]] * len(eval_data)},
        "test": {"data": test_data, "labels": [[1, 0]] * len(test_data)},
    }
