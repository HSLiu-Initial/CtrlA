# -*- coding:utf-8 -*-
import re
import string

import numpy as np
import spacy
import torch
from matplotlib.colors import Normalize

# Load the English language model for SpaCy
nlp = spacy.load("en_core_web_sm")

# Get the default stop words for SpaCy's English language model
spacy_stopwords = nlp.Defaults.stop_words


def format_list_as_numbered_string(lst):
    """
    Formats a list as a numbered string.

    Args:
        lst (list): The list to format.

    Returns:
        str: The formatted string.
    """
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(lst))


def compare_strings_except_punctuation(str1, str2):
    """
    Compares two strings, ignoring trailing punctuation.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.

    Returns:
        bool: True if the strings are equal when trailing punctuation is removed, False otherwise.
    """
    punctuation = string.punctuation

    str1_cleaned = str1.rstrip(punctuation)
    str2_cleaned = str2.rstrip(punctuation)

    return str1_cleaned == str2_cleaned


def transform_data(data):
    """
    Transforms data by splitting it into columns.

    Args:
        data (list): The data to transform. The first item in the list should be a dictionary.

    Returns:
        list: The transformed data. Each item in the list is a list containing a dictionary.
    """
    num_columns = data[0][-1].shape[1]
    transformed_data = []

    for col in range(num_columns):
        column_data = {}
        for key in data[0]:
            column_data[key] = np.array(
                [data[0][key][0, col]], dtype=np.float32)
        transformed_data.append([column_data])

    return transformed_data


def process_batch(
        test_data, confidence_monitoring_pipeline, confidence_tokens, confidence_monitoring_reader, hidden_layer_id
):
    """
    Processes a batch of test data.

    Args:
        test_data (list): The test data to process.
        confidence_monitoring_pipeline (function): The function to use for processing the test data.
        confidence_tokens (list): The tokens to use for processing the test data.
        rep_reader (Reader): The confidence reader to use for processing the test data.
        hidden_layer_id (int): The IDs of the hidden layer to use for processing the test data.

    Returns:
        list: The processed test data. Each item in the list is a list containing a dictionary.
    """
    H_tests_batch = confidence_monitoring_pipeline(
        [test_data],
        rep_reader=confidence_monitoring_reader,
        rep_token=confidence_tokens,
        hidden_layers=hidden_layer_id,
    )

    results_batch = transform_data(H_tests_batch)
    return results_batch


def process_in_batches(
        test_data, confidence_monitoring_pipeline, input_ids, batch_size, confidence_monitoring_reader, hidden_layer_id
):
    """
    Processes test data in batches.

    Args:
        test_data (list): The test data to process.
        confidence_monitoring_pipeline (function): The function to use for processing the test data.
        input_ids (list): The input IDs to use for processing the test data.
        batch_size (int): The size of the batches.
        confidence_monitoring_reader (Reader): The reader to use for processing the test data.
        hidden_layer_id (int): The ID of the hidden layer to use for processing the test data.

    Returns:
        list: The processed test data. Each item in the list is a list containing a dictionary.
    """
    results = []
    total_length = len(input_ids)
    for start_idx in range(0, total_length, batch_size):
        end_idx = min(start_idx + batch_size, total_length)
        confidence_tokens = [-total_length +
                             ice_pos for ice_pos in range(start_idx, end_idx)]
        batch_results = process_batch(
            test_data, confidence_monitoring_pipeline, confidence_tokens, confidence_monitoring_reader, hidden_layer_id
        )
        results.extend(batch_results)
    return results


def postprocess_answer_option_conditioned(answer):
    """
    Postprocess an answer by removing certain substrings.

    Args:
        answer (str): The answer to postprocess.

    Returns:
        str: The postprocessed answer.
    """
    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer


def sentence_matches_pattern(sentence):
    """
    Checks if a sentence matches a specific pattern. The pattern is a regular expression that matches various phrases
    that indicate a lack of information or a request for more information.

    Args:
        sentence (str): The sentence to check.

    Returns:
        bool: True if the sentence matches the pattern, False otherwise.
    """
    pattern = re.compile(
        r"\b(not specified|not mentioned|don't have information|do not know|please provide more context|"
        r"is not clear|cannot confirm|no information|no details|unknown|unspecified|unclear|not available|"
        r"lack of information|without details|needs more clarification|is not provided|is not disclosed|"
        r"awaiting more information|requires further details|information is lacking|details are scarce|"
        r"not been disclosed|absence of information|information is absent|beyond the scope|not covered|"
        r"lacks details|not defined|information is vague|not determined|not ascertainable|information not found|"
        r"missing information|insufficient information|cannot be verified|"
        r"has not been specified|does not mention|no data available|unable to determine|unable to answer|"
        r"inadequate information|details not found|details unavailable|outside the scope of this|not enough details|"
        r"need further info|no further details|not enough context|can't say for certain|can't be certain|"
        r"can't give a definite answer|don't have enough context|unable to say|unable to confirm|unable to provide|"
        r"cannot provide an answer|cannot give a clear answer|no clear answer|can't find the information|"
        r"limited information available|no conclusive information|could not find details|could not determine|"
        r"could not confirm|could not verify|does not provide information|no information about|"
        r"lacking detail on|information not specified about|details not provided|absent information on|"
        r"no disclosure of|not elaborated on|withholding information on|omission of details about|"
        r"not forthcoming with information about|evasive about|information is missing regarding|"
        r"details are lacking about|not specific about|vague about|silence on|leaves out information about|"
        r"skips over details of|avoids mentioning|provide more specific information|provide a more detailed answer|"
        r"could you specify|need more specific|require more details|if you could provide|seeking more information)\b",
        re.IGNORECASE,
    )

    if pattern.search(sentence):
        return True
    else:
        return False


def remove_punctuation(word):
    return word.translate(str.maketrans("", "", string.punctuation))


def split_contractions(word):
    parts = re.split(r"('s|'t|'re|'ve|'m|'d|'ll|n't)", word)
    return [p for p in parts if p]


def process_sentence(sentence):
    """
    Processes a sentence by splitting it into words, splitting contractions, removing punctuation, and converting to lowercase.

    Args:
        sentence (str): The sentence to process.

    Returns:
        list: The processed words.
    """
    words = sentence.split()

    processed_words = []
    for word in words:
        parts = split_contractions(word)
        cleaned_parts = [remove_punctuation(part).lower() for part in parts]
        processed_words.extend(cleaned_parts)
    return processed_words


def tokenize_with_hf(text, tokenizer):
    """
    Tokenizes the input text using the provided tokenizer and returns the tokens and their offset mappings.

    Args:
        text (str): The text to tokenize.
        tokenizer (Tokenizer): The tokenizer to use.

    Returns:
        list: The tokens.
        list: The offset mappings for the tokens.
    """
    encoded_input = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"])
    offset_mapping = encoded_input["offset_mapping"]
    return tokens, offset_mapping


def build_token_to_words_mapping(text, tokens, offset_mapping):
    """
    Builds a mapping from each token to the words that correspond to it.

    Args:
        text (str): The original text.
        tokens (list): The tokens.
        offset_mapping (list): The offset mappings for the tokens.

    Returns:
        dict: A mapping from each token to the words that correspond to it.
    """
    words = text.split()
    word_positions = []
    position = 0
    for word in words:
        start = text.find(word, position)
        end = start + len(word)
        word_positions.append((start, end))
        position = end

    token_to_words = {}
    for idx, (token, (token_start, token_end)) in enumerate(
            zip(tokens, offset_mapping)
    ):
        corresponding_words = []
        for word_idx, (word_start, word_end) in enumerate(word_positions):
            if word_start >= token_end:
                break
            if word_end > token_start:
                corresponding_words.append((words[word_idx], word_idx))

        if corresponding_words:
            token_to_words[idx] = corresponding_words

    return token_to_words


def retain_first_sentence(text, prompt=None):
    """
    Retains the first valid sentence in the text. If a prompt is provided, the first sentence after the prompt is retained.
    If no prompt is provided, the text is expected to contain the string "[/INST]", and the first sentence after this string is retained.

    Args:
        text (str): The text to process.
        prompt (str, optional): An optional prompt that precedes the text.

    Returns:
        str: The processed text, which contains the first valid sentence.
        str: The first valid sentence.
        bool: A flag indicating whether the text contained only one sentence.
    """
    one_sent_flag = False
    nlp = spacy.load("en_core_web_sm")

    def get_first_valid_sentence(doc):
        for sentence in doc.sents:
            if sentence.text.strip(string.punctuation).strip():
                return sentence.text.strip(), True
        return "", False

    if prompt:
        parts = text.replace(prompt, "").strip()
        doc = nlp(parts.strip())
        first_sentence, valid = get_first_valid_sentence(doc)
        if valid and len(list(doc.sents)) == 1:
            one_sent_flag = True
        result = prompt + " " + first_sentence if first_sentence else prompt
    else:
        parts = text.split("[/INST]")
        if len(parts) == 2:
            doc = nlp(parts[1].strip())
            first_sentence, valid = get_first_valid_sentence(doc)
            if valid and len(list(doc.sents)) == 1:
                one_sent_flag = True
            result = (
                parts[0] + "[/INST] " +
                first_sentence if first_sentence else parts[0]
            )
        else:
            raise ValueError("No [/INST]")

    return result, first_sentence, one_sent_flag


def confidence_monitor(
        tokenizer,
        input_ids,
        confidence_monitoring_reader_scores_dict,
        THRESHOLD,
        start_answer_token=":",
        last_sentence="",
):
    """
    Performs confidence_monitor based on the given parameters.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        input_ids (list): The input IDs.
        confidence_monitoring_reader_scores_dict (dict): A dictionary of reader scores.
        THRESHOLD (float): The threshold to use.
        start_answer_token (str, optional): The start answer token. Defaults to ":".
        last_sentence (str, optional): The last sentence. Defaults to "".

    Raises:
        ValueError: If both start_answer_token and last_sentence are None or both have a value.

    Returns:
        dict: A dictionary of pairs of tokens and their corresponding words.
        str: The text generated from the tokens.
    """
    selected_concepts = ["confidence"]
    norm_style = ["mean"]
    selection_style = ["neg"]

    for rep, s_style, n_style in zip(selected_concepts, selection_style, norm_style):
        rep_scores = np.array(confidence_monitoring_reader_scores_dict[rep])
        mean, std = np.median(rep_scores), rep_scores.std()
        rep_scores[(rep_scores > mean + 5 * std) | (rep_scores < mean - 5 * std)] = (
            mean
        )

        mag = max(
            0.3, np.abs(rep_scores).std() / 10
        )
        min_val, max_val = -mag, mag
        norm = Normalize(
            vmin=min_val, vmax=max_val
        )

        if "mean" in n_style:
            rep_scores = (
                rep_scores - THRESHOLD
            )
            rep_scores = rep_scores / np.std(
                rep_scores[5:]
            )
            rep_scores = np.clip(
                rep_scores, -mag, mag
            )
        if "flip" in n_style:
            rep_scores = -rep_scores
        rep_scores[np.abs(rep_scores) < 0.0] = (
            0
        )

        if s_style == "neg":
            rep_scores = np.clip(rep_scores, -np.inf, 0)
            rep_scores[rep_scores == 0] = mag
        elif s_style == "pos":
            rep_scores = np.clip(rep_scores, 0, np.inf)
        pairs = {}
        tokens_gen = []
        if start_answer_token is not None and last_sentence is None:
            start_token_length = len(start_answer_token)
            buffer = []
            started = False
            for word, score in zip(input_ids[5:], rep_scores[5:]):
                buffer.append(word)
                if len(buffer) > start_token_length:
                    buffer.pop(0)

                if buffer == start_answer_token and not started:
                    started = True
                    buffer.clear()
                    idx = 1
                    continue

                if not started:
                    continue

                tokens_gen.append(word)
                pairs[idx] = {"token": word, "score": norm(score)}
                idx += 1
        elif last_sentence is not None and start_answer_token is None:
            last_sentence_ids = tokenizer.tokenize(last_sentence)
            start_token_length = len(last_sentence_ids)
            idx = 1
            started = True
            for word, score in zip(
                    input_ids[-start_token_length:], rep_scores[-start_token_length:]
            ):
                tokens_gen.append(word)
                pairs[idx] = {"token": word, "score": norm(score)}
                idx += 1
        else:
            raise ValueError(
                "start_answer_token and last_sentence cannot both be None and cannot both have a value."
            )

        if started:
            text = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(tokens_gen))
            tokens, offset_mapping = tokenize_with_hf(text, tokenizer)
            token_to_words_mapping = build_token_to_words_mapping(
                text, tokens, offset_mapping
            )

            for key in pairs:
                try:
                    pairs[key]["word_position"] = token_to_words_mapping[key]
                except KeyError:
                    continue

        return pairs, text


def result_given_prompt_with_control(
        prompt,
        confidence_monitoring_reader,
        honesty_control_reader,
        coeff,
        hidden_layer_id,
        control_layer_id,
        monitoring_layer_id,
        confidence_monitoring_pipeline,
        honesty_control_pipeline,
        tokenizer,
        model,
        repetition_penalty=1.1,
        only_return_first_sentence=True,
        THRESHOLD=0,
        max_new_tokens=100,
        beam_size=1,
        collect_pairs=True,
        mode="adaptive_retrieval",
):
    """
    Generates text based on a given prompt and a set of parameters. It uses a confidence reader and an honesty
    controller to guide the generation process. The function can operate in different modes, and can return either
    the first sentence of the generated text or the entire text.

    Args:
        prompt (str): The prompt to base the generation on.
        confidence_monitoring_reader (Reader): The confidence reader to use for the generation.
        honesty_control_reader (Reader): The honesty control reader to use for the generation.
        coeff (float): The coefficient to use for the honesty control reader.
        hidden_layer_id (list): The IDs of the hidden layers to use for the generation.
        control_layer_id (list): The IDs of the control layers to use for the generation.
        monitoring_layer_id (list): The IDs of the reading layers to use for the generation.
        confidence_monitoring_pipeline (function): The function to use for the confidence reading pipeline.
        honesty_control_pipeline (function): The function to use for the honesty control pipeline.
        tokenizer (Tokenizer): The tokenizer to use for the generation.
        model (Model): The model to use for the generation.
        repetition_penalty (float, optional): The penalty to apply for repetition. Defaults to 1.1.
        only_return_first_sentence (bool, optional): Whether to return only the first sentence of the generated text. Defaults to True.
        THRESHOLD (float, optional): The threshold to use for the confidence_monitor of confidence monitoring. Defaults to 0.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 100.
        beam_size (int, optional): The size of the beam to use for the generation. Defaults to 1.
        collect_pairs (bool, optional): Whether to collect pairs of tokens and their corresponding words. Defaults to True.
        mode (str, optional): The mode to use for the generation. Defaults to "adaptive_retrieval".

    Returns:
        list: The pairs of tokens and their corresponding words, if collect_pairs is True.
        str: The first sentence of the generated text, if only_return_first_sentence is True. Otherwise, the entire generated text.
        bool: A flag indicating whether the generation should stop.
    """
    confidence_monitoring_reader_scores_dict = {}
    confidence_monitoring_reader_scores_mean_dict = {}
    activations = {}
    stop_gen_flag = False

    if coeff != 0:
        for layer in control_layer_id:
            activations[layer] = (
                torch.tensor(
                    coeff
                    * honesty_control_reader.directions[layer]
                    * honesty_control_reader.direction_signs[layer]
                )
                .to(model.device)
                .half()
            )
        control_outputs = honesty_control_pipeline(
            prompt,
            activations=activations,
            batch_size=4,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            num_beams=beam_size,
        )
        test_data = control_outputs[0]["generated_text"]

    else:
        with torch.no_grad():
            output = model.generate(
                **tokenizer(prompt, return_tensors="pt").to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=beam_size,
                repetition_penalty=repetition_penalty,
            )
        test_data = tokenizer.decode(output[0], skip_special_tokens=True)

    if only_return_first_sentence:
        if test_data == prompt or compare_strings_except_punctuation(
                test_data, prompt
        ):
            return None, None, True
        test_data, first_sentence, one_sent_flag = retain_first_sentence(
            test_data, prompt
        )
        if (
                one_sent_flag
        ):
            stop_gen_flag = True
    input_ids = tokenizer.tokenize(test_data)
    if collect_pairs:
        results = process_in_batches(
            test_data,
            confidence_monitoring_pipeline,
            input_ids,
            8192,
            confidence_monitoring_reader,
            hidden_layer_id,
        )

        confidence_scores = []
        confidence_scores_means = []
        for pos in range(len(results)):
            tmp_scores = []
            tmp_scores_all = []
            for layer in hidden_layer_id:
                tmp_scores_all.append(
                    results[pos][0][layer][0] *
                    confidence_monitoring_reader.direction_signs[layer][0]
                )
                if layer in monitoring_layer_id:
                    tmp_scores.append(
                        results[pos][0][layer][0]
                        * confidence_monitoring_reader.direction_signs[layer][0]
                    )
            confidence_scores.append(tmp_scores_all)
            confidence_scores_means.append(np.mean(tmp_scores))

        confidence_monitoring_reader_scores_dict["confidence"] = confidence_scores
        confidence_monitoring_reader_scores_mean_dict["confidence"] = confidence_scores_means
        pairs, text = confidence_monitor(
            tokenizer,
            input_ids,
            confidence_monitoring_reader_scores_mean_dict,
            THRESHOLD,
            start_answer_token=None,
            last_sentence=first_sentence,
        )
        torch.cuda.empty_cache()
    else:
        pairs = []
    if only_return_first_sentence:
        return pairs, first_sentence, stop_gen_flag
    else:
        return pairs, test_data
