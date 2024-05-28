import datetime
import json
import logging

import numpy as np
import pytz
import torch
from code.dataset.utils import PROMPT_DICT
from code.model.utils import (
    process_in_batches,
    postprocess_answer_option_conditioned,
    format_list_as_numbered_string,
    retain_first_sentence,
    confidence_monitor,
    process_sentence,
    split_contractions,
    spacy_stopwords,
    remove_punctuation,
    sentence_matches_pattern,
)
from code.model.utils import result_given_prompt_with_control
from code.retrieval import retrieve, tvq, RHM

# Get the current date in the "America/Los_Angeles" timezone and format it as "Month Day, Year"
current_date = datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime(
    "%B %d, %Y"
)


def format_prompt(dataset, row, contents=None, prev_gen=None):
    """
    Formats the prompt based on the dataset, row, contents, and previous generation.

    Args:
        dataset (str): The dataset to use for formatting the prompt.
        row (dict): The row of data to use for formatting the prompt.
        contents (str, optional): The contents to include in the prompt. Defaults to None.
        prev_gen (str, optional): The previous generation to include in the prompt. Defaults to None.

    Raises:
        NotImplementedError: If the dataset is not recognized.

    Returns:
        str: The formatted prompt.
    """
    if contents is None:
        if prev_gen is None:
            if dataset == "asqa":
                prompt = PROMPT_DICT["mistral_prompt_ada"].format(
                    instruction=row["instruction"], question=row["question"]
                )
            elif (
                    dataset == "popqa"
                    or dataset == "triviaqa"
                    or dataset == "fact"
                    or dataset == "fresh"
            ):
                prompt = PROMPT_DICT["mistral_prompt"].format(
                    instruction=row["instruction"], question=row["question"]
                )
            else:
                raise NotImplementedError
        else:
            if dataset == "asqa":
                prompt = PROMPT_DICT["mistral_prompt_with_prevgen_ada"].format(
                    instruction=row["instruction"],
                    question=row["question"],
                    prev_gen=prev_gen,
                )
            elif (
                    dataset == "popqa"
                    or dataset == "triviaqa"
                    or dataset == "fact"
                    or dataset == "fresh"
            ):
                prompt = PROMPT_DICT["mistral_prompt_with_prevgen"].format(
                    instruction=row["instruction"],
                    question=row["question"],
                    prev_gen=prev_gen,
                )
            else:
                raise NotImplementedError
    else:
        if prev_gen is None:
            if dataset == "asqa":
                prompt = PROMPT_DICT["mistral_prompt_retrieval_ada"].format(
                    instruction=row["instruction"],
                    question=row["question"],
                    contents=contents,
                )
            elif (
                    dataset == "popqa"
                    or dataset == "triviaqa"
                    or dataset == "fact"
                    or dataset == "fresh"
            ):
                prompt = PROMPT_DICT["mistral_prompt_retrieval"].format(
                    instruction=row["instruction"],
                    question=row["question"],
                    contents=contents,
                )
            else:
                raise NotImplementedError
        else:
            if dataset == "asqa":
                prompt = PROMPT_DICT[
                    "mistral_prompt_retrieval_with_prevgen_ada"
                ].format(
                    instruction=row["instruction"],
                    question=row["question"],
                    prev_gen=prev_gen,
                    contents=contents,
                )
            elif (
                    dataset == "popqa"
                    or dataset == "triviaqa"
                    or dataset == "fact"
                    or dataset == "fresh"
            ):
                prompt = PROMPT_DICT[
                    "mistral_prompt_retrieval_with_prevgen"
                ].format(
                    instruction=row["instruction"],
                    question=row["question"],
                    prev_gen=prev_gen,
                    contents=contents,
                )
            else:
                raise NotImplementedError
        if dataset == "fresh":
            prompt = prompt.replace(
                "Contents:",
                f"As of today {
                    current_date}, the most up-to-date and information "
                f"regarding this question is as follows.\nContents:",
            )
    return prompt


def adaptive_retrieve(
        model,
        tokenizer,
        n_docs,
        row,
        max_new_tokens,
        beam_size,
        confidence_monitoring_reader,
        honesty_control_reader,
        coeff,
        hidden_layer_id,
        control_layer_id,
        monitoring_layer_id,
        confidence_monitoring_pipeline,
        honesty_control_pipeline,
        repetition_penalty,
        retrieve_method,
        use_tvq,
        query_exclude_question,
        query_exclude_old_info,
        continue_gen_without_contents,
        search_initial,
        THRESHOLD=0,
        mode="adaptive_retrieval",
        debug=False,
        dataset="popqa",
):
    """
        Performs adaptive retrieval based on the given parameters.

        Args:
            model (Model): The model to use for generation.
            tokenizer (Tokenizer): The tokenizer to use for generation.
            n_docs (int): The number of documents to retrieve.
            row (dict): The row of data.
            max_new_tokens (int): The maximum number of new tokens to generate.
            beam_size (int): The beam size to use for generation.
            confidence_monitoring_reader (Reader): The reader to use for generation.
            honesty_control_reader (Reader): The control reader to use for generation.
            coeff (float): The coefficient to use for honesty control.
            hidden_layer_id (int): The IDs of the hidden layer.
            control_layer_id (int): The IDs of the control layer to use for honesty control.
            monitoring_layer_id (int): The IDs of the reading layer to use for confidence monitoring.
            confidence_monitoring_pipeline (Pipeline): The reading pipeline to use for confidence reading.
            honesty_control_pipeline (Pipeline): The control pipeline to use for honesty control.
            repetition_penalty (float): The penalty to apply for repetition.
            retrieve_method (str): The method to use for retrieval.
            use_tvq (bool): Whether to rewrite the query.
            query_exclude_question (bool): Whether to exclude the question from the query.
            query_exclude_old_info (bool): Whether to exclude old information from the query.
            continue_gen_without_contents (bool): Whether to continue generation without contents.
            search_initial (bool): Whether to search initially.
            THRESHOLD (float, optional): The threshold to use for retrieval. Defaults to 0.
            mode (str, optional): The mode to use for retrieval. Defaults to "adaptive_retrieval".
            debug (bool, optional): Whether to print debug information. Defaults to False.
            dataset (str, optional): The dataset to use for retrieval. Defaults to "popqa".

        Returns:
            tuple: The post-processed prediction, results, and a boolean indicating whether retrieval was performed.
        """
    google_not_retrieve = False
    gen = []
    if dataset == "fresh" or dataset == "fact" and search_initial:
        query = row["question"]

        contents, empty_flag = retrieve(
            query, n_docs, retrieve_method
        )
        prompt = format_prompt(dataset, row, contents=contents, prev_gen=None)
    else:
        prompt = format_prompt(dataset, row, contents=None, prev_gen=None)
    if debug:
        print("adaptive_retrieval prompt: ", prompt)
    retrieve_count = 0
    retrieve_count_tol = 0
    total_tokens = 0
    sentence_count = 0
    repetiton_flag = False
    generation_details = []
    while total_tokens < max_new_tokens and retrieve_count_tol - retrieve_count <= 6:

        if debug:
            print("original prompt", prompt)
        pairs, text, stop_gen_flag = result_given_prompt_with_control(
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
            repetition_penalty=repetition_penalty,
            only_return_first_sentence=True,
            THRESHOLD=THRESHOLD,
            max_new_tokens=64,
            beam_size=beam_size,
            collect_pairs=True,
        )
        if pairs is None and text is None and stop_gen_flag is True:
            print("Return None!")
            break
        if gen:
            try:
                if text == gen[-1] or text == gen[-2]:
                    repetiton_flag = True
                    break
            except:
                if text == gen[-1]:
                    repetiton_flag = True
                    break

        if debug:
            print("text: ", text)
            print("stop_gen_flag: ", stop_gen_flag)
        stop = stop_gen_flag
        step_details = {
            "prompt": prompt,
            "generated_text": text,
            "stop_generation": stop,
            "low_score_tokens": [],
        }

        need_retrieval = False
        if sentence_matches_pattern(text):
            need_retrieval = True
        text_list = text.split()
        cleaned_instruction_list = process_sentence(row["question"])

        if gen:
            temp_gen_words = []
            for sentence in gen:
                temp_gen_words.extend(process_sentence(sentence))
            gen_words = set(temp_gen_words)
        for key, value in pairs.items():
            if value["score"] < 1 and "word_position" in value:
                cleaned_words = []
                for word, pos in value["word_position"]:
                    parts = split_contractions(word)
                    main_part = parts[0] if parts else cleaned_word
                    cleaned_words.append((remove_punctuation(main_part), pos))

                for idx, (cleaned_word, pos) in enumerate(cleaned_words):
                    main_part = cleaned_word.lower()
                    if gen:
                        if query_exclude_old_info:
                            step_details["low_score_tokens"].append(
                                {
                                    "token": value["token"],
                                    "score": value["score"],
                                    "word_position": value["word_position"],
                                }
                            )
                            text_list[pos] = " "
                            need_retrieval = True
                        else:
                            if (
                                    main_part not in cleaned_instruction_list
                                    and main_part not in spacy_stopwords
                                    and main_part not in gen_words
                            ):
                                step_details["low_score_tokens"].append(
                                    {
                                        "token": value["token"],
                                        "score": value["score"],
                                        "word_position": value["word_position"],
                                    }
                                )
                                text_list[pos] = " "
                                need_retrieval = True
                    else:
                        if query_exclude_old_info:
                            step_details["low_score_tokens"].append(
                                {
                                    "token": value["token"],
                                    "score": value["score"],
                                    "word_position": value["word_position"],
                                }
                            )
                            text_list[pos] = " "
                            need_retrieval = True
                        else:
                            if (
                                    main_part not in cleaned_instruction_list
                                    and main_part not in spacy_stopwords
                            ):
                                step_details["low_score_tokens"].append(
                                    {
                                        "token": value["token"],
                                        "score": value["score"],
                                        "word_position": value["word_position"],
                                    }
                                )
                                text_list[pos] = " "
                                need_retrieval = True

        if need_retrieval:

            retrieve_count += 1
            retrieve_count_tol += 1
            filtered_text_list = [item for item in text_list if item != ""]
            text_masked = " ".join(filtered_text_list)
            if query_exclude_question:
                query = text_masked
            else:
                query = row["question"] + " " + text_masked
            if use_tvq:
                if query_exclude_question:
                    query = tvq(
                        question=row["question"],
                        answer=text,
                        model=model,
                        tokenizer=tokenizer,
                        dataset=dataset,
                    )
                else:
                    query_temp = tvq(
                        question=row["question"],
                        answer=text,
                        model=model,
                        tokenizer=tokenizer,
                        dataset=dataset,
                    )

                    query = [row["question"], query_temp]
                if debug:
                    print("temp answer: ", text)
            if debug:
                print("query: ", query)

            contents, empty_flag = retrieve(
                query,
                n_docs,
                retrieve_method,
                model=model,
                tokenizer=tokenizer,
            )
            if empty_flag is True:
                logging.error(f"Question id: {
                              row['question_id']}-Google return none")
                google_not_retrieve = True

            step_details["retrieval_needed"] = True
            step_details["retrieval_query"] = query
            step_details["retrieval_content"] = contents
            if gen:
                prompt = format_prompt(
                    dataset,
                    row,
                    contents=contents,
                    prev_gen=" ".join(gen),
                )
                if debug:
                    print("prev_gen: \n", " ".join(gen))
            else:
                prompt = format_prompt(
                    dataset, row, contents=contents,  prev_gen=None
                )

                if debug:
                    print("prev_gen: \n", " ".join(gen))

            _, text, stop_gen_flag = result_given_prompt_with_control(
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
                only_return_first_sentence=True,
                THRESHOLD=THRESHOLD,
                max_new_tokens=64,
                collect_pairs=False,
            )
            if text is None and stop_gen_flag is True:
                break
            if gen:
                try:
                    if text == gen[-1] or text == gen[-2]:
                        repetiton_flag = True
                        break
                except:
                    if text == gen[-1]:
                        repetiton_flag = True
                        break
            if debug:
                print("new prompt: ", prompt)
                print("new text: ", text)
                print("new stop_gen_flag: ", stop_gen_flag)
            stop = stop_gen_flag
            step_details["stop_generation"] = stop
            if sentence_matches_pattern(text):  #
                if retrieve_count_tol - retrieve_count == 5:

                    try:
                        contents, empty_flag = retrieve(
                            row["question"],
                            n_docs,
                            retrieve_method,
                            model=model,
                            tokenizer=tokenizer,
                        )
                    except:
                        contents, empty_flag = retrieve(
                            row["question"],
                            n_docs,
                            retrieve_method,
                            model=model,
                            tokenizer=tokenizer,
                        )
                    if empty_flag is True:
                        logging.error(f"Question id: {
                                      row['question_id']}-Google return none!")
                        google_not_retrieve = True
                else:
                    if isinstance(query, list):
                        query_now = []
                        for q in query:
                            temp = RHM(
                                Question=row["question"],
                                Query=q,
                                model=model,
                                tokenizer=tokenizer,
                                dataset=dataset,
                            )
                            query_now.append(temp)
                    elif isinstance(query, str):
                        query_now = RHM(
                            Question=row["question"],
                            Query=query,
                            model=model,
                            tokenizer=tokenizer,
                            dataset=dataset,
                        )
                    else:
                        raise NotImplementedError

                    contents, empty_flag = retrieve(
                        query_now,
                        n_docs,
                        retrieve_method,
                        model=model,
                        tokenizer=tokenizer,
                    )
                    if empty_flag is True:
                        logging.error(f"Question id: {
                                      row['question_id']}-Google return none")
                        google_not_retrieve = True
                retrieve_count_tol += 1
            else:
                gen.append(text)
                if continue_gen_without_contents:
                    if gen:
                        prompt = format_prompt(
                            dataset,
                            row,
                            contents=None,
                            prev_gen=" ".join(gen),
                        )

                        if debug:
                            print("prev_gen: \n", " ".join(gen))
                    else:
                        prompt = format_prompt(
                            dataset,
                            row,
                            contents=None,
                            prev_gen=None,
                        )

                        if debug:
                            print("prev_gen: \n", " ".join(gen))
                    sentence_count += 2
                    step_details["actual prompt"] = prompt
                    step_details["generated_text_actual_keep"] = text
                    continue

            if gen:
                prompt = format_prompt(
                    dataset,
                    row,
                    contents=contents,
                    prev_gen=" ".join(gen),
                )

            else:
                prompt = format_prompt(
                    dataset, row, contents=contents, prev_gen=None
                )

            step_details["actual prompt"] = prompt
            step_details["generated_text_actual_keep"] = text

        else:
            if debug:
                print("no retrieval")
                print("no retrieval stop gen flag:", stop)
            step_details["retrieval_needed"] = False
            gen.append(text)
            sentence_count += 1

            if continue_gen_without_contents:
                if gen:
                    prompt = format_prompt(
                        dataset,
                        row,
                        contents=None,
                        prev_gen=" ".join(gen),
                    )

                else:
                    prompt = format_prompt(
                        dataset, row, contents=None, prev_gen=None
                    )

            else:

                prompt = prompt + f" {text}"

        total_tokens = len(tokenizer.tokenize(" ".join(gen)))
        generation_details.append(step_details)
        if debug:
            print("total_tokens: ", total_tokens)
    if gen == []:
        query = row["question"]
        contents, empty_flag = retrieve(
            query,
            n_docs,
            retrieve_method,
            model=model,
            tokenizer=tokenizer,

        )

        if empty_flag is True:
            logging.error(f"Question id: {
                          row['question_id']}-Google return none")
            google_not_retrieve = True
        prompt = format_prompt(
            dataset, row, contents=contents, prev_gen=None
        )

        with torch.no_grad():
            prompt_ids = tokenizer(
                prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                **prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=beam_size,
                repetition_penalty=repetition_penalty,
            )
        pred = tokenizer.decode(
            output[0][len(prompt_ids["input_ids"][0])
                          :], skip_special_tokens=True
        )
        gen.append(pred)
        prompt = format_prompt(dataset, row, contents=None, prev_gen=None)

        if coeff == 0:
            with torch.no_grad():
                prompt_ids = tokenizer(
                    prompt, return_tensors="pt").to(model.device)
                output = model.generate(
                    **prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=beam_size,
                    repetition_penalty=repetition_penalty,
                )
            pred = tokenizer.decode(
                output[0][len(prompt_ids["input_ids"][0])
                              :], skip_special_tokens=True
            )
        else:
            activations = {}
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
            pred = control_outputs[0]["generated_text"].replace(prompt, "")
        gen.append(pred)

    pred = " ".join(gen)
    keep_sentence_count = len(gen)
    retrieval_frequency_gen = (
        retrieve_count / sentence_count if sentence_count > 0 else 0
    )
    retrieval_frequency_keep = (
        retrieve_count / keep_sentence_count if keep_sentence_count > 0 else 0
    )
    if retrieval_frequency_gen > 1:
        retrieval_frequency_gen = 1
    if retrieval_frequency_keep > 1:
        retrieval_frequency_keep = 1

    results = {
        "retrieve_count": retrieve_count,
        "retrieve_count_tol": retrieve_count_tol,
        "retrieval_frequency": retrieval_frequency_gen,
        "retrieval_frequency_keep": retrieval_frequency_keep,
        "generation_details": generation_details,
        "google_not_retrieve": google_not_retrieve,
        "repetition_flag": repetiton_flag,
    }
    do_retrieve = retrieve_count > 0
    post_process_pred = postprocess_answer_option_conditioned(pred)
    return post_process_pred, results, do_retrieve
