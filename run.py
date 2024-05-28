import argparse
import json
import logging
import pickle
from datetime import datetime

import numpy as np
import torch
from code.model.generation import generate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from code.dataset.utils import preprocess_input_data
from code.repe import repe_pipeline_registry
from code.utils import (
    ConfigParser,
    setup_seed,
    init_method,
    load_jsonlines, match,
)

# Register the pipeline
repe_pipeline_registry()
import os


def main(args=None):
    """
    Main function to run the model.

    Args:
        args: Command line arguments parsed via argparse.
    """
    # Set up the seed for reproducibility
    if args.seed != -1:
        try:
            setup_seed(args.seed + args.rank)
        except:
            setup_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your setup.")
    # Define the prefix for saving the results
    start_time_date = datetime.now().strftime("%m%d")
    start_time_time = datetime.now().strftime("%H_%M_%S")

    if args.prefix == "auto define":
        PREFIX = os.path.join(
            file_dir_path,
            args.out_path,
            "run",
            args.model,
            args.dataset,
            args.mode,
            start_time_date,
            start_time_time,
        )
    else:
        PREFIX = args.prefix
    img_dir, save_dir, model_save_dir, results_save_dir = init_method(PREFIX, args)
    logging.info("Prefix: " + PREFIX)
    # Load the model
    model = (
        AutoModelForCausalLM.from_pretrained(
            os.path.join(file_dir_path, "model", args.model_name_or_path),
            torch_dtype=torch.float16,
            device_map="auto",
            token=True,
        ).eval()

    )
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(file_dir_path, "model", args.model_name_or_path),
        padding_side="left",
        legacy=False,
        use_fast=True,
        token=True,
    )
    tokenizer.pad_token_id = 0
    # Define the layers for the model
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    layers = range(
        -int(args.begin_reading_layer), -int(args.end_reading_layer), -1
    )
    control_layers = list(
        range(
            -int(args.begin_control_layer),
            -int(args.end_control_layer),
            -int(args.control_step),
        )
    )
    # Setting up pipelines
    honesty_control_pipeline = pipeline(
        "rep-control",
        model=model,
        tokenizer=tokenizer,
        layers=control_layers,
        control_method=args.control_method,
    )
    confidence_monitoring_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    # Load readers
    confidence_monitoring_reader, honesty_control_reader = None, None
    try:
        with open(args.monitoring_reader_dir, "rb") as file:
            confidence_monitoring_reader = pickle.load(file)
        with open(args.control_reader_dir, "rb") as file:
            honesty_control_reader = pickle.load(file)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    # Define the input path based on the dataset
    if args.dataset == "popqa":
        input_path = os.path.join(
            file_dir_path, "eval_data", "popqa_longtail.jsonl"
        )
    elif args.dataset == "triviaqa":
        input_path = os.path.join(
            file_dir_path, "eval_data", "triviaqa_test.jsonl"
        )
    elif args.dataset == "asqa":
        input_path = os.path.join(
            file_dir_path, "eval_data", "asqa_eval.json"
        )
    elif args.dataset == "fact":
        input_path = os.path.join(
            file_dir_path, "eval_data", "bio_gen.jsonl"
        )
    elif args.dataset == "fresh":
        input_path = os.path.join(
            file_dir_path, "eval_data", "FreshQA_v04082024.jsonl"
        )
    else:
        raise NotImplementedError
    input_data = (
        json.load(open(input_path))
        if input_path.endswith(".json")
        else load_jsonlines(input_path)
    )
    if args.dataset == "fresh":
        input_data = [i for i in input_data if i['split'] == 'TEST']
    input_data = preprocess_input_data(
        input_data,
        task=args.task,
    )
    try:
        with open(
                os.path.join(args.continue_answer_result_dir, "results.json"), "r"
        ) as f:
            continue_answer_result = json.load(f)

            already_answer_id = [
                i["question_id"]
                for i in continue_answer_result
                if not i["retrieve_freq_results"]["google_not_retrieve"]
            ]
        logging.info("Continue Generate")
    except:
        continue_answer_result = None

    preds, prompts, metric_results, all_retrieve_freq_results = [], [], [], []
    count = 0

    tqdm_style = {
        "colour": "green",
        "ascii": False,
        "desc": "Processing",
    }
    try:
        with open(
                os.path.join(args.continue_answer_result_dir, "results.json"), "r"
        ) as f:
            continue_answer_result = json.load(f)
        if len(continue_answer_result) < len(input_data):
            results_list = continue_answer_result
            start_row = len(continue_answer_result)
            logging.info(f"start row: {start_row}")
    except:
        results_list = []
        start_row = args.start_row
        logging.info(f"start row: {start_row}")
    total_iterations = len(input_data)
    temp_metric_results = []
    google_not_retrieval_count = 0
    for i, row in tqdm(
            enumerate(input_data[start_row:], start=start_row),
            total=total_iterations - start_row,
            **tqdm_style,
    ):

        try:
            if (
                    args.continue_answer_result_dir
                    and row["question_id"] in already_answer_id
            ):
                logging.info(f"Already Answer id: {row['question_id']}. Skip!!")
                results_list.append(
                    continue_answer_result[already_answer_id.index(row["question_id"])]
                )
                assert (
                        continue_answer_result[already_answer_id.index(row["question_id"])][
                            "question_id"
                        ]
                        == row["question_id"]
                ), f"continue_answer_result question_id: {continue_answer_result[already_answer_id.index(row['question_id'])]['question_id']} != row id {row['question_id']}"
                continue
        except:
            pass
        if args.debug:
            logging.debug(f"Row: {row}")
        pred, retrieve_freq_results, do_retrieve = generate(
            model=model,
            tokenizer=tokenizer,
            n_docs=args.ndocs,
            row=row,
            max_new_tokens=args.max_new_tokens,
            beam_size=args.beam_size,
            confidence_monitoring_reader=confidence_monitoring_reader,
            honesty_control_reader=honesty_control_reader,
            coeff=args.coeff,
            hidden_layer_id=hidden_layers,
            control_layer_id=control_layers,
            monitoring_layer_id=layers,
            confidence_monitoring_pipeline=confidence_monitoring_pipeline,
            honesty_control_pipeline=honesty_control_pipeline,
            repetition_penalty=args.repetition_penalty,
            query_exclude_question=args.query_exclude_question,
            query_exclude_old_info=args.query_exclude_old_info,
            continue_gen_without_contents=args.continue_gen_without_contents,
            THRESHOLD=args.threshold,
            retrieve_method=args.retrieve_method,
            use_tvq=args.use_tvq,
            mode=args.mode,
            debug=args.debug,
            dataset=args.dataset,
            search_initial=args.search_initial
        )
        if retrieve_freq_results["google_not_retrieve"] is True:
            google_not_retrieval_count += 1
            logging.error(f"Question id: {row['question_id']}-Google return none. Skip!!")

        count += 1 if do_retrieve else 0
        if args.dataset == "triviaqa" or args.dataset == "popqa":
            if "answers" not in row and "answer" in row:
                row["answers"] = (
                    [row["answer"]] if type(row["answer"]) is str else row["answer"]
                )
            if args.metric == "match":
                metric_result = match(pred, row["answers"])
            else:
                raise NotImplementedError
            metric_results.append(metric_result)
            temp_metric_results.append(metric_result)
            row["output"] = pred
            row["retrieve_freq_results"] = retrieve_freq_results
            row["metric_result"] = metric_result
            row["metric_mean"] = np.mean(metric_results)
            results_list.append(row)
            if i % 20 == 0 and i != 0:
                logging.info(
                    f"Iteration {i} done! Average: {np.mean(metric_results)}, Average of last 20: {np.mean(temp_metric_results)}"
                )
                logging.info(f"Retrieval Frequencies: {count / len(results_list)}")
                temp_metric_results = []
                with open(os.path.join(results_save_dir, "results.json"), "w") as outfile:
                    json.dump(results_list, outfile)
        else:
            row["output"] = pred
            row["retrieve_freq_results"] = retrieve_freq_results
            results_list.append(row)

            if i % 20 == 0 and i != 0:
                logging.info(
                    f"Iteration {i} done!"
                )
                with open(os.path.join(results_save_dir, "results.json"), "w") as outfile:
                    json.dump(results_list, outfile)

    with open(os.path.join(results_save_dir, "results.json"), "w") as outfile:
        json.dump(results_list, outfile)
    if args.dataset == "triviaqa" or args.dataset == "popqa":
        metric_results_all = [row["metric_result"] for row in results_list]
        logging.info("Final result: {0}".format(np.mean(metric_results_all)))
    logging.info(f"google_not_retrieval_count: {google_not_retrieval_count}")
    logging.info("Done!")


if __name__ == "__main__":
    """
    Entry point of the script. Parses the command line arguments and calls the main function.
    """
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", type=str, help="Location of the config file")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Model name and parameter file location, default is Mistral-7B-Instruct-v0.1",
    )
    parser.add_argument("--p" "prefix", type=str, help="Prefix")
    parser.add_argument("--seed", type=int, help="Seed for random number generation")
    parser.add_argument("--model", type=str, help="Experiment name")
    parser.add_argument("--out_path", type=str, help="Path to save the output")
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument(
        "--continue_answer_result_dir",
        type=str,
        help="Path of the already generated answers if continuing generation",
    )
    parser.add_argument("--task", type=str, help="Task to perform")
    parser.add_argument("--max_new_tokens", type=int, help="Maximum number of new tokens to generate")
    parser.add_argument(
        "--ndocs", type=int, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="adaptive_retrieval",
        help="Mode.",
        choices=[
            "adaptive_retrieval",
        ],
    )
    parser.add_argument(
        "--retrieve_method",
        type=str,
        help="Retrieval method",
    )
    parser.add_argument(
        "--begin_reading_layer", type=int, help="The starting layer for reading, default is 5"
    )
    parser.add_argument(
        "--end_reading_layer",
        type=int,
        help="The ending layer for reading, default is 25 depending on the chosen reading",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold for reading, default is 0.0 depending on the chosen reading",
    )
    parser.add_argument("--repetition_penalty", type=float, help="Penalty for repetition")
    parser.add_argument(
        "--monitoring_reader_dir", type=str, help="Directory related to confidence monitoring"
    )
    parser.add_argument("--start_row", type=int, default=0, help="Starting row")
    # control专属
    parser.add_argument(
        "--control_reader_dir", type=str, help="Directory related to honesty control, default is"
    )
    parser.add_argument(
        "--begin_control_layer", type=int, help="The starting layer for honesty control, default is 5"
    )
    parser.add_argument(
        "--end_control_layer", type=int, help="The ending layer for honesty control, default is 18"
    )
    parser.add_argument("--control_step", type=int, help="Step, default is 1")
    parser.add_argument("--beam_size", type=int, help="Beam search size")
    parser.add_argument("--block_name", type=str, help="Default is decoder_block")
    parser.add_argument("--control_method", type=str, help="Reading vector")
    parser.add_argument("--coeff", type=float, help="Coefficient for honesty control")
    parser.add_argument(
        "--query_exclude_question",
        action="store_true",
        help="Whether the query includes the question or just that sentence, default includes the question",
    )
    parser.add_argument(
        "--continue_gen_without_contents",
        action="store_true",
        help="Whether to discard the contents after generating a sentence",
    )
    parser.add_argument(
        "--query_exclude_old_info",
        action="store_true",
        help="Whether to exclude old information from the query, default does not exclude",
    )
    parser.add_argument(
        "--use_tvq",
        action="store_true",
        help="Whether to rewrite the Query, default does not rewrite",
    )
    parser.add_argument(
        "--search_initial",
        action="store_true",
        help="Whether to use question as the initial query before generation.",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--metric", type=str, help="Metric to be used during evaluation"
    )
    opts, _ = parser.parse_known_args()
    file_path = os.path.abspath(__file__)
    file_dir_path = os.path.dirname(file_path)
    if opts.config is not None:
        with open(os.path.join(file_dir_path, opts.config)) as f:
            options = json.load(f)
            args = ConfigParser(options)
    else:
        with open(
                os.path.join(file_dir_path, "configs", "run.json"),
                "r",
                encoding="UTF-8",
        ) as f:
            options = json.load(f)
            args = ConfigParser(options)
    args.update(opts)
    main(args)
