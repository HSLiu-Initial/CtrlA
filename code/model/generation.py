# -*- coding:utf-8 -*-

import datetime
import json
import logging

import numpy as np
import torch
from code.model.adaptive_retrieval import adaptive_retrieve


def generate(
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
    if mode == "adaptive_retrieval":
        post_process_pred, results, do_retrieve = adaptive_retrieve(
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
            THRESHOLD,
            mode,
            debug,
            dataset,
        )
    else:
        raise NotImplementedError
    if debug:
        print("post_process_pred:\n", post_process_pred)
    return post_process_pred, results, do_retrieve
