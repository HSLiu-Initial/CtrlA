# -*- coding:utf-8 -*-
import datetime

import pytz

# Get the current date in the "America/Los_Angeles" timezone and format it as "Month Day, Year"
current_date = datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime(
    "%B %d, %Y"
)
# A string containing further instructions for the user
further_instruction = "Please focus on generating a detailed, thorough, and informative answer that directly " \
                      "addresses the question asked. Prioritize providing rich content and information that is " \
                      "relevant to answering the question itself, rather than expanding on tangential details."

# A dictionary containing different prompt formats
PROMPT_DICT = {
    "mistral_prompt": (
        "[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\nQuestion: {question} [/INST]"
    ),
    "mistral_prompt_with_prevgen": (
        "[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\nQuestion: {question} [/INST] {prev_gen}"
    ),
    "mistral_prompt_retrieval": (
        "[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\nContents:\n{contents}\n\nQuestion: {question} [/INST]"
    ),
    "mistral_prompt_retrieval_with_prevgen": (
        "[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\nContents:\n{contents}\n\nQuestion: {question} [/INST] {prev_gen}"
    ),
    "mistral_prompt_ada": (
            "[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\nQuestion: {question}\n"
            + further_instruction
            + " [/INST]"
    ),
    "mistral_prompt_with_prevgen_ada": (
            "[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\nQuestion: {question}\n"
            + further_instruction
            + " [/INST] {prev_gen}"
    ),
    "mistral_prompt_retrieval_ada": (
            "[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\nContents:\n{contents}\n\nQuestion: {question}\n"
            + further_instruction
            + " [/INST]"
    ),
    "mistral_prompt_retrieval_with_prevgen_ada": (
            "[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\nContents:\n{contents}\n\nQuestion: {question}\n"
            + further_instruction
            + " [/INST] {prev_gen}"
    ),
    "qd": [
        """{instruction}\n
Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali 

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball 

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Are follow up questions needed here: Yes. 
Follow up: Who is the director of Jaws? 
Intermediate Answer: The director of Jaws is Steven Spielberg. 
Follow up: Where is Steven Spielberg from? 
Intermediate Answer: The United States. 
Follow up: Who is the director of Casino Royale? 
Intermediate Answer: The director of Casino Royale is Martin Campbell. 
Follow up: Where is Martin Campbell from? 
Intermediate Answer: New Zealand. 
So the final answer is: No

Question: """,
        """
Are follow up questions needed here:""",
    ],
}

# A dictionary containing instructions for different tasks
TASK_INST = {
    "popqa": "You are a response generation assistant, designed to provide accurate and clear answers to questions "
             "based on the given content. Please complete the answer if the question is partially answered.",
    "triviaqa": "You are a response generation assistant, designed to provide accurate and clear answers to questions "
                "based on the given content. Please complete the answer if the question is partially answered.",
    "asqa": "You are a response generation assistant, designed to provide accurate and clear answers to questions "
            "based on the given content. The questions are ambiguous and have multiple correct answers, and in that "
            "case, you have to provide a long-form answer including all correct answers.",
    "fact": "You are a biography generation assistant, designed to generate accurate and concise biographies about a "
            "person based on the given content. Please complete the answer if the question is partially answered.",
    "fresh": "You are a response generation assistant, designed to provide accurate and clear answers to questions "
             f"based on the given content. Answer as concisely as possible. Knowledge cutoff: {current_date}. Today is {current_date} in Pacific Standard Time."
             f"The question is time-sensitive, please pay attention to identifying outdated information.",
}


def preprocess_input_data(
        dataset, task=None
):
    """
    Preprocesses the input data by adding an instruction to each item in the dataset.

    Args:
        dataset (list): The dataset to preprocess. Each item in the dataset is a dictionary.
        task (str, optional): The task to get instructions for. If the task is not in TASK_INST, no instruction is added.

    Returns:
        list: The preprocessed dataset. Each item in the dataset is a dictionary that has an additional "instruction" key.
    """
    new_data = []
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
    for item in dataset:
        item["instruction"] = instruction
        new_data.append(item)

    return new_data
