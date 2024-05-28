import re
from langchain_community.utilities import GoogleSerperAPIWrapper
import logging
import os
import time
import torch

os.environ["SERPER_API_KEY"] = "12345"


def extract_query(s):
    """
    Extracts the query from a string.

    Args:
        s (str): The string to extract the query from.

    Returns:
        str: The extracted query.
    """
    match = re.search(r"(Query:|Generated Query:)(.*)", s)
    if match:
        return match.group(2).strip()
    return s


tvq_prompt_original = """[INST] Given a question and its corresponding temporary answer, your task is to generate a search query based on the answer. The goal is to generate a query that verifies the accuracy of the answer. Follow these guidelines:

1. Create a Relevant and Concise Search Query: The query should directly relate to the answer provided, aiming to verify its accuracy. It must not mirror the initial question but should aid in refining or better answering it.
2. Utilize Given Information: Only generate content related to the information provided. Avoid extrapolating beyond the given details.
3. Direct Utilization of Question: If the answer includes phrases like "I don't have information" or "is not specified," use the question itself for your search query, ignoring the provided answer.
4. Professional and Creative Approach: Maintain a professional tone while being creative in query formulation.
5. Short and Concise: The query should be short but need to be specific enough to promise search engine and dense retriever can find related knowledge.

Examples

Question: What is Franz Seitz Sr.'s occupation?
Answer: Franz Seitz Sr.'s occupation is not specified in the given content.
Generated Query: What was Franz Seitz Sr.'s profession?

Question: What is the generally accepted name for the Puritans who became the earliest settlers in the Plymouth colony in America, in 1620?
Answer: I'm sorry, but I don't have information about who Jim is or where he was born. Could you please provide more context or clarify your question?
Generated Query: What is the generally accepted name for the Puritans who became the earliest settlers in the Plymouth colony in America, in 1620?

Question: Who played the weasley brothers in harry potter?
Answer: Fred Weasley was played by James and Oliver Phelps.
Generated Query: Who portrayed Fred Weasley in the Harry Potter film series?

Question: Who was the ruler of France in 1830?
Answer: Until 2 August 1830, the ruler of France was King Charles X.
Generated Query: Who was the ruler of France until 2 August 1830?

Question: Who won the mayor race in st petersburg florida?
Answer: In the 2017 mayoral election held in St. Petersburg, Florida, Rick Kriseman won re-election with 57.3% of the vote.
Generated Query: Who won the mayor race in st petersburg florida in 2017?

Question: When did toronto host the mlb all-star game? 
Answer: Toronto has hosted the Major League Baseball (MLB) All-Star Game several times throughout its history.
Generated Query: What years did Toronto host the MLB All-Star Game?

Question: {Question}
Answer: {Answer}
Generated Query: [/INST]"""

RHM_prompt_original = """[INST] Given an original question and a reference query that may not fully align with the original's intent, your task is to create a better, short and concise query for search engine and dense retriever to answer the original question. This generated query should be a question that directly addresses the original question's main point, incorporating details and information from the reference query. Begin with an interrogative word. Aim to make the generated query more precise and relevant to the original question.

Examples

Original Question: Rita Coolidge sang the title song for which Bond film?
Reference Query: What James Bond film did Rita Coolidge contribute a song to?
Generated Query: In which James Bond movie is the title song sung by Rita Coolidge?

Original Question: Who was the lead singer of the band whose only UK chart topper was "So you win again"?
Reference Query: What band had "So you win again" as a UK chart topper?
Generated Query: Who was the lead vocalist for the band that achieved a UK number one hit with "So you win again"?

Original Question: What was Xanadu in the title of the film?
Reference Query: What genre does the film Xanadu belong to?
Generated Query: What is the significance or meaning of "Xanadu" in the film's title?

Original Question: How does caffeine affect the brain?
Reference Query: What are the effects of drinking coffee?
Generated Query: What specific effects does caffeine have on brain function?

Original Question: What is the oldest university in the United States?
Reference Query: List of universities in the United States by founding date.
Generated Query: Which university is recognized as the first or oldest institution of higher education established in the United States?

Original Question: Who wrote the novel "Moby-Dick"?
Reference Query: Information on the book Moby-Dick.
Generated Query: Who is the author of the novel "Moby-Dick"?

Original Question: {Question}
Reference Query: {Query}
Generated Query: [/INST]"""

tvq_prompt = """[INST] Given a question and its corresponding temporary answer, your task is to generate a search query based on the answer. The goal is to generate a query that verifies the accuracy of the answer. Follow these guidelines:

1. Create a Relevant and Concise Search Query: The query should directly relate to the answer provided, aiming to verify its accuracy. It must not mirror the initial question but should aid in refining or better answering it.
2. Utilize Given Information: Only generate content related to the information provided. Avoid extrapolating beyond the given details.
3. Direct Utilization of Question: If the answer includes phrases like "I don't have information" or "is not specified," use the question itself for your search query, ignoring the provided answer.
4. Professional and Creative Approach: Maintain a professional tone while being creative in query formulation.
5. Short and Concise: The query should be short but need to be specific enough to promise search engine and dense retriever can find related knowledge.

Examples

Question: What is Franz Seitz Sr.'s occupation?
Answer: Franz Seitz Sr.'s occupation is not specified in the given content.
Generated Query: What was Franz Seitz Sr.'s profession?

Question: What is the generally accepted name for the Puritans who became the earliest settlers in the Plymouth colony in America, in 1620?
Answer: I'm sorry, but I don't have information about who Jim is or where he was born. Could you please provide more context or clarify your question?
Generated Query: What is the generally accepted name for the Puritans who became the earliest settlers in the Plymouth colony in America, in 1620?

Question: Who played the weasley brothers in harry potter?
Answer: Fred Weasley was played by James and Oliver Phelps.
Generated Query: Who portrayed Fred Weasley in the Harry Potter film series?

Question: Who was the ruler of France in 1830?
Answer: Until 2 August 1830, the ruler of France was King Charles X.
Generated Query: Who was the ruler of France until 2 August 1830?

Question: Who won the mayor race in st petersburg florida?
Answer: In the 2017 mayoral election held in St. Petersburg, Florida, Rick Kriseman won re-election with 57.3% of the vote.
Generated Query: Who won the mayor race in st petersburg florida in 2017?

Question: When did toronto host the mlb all-star game? 
Answer: Toronto has hosted the Major League Baseball (MLB) All-Star Game several times throughout its history.
Generated Query: What years did Toronto host the MLB All-Star Game?

Question: {Question}
Answer: {Answer} 
Directly provide the Generated Query. [/INST]"""

RHM_prompt = """[INST] Given an original question and a reference query that may not fully align with the original's intent, your task is to create a better, short and concise query for search engine and dense retriever to answer the original question. This generated query should be a question that directly addresses the original question's main point, incorporating details and information from the reference query. Begin with an interrogative word. Aim to make the generated query more precise and relevant to the original question.

Examples

Original Question: Rita Coolidge sang the title song for which Bond film?
Reference Query: What James Bond film did Rita Coolidge contribute a song to?
Generated Query: In which James Bond movie is the title song sung by Rita Coolidge?

Original Question: Who was the lead singer of the band whose only UK chart topper was "So you win again"?
Reference Query: What band had "So you win again" as a UK chart topper?
Generated Query: Who was the lead vocalist for the band that achieved a UK number one hit with "So you win again"?

Original Question: What was Xanadu in the title of the film?
Reference Query: What genre does the film Xanadu belong to?
Generated Query: What is the significance or meaning of "Xanadu" in the film's title?

Original Question: How does caffeine affect the brain?
Reference Query: What are the effects of drinking coffee?
Generated Query: What specific effects does caffeine have on brain function?

Original Question: What is the oldest university in the United States?
Reference Query: List of universities in the United States by founding date.
Generated Query: Which university is recognized as the first or oldest institution of higher education established in the United States?

Original Question: Who wrote the novel "Moby-Dick"?
Reference Query: Information on the book Moby-Dick.
Generated Query: Who is the author of the novel "Moby-Dick"?

Original Question: {Question}
Reference Query: {Query} 
Directly provide the Generated Query. [/INST]"""

temp_str_3 = """[INST] Provide a better search query for web search engine to answer the given question.

Examples

Question: Ezzard Charles was a world champion in which sport? 
Query: Ezzard Charles world champion sport

Question: What is the correct name of laughing gas? 
Query: laughing gas name

Question: Rita Coolidge sang the title song for which Bond film?
Query: Rita Coolidge Bond film title song

Question: Who was the lead singer of the band whose only UK chart topper was "So you win again"?
Query: lead singer band 'So you win again' only UK chart topper

Question: How does caffeine affect the brain?
Query: effects of caffeine on brain function and neurotransmitters

Question: What is the oldest university in the United States?
Query: oldest university in the United States

Question: Who wrote the novel "Moby-Dick"?
Query: author of Moby-Dick novel

Question: {Question} 
Directly provide the Query. [/INST]"""


def tvq(question, answer, model, tokenizer, dataset):
    """
    Rewrites a query based on a given question and answer.

    Args:
        question (str): The original question.
        answer (str): The answer to the question.
        model (Model): The model to use for rewriting the query.
        tokenizer (Tokenizer): The tokenizer to use for rewriting the query.
        dataset (str): The dataset to use for rewriting the query.

    Returns:
        str: The rewritten query.
    """
    if dataset == "fact" or dataset == "fresh":
        prompt = tvq_prompt.format(
            Question=question,
            Answer=answer,
        )
        if dataset == "fact":
            beam_size = 1
        elif dataset == "fresh":
            beam_size = 5
        else:
            raise NotImplementedError
    elif dataset == "popqa" or dataset == "triviaqa" or dataset == "asqa":
        prompt = tvq_prompt_original.format(
            Question=question,
            Answer=answer,
        )
        beam_size = 5
    else:
        raise NotImplementedError
    with torch.no_grad():
        prompt_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **prompt_ids,
            max_new_tokens=64,
            do_sample=False,
            num_beams=beam_size,
            repetition_penalty=1,
        )
    pred = tokenizer.decode(
        output[0][len(prompt_ids["input_ids"][0]):], skip_special_tokens=True
    )
    if dataset == "fact" or dataset == "fresh":
        return extract_query(pred)
    else:
        return pred


def RHM(Question, Query, model, tokenizer, dataset):
    """
    Rewrites a query based on a given question and a reference query.

    Args:
        Question (str): The original question.
        Query (str): The reference query.
        model (Model): The model to use for rewriting the query.
        tokenizer (Tokenizer): The tokenizer to use for rewriting the query.
        dataset (str): The dataset to use for rewriting the query.

    Returns:
        str: The rewritten query.
    """
    if dataset == "fact" or dataset == "fresh":
        prompt = RHM_prompt.format(
            Question=Question,
            Query=Query,
        )
        beam_size = 1
    elif dataset == "popqa" or dataset == "triviaqa" or dataset == "asqa":
        prompt = RHM_prompt_original.format(
            Question=Question,
            Query=Query,
        )
        beam_size = 5
    else:
        raise NotImplementedError
    with torch.no_grad():
        prompt_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **prompt_ids,
            max_new_tokens=64,
            do_sample=False,
            num_beams=beam_size,
            repetition_penalty=1,
        )
    pred = tokenizer.decode(
        output[0][len(prompt_ids["input_ids"][0]):], skip_special_tokens=True
    )
    if dataset == "fact" or dataset == "fresh":
        return extract_query(pred)
    else:
        return pred


def rewrite_serper(Question, model, tokenizer):
    """
    Rewrites a query based on a given question.

    Args:
        Question (str): The original question.
        model (Model): The model to use for rewriting the query.
        tokenizer (Tokenizer): The tokenizer to use for rewriting the query.

    Returns:
        str: The rewritten query.
    """
    prompt = temp_str_3.format(
        Question=Question,
    )

    with torch.no_grad():
        prompt_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **prompt_ids,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1,
        )
    pred = tokenizer.decode(
        output[0][len(prompt_ids["input_ids"][0]):], skip_special_tokens=True
    )
    return extract_query(pred)


def format_list_as_numbered_string(lst):
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(lst))


def retrieve_dpr(query, nprobe, top_k):
    pass


def retrieve_bge(query, top_k, use_prefix=True):
    pass


def wiki_search(query, size=10):
    pass


def format_quotes(quotes):
    snippets_ = []
    quotes_ = []
    for q in quotes:
        snippet = q["snippet"].strip()
        quote = q["quote"].strip()
        snippets_.append(snippet)
        quotes_.append(quote)
    return quotes_, snippets_


def google_search(query, quote_num=5):
    pass


def serper(query, top_k):
    search = GoogleSerperAPIWrapper()
    results = search.results(query)
    organic_results = results["organic"]
    time.sleep(0.5)
    if len(organic_results) < top_k:
        result = [item["snippet"] for item in organic_results]
    else:
        result = [item["snippet"] for item in organic_results[:top_k]]
    return result


def retrieve(
        query,
        n_docs,
        retrieve_method,
        model=None,
        tokenizer=None,
        rewrite_serper=True,
):
    """
    Retrieves documents based on a given query.

    Args:
        query (str): The query to use for retrieving documents.
        n_docs (int): The number of documents to retrieve.
        retrieve_method (str): The method to use for retrieving documents.
        model (Model, optional): The model to use for retrieving documents. Defaults to None.
        tokenizer (Tokenizer, optional): The tokenizer to use for retrieving documents. Defaults to None.
        rewrite_serper (bool, optional): Whether to rewrite the query using the SERPER method. Defaults to True.

    Returns:
        str: The retrieved documents as a numbered string.
        bool: A flag indicating whether the retrieval was successful.
    """
    contents = []
    empty_flag = False

    try:
        if isinstance(query, list):
            query = [q.strip("\"'") for q in query]
        elif isinstance(query, str):
            query = query.strip("\"'")
        else:
            raise NotImplementedError
        if "bm25" in retrieve_method:
            if isinstance(query, list):
                quotes = wiki_search(query=query[0], size=4)
                if not quotes:
                    logging.error(f"bm25 return none. query: {query[0]}")
                    empty_flag = True
                else:
                    contents.extend(quotes)
                quotes = wiki_search(query=query[1], size=4)
                if not quotes:
                    logging.error(f"bm25 return none.query: {query[1]}")
                    empty_flag = True
                else:
                    contents.extend(quotes)
            elif isinstance(query, str):
                quotes = wiki_search(query=query, size=n_docs)
                if not quotes:
                    logging.error(f"bm25 return none.query: {query}")
                    empty_flag = True
                else:
                    contents.extend(quotes)
            else:
                raise NotImplementedError
        if "dpr" in retrieve_method:
            quotes = retrieve_dpr(query=query, nprobe=20000, top_k=n_docs)
            contents.extend(quotes["response"])
        if "bge" in retrieve_method:
            if isinstance(query, list):
                quotes = retrieve_bge(query=query[0], top_k=4)
                if not quotes:
                    logging.error(f"return none.query: {query[0]}")
                    empty_flag = True
                else:
                    contents.extend(quotes["response"])
                quotes = retrieve_bge(query=query[1], top_k=4)
                if not quotes:
                    logging.error(f"return none.query: {query[1]}")
                    empty_flag = True
                else:
                    contents.extend(quotes["response"])
            elif isinstance(query, str):
                quotes = retrieve_bge(query=query, top_k=n_docs)
                if not quotes:
                    logging.error(f"return none.query: {query}")
                    empty_flag = True
                else:
                    contents.extend(quotes["response"])
            else:
                raise NotImplementedError

        if "serper" in retrieve_method:
            if isinstance(query, list):
                if model:
                    query_1 = rewrite_serper(query[0], model, tokenizer)
                    query_2 = rewrite_serper(query[1], model, tokenizer)
                else:
                    query_1 = query[0]
                    query_2 = query[1]
                quotes = serper(query=query_1, top_k=3)
                if not quotes:
                    logging.error(f"SERPER return none. query: {query_1}")
                    empty_flag = True
                else:
                    contents.extend(quotes)
                quotes = serper(query=query_2, top_k=3)
                if not quotes:
                    logging.error(f"SERPER return none. query: {query_2}")
                    empty_flag = True
                else:
                    contents.extend(quotes)
            elif isinstance(query, str):
                if model:
                    query_1 = rewrite_serper(query, model, tokenizer)
                else:
                    query_1 = query
                quotes = serper(query=query_1, top_k=n_docs)
                if not quotes:
                    logging.error(f"SERPER return none. query: {query_1}")
                    empty_flag = True
                else:
                    contents.extend(quotes)
            else:
                raise NotImplementedError
    except:
        if isinstance(query, list):
            logging.error(f"Error query 1: {query[0]}")
            logging.error(f"Error query 2: {query[1]}")
            empty_flag = True
        elif isinstance(query, str):
            logging.error(f"Error query: {query[1]}")
            empty_flag = True
    unique_list = []
    [unique_list.append(x) for x in contents if x not in unique_list]
    return format_list_as_numbered_string(unique_list), empty_flag
