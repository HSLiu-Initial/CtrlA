# CtrlA: Adaptive Retrieval-Augmented Generation via Probe-Guided Control

The original implementation of **CtrlA: Adaptive Retrieval-Augmented Generation via Probe-Guided Control**.

![ctrla](assets/framework.png)

This work introduces an effective probe-guided adaptive RAG framework, termed CtrlA, to enhance retrieval-augmented generation for LLM, balancing its internal and external knowledge. CtrlA characterize LLM’s internal states and intervene in the LLM generation from two perspectives: honesty control and confidence monitoring via simple yet effective probes.

## Installation

Install dependenices by running the command below.
```
pip install -r requirements.txt
```

## Datasets and model

The dataset used for training the Confidence and Honesty Probes, as well as for our evaluation, is available [here](https://drive.google.com/drive/folders/1DlIDkYvo1C_d5Nb8j589Jv7Hhe5Guk9T?usp=sharing). Please create a `eval_data/` directory and place all the data files within it.

Please download the model file from [mistralai/Mistral-7B-Instruct-v0.1 on Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) and place it in the `model/` directory.

## Confidence and Honesty Probe

The pre-trained probes are stored in the `trained_probe/` directory.

To train the probes, refer to the `train_confidence_probe.ipynb` notebook for the confidence probe, and the `train_honesty_probe.ipynb` notebook for the honesty probe.

## Retriever Setup

All the code related to the retriever setup is in the `code/retrievers` directory. We provide two retrieval services
as reported in our paper:

1. **BM25** Retrieval Service using ElasticSearch
2. **BGE** Retrieval Service using FAISS

### Downloads

1. Wikipedia 2018 Snippets: `wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz`
2. BGE Embedding Model Weights: `https://huggingface.co/BAAI/bge-large-en-v1.5`

### Retriever Dependencies

- FAISS : `https://github.com/facebookresearch/faiss` or `https://pypi.org/project/faiss/`
- SentenceTransformers: `https://github.com/UKPLab/sentence-transformers`
- Flask
- PyTorch
- ElasticSearch

### Quick Start to set up **BGE** Retrieval Service

```bash
cd code/retrievers/bge_retrieval_service  # go to the target directory
python encode_wiki_bge.py  # encode snippets into embeddings
python bge_faiss.py  # set up bge-retrieval service
```

The sample code to call bge-retrieval service: 
```bash
python send_req_bge_wiki.py -q <query> -k <stop_k> --use_prefix
```
`--use_prefix` is optional, which appends the prefix `Represent this sentence for searching relevant passages:` in front of queries for asymmetric encoding of queries and passages

### Quick Start to set up ES (Elasticsearch) Retrieval Service (**BM25**)
```bash
cd code/retrievers/es_retrieval_service  # go to the target directory
python es_dictionary.py  # convert passages in tsv to desired dictionary format.
python es_service.py  # set up Elasticsearch Retrieval Service
```

The sample code to call es-retrieval service: 
```bash
python send_es_req.py -q <query> -k <stop_k>
```

After deploying the retrieval service, please complete the corresponding retrieval functions in `code/retrieval.py`.

## Evaluation

All the commands can be found in `./run.sh`

### TriviaQA
```bash
python run.py --config configs/run.json --model run_short_form --dataset triviaqa --task triviaqa --max_new_tokens 1024 --retrieve_method bge_serper --metric match --use_tvq
```

### PopQA
```bash
python run.py --config configs/run.json --model run_short_form --dataset popqa --task popqa --max_new_tokens 1024 --retrieve_method bge_serper --metric match --use_tvq --continue_gen_without_contents
```

### ASQA
```bash
python run.py --config configs/run.json --model run_long_form --dataset asqa --task asqa --max_new_tokens 130 --retrieve_method bge --use_tvq
```
[ALCE/ASQA](https://github.com/princeton-nlp/ALCE) offers a thorough evaluation of long-form QA using various metrics. To conduct your initial evaluation, install the ALCE repository and download the necessary data.
```bash
git clone https://github.com/princeton-nlp/ALCE.git
python3 -m alce_env
cd ALCE
bash download_data.sh
```

### Bio Generation
```bash
python run.py --config configs/run.json --model run_long_form --dataset fact --task fact --max_new_tokens 300 --retrieve_method bge_serper --use_tvq
```
Please follow the instructions in the [FactScore](https://github.com/shmsw25/FActScore) official repository to set up your environment. Since the original repository is no longer maintained, consider using alternative sources like [wj210's fork](https://github.com/wj210/factscore) or [armingh2000's FactScoreLite](https://github.com/armingh2000/FactScoreLite) for evaluations. To proceed, use the command below:
```bash
python -m factscore.factscorer --data_path <output_file>  --model_name retrieval+ChatGPT --cache_dir <cache_dir> --openai_key <openai_key> --verbose
```

### FreshQA
```bash
python run.py --config configs/run.json --model run_long_form --dataset fresh --task fresh --max_new_tokens 1024 --retrieve_method serper --use_tvq
```
Please follow the instructions provided in the [freshllms/freshqa](https://github.com/freshllms/freshqa) repository, which includes complete data and codes of [FreshLLMs](https://arxiv.org/abs/2310.03214), to conduct your evaluation.

## Citation
If this work is helpful for you, please kindly cite it as follows:
```bibtex
TBA
```
