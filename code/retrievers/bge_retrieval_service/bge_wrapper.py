import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class BGE():
    def __init__(self, bServer=False):
        self.model_path = "bge-english"
        self.model = SentenceTransformer(self.model_path)
        self.use_gpu = True
        self.eval_batch_size = 256
        self.inference_bs = self.eval_batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")
        self.model.to(self.device)
        print("BGE initialized. Model loaded from {}".format(self.model_path))
        self.model.eval()

    def embed_query(self, query, include_prefix=False):
        prefix = ""
        if include_prefix:
            prefix = "Represent this sentence for searching relevant passages: "
        query = prefix + query
        with torch.no_grad():
            query_embedding = self.model.encode(
                [query], convert_to_tensor=True, normalize_embeddings=True)
        torch.cuda.empty_cache()
        return query_embedding.tolist(), query

    def embed_quotes(self, quotes):
        quote_embeddings = []
        with torch.no_grad():
            for i in range(0, len(quotes), self.inference_bs):
                batch_quotes = quotes[i:i + self.inference_bs]
                quote_emb = self.model.encode(
                    batch_quotes, normalize_embeddings=True)
                quote_embeddings.extend(quote_emb.tolist())
        return quote_embeddings

    def score(self, query, quotes):
        query_emb = np.asarray(self.embed_query(query)[0])
        quote_emb = np.asarray(self.embed_quotes(quotes))
        scores = np.dot(quote_emb, query_emb.T).tolist()[0]
        return scores

    def get_query_tok_len(self, query):
        return self._get_tok_len("Represent this sentence for searching relevant passages: " + query)

    def get_quotes_tok_len(self, quote):
        return self._get_tok_len(quote)

    def get_max_query_len(self):
        return self.model.get_max_seq_length()

    def get_max_quotes_len(self):
        return self.model.get_max_seq_length()

    def _get_tok_len(self, text_input):
        return self.model.first_module().tokenizer(
            text=text_input, truncation=False, max_length=False, 
            return_tensors="pt")["input_ids"].size()[-1]
