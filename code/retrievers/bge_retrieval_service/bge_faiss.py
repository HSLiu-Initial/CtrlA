import os
import pickle
import time

import faiss
import numpy as np
from flask import Flask, jsonify, request

from bge_wrapper import BGE

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # set number of GPU you want to use


class Indexer():
    def __init__(self, path='wiki_passage_bge_encode.pkl'):
        self.passage, self.index = None, None
        self.build_index(path)
        self.model = BGE()

    def build_index(self, path):
        g = open(path, 'rb')
        passages, embs = pickle.load(g)
        emb_norm = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / emb_norm
        embs = np.asarray(embs, dtype=np.float32)
        index = faiss.IndexFlatIP(len(embs[0]))
        co = faiss.GpuMultipleClonerOptions()
        co.shard, co.useFloat16 = True, True
        resources = [faiss.StandardGpuResources() for i in range(4)]
        index = faiss.index_cpu_to_gpu_multiple(
            resources, index, co=co, gpus=[0, 1])
        index.add(embs)
        self.passages = passages
        self.index = index

    def retrieve(self, query, top_k=5, use_prefix=False):
        query_vec, query = self.model.embed_query(query, use_prefix)
        query_vec = np.asarray(query_vec, dtype=np.float32)
        scores, indices = self.index.search(query_vec, top_k)
        results = [self.passages[j] for j in indices[0]]
        return results, scores.tolist()[0], query


app = Flask(__name__)
index = Indexer('wiki_passage_bge_encode.pkl')


@app.route('/bge_wiki', methods=['POST'])
def bge_wiki():
    json_data = request.get_json(force=True)
    query = json_data.get('query')
    print(f'query received is : {query}')
    top_k = json_data.get('top_k')
    top_k = int(top_k) if top_k else 10
    use_prefix = json_data.get('use_prefix', False)
    start = time.time()
    results, scores, query_final = index.retrieve(
        query, top_k=top_k, use_prefix=use_prefix)
    end = time.time()
    return jsonify(
        {"response": results, "scores": scores, "time_elapsed": end - start, "query_used_for_search": query_final})


if __name__ == '__main__':
    HOST_IP, port = '', 1234
    app.run(host=HOST_IP, port=port, debug=False)
