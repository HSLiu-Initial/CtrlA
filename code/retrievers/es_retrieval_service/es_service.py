import json

from elasticsearch import Elasticsearch, helpers
from flask import Flask, jsonify, request
from tqdm import tqdm


# -*- coding: utf-8 -*-

def create_index(es, index_name):
    es.indices.delete(index=index_name)
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)
        print(f"Created index: {index_name}")
    else:
        print(f"Index {index_name} already exists")


def generate_documents(file_path):
    l = []
    with open(file_path, encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            d = {
                '_index': index_name,
                '_id': data.get('id', ''),
                '_source': {
                    'text': data.get('text', ''),
                    'title': data.get('title', '')
                }
            }
            yield d


def bulk_index_data(es, file_path, index_name):
    actions = generate_documents(file_path)
    progress = tqdm(actions, total=21015324)
    # progress = tqdm(actions, total=1116662)
    _, a = helpers.bulk(es, progress)
    print(f"Successfully indexed {a} documents.")


app = Flask(__name__)
es = Elasticsearch("http://localhost:9200")
index_name = 'wiki_2018'
file_path = 'text-list-wiki2018.json'
# file_path = 'sample.json'

create_index(es, index_name)
bulk_index_data(es, file_path, index_name)


@app.route('/bm25', methods=['POST'])
def bm25():
    json_data = request.get_json(force=True)
    query = json_data.get('query')
    size = 10
    if 'size' in json_data:
        size = json_data.get('size')
        # defaults to 10
        if size < 0:
            size = 10
    print(f"query received are :{query}")
    resp = es.search(
        index='wiki_2018',
        query={
            'match': {
                'text': {
                    'query': query
                }
            },
        },
        size=size
    )
    results = {
        'results_text': [r['_source']['text'] for r in resp['hits']['hits']],
        'results_title': [r['_source']['title'] for r in resp['hits']['hits']],
        'results_score': [r['_score'] for r in resp['hits']['hits']]
    }
    return jsonify(results)


if __name__ == '__main__':
    HOST_IP, port = '', 8888
    app.run(host=HOST_IP, port=port, debug=False)
