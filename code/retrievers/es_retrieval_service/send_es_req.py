import argparse
import json

import requests

ES_URL = ''


def rank(query, top_k=5):
    d = {
        'query': query,
        'size': top_k
    }
    args = {
        'url': ES_URL,
        'data': json.dumps(d)
    }
    x = requests.post(**args)
    return x.json()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query', type=str, required=True)
    parser.add_argument('-k', '--top_k', type=int, required=True)
    args = parser.parse_args()
    res = rank(args.query.strip(), args.top_k)
    print('scores for query:', args.query.strip())

    for p in range(5):
        print(
            f"Title : {str(res['results_title'][p])} | Text: {str(res['results_text'][p])} | Score : {str(res['results_score'][p])}")
        print()
