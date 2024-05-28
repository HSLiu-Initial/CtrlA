import argparse
import json

import requests

SERVICE_URL = ''


def retrieve_dpr(query, top_k, use_prefix):
    d = {
        'query': query,
        'top_k': top_k,
        'use_prefix': use_prefix,
    }
    args = {
        'url': SERVICE_URL,
        'data': json.dumps(d, ensure_ascii=False).encode('utf-8')
    }
    x = requests.post(**args)
    return x.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query', type=str, required=True)
    parser.add_argument('-k', '--top_k', type=int, default=10)
    parser.add_argument('-p', '--use_prefix',
                        action='store_true', default=False)
    args = parser.parse_args()
    res = retrieve_dpr(args.query, args.top_k, args.use_prefix)
    print(res)
