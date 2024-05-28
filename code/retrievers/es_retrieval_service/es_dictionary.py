# -*- coding:utf-8 -*-
import json
from tqdm import tqdm

f = open('psgs_w100.tsv', 'r')

passages = []
for idx, t in tqdm(enumerate(f.readlines())):
    if idx == 0:
        continue
    iddx, text, title = t.split('\t')
    d = {
        "id": iddx,
        "text": text,
        "section": "",
        "title": title
    }
    passages.append(d)

g = open('text-list-wiki2018.json', 'w')

for d in passages:
    g.write(json.dumps(d, ensure_ascii=False) + '\n')

g.close()
f.close()
