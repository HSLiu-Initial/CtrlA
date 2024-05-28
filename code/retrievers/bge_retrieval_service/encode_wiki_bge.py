import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BATCH_SIZE = 65536
MODEL_LOC = 'bge-english/'

if __name__ == "__main__":
    passages = []
    batched_passage_embs = []
    f = open('psgs_w100.tsv', 'r')

    for idx, t in tqdm(enumerate(f.readlines())):
        if idx == 0:
            continue
        idx, txt, _ = t.split('\t')
        passages.append(txt)

    model = SentenceTransformer(MODEL_LOC)

    # read the passages
    for i in tqdm(range(0, len(passages), BATCH_SIZE)):
        processes = model.start_multi_process_pool()
        embs = model.encode_multi_process(
            passages[i: i + BATCH_SIZE], processes)
        model.stop_multi_process_pool(processes)
        batched_passage_embs.append(embs)
        del embs

    # concatenate embeddings
    passage_embs = np.concatenate(batched_passage_embs, axis=0)

    # save to pickle
    g = open('wiki_passage_bge_encode.pkl', 'wb')
    pickle.dump((passages, passage_embs), g, 8)
    g.close()
