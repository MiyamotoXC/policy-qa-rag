# -*- coding: utf-8 -*-
"""解析制度文档 -> 切分 -> 本地向量化 -> FAISS 落盘。"""
import os
import glob
import json

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config


def load_docs(data_dir=config.DATA_DIR):
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        title = os.path.basename(path)[:-3]
        docs.append((title, text))
    return docs


def split_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """按段落优先切分；超长段落用滑动窗口兜底。"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) <= size:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) > size:
                for i in range(0, len(p), size - overlap):
                    chunks.append(p[i : i + size])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks


def build():
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    records = []
    for title, text in load_docs():
        for c in split_text(text):
            records.append((title, c))

    texts = [r[1] for r in records]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(config.INDEX_DIR, "policy.index"))
    with open(os.path.join(config.INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"已索引 {len(records)} 个切片，存至 {config.INDEX_DIR}")


if __name__ == "__main__":
    build()
