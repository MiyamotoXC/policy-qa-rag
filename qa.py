# -*- coding: utf-8 -*-
"""加载索引 -> 检索 -> LLM 生成（带引用溯源与越界拒答）。"""
import os
import sys
import json

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config


def load():
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    index = faiss.read_index(os.path.join(config.INDEX_DIR, "policy.index"))
    with open(os.path.join(config.INDEX_DIR, "meta.json"), encoding="utf-8") as f:
        records = json.load(f)
    return model, index, records


def retrieve(model, index, records, query, top_k=config.TOP_K):
    q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    scores, ids = index.search(np.asarray(q, dtype="float32"), top_k)
    return [(records[i], float(scores[0][j])) for j, i in enumerate(ids[0])]


def answer(query):
    model, index, records = load()
    hits = retrieve(model, index, records, query)

    context = "\n\n".join(f"[来源：{t}]\n{c}" for (t, c), _ in hits)
    prompt = (
        "你是企业制度问答助手。只能依据下方[来源]给出的制度条文回答，"
        "不得编造未提及的条款；若材料无法回答，明确说'制度中未找到相关条款'。\n\n"
        f"{context}\n\n问题：{query}"
    )

    if not config.DASHSCOPE_API_KEY:
        print("（未设置 DASHSCOPE_API_KEY，仅返回检索片段）\n")
        for (t, c), s in hits:
            print(f"[{t}] 相似度 {s:.3f}: {c[:160]}\n")
        return

    import dashscope
    from dashscope import Generation

    dashscope.api_key = config.DASHSCOPE_API_KEY
    resp = Generation.call(model=config.LLM_MODEL, prompt=prompt, temperature=0)
    print(resp.output.text if resp.output else "生成失败")
    print("\n引用来源：")
    for (t, c), s in hits:
        print(f" - {t}（相似度 {s:.3f}）")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "出差报销需要哪些材料？"
    answer(q)
