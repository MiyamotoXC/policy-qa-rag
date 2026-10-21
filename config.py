# -*- coding: utf-8 -*-
"""企业制度问答 RAG：统一配置（密钥走环境变量）。"""
import os

# LLM 生成所需密钥（可选：未设置时仅返回检索片段）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 本地向量化模型（sentence-transformers，无需联网调用，保护制度文本隐私）
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "qwen-plus"

DATA_DIR = "example_docs"
INDEX_DIR = "vector_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4
