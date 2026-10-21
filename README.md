# 企业制度问答 RAG（policy-qa-rag）

> 一个**企业制度问答** RAG 系统：把制度文档（报销 / 假期 / 考勤 / 信息安全）向量化入库，员工用自然语言提问，
> 系统检索最相关条款并让 LLM 生成答案，**强制带引用溯源**，且**只依据库内条文回答、不编造**，答不出时明确说"制度中未找到相关条款"。
> 向量化使用**本地** `sentence-transformers`，制度文本不出本地环境，契合企业内部"数据不外流"的合规要求。

> 作者：[@MiyamotoXC](https://github.com/MiyamotoXC) ｜ 许可证：MIT ｜ 欢迎 Star / Fork 二次开发。

## 一、特性

- **本地向量化**：embedding 在本地完成，制度文本不上传第三方，保护内部制度隐私
- **引用溯源**：每个答案回显命中条款的文档名与相似度，便于人工核验
- **越界拒答**：LLM 被约束为"只依据检索片段作答"，避免编造未存在的条款
- **零密钥可跑检索**：未配置 LLM Key 时退化为"纯检索模式"，仍可验证召回效果

## 二、目录结构

```
policy-qa-rag/
├── config.py          # 配置（密钥走环境变量）
├── ingest.py          # 解析 .md -> 切分 -> 本地向量化 -> FAISS 落盘
├── qa.py              # 加载 -> 检索 -> LLM 生成（带引用）
├── example_docs/      # 脱敏模拟制度（报销 / 假期 / 考勤 / 信息安全）
├── requirements.txt
└── LICENSE
```

## 三、快速开始

```bash
pip install -r requirements.txt

# 1) 构建索引（首次会下载本地 embedding 模型）
python ingest.py

# 2) 提问
python qa.py "出差报销需要哪些材料？"
export DASHSCOPE_API_KEY="你的密钥"   # 配置后可获得 LLM 生成答案
python qa.py "年假最多能休几天？"
```

## 四、配置（config.py）

| 项 | 默认 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 环境变量（可选） | LLM 生成密钥；未设置则仅检索 |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | 本地句向量模型 |
| `LLM_MODEL` | `qwen-plus` | 生成答案的 LLM |
| `DATA_DIR` | `example_docs` | 制度文档目录 |
| `INDEX_DIR` | `vector_store` | 索引落盘目录（已被 .gitignore 忽略） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 80 | 切分参数 |

## 五、能力边界（诚实说明）

- 示例制度为**脱敏的通用模板**，仅作技术演示，不构成任何真实公司制度
- 检索质量上限取决于制度文档的覆盖度与切分粒度
- LLM 回答受"只依据检索片段"约束，但生成仍有概率性，关键条款以原文为准

## 六、License

代码以 MIT 发布。
