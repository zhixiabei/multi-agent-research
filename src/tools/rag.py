import os
import uuid
import yaml
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# ========== 固定写法：自动找到项目根目录的 config.yaml ==========
# 当前文件：src/tools/rag.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(BASE_DIR, "config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
# ==========================================================

PERSIST_DIR = cfg["chroma"]["persist_directory"]
COLLECTION_NAME = cfg["chroma"]["collection_name"]
CHUNK_SIZE = cfg["chroma"]["chunk_size"]
CHUNK_OVERLAP = cfg["chroma"]["chunk_overlap"]

# 初始化向量库与向量化模型
client = chromadb.PersistentClient(path=PERSIST_DIR)
embedding_func = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func
)

# 语义分块器：按段落 → 句子 → 词语 的优先级切割，保持语义完整
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", ".", "；", ";", " ", ""],
    keep_separator=True,
)


def split_long_text(text: str) -> list[str]:
    """语义分块：按段落/句子边界切割，保证每块语义完整"""
    if not text or len(text.strip()) == 0:
        return []
    return _text_splitter.split_text(text)


@tool
async def add_knowledge(text: str, doc_source: str = "user_upload") -> str:
    """
    将文本存入本地向量知识库，用于RAG检索
    Args:
        text: 待入库文本
        doc_source: 文档来源标记
    """
    text = text.strip()
    chunks = split_long_text(text)
    if not chunks:
        return "入库失败：无有效文本内容"

    ids = []
    documents = []
    metadatas = []
    for idx, chunk in enumerate(chunks):
        unique_id = f"{doc_source}_{uuid.uuid4()}_{idx}"
        ids.append(unique_id)
        documents.append(chunk)
        metadatas.append({"source": doc_source})

    try:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return f"入库成功，共分片 {len(chunks)} 条内容"
    except Exception as e:
        return f"入库异常：{str(e)}"


@tool
async def query_knowledge(query: str, top_k: int = 3) -> str:
    """
    检索本地知识库，返回相关内容片段
    Args:
        query: 查询问题
        top_k: 召回片段数量
    """
    query = query.strip()
    if not query:
        return "检索失败：查询内容不能为空"

    try:
        results = collection.query(query_texts=[query], n_results=top_k)
        doc_list = results.get("documents", [[]])[0]
        if not doc_list:
            return "知识库中未查询到相关内容"

        res_text = "【知识库相关内容】\n"
        for idx, content in enumerate(doc_list, 1):
            res_text += f"{idx}. {content}\n"
        return res_text
    except Exception as e:
        return f"检索异常：{str(e)}"


# 对外暴露工具集合
RAG_TOOLS = [add_knowledge, query_knowledge]