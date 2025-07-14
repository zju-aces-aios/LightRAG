import os
import asyncio
import logging
import logging.config
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.llm.openai import openai_embed, openai_complete_if_cache
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./dickens"

def configure_logging():
    """Configure logging for the application"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))
    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(log_dir, exist_ok=True)
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )
    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 推荐用环境变量方式设置你的 key 和 base_url
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "qwen-max-latest",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="text-embedding-v4",
        api_key=API_KEY,
        base_url=BASE_URL
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=embedding_func
        ),
        llm_model_func=llm_model_func,
        embedding_batch_num=10,
        vector_storage="LightFaissVectorDBStorage"

    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def main(logs=None):
    if logs is None:
        logs = []
    rag = None
    try:
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]
        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                logs.append(f"已删除旧文件: {file_path}")

        rag = await initialize_rag()

        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        logs.append("=======================")
        logs.append("Test embedding function")
        logs.append("=======================")
        logs.append(f"Test dict: {test_text}")
        logs.append(f"Detected embedding dimension: {embedding_dim}")
        logs.append(f"Embedding result: {embedding}")

        book_path = os.path.join(os.getcwd(), "book.txt")
        if not os.path.exists(book_path):
            logs.append(f"未找到 {book_path} 文件，请先准备好文本文件。")
            return "\n".join(logs)
        with open(book_path, "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # naive 查询
        logs.append("=====================")
        logs.append("Query mode: naive")
        logs.append("=====================")
        result_naive = await rag.aquery(
            "这个故事的主要主题是什么？", param=QueryParam(mode="naive")
        )
        logs.append(str(result_naive))

        # local 查询
        logs.append("=====================")
        logs.append("Query mode: local")
        logs.append("=====================")
        result_local = await rag.aquery(
            "这个故事的主要主题是什么？", param=QueryParam(mode="local")
        )
        logs.append(str(result_local))

        # global 查询
        logs.append("=====================")
        logs.append("Query mode: global")
        logs.append("=====================")
        result_global = await rag.aquery(
            "这个故事的主要主题是什么？",
            param=QueryParam(mode="global"),
        )
        logs.append(str(result_global))

        # hybrid 查询
        logs.append("=====================")
        logs.append("Query mode: hybrid")
        logs.append("=====================")
        result_hybrid = await rag.aquery(
            "这个故事的主要主题是什么？",
            param=QueryParam(mode="hybrid"),
        )
        logs.append(str(result_hybrid))

    except Exception as e:
        logs.append(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()
    return "\n".join(logs)

def run_main_sync():
    configure_logging()
    logs = []
    result = asyncio.run(main(logs))
    print(result)
    print("\nDone!")

if __name__ == "__main__":
    run_main_sync()