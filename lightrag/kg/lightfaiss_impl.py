import os
import time
import numpy as np
import asyncio
from typing import Any, final
import json

from dataclasses import dataclass
import pipmaster as pm

from lightrag.utils import logger, compute_mdhash_id
from lightrag.base  import BaseVectorStorage

from .shared_storage import (
    get_storage_lock,
    get_update_flag,
    set_all_update_flags,
)

import lightrag.kg.edgevecdb.edgevecdb_core as lf

@final
@dataclass
class LightFaissVectorDBStorage(BaseVectorStorage):
    """
    A LightFaissVectorDBStorage class that implements a vector storage
    Uses cosine similarity for vector comparisons
    """

    def __post_init__(self):
        # 获取全局变量
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # 生成对应文件路径
        self._lightfaiss_index_file = os.path.join(
            self.global_config["working_dir"], f"lightfaiss_index_{self.namespace}.index"
        )
        self._meta_file = self._lightfaiss_index_file + ".meta.json"

        self._max_batch_size = self.global_config["embedding_batch_num"]
        # 向量维度
        self._dim = self.embedding_func.embedding_dim

        # 初始化一个使用内积的向量索引
        # self._mgr = kp.Maneger()
        self._index = lf.FlatIndex(self._dim, None, lf.MetricType.METRIC_INNER_PRODUCT)
        self._id_to_meta = {}

        self._load_lightfaiss_index()

    async def initialize(self):
        """ 初始化storage数据 """
        self.storage_updated = await get_update_flag(self.namespace)
        self._storage_lock = get_storage_lock()
    
    async def _get_index(self):
        """检查数据库是否需要更新"""
        async with self._storage_lock:
            # 检查数据库是否被别的线程更新
            if self.storage_updated:
                logger.info(
                    f"Process {os.getpid()} LIGHTFAISS reloading {self.namespace} due to update by another process."
                )
                # 重新载入数据
                self._index = lf.FlatIndex(self._dim, None, lf.MetricType.METRIC_INNER_PRODUCT)
                self._id_to_meta = {}
                self._load_lightfaiss_index()
                self.storage_updated.value = False
            return self._index

    async def upsert(self, data: dict[str, dict[str, Any]]):
        """
        Insert or update vectors in the LightFaiss index.

        data: {
           "custom_id_1": {
               "content": <text>,
               ...metadata...
           },
           "custom_id_2": {
               "content": <text>,
               ...metadata...
           },
           ...
        }
        """
        logger.debug(f"LIGHTFAISS: Inserting {len(data)} to {self.namespace}")
        if not data:
            return
        
        current_time = int(time.time())

        # 准备向量（从文本变为向量数据
        list_data = []
        contents = []
        for k, v in data.items():
            # 存储知道的meta数据
            meta = {mf: v[mf] for mf in self.meta_fields if mf in v}
            meta["__id__"] = k
            meta["__created_at__"] = current_time
            list_data.append(meta)
            contents.append(v["content"])

        # 按照batches分批处理数据
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embedding_list = await asyncio.gather(*embedding_tasks)

        # 把数据摊平
        embeddings = np.concatenate(embedding_list, axis=0)
        if len(embeddings) != len(list_data):
            raise ValueError(
                f"Embedding length mismatch: {len(embeddings)} vs {len(list_data)}"
            )
            return []
        
        # 数据转换为float32类型，并进行L2 norm方便后续进行cosine相似度计算
        embeddings = embeddings.astype(np.float32)
        # TODO: embeddings的形状到底是什么样子的？运行时确认一下
        lf.normalized_L2_cpu(embeddings, embeddings.shape[0], embeddings.shape[1])

        # 插入逻辑
        # 1. 检查哪些向量已经存在
        # 2. 移除已经存在的向量
        existing_ids_to_remove = []
        for meta, emb in zip(list_data, embeddings):
            lightfaiss_internal_id = self._find_lightfaiss_id_by_custom_id(meta["__id__"])
            if lightfaiss_internal_id is not None:
                existing_ids_to_remove.append(lightfaiss_internal_id)

        if existing_ids_to_remove:
            await self._remove_lightfaiss_ids(existing_ids_to_remove)
        
        # 3. 插入新的向量
        index = await self._get_index()
        start_idx = index.get_num()
        index.add_vectors(embeddings)

        # 4. 重新储存现有的向量和meta数据
        for i, meta in enumerate(list_data):
            fid = start_idx + i
            meta["__vector__"] = embeddings[i].tolist()  # 转换为列表以便JSON序列化
            self._id_to_meta.update({fid: meta})

        logger.info(f"Upserted {len(list_data)} vectors to LightFaiss index {self.namespace}")

        return [m["__id__"] for m in list_data]

    async def query(
            self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        # 传入string，进行单个查询
        embedding = await self.embedding_func( 
            [query], _priority=5
        )
        embedding = np.array(embedding, dtype=np.float32)
        # TODO: embedding的形状到底是什么样子的？运行时确认一下
        lf.normalized_L2_cpu(embedding, embedding.shape[0], embedding.shape[1])

        logger.info(
            f"Query: {query}, top_k: {top_k}, threshold: {self.cosine_better_than_threshold}"
        )

        # 执行查询
        index = await self._get_index()
        # TODO: 返回值的形状怎么样？
        indices, distances = index.search(embedding, top_k)

        distances = distances[0]
        indices = indices[0]

        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                # TODO: search的返回结果如果是空，应该表达成-1
                continue

            if dist < self.cosine_better_than_threshold:
                # 如果距离小于阈值，则认为没有足够相似的向量
                continue

            meta = self._id_to_meta.get(idx, {})
            results.append(
                {
                    **meta,
                    "id": meta.get("__id__"),
                    "distance": float(dist),
                    "created_at": meta.get("__created_at__"),
                }
            )
        
        return results

    @property
    def client_storage(self):
        return {"data": list(self._id_to_meta.values())}

    async def delete(self, ids: list[str]):
        """
        因为FlatIndex不提供删除方法，所以删除时新建一个FlatIndex
        """
        logger.info(f"Deleting {len(ids)} from LightFaiss index {self.namespace}")
        to_remove = []
        for cid in ids:
            fid = self._find_lightfaiss_id_by_custom_id(cid)
            if fid is not None:
                to_remove.append(fid)
        
        if to_remove:
            await self._remove_lightfaiss_ids(to_remove)
        
        logger.debug(
            f"Successfully deleted {len(to_remove)} vectors from {self.namespace}"
        )

    async def delete_entity(self, entity_name: str) -> None:
        """
        1. 修改并不会直接写入磁盘，二十调用index_do_callback之后才会写入
        2. 在 index_done_callback 被调用之前，同一时刻只能有一个进程更新存储，
           应该使用 KG-storage-log 来避免数据损坏（data corruption）
        """
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        logger.debug(f"Attempting to delete entity {entity_name} with ID {entity_id} from LightFaiss index {self.namespace}")
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        """
        注意事项同上
        """
        logger.debug(f"Searching relations for entity {entity_name} in LightFaiss index {self.namespace}")
        relations = []
        for fid, meta in self._id_to_meta.items():
            if meta.get("src_id") == entity_name or meta.get("tgt_id") == entity_name:
                relations.append(fid)
        
        logger.debug(f"Found {len(relations)} relations for entity {entity_name} in LightFaiss index {self.namespace}")
        if relations:
            await self._remove_lightfaiss_ids(relations)
            logger.debug(f"Deleted {len(relations)} relations for entity {entity_name} in LightFaiss index {self.namespace}")

    # --------------------------------------------------------
    # Internal methods for LightFaiss
    # ---------------------------------------------------------

    def _find_lightfaiss_id_by_custom_id(self, custom_id: str):
        for fid, meta in self._id_to_meta.items():
            if meta.get("__id__") == custom_id:
                return fid
        return None
    
    async def _remove_lightfaiss_ids(self, fid_list):
        """
        因为FlatIndex不支持remove方法，此处新建一个Index
        """
        keep_fids = [fid for fid in self._id_to_meta if fid not in fid_list]

        # 重建
        vectors_to_keep = []
        new_id_to_meta = {}
        for new_fid, old_fid in enumerate(keep_fids):
            vec_meta = self._id_to_meta[old_fid]
            vectors_to_keep.append(vec_meta["__vector__"])
            new_id_to_meta[new_fid] = vec_meta
        
        async with self._storage_lock:
            self._index = lf.FlatIndex(self._dim, None, lf.MetricType.METRIC_INNER_PRODUCT)
            if vectors_to_keep:
                array = np.array(vectors_to_keep, dtype=np.float32)
                self._index.add_vectors(array)
            
            self._id_to_meta = new_id_to_meta

    def _save_lightfaiss_index(self):
        """
        保存当前的lightFaiss和meta data到磁盘
        """
        self._index.save(self._lightfaiss_index_file)

        # Save metadata dict to JSON. Convert all keys to strings for JSON storage.
        # _id_to_meta is { int: { '__id__': doc_id, '__vector__': [float,...], ... } }
        # We'll keep the int -> dict, but JSON requires string keys.
        serializable_dict = {}
        for fid, meta in self._id_to_meta.items():
            # Convert int fid to string for JSON serialization
            serializable_dict[str(fid)] = meta
        
        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(serializable_dict, f)

    def _load_lightfaiss_index(self):
        if not os.path.exists(self._lightfaiss_index_file):
            logger.warning("No existing LightFaiss index file found. Starting fresh.")
            return

        try:
            self._index.load(self._lightfaiss_index_file)
            with open(self._meta_file, "r", encoding="utf-8") as f:
                stored_dict = json.load(f)
            
            self._id_to_meta = {}
            for fid_str, meta in stored_dict.items():
                fid = int(fid_str)  # Convert string key back to int
                self._id_to_meta[fid] = meta
            
            logger.info(
                f"LightFaiss index loaded with {self._index.get_num()} vectors from {self._lightfaiss_index_file}"
            )
        except Exception as e:
            logger.error(f"Failed to load LightFaiss index or metadata: {e}")
            logger.warning("Starting with an empty LightFaiss index.")
            self._index = lf.FlatIndex(self._dim, None, lf.MetricType.METRIC_INNER_PRODUCT)
            self._id_to_meta = {}

    async def index_done_callback(self):
        async with self._storage_lock:
            if self.storage_updated.value:
                logger.warning(
                    f"Storage for LIGHTFAISS {self.namespace} was updated by another process, reloading..."
                )
                self._index = lf.FlatIndex(self._dim, None, lf.MetricType.METRIC_INNER_PRODUCT)
                self._id_to_meta = {}
                self._load_lightfaiss_index()
                self.storage_updated.value = False
                return False
            
        async with self._storage_lock:
            try:
                # 保存到磁盘
                self._save_lightfaiss_index()
                await set_all_update_flags(self.namespace)
                self.storage_updated.value = False
            except Exception as e:
                logger.error(f"Failed to save LightFaiss index for {self.namespace}: {e}")
                return False
            
        return True

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        fid = self._find_lightfaiss_id_by_custom_id(id)
        if fid is None:
            return None
        
        metadata = self._id_to_meta.get(fid, {})
        if not metadata:
            return None
        
        return {
            **metadata,
            "id" : metadata.get("__id__"),
            "created_at": metadata.get("__created_at__"),
        }
    
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        
        results = []
        for id in ids:
            fid = self._find_lightfaiss_id_by_custom_id(id)
            if fid is not None:
                metadata = self._id_to_meta.get(fid, {})
                if metadata:
                    results.append({
                        **metadata,
                        "id": metadata.get("__id__"),
                        "created_at": metadata.get("__created_at__"),
                    })

        return results
    
    async def drop(self) -> dict[str, str]:
        try:
            async with self._storage_lock:
                self._index = lf.FlatIndex(self._dim, None, lf.MetricType.METRIC_INNER_PRODUCT)
                self._id_to_meta = {}

                if os.path.exists(self._lightfaiss_index_file):
                    os.remove(self._lightfaiss_index_file)
                if os.path.exists(self._meta_file):
                    os.remove(self._meta_file)

                self._id_to_meta = {}
                self._load_lightfaiss_index()

                await set_all_update_flags(self.namespace)
                self.storage_updated.value = False

                logger.info(f"Process {os.getpid()} drop LIGHT_FAISS index {self.namespace}")
            return {"status": "success", "message": "index dropped"}
        except Exception as e:
            logger.error(f"Failed to drop LightFaiss index {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
