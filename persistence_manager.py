# -*- coding: utf-8 -*-
"""
向量数据持久化管理器
处理索引、向量和元数据的保存与加载
"""

import os
import json
import time
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PersistenceManager:
    """管理向量数据的持久化"""

    def __init__(self, cache_dir: str = r"D:\MNN_RAG_Project\cache"):
        """
        初始化持久化管理器

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        self.index_path = os.path.join(cache_dir, "faiss_index.bin")
        self.embeddings_path = os.path.join(cache_dir, "embeddings.npy")
        self.fragments_path = os.path.join(cache_dir, "fragments.json")
        self.metadata_path = os.path.join(cache_dir, "metadata.json")

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"✅ 缓存目录已准备: {cache_dir}")

    def save_index(self, faiss_index) -> bool:
        """
        保存 Faiss 索引

        Args:
            faiss_index: Faiss 索引对象

        Returns:
            是否保存成功
        """
        try:
            import faiss
            logger.info("💾 正在保存 Faiss 索引...")
            faiss.write_index(faiss_index, self.index_path)
            logger.info(f"✅ 索引已保存: {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"❌ 索引保存失败: {e}")
            return False

    def load_index(self):
        """
        加载 Faiss 索引

        Returns:
            Faiss 索引对象，如果不存在返回 None
        """
        if not os.path.exists(self.index_path):
            logger.warning(f"⚠️  索引文件不存在: {self.index_path}")
            return None

        try:
            import faiss
            logger.info("📂 正在加载 Faiss 索引...")
            start_time = time.time()

            faiss_index = faiss.read_index(self.index_path)

            load_time = time.time() - start_time
            logger.info(f"✅ 索引已加载 ({load_time:.2f}s)")
            logger.info(f"   索引中的向量数: {faiss_index.ntotal}")

            return faiss_index
        except Exception as e:
            logger.error(f"❌ 索引加载失败: {e}")
            return None

    def save_embeddings(self, embeddings: np.ndarray) -> bool:
        """
        保存向量数据

        Args:
            embeddings: numpy 数组

        Returns:
            是否保存成功
        """
        try:
            logger.info("💾 正在保存向量数据...")
            np.save(self.embeddings_path, embeddings)

            file_size = os.path.getsize(self.embeddings_path) / 1024 / 1024
            logger.info(f"✅ 向量已保存: {self.embeddings_path}")
            logger.info(f"   文件大小: {file_size:.2f} MB")

            return True
        except Exception as e:
            logger.error(f"❌ 向量保存失败: {e}")
            return False

    def load_embeddings(self) -> Optional[np.ndarray]:
        """
        加载向量数据

        Returns:
            numpy 数组，如果不存在返回 None
        """
        if not os.path.exists(self.embeddings_path):
            logger.warning(f"⚠️  向量文件不存在: {self.embeddings_path}")
            return None

        try:
            logger.info("📂 正在加载向量数据...")
            start_time = time.time()

            embeddings = np.load(self.embeddings_path)

            load_time = time.time() - start_time
            logger.info(f"✅ 向量已加载 ({load_time:.2f}s)")
            logger.info(f"   向量形状: {embeddings.shape}")
            logger.info(f"   数据类型: {embeddings.dtype}")

            return embeddings
        except Exception as e:
            logger.error(f"❌ 向量加载失败: {e}")
            return None

    def save_fragments(self, fragments: List[str]) -> bool:
        """
        保存知识库片段

        Args:
            fragments: 文本片段列表

        Returns:
            是否保存成功
        """
        try:
            logger.info("💾 正在保存知识库片段...")
            with open(self.fragments_path, 'w', encoding='utf-8') as f:
                json.dump(fragments, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 片段已保存: {self.fragments_path}")
            logger.info(f"   片段数量: {len(fragments)}")

            return True
        except Exception as e:
            logger.error(f"❌ 片段保存失败: {e}")
            return False

    def load_fragments(self) -> Optional[List[str]]:
        """
        加载知识库片段

        Returns:
            文本片段列表，如果不存在返回 None
        """
        if not os.path.exists(self.fragments_path):
            logger.warning(f"⚠️  片段文件不存在: {self.fragments_path}")
            return None

        try:
            logger.info("📂 正在加载知识库片段...")
            with open(self.fragments_path, 'r', encoding='utf-8') as f:
                fragments = json.load(f)

            logger.info(f"✅ 片段已加载")
            logger.info(f"   片段数���: {len(fragments)}")

            return fragments
        except Exception as e:
            logger.error(f"❌ 片段加载失败: {e}")
            return None

    def save_metadata(self, metadata: Dict) -> bool:
        """
        保存元数据（版本、时间戳等）

        Args:
            metadata: 元数据字典

        Returns:
            是否保存成功
        """
        try:
            logger.info("💾 正在保存元数据...")

            # 添加时间戳
            metadata['saved_at'] = datetime.now().isoformat()
            metadata['cache_version'] = '1.0'

            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 元数据已保存: {self.metadata_path}")

            return True
        except Exception as e:
            logger.error(f"❌ 元数据保存失败: {e}")
            return False

    def load_metadata(self) -> Optional[Dict]:
        """
        加载元数据

        Returns:
            元数据字典，如果不存在返回 None
        """
        if not os.path.exists(self.metadata_path):
            logger.warning(f"⚠️  元数据文件不存在: {self.metadata_path}")
            return None

        try:
            logger.info("📂 正在加载元数据...")
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            logger.info(f"✅ 元数据已加载")
            logger.info(f"   保存时间: {metadata.get('saved_at', 'Unknown')}")
            logger.info(f"   向量数量: {metadata.get('num_fragments', 'Unknown')}")

            return metadata
        except Exception as e:
            logger.error(f"❌ 元数据加载失败: {e}")
            return None

    def save_all(self, faiss_index, embeddings: np.ndarray,
                 fragments: List[str], metadata: Dict = None) -> bool:
        """
        一次性保存所有数据

        Args:
            faiss_index: Faiss 索引
            embeddings: 向量数据
            fragments: 文本片段
            metadata: 元数据

        Returns:
            是否全部保存成功
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔄 开始完整保存流程...")
        logger.info("=" * 80)

        start_time = time.time()

        # 保存各类数据
        results = [
            self.save_index(faiss_index),
            self.save_embeddings(embeddings),
            self.save_fragments(fragments),
        ]

        # 保存元数据（如果提供）
        if metadata is None:
            metadata = {}
        metadata['num_fragments'] = len(fragments)
        metadata['embedding_shape'] = list(embeddings.shape)
        self.save_metadata(metadata)

        total_time = time.time() - start_time

        if all(results):
            logger.info("=" * 80)
            logger.info(f"✅ 完整保存成功！耗时 {total_time:.2f}s")
            logger.info("=" * 80 + "\n")
            return True
        else:
            logger.error("❌ 部分数据保存失败")
            return False

    def load_all(self) -> Tuple[Any, Optional[np.ndarray], Optional[List[str]], Optional[Dict]]:
        """
        一次性加载所有数据

        Returns:
            (faiss_index, embeddings, fragments, metadata) 元组
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔄 开始完整加载流程...")
        logger.info("=" * 80)

        start_time = time.time()

        faiss_index = self.load_index()
        embeddings = self.load_embeddings()
        fragments = self.load_fragments()
        metadata = self.load_metadata()

        total_time = time.time() - start_time

        if all([faiss_index, embeddings is not None, fragments]):
            logger.info("=" * 80)
            logger.info(f"✅ 完整加载成功！耗时 {total_time:.2f}s")
            logger.info("=" * 80 + "\n")
            return faiss_index, embeddings, fragments, metadata
        else:
            logger.info("❌ 无法完整加载缓存，需要重新构建")
            return None, None, None, None

    def is_cache_valid(self) -> bool:
        """
        检查缓存是否有效（所有文件都存在）

        Returns:
            缓存是否有效
        """
        required_files = [
            self.index_path,
            self.embeddings_path,
            self.fragments_path,
            self.metadata_path
        ]

        is_valid = all(os.path.exists(f) for f in required_files)

        if is_valid:
            logger.info("✅ 缓存有效，所有文件都存在")
        else:
            missing_files = [f for f in required_files if not os.path.exists(f)]
            logger.warning(f"❌ 缓存不完整，缺少文件:")
            for f in missing_files:
                logger.warning(f"   - {os.path.basename(f)}")

        return is_valid

    def clear_cache(self) -> bool:
        """
        清除所有缓存

        Returns:
            是否成功清除
        """
        try:
            logger.warning("🗑️  正在清除缓存...")
            files_to_remove = [
                self.index_path,
                self.embeddings_path,
                self.fragments_path,
                self.metadata_path
            ]

            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"   ✅ 已删除: {os.path.basename(file_path)}")

            logger.info("✅ 缓存已清除")
            return True
        except Exception as e:
            logger.error(f"❌ 清除缓存失败: {e}")
            return False

    def get_cache_info(self) -> Dict:
        """
        获取缓存统计信息

        Returns:
            包含缓存大小、文件数量等信息的字典
        """
        total_size = 0
        file_info = {}

        files = {
            'Faiss 索引': self.index_path,
            '向量数据': self.embeddings_path,
            '片段数据': self.fragments_path,
            '元数据': self.metadata_path
        }

        for name, path in files.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                total_size += size
                file_info[name] = f"{size / 1024 / 1024:.2f} MB"

        return {
            'total_size_mb': total_size / 1024 / 1024,
            'file_info': file_info,
            'is_valid': self.is_cache_valid()
        }