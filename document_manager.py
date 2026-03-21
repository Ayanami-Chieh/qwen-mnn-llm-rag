"""
文档管理器模块 - 支持多种文件格式的加载和处理
支持格式：TXT, Markdown, DOCX, PDF
分块策略：LangChain 递归分块 / 滑动窗口分块
"""
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import json
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# 分块策略枚举
# ============================================================================

class ChunkStrategy(Enum):
    """文本分块策略"""
    RECURSIVE = "recursive"        # LangChain 递归字符分块
    SLIDING_WINDOW = "sliding_window"  # 滑动窗口分块


# ============================================================================
# 文档加载器基类
# ============================================================================

class DocumentLoader(ABC):
    """文档加载器基类"""

    @abstractmethod
    def load(self, file_path: str) -> List[str]:
        """
        加载文档并返回文本内容

        Args:
            file_path: 文件路径

        Returns:
            文本列表，每个元素是一个段落
        """
        pass

    @staticmethod
    def get_file_info(file_path: str) -> Dict:
        """获取文件信息"""
        path = Path(file_path)
        return {
            'name': path.name,
            'size': path.stat().st_size,
            'created': datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }


# ============================================================================
# TXT 文档加载器
# ============================================================================

class TxtLoader(DocumentLoader):
    """文本文档加载器"""

    def load(self, file_path: str) -> List[str]:
        """加载TXT文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 按行分割，过滤空行
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            logger.info(f"✅ TXT文件加载成功: {len(paragraphs)} 段")
            return paragraphs

        except Exception as e:
            logger.error(f"❌ TXT文件加载失败: {e}")
            return []


# ============================================================================
# Markdown 文档加载器
# ============================================================================

class MarkdownLoader(DocumentLoader):
    """Markdown文档加载器"""

    def load(self, file_path: str) -> List[str]:
        """加载Markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            paragraphs = self._parse_markdown(content)
            logger.info(f"✅ Markdown文件加载成功: {len(paragraphs)} 段")
            return paragraphs

        except Exception as e:
            logger.error(f"❌ Markdown文件加载失败: {e}")
            return []

    @staticmethod
    def _parse_markdown(content: str) -> List[str]:
        """解析Markdown内容"""
        # 移除Markdown特殊字符但保留内容
        content = re.sub(r'#+\s+', '', content)  # 移除标题符号
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)  # 转换链接
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # 移除粗体
        content = re.sub(r'__([^_]+)__', r'\1', content)  # 移除粗体
        content = re.sub(r'\*([^*]+)\*', r'\1', content)  # 移除斜体
        content = re.sub(r'_([^_]+)_', r'\1', content)  # 移除斜体
        content = re.sub(r'`([^`]+)`', r'\1', content)  # 移除代码块标记
        content = re.sub(r'```[\s\S]*?```', '', content)  # 移除代码块

        # 按段落分割
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        return paragraphs


# ============================================================================
# DOCX 文档加载器
# ============================================================================

class DocxLoader(DocumentLoader):
    """DOCX文档加载器"""

    def load(self, file_path: str) -> List[str]:
        """加载DOCX文件"""
        try:
            from docx import Document

            doc = Document(file_path)
            paragraphs = []

            # 提取段落
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text.strip())

            # 提取表格内容
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' '.join([cell.text.strip() for cell in row.cells])
                    if row_text.strip():
                        paragraphs.append(row_text)

            logger.info(f"✅ DOCX文件加载成功: {len(paragraphs)} 段")
            return paragraphs

        except ImportError:
            logger.error("❌ 缺少python-docx库，请运行: pip install python-docx")
            return []
        except Exception as e:
            logger.error(f"❌ DOCX文件加载失败: {e}")
            return []


# ============================================================================
# PDF 文档加载器
# ============================================================================

class PdfLoader(DocumentLoader):
    """PDF文档加载器"""

    def load(self, file_path: str) -> List[str]:
        """加载PDF文件"""
        try:
            import pypdf

            paragraphs = []

            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()

                    if text.strip():
                        # 按行分割，过滤空行
                        lines = [line.strip() for line in text.split('\n') if line.strip()]
                        paragraphs.extend(lines)

            logger.info(f"✅ PDF文件加载成功: {len(paragraphs)} 段")
            return paragraphs

        except ImportError:
            logger.error("❌ 缺少pypdf库，请运行: pip install pypdf")
            return []
        except Exception as e:
            logger.error(f"❌ PDF文件加载失败: {e}")
            return []


# ============================================================================
# 文档管理器
# ============================================================================

class DocumentManager:
    """文档管理器 - 支持多种格式的文档加载和处理"""

    # 支持的文件格式
    SUPPORTED_FORMATS = {
        '.txt': TxtLoader,
        '.md': MarkdownLoader,
        '.markdown': MarkdownLoader,
        '.docx': DocxLoader,
        '.doc': DocxLoader,
        '.pdf': PdfLoader,
    }

    def __init__(
        self,
        chunk_size: int = 200,
        overlap: int = 50,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    ):
        """
        初始化文档管理器

        Args:
            chunk_size: 分块大小（字符数）
            overlap:    分块重叠字符数（递归分块时为重叠量；
                        滑动窗口时为步长 = chunk_size - overlap）
            strategy:   分块策略，ChunkStrategy.RECURSIVE 或
                        ChunkStrategy.SLIDING_WINDOW
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.documents = {}       # 存储已加载的文档
        self.file_metadata = {}   # 存储文件元数据

        # 初始化 LangChain 分块器（懒加载，首次使用时创建）
        self._recursive_splitter = None
        self._sliding_splitter = None

        logger.info(
            f"✅ DocumentManager 初始化完成 | 策略={strategy.value} | "
            f"chunk_size={chunk_size} | overlap={overlap}"
        )

    # ------------------------------------------------------------------
    # LangChain 分块器（内部工厂）
    # ------------------------------------------------------------------

    def _get_recursive_splitter(self):
        """
        获取（或懒创建）LangChain RecursiveCharacterTextSplitter。

        分隔符优先级（中英文混合友好）：
          段落空行 → 句号/问号/叹号 → 逗号/分号 → 空格 → 单字符兜底
        """
        if self._recursive_splitter is None:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError:
                from langchain.text_splitter import RecursiveCharacterTextSplitter

            self._recursive_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n",          # 段落
                    "\n",            # 换行
                    "。", "！", "？",  # 中文句末
                    ".", "!", "?",   # 英文句末
                    "；", "；",       # 中文分号
                    ";",
                    "，", ",",        # 逗号
                    " ",             # 空格
                    "",              # 兜底：按字符切
                ],
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                length_function=len,
                is_separator_regex=False,
            )
            logger.info("✅ RecursiveCharacterTextSplitter 初始化完成")
        return self._recursive_splitter

    def _get_sliding_splitter(self):
        """
        获取（或懒创建）LangChain CharacterTextSplitter（用作滑动窗口）。

        CharacterTextSplitter 配合 chunk_overlap 等效于固定步长滑动窗口：
          步长 = chunk_size - overlap
        separator="" 表示按字符切割，不依赖分隔符。
        """
        if self._sliding_splitter is None:
            try:
                from langchain_text_splitters import CharacterTextSplitter
            except ImportError:
                from langchain.text_splitter import CharacterTextSplitter

            self._sliding_splitter = CharacterTextSplitter(
                separator="",               # 不依赖分隔符，纯字符滑动
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                length_function=len,
            )
            step = self.chunk_size - self.overlap
            logger.info(
                f"✅ CharacterTextSplitter（滑动窗口）初始化完成 | "
                f"window={self.chunk_size} | step={step}"
            )
        return self._sliding_splitter

    # ------------------------------------------------------------------
    # 公共分块入口
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        strategy: Optional[ChunkStrategy] = None,
    ) -> List[str]:
        """
        将文本分块（统一入口，根据 strategy 调度到对应实现）。

        Args:
            text:       输入文本
            chunk_size: 分块大小，None 则使用实例默认值
            overlap:    重叠/步长，None 则使用实例默认值
            strategy:   分块策略，None 则使用实例默认策略

        Returns:
            分块列表
        """
        if not text:
            return []

        # 若调用时临时指定了不同的参数，则新建临时分块器
        use_strategy = strategy or self.strategy
        use_size = chunk_size or self.chunk_size
        use_overlap = overlap if overlap is not None else self.overlap

        # 参数与实例一致时直接复用缓存的分块器
        same_params = (use_size == self.chunk_size and use_overlap == self.overlap)

        if use_strategy == ChunkStrategy.RECURSIVE:
            if same_params:
                splitter = self._get_recursive_splitter()
            else:
                try:
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                except ImportError:
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?",
                                "；", ";", "，", ",", " ", ""],
                    chunk_size=use_size,
                    chunk_overlap=use_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
            chunks = splitter.split_text(text)

        elif use_strategy == ChunkStrategy.SLIDING_WINDOW:
            if same_params:
                splitter = self._get_sliding_splitter()
            else:
                try:
                    from langchain_text_splitters import CharacterTextSplitter
                except ImportError:
                    from langchain.text_splitter import CharacterTextSplitter
                splitter = CharacterTextSplitter(
                    separator="",
                    chunk_size=use_size,
                    chunk_overlap=use_overlap,
                    length_function=len,
                )
            chunks = splitter.split_text(text)

        else:
            logger.warning(f"未知分块策略 {use_strategy}，回退到递归分块")
            chunks = self._get_recursive_splitter().split_text(text)

        logger.debug(
            f"chunk_text | strategy={use_strategy.value} | "
            f"input_len={len(text)} | chunks={len(chunks)}"
        )
        return chunks

    # ------------------------------------------------------------------
    # 文档加载
    # ------------------------------------------------------------------

    def load_document(self, file_path: str) -> Tuple[bool, str, List[str]]:
        """
        加载单个文档

        Args:
            file_path: 文件路径

        Returns:
            (是否成功, 消息, 文本列表)
        """
        file_path = str(file_path)

        if not os.path.exists(file_path):
            msg = f"文件不存在: {file_path}"
            logger.error(f"❌ {msg}")
            return False, msg, []

        ext = Path(file_path).suffix.lower()

        if ext not in self.SUPPORTED_FORMATS:
            supported = ', '.join(self.SUPPORTED_FORMATS.keys())
            msg = f"不支持的文件格式: {ext}。支持的格式: {supported}"
            logger.error(f"❌ {msg}")
            return False, msg, []

        loader = self.SUPPORTED_FORMATS[ext]()
        paragraphs = loader.load(file_path)

        if not paragraphs:
            msg = f"文件加载失败或内容为空: {file_path}"
            logger.error(f"❌ {msg}")
            return False, msg, []

        doc_name = Path(file_path).stem
        self.documents[doc_name] = paragraphs
        self.file_metadata[doc_name] = {
            'path': file_path,
            'format': ext,
            'file_info': loader.get_file_info(file_path),
            'paragraph_count': len(paragraphs),
            'total_chars': sum(len(p) for p in paragraphs),
            'loaded_at': datetime.now().isoformat(),
        }

        msg = (
            f"文档加载成功: {len(paragraphs)} 段，"
            f"总计 {self.file_metadata[doc_name]['total_chars']} 字符"
        )
        logger.info(f"✅ {msg}")
        return True, msg, paragraphs

    def load_documents_from_directory(
        self, directory: str
    ) -> Dict[str, Tuple[bool, str, List[str]]]:
        """
        从目录加载所有支持的文档

        Args:
            directory: 目录路径

        Returns:
            {文档名: (是否成功, 消息, 文本列表)}
        """
        results = {}

        if not os.path.isdir(directory):
            logger.error(f"❌ 目录不存在: {directory}")
            return results

        for ext in self.SUPPORTED_FORMATS.keys():
            for file_path in Path(directory).glob(f'*{ext}'):
                success, msg, paragraphs = self.load_document(str(file_path))
                doc_name = file_path.stem
                results[doc_name] = (success, msg, paragraphs)

        logger.info(f"📁 目录加载完成: {len(results)} 个文件")
        return results

    # ------------------------------------------------------------------
    # 文档处理（分块 + 清理）
    # ------------------------------------------------------------------

    def process_documents(
        self,
        chunk_size: Optional[int] = None,
        strategy: Optional[ChunkStrategy] = None,
    ) -> List[str]:
        """
        处理所有已加载的文档，进行清理和分块。

        Args:
            chunk_size: 分块大小，None 则使用实例默认值
            strategy:   分块策略，None 则使用实例默认策略

        Returns:
            处理后的文本分块列表（已去重）
        """
        use_strategy = strategy or self.strategy
        all_chunks = []

        logger.info(
            f"🔄 开始处理文档 | 文档数={len(self.documents)} | "
            f"策略={use_strategy.value}"
        )

        for doc_name, paragraphs in self.documents.items():
            logger.info(f"  处理文档: {doc_name} ({len(paragraphs)} 段)")
            doc_chunks = []

            for para in paragraphs:
                cleaned = self._clean_text(para)
                if cleaned:
                    chunks = self.chunk_text(
                        cleaned,
                        chunk_size=chunk_size,
                        strategy=use_strategy,
                    )
                    doc_chunks.extend(chunks)

            logger.info(f"  ✅ {doc_name}: {len(doc_chunks)} 个分块")
            all_chunks.extend(doc_chunks)

        # 全局去重
        all_chunks = self._remove_duplicates(all_chunks)
        logger.info(f"✅ 文档处理完成: 生成 {len(all_chunks)} 个文本块")
        return all_chunks

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(
            r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：、\'"\'""''（）【】《》]',
            '',
            text,
        )
        return text.strip()

    def _remove_duplicates(self, fragments: List[str]) -> List[str]:
        """移除重复片段"""
        seen = set()
        unique_fragments = []
        for frag in fragments:
            if frag not in seen:
                seen.add(frag)
                unique_fragments.append(frag)
        removed = len(fragments) - len(unique_fragments)
        logger.info(f"⚙️ 去重完成，共移除 {removed} 个重复片段")
        return unique_fragments

    # ------------------------------------------------------------------
    # 统计 / 元数据
    # ------------------------------------------------------------------

    def get_document_stats(self) -> Dict:
        """获取文档统计信息"""
        stats = {
            'total_documents': len(self.documents),
            'total_paragraphs': sum(len(p) for p in self.documents.values()),
            'total_characters': sum(
                self.file_metadata[doc_name]['total_chars']
                for doc_name in self.documents.keys()
            ),
            'chunk_strategy': self.strategy.value,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'documents': {},
        }

        for doc_name, metadata in self.file_metadata.items():
            stats['documents'][doc_name] = {
                'format': metadata['format'],
                'paragraphs': metadata['paragraph_count'],
                'characters': metadata['total_chars'],
                'path': metadata['path'],
            }

        return stats

    def export_metadata(self, output_path: str):
        """导出文件元数据为JSON"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.file_metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 元数据已导出: {output_path}")
        except Exception as e:
            logger.error(f"❌ 元数据导出失败: {e}")

    def clear_documents(self):
        """清空所有已加载的文档"""
        self.documents.clear()
        self.file_metadata.clear()
        logger.info("✅ 所有文档已清空")

    def get_loaded_documents_count(self) -> int:
        """获取已加载的文档数量"""
        return len(self.documents)

    # ------------------------------------------------------------------
    # 策略切换（运行时）
    # ------------------------------------------------------------------

    def set_strategy(
        self,
        strategy: ChunkStrategy,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ):
        """
        运行时切换分块策略（同时可更新 chunk_size / overlap）。
        切换后会重置对应缓存的分块器，下次使用时重新创建。

        Args:
            strategy:   新的分块策略
            chunk_size: 新的分块大小（可选）
            overlap:    新的重叠大小（可选）
        """
        self.strategy = strategy
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if overlap is not None:
            self.overlap = overlap

        # 重置缓存的分块器
        self._recursive_splitter = None
        self._sliding_splitter = None

        logger.info(
            f"✅ 分块策略已切换 | strategy={strategy.value} | "
            f"chunk_size={self.chunk_size} | overlap={self.overlap}"
        )