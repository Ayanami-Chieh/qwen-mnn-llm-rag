# -*- coding: utf-8 -*-
"""
知识库管理器
支持用户对知识库进行：新增、删除、更新、查看、导入、导出操作
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class KBManager:
    """知识库管理器 - 支持对知识库片段的增删改查"""

    def __init__(self, kb_path: str, chunk_fn=None):
        """
        初始化知识库管理器

        Args:
            kb_path:  知识库文件路径（.txt）
            chunk_fn: 文本分块函数（传入 document_manager.chunk_text），
                      为 None 时退化为按句号硬切（兼容旧行为）
        """
        self.kb_path = kb_path
        self.chunk_fn = chunk_fn
        self.backup_dir = str(Path(kb_path).parent / "kb_backups")
        os.makedirs(self.backup_dir, exist_ok=True)
        logger.info(f"✅ KBManager 初始化完成 | 路径={kb_path}")

    # ------------------------------------------------------------------
    # 读写底层
    # ------------------------------------------------------------------

    def _read_raw(self) -> str:
        """读取知识库原始文本"""
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _write_raw(self, content: str):
        """写入知识库原始文本"""
        with open(self.kb_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _parse_fragments(self, raw: str) -> List[str]:
        """将原始文本解析为片段列表。
        若传入了 chunk_fn 则使用与主系统一致的分块逻辑；
        否则退化为按句号硬切（兼容旧行为）。
        """
        if self.chunk_fn is not None:
            return self.chunk_fn(raw)
        # 兼容旧行为
        return [frag.strip() + '。' for frag in raw.split('。') if frag.strip()]

    def _fragments_to_raw(self, fragments: List[str]) -> str:
        """将片段列表还原为原始文本（去掉末尾句号后用句号拼接）"""
        # 去掉每段末尾的句号后重新拼接，保持格式统一
        parts = [frag.rstrip('。') for frag in fragments if frag.strip()]
        return '。'.join(parts) + ('。' if parts else '')

    # ------------------------------------------------------------------
    # 备份
    # ------------------------------------------------------------------

    def _backup(self):
        """备份当前知识库"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"kb_backup_{timestamp}.txt")
        content = self._read_raw()
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"✅ 知识库已备份: {backup_path}")
        return backup_path

    # ------------------------------------------------------------------
    # 查看
    # ------------------------------------------------------------------

    def list_fragments(
        self, keyword: Optional[str] = None, page: int = 1, page_size: int = 20
    ) -> Tuple[List[Tuple[int, str]], int]:
        """
        列出知识库片段（支持关键字筛选和分页）

        Args:
            keyword:   筛选关键字，None 则列出全部
            page:      页码（从 1 开始）
            page_size: 每页条数

        Returns:
            ([(序号, 片段), ...], 总条数)
        """
        raw = self._read_raw()
        fragments = self._parse_fragments(raw)

        if keyword:
            indexed = [(i, f) for i, f in enumerate(fragments) if keyword in f]
        else:
            indexed = list(enumerate(fragments))

        total = len(indexed)
        start = (page - 1) * page_size
        end = start + page_size
        return indexed[start:end], total

    def search_fragments(self, keyword: str) -> List[Tuple[int, str]]:
        """
        搜索包含关键字的片段

        Args:
            keyword: 搜索关键字

        Returns:
            [(原始序号, 片段), ...]
        """
        raw = self._read_raw()
        fragments = self._parse_fragments(raw)
        results = [(i, f) for i, f in enumerate(fragments) if keyword in f]
        return results

    # ------------------------------------------------------------------
    # 新增
    # ------------------------------------------------------------------

    def add_fragment(self, text: str) -> Tuple[bool, str]:
        """
        在知识库末尾追加一条新片段

        Args:
            text: 新知识片段文本（可以不带句号）

        Returns:
            (是否成功, 提示信息)
        """
        text = text.strip()
        if not text:
            return False, "片段内容不能为空"

        # 保证末尾有句号
        if not text.endswith('。'):
            text += '。'

        try:
            self._backup()
            raw = self._read_raw()
            # 在末尾追加，确保换行分隔
            new_raw = raw.rstrip() + text
            self._write_raw(new_raw)

            fragments = self._parse_fragments(new_raw)
            idx = len(fragments) - 1
            msg = f"已添加第 {idx} 条片段: {text[:40]}{'...' if len(text) > 40 else ''}"
            logger.info(f"✅ {msg}")
            return True, msg
        except Exception as e:
            logger.error(f"❌ 添加失败: {e}")
            return False, f"添加失败: {e}"

    def add_fragments_batch(self, texts: List[str]) -> Tuple[int, int, List[str]]:
        """
        批量添加多条片段

        Args:
            texts: 片段列表

        Returns:
            (成功数, 失败数, 错误信息列表)
        """
        success_count = 0
        fail_count = 0
        errors = []

        self._backup()
        raw = self._read_raw()
        fragments = self._parse_fragments(raw)

        for text in texts:
            text = text.strip()
            if not text:
                fail_count += 1
                errors.append("空字符串，已跳过")
                continue
            if not text.endswith('。'):
                text += '。'
            fragments.append(text)
            success_count += 1

        try:
            self._write_raw(self._fragments_to_raw(fragments))
            logger.info(f"✅ 批量添加完成: 成功 {success_count} 条，失败 {fail_count} 条")
        except Exception as e:
            logger.error(f"❌ 批量写入失败: {e}")
            errors.append(f"写入失败: {e}")
            return 0, len(texts), errors

        return success_count, fail_count, errors

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    def delete_fragment(self, index: int) -> Tuple[bool, str]:
        """
        按序号删除一条片段

        Args:
            index: 片段序号（0 起始）

        Returns:
            (是否成功, 提示信息)
        """
        try:
            raw = self._read_raw()
            fragments = self._parse_fragments(raw)

            if index < 0 or index >= len(fragments):
                return False, f"序号超出范围，当前共 {len(fragments)} 条（0~{len(fragments)-1}）"

            self._backup()
            deleted = fragments[index]
            fragments.pop(index)
            self._write_raw(self._fragments_to_raw(fragments))

            msg = f"已删除第 {index} 条: {deleted[:40]}{'...' if len(deleted) > 40 else ''}"
            logger.info(f"✅ {msg}")
            return True, msg
        except Exception as e:
            logger.error(f"❌ 删除失败: {e}")
            return False, f"删除失败: {e}"

    def delete_by_keyword(self, keyword: str) -> Tuple[bool, str, int]:
        """
        删除所有包含关键字的片段

        Args:
            keyword: 关键字

        Returns:
            (是否成功, 提示信息, 删除数量)
        """
        try:
            raw = self._read_raw()
            fragments = self._parse_fragments(raw)
            before = len(fragments)

            self._backup()
            fragments = [f for f in fragments if keyword not in f]
            after = len(fragments)
            deleted_count = before - after

            self._write_raw(self._fragments_to_raw(fragments))

            msg = f"已删除包含 '{keyword}' 的片段 {deleted_count} 条"
            logger.info(f"✅ {msg}")
            return True, msg, deleted_count
        except Exception as e:
            logger.error(f"❌ 关键字删除失败: {e}")
            return False, f"删除失败: {e}", 0

    # ------------------------------------------------------------------
    # 修改
    # ------------------------------------------------------------------

    def update_fragment(self, index: int, new_text: str) -> Tuple[bool, str]:
        """
        更新指定序号的片段内容

        Args:
            index:    片段序号（0 起始）
            new_text: 新内容

        Returns:
            (是否成功, 提示信息)
        """
        new_text = new_text.strip()
        if not new_text:
            return False, "新内容不能为空"
        if not new_text.endswith('。'):
            new_text += '。'

        try:
            raw = self._read_raw()
            fragments = self._parse_fragments(raw)

            if index < 0 or index >= len(fragments):
                return False, f"序号超出范围，当前共 {len(fragments)} 条（0~{len(fragments)-1}）"

            self._backup()
            old = fragments[index]
            fragments[index] = new_text
            self._write_raw(self._fragments_to_raw(fragments))

            msg = (
                f"已更新第 {index} 条\n"
                f"  旧: {old[:60]}{'...' if len(old) > 60 else ''}\n"
                f"  新: {new_text[:60]}{'...' if len(new_text) > 60 else ''}"
            )
            logger.info(f"✅ 更新成功")
            return True, msg
        except Exception as e:
            logger.error(f"❌ 更新失败: {e}")
            return False, f"更新失败: {e}"

    # ------------------------------------------------------------------
    # 导入外部文件追加到知识库
    # ------------------------------------------------------------------

    def import_from_file(self, file_path: str) -> Tuple[bool, str, int]:
        """
        从外部 TXT 文件导入内容追加到知识库
        文件中每行视为一条片段

        Args:
            file_path: 外部文件路径

        Returns:
            (是否成功, 提示信息, 新增条数)
        """
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}", 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            if not lines:
                return False, "文件内容为空", 0

            success, fail, errors = self.add_fragments_batch(lines)
            msg = f"从文件导入完成: 成功 {success} 条，失败 {fail} 条"
            if errors:
                msg += f"\n错误: {'; '.join(errors[:3])}"
            return True, msg, success
        except Exception as e:
            logger.error(f"❌ 导入失败: {e}")
            return False, f"导入失败: {e}", 0

    # ------------------------------------------------------------------
    # 导出
    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        raw = self._read_raw()
        fragments = self._parse_fragments(raw)
        return {
            'total_fragments': len(fragments),
            'total_chars': len(raw),
            'avg_fragment_len': round(sum(len(f) for f in fragments) / max(len(fragments), 1), 1),
            'kb_path': self.kb_path,
        }
