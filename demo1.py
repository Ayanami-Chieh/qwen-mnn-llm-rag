"""
完整的 MNN RAG 系统
集成文档管理、命令解析和智能检索
核心特性：
- 多格式文档导入支持（TXT, Markdown, DOCX, PDF）
- 智能命令解析和验证
- 会话级向量缓存（系统退出时清除）
- 向量索引和缓存持久化
- 知识库增删改查（kbadd / kbdel / kbupdate / kbsearch / kbimport / kbclearcache）
"""
import os
import re
import sys
import time
import json
import numpy as np
from typing import List, Tuple, Optional
from contextlib import redirect_stdout, redirect_stderr
import io
import logging

from persistence_manager import PersistenceManager
from document_manager import DocumentManager, ChunkStrategy
from command_parser import CommandParser, CommandValidator, CommandType
from kb_manager import KBManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import MNN.llm as llm
    from sentence_transformers import SentenceTransformer
    import faiss

    FAISS_AVAILABLE = True
except ImportError as e:
    print(f"警告：Faiss不可用 ({e})，将使用基础检索方法")
    FAISS_AVAILABLE = False
    import MNN.llm as llm
    from sentence_transformers import SentenceTransformer


class RAGSystem:
    """增强的 RAG 系统"""

    def __init__(self):
        """初始化系统"""
        # 配置路径
        self.config = {
            'llm_config': r"D:\MNN_RAG_Project\mnn_model\model\config.json",
            'bge_model': r"D:\MNN_RAG_Project\models\bge-m3",
            'knowledge_base': r"D:\MNN_RAG_Project\src\ipl6.md",
            'documents_dir': r"D:\MNN_RAG_Project\documents",
        }

        # 初始化管理器
        self.document_manager = DocumentManager(
            chunk_size=200, overlap=50,
            strategy=ChunkStrategy.RECURSIVE
        )
        self.command_parser = CommandParser()
        self.command_validator = CommandValidator(self.command_parser)
        self.persistence = PersistenceManager()

        # ⭐ 知识库管理器
        self.kb_manager = KBManager(self.config['knowledge_base'], chunk_fn=self.document_manager.chunk_text)

        # 初始化变量
        self.llm_model = None
        self.embedder = None

        # ⭐ 核心数据结构
        # base_knowledge_fragments: 只存知识库原始片段，永远不被文档内容污染
        # knowledge_fragments:      实际检索用 = 知识库片段 + 带标签的文档片段
        self.base_knowledge_fragments = []
        self.knowledge_fragments = []
        self.fragment_embeddings = None
        self.faiss_index = None

        # ⭐ 会话状态标记
        self.has_loaded_documents = False

        # 启动系统
        self.startup()

    # ------------------------------------------------------------------
    # 文档片段打标签（统一入口）
    # ------------------------------------------------------------------

    def _build_tagged_doc_chunks(self) -> List[str]:
        """
        将所有已加载文档的片段打上来源文件名标签。
        同时在每个文档前插入一条索引片段，使"上传了什么/有哪些文档"类问题也能被检索到。
        只写入 knowledge_fragments，不触碰 base_knowledge_fragments，隔离完全安全。
        """
        result = []
        for doc_name, paragraphs in self.document_manager.documents.items():
            meta = self.document_manager.file_metadata.get(doc_name, {})
            filename = os.path.basename(meta.get('path', doc_name))
            chunks = self.document_manager.chunk_text("\n".join(paragraphs))
            total_chars = sum(len(p) for p in paragraphs)
            # 索引片段
            result.append(
                f"[来源文件:{filename}] 用户已加载文档：{filename}，"
                f"共 {len(paragraphs)} 段，{total_chars} 个字符。"
            )
            # 内容片段
            result.extend([f"[来源文件:{filename}] {chunk}" for chunk in chunks])
        return result

    # ------------------------------------------------------------------
    # 启动流程
    # ------------------------------------------------------------------

    def startup(self):
        """启动系统"""
        self.display_welcome()
        self.validate_files()
        self.load_models()
        self.load_knowledge_base()
        self.build_faiss_index()
        self.display_system_info()
        self.start_chat_session()

    def display_welcome(self):
        """显示欢迎信息"""
        print("\n" + "=" * 80)
        print(" " * 25 + "🚀 MNN RAG 系统 ")
        if FAISS_AVAILABLE:
            print(" " * 15 + "LLM: Qwen2-1.5B | Embedding: BGE-M3 | Index: Faiss")
        else:
            print(" " * 15 + "LLM: Qwen2-1.5B | Embedding: BGE-M3 | Index: Basic")
        print("=" * 80 + "\n")

    def validate_files(self):
        """验证文件存在"""
        print("🔍 检查必要文件...")
        all_exist = True

        for key, path in self.config.items():
            if key == 'documents_dir':
                continue
            status = "✅" if os.path.exists(path) else "❌"
            print(f"  {status} {key}: {path}")
            if not os.path.exists(path):
                all_exist = False

        if not all_exist:
            print("\n❌ 一个或多个文件不存在，请检查路径配置")
            sys.exit(1)

        if not os.path.exists(self.config['documents_dir']):
            os.makedirs(self.config['documents_dir'], exist_ok=True)
            print(f"  ✅ 文档目录已创建: {self.config['documents_dir']}")

        print()

    def load_models(self):
        """加载模型"""
        print("📦 正在加载模型...")

        print("\n[1/2] 加载Qwen2-1.5B...")
        start_time = time.time()
        try:
            self.llm_model = llm.create(self.config['llm_config'])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.llm_model.load()
            load_time = time.time() - start_time
            print(f"     ✅ 加载完成 ({load_time:.2f}s)")
        except Exception as e:
            print(f"     ❌ LLM加载失败: {e}")
            sys.exit(1)

        print("\n[2/2] 加载BGE-M3嵌入模型...")
        start_time = time.time()
        try:
            self.embedder = SentenceTransformer(self.config['bge_model'])
            load_time = time.time() - start_time
            print(f"     ✅ 加载完成 ({load_time:.2f}s)")
        except Exception as e:
            print(f"     ❌ BGE-M3加载失败: {e}")
            sys.exit(1)

        print()

    def load_knowledge_base(self):
        """加载知识库"""
        print("📚 正在加载知识库...")

        try:
            with open(self.config['knowledge_base'], 'r', encoding='utf-8') as f:
                content = f.read()

            self.base_knowledge_fragments = self.document_manager.chunk_text(content)
            self.knowledge_fragments = self.base_knowledge_fragments.copy()

            print(f"     已加载 {len(self.knowledge_fragments)} 个知识片段")

            if self.persistence.is_cache_valid():
                print("     📂 发现缓存，尝试从磁盘加载向量...")
                faiss_index, embeddings, fragments_cache, metadata = self.persistence.load_all()

                if faiss_index is not None and embeddings is not None and fragments_cache is not None:
                    self.fragment_embeddings = embeddings
                    self.faiss_index = faiss_index
                    self.knowledge_fragments = fragments_cache
                    print("     ✅ 从缓存加载成功，跳过向量计算")
                    print()
                    return
                else:
                    print("     ⚠️  缓存加载失败，重新计算向量...")

            print("     正在计算向量表示...")
            start_time = time.time()
            self.fragment_embeddings = self.embedder.encode(
                self.knowledge_fragments,
                batch_size=32,
                show_progress_bar=False
            )
            embed_time = time.time() - start_time
            print(f"     ✅ 向量计算完成 ({embed_time:.2f}s)")

            print("     💾 正在保存向量缓存到磁盘...")
            self.persistence.save_embeddings(self.fragment_embeddings)
            self.persistence.save_fragments(self.knowledge_fragments)
            print("     ✅ 向量缓存已保存")

            print()

        except Exception as e:
            print(f"     ❌ 知识库加载失败: {e}")
            sys.exit(1)

    def build_faiss_index(self):
        """构建Faiss索引"""
        if FAISS_AVAILABLE:
            print("🎯 构建Faiss索引...")
            start_time = time.time()

            embeddings = self.fragment_embeddings.astype('float32')
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings)

            build_time = time.time() - start_time
            print(f"     ✅ Faiss索引构建完成 ({build_time:.2f}s)")
            print(f"     索引向量数: {self.faiss_index.ntotal}")

            print("     💾 正在保存Faiss索引到磁盘...")
            self.persistence.save_index(self.faiss_index)
            self.persistence.save_metadata({
                'num_fragments': len(self.knowledge_fragments),
                'embedding_shape': list(self.fragment_embeddings.shape),
            })
            print("     ✅ Faiss索引已保存")

        else:
            print("⚠️  Faiss不可用，使用基础检索方法")

        print()

    def retrieve_relevant_fragments(self, query: str, top_k: int = 3) -> List[str]:
        """检索相关知识片段"""
        if FAISS_AVAILABLE and self.faiss_index is not None:
            query_embedding = self.embedder.encode([query]).astype('float32')
            faiss.normalize_L2(query_embedding)
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            relevant_fragments = [self.knowledge_fragments[i] for i in indices[0]]
            return relevant_fragments
        else:
            query_embedding = self.embedder.encode([query])[0]
            similarities = []
            for frag_embedding in self.fragment_embeddings:
                similarity = np.dot(query_embedding, frag_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(frag_embedding) + 1e-8
                )
                similarities.append(similarity)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            relevant_fragments = [self.knowledge_fragments[i] for i in top_indices]
            return relevant_fragments

    def display_system_info(self):
        """显示系统信息"""
        print("=" * 80)
        print("✅ 系统已就绪！")
        print("=" * 80)
        print(f"\n📊 系统信息:")
        print(f"  - LLM: Qwen2-1.5B (MNN)")
        print(f"  - Embedding: BGE-M3")
        print(f"  - 知识库: {len(self.knowledge_fragments)} 个片段")
        print(f"  - 检索方式: {'Faiss' if FAISS_AVAILABLE and self.faiss_index else 'Basic'}")
        print(f"  - 文档状态: {'已加载' if self.has_loaded_documents else '未加载'}")
        print(f"\n💡 提示：输入 'help' 查看所有命令\n")

    def generate_response(self, query: str) -> Tuple[str, List[str]]:
        """生成响应"""
        # 修复: 检测问题中是否包含文件名，优先精确匹配该文件的所有片段
        filename_hit = re.search(
            r'[\w\u4e00-\u9fa5\-]+\.(txt|md|pdf|docx)', query, re.IGNORECASE
        )
        if filename_hit:
            fname = filename_hit.group(0)
            file_frags = [f for f in self.knowledge_fragments if f"[来源文件:{fname}]" in f]
            if file_frags:
                relevant_fragments = file_frags[:9]  # 最多取9条覆盖文件内容
            else:
                relevant_fragments = self.retrieve_relevant_fragments(query, top_k=3)
        else:
            relevant_fragments = self.retrieve_relevant_fragments(query, top_k=3)

        # 修复: 检索为空时直接返回提示，不把空 context 送给模型造成胡乱回答
        if not relevant_fragments:
            return "未能在知识库中检索到相关内容，请确认文档已正确加载，或换一种提问方式。", []

        context = "\n".join([f"- {frag}" for frag in relevant_fragments])
        prompt = f"""根据以下知识库信息回答问题：

【知识库信息】
{context}

【问题】
{query}

【回答】
"""
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            response = self.llm_model.response(prompt, stream=False)

        if hasattr(response, 'text'):
            answer = response.text
        else:
            answer = str(response)

        if "【回答】" in answer:
            answer_start = answer.find("【回答】") + len("【回答】")
            actual_answer = answer[answer_start:].strip()
        else:
            actual_answer = answer

        return actual_answer, relevant_fragments

    def start_chat_session(self):
        """开始聊天会话"""
        print("💬 开始对话")
        print("-" * 80)
        print("输入 'help' 查看命令列表，输入 'quit' 退出")
        print()

        while True:
            try:
                user_input = input("👤 你: ").strip()

                if not user_input:
                    continue

                # 解析命令
                cmd, cmd_type, args = self.command_parser.parse(user_input)

                # 处理非命令输入（普通问题）
                if cmd_type == CommandType.QUERY:
                    print("🤖 正在思考...")
                    start_time = time.time()
                    answer, sources = self.generate_response(user_input)
                    response_time = time.time() - start_time
                    self._display_response(answer, sources, response_time)
                    continue

                # 验证命令
                is_valid, error_msg = self.command_validator.validate(cmd, args)
                if not is_valid:
                    print(error_msg)
                    continue

                # 执行命令
                self._execute_command(cmd, args)

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                self._cleanup_on_exit()
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()

    # ------------------------------------------------------------------
    # 命令执行
    # ------------------------------------------------------------------

    def _execute_command(self, cmd: str, args: list):
        """执行命令"""
        if cmd == 'help':
            self._show_help()

        elif cmd == 'quit':
            print("\n👋 再见！")
            self._cleanup_on_exit()
            sys.exit(0)

        elif cmd == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')

        elif cmd == 'kb':
            self._list_knowledge_base()

        elif cmd == 'cache':
            self._show_document_stats()

        elif cmd == 'doc':
            self._show_document_help()

        elif cmd == 'load':
            file_path = args[0] if args else ''
            self._handle_load_document(file_path)

        elif cmd == 'loaddir':
            self._handle_load_directory()

        elif cmd == 'docs':
            self._show_document_stats()

        elif cmd == 'kblist':
            self._kb_list(args)

        elif cmd == 'kbsearch':
            keyword = ' '.join(args) if args else ''
            self._kb_search(keyword)

        elif cmd == 'kbadd':
            text = ' '.join(args) if args else ''
            self._kb_add(text)

        elif cmd == 'kbdel':
            self._kb_delete(args)

        elif cmd == 'kbupdate':
            self._kb_update(args)

        elif cmd == 'kbimport':
            file_path = args[0] if args else ''
            self._kb_import(file_path)

        elif cmd == 'kbstats':
            self._kb_stats()

        elif cmd == 'kbclearcache':
            self._kb_clear_cache()

        else:
            suggestion = self.command_validator.suggest_command(cmd)
            if suggestion:
                print(f"❌ 未知命令: {cmd}")
                print(f"💡 你是不是想用: {suggestion}?")
            else:
                print(f"❌ 未知命令: {cmd}")
                print(f"   输入 'help' 查看可用命令")

    # ------------------------------------------------------------------
    # 知识库管理命令实现
    # ------------------------------------------------------------------

    def _kb_list(self, args: list):
        """列出知识库片段（支持分页和关键字过滤）"""
        page = 1
        keyword = None

        for arg in args:
            if arg.isdigit():
                page = int(arg)
            else:
                keyword = arg

        items, total = self.kb_manager.list_fragments(keyword=keyword, page=page, page_size=20)

        filter_tip = f" (筛选: '{keyword}')" if keyword else ""
        print(f"\n📚 知识库片段{filter_tip} — 共 {total} 条，第 {page} 页:")
        print("-" * 70)
        if not items:
            print("  (无结果)")
        for idx, frag in items:
            snippet = frag[:60] + '...' if len(frag) > 60 else frag
            print(f"  [{idx:>4}] {snippet}")
        print("-" * 70)
        total_pages = (total + 19) // 20
        print(f"  第 {page}/{total_pages} 页  |  输入 'kblist {page+1}' 查看下一页")
        print()

    def _kb_search(self, keyword: str):
        """搜索知识库"""
        if not keyword:
            print("❌ 用法: kbsearch <关键字>")
            return

        results = self.kb_manager.search_fragments(keyword)
        print(f"\n🔍 搜索 '{keyword}' — 找到 {len(results)} 条:")
        print("-" * 70)
        if not results:
            print("  (无结果)")
        for idx, frag in results[:30]:
            snippet = frag[:70] + '...' if len(frag) > 70 else frag
            print(f"  [{idx:>4}] {snippet}")
        if len(results) > 30:
            print(f"  ... 还有 {len(results)-30} 条，请使用更精确的关键字")
        print()

    def _kb_add(self, text: str):
        """向知识库添加一条新片段"""
        if not text:
            print("❌ 用法: kbadd <片段内容>")
            print("   例:  kbadd 北京是中国的首都，位于华北平原北部")
            return

        success, msg = self.kb_manager.add_fragment(text)
        if success:
            print(f"✅ {msg}")
            self._reload_kb_and_rebuild()
        else:
            print(f"❌ {msg}")

    def _kb_delete(self, args: list):
        """删除知识库片段"""
        if not args:
            print("❌ 用法:")
            print("   kbdel <序号>          按序号删除")
            print("   kbdel keyword <关键字> 删除所有包含关键字的片段")
            return

        if args[0] == 'keyword':
            keyword = ' '.join(args[1:]) if len(args) > 1 else ''
            if not keyword:
                print("❌ 请提供关键字，例: kbdel keyword 北京")
                return

            results = self.kb_manager.search_fragments(keyword)
            if not results:
                print(f"  没有找到包含 '{keyword}' 的片段")
                return

            print(f"\n⚠️  将删除以下 {len(results)} 条片段:")
            for idx, frag in results[:10]:
                print(f"  [{idx}] {frag[:60]}...")
            if len(results) > 10:
                print(f"  ... 共 {len(results)} 条")

            confirm = input("\n确认删除？(输入 yes 确认): ").strip().lower()
            if confirm != 'yes':
                print("  已取消")
                return

            success, msg, count = self.kb_manager.delete_by_keyword(keyword)
            if success:
                print(f"✅ {msg}")
                self._reload_kb_and_rebuild()
            else:
                print(f"❌ {msg}")

        elif args[0].isdigit():
            index = int(args[0])
            items, _ = self.kb_manager.list_fragments(page=1, page_size=99999)
            target = next((f for i, f in items if i == index), None)
            if target:
                print(f"\n⚠️  将删除第 {index} 条:")
                print(f"   {target[:80]}")
                confirm = input("确认删除？(输入 yes 确认): ").strip().lower()
                if confirm != 'yes':
                    print("  已取消")
                    return

            success, msg = self.kb_manager.delete_fragment(index)
            if success:
                print(f"✅ {msg}")
                self._reload_kb_and_rebuild()
            else:
                print(f"❌ {msg}")
        else:
            print("❌ 序号必须是数字，或使用 'keyword' 按关键字删除")

    def _kb_update(self, args: list):
        """更新知识库某条片段"""
        if len(args) < 2 or not args[0].isdigit():
            print("❌ 用法: kbupdate <序号> <新内容>")
            print("   例:  kbupdate 3 北京是中华人民共和国的首都")
            return

        index = int(args[0])
        new_text = ' '.join(args[1:])

        success, msg = self.kb_manager.update_fragment(index, new_text)
        if success:
            print(f"✅ {msg}")
            self._reload_kb_and_rebuild()
        else:
            print(f"❌ {msg}")

    def _kb_import(self, file_path: str):
        """从外部 TXT 文件导入知识片段（每行一条）"""
        if not file_path:
            print("❌ 用法: kbimport <文件路径>")
            print("   例:  kbimport D:\\data\\new_knowledge.txt")
            print("   文件格式：每行一条知识片段")
            return

        print(f"\n📥 从文件导入: {file_path}")
        success, msg, count = self.kb_manager.import_from_file(file_path)
        if success:
            print(f"✅ {msg}")
            if count > 0:
                self._reload_kb_and_rebuild()
        else:
            print(f"❌ {msg}")

    def _kb_stats(self):
        """显示知识库统计信息"""
        stats = self.kb_manager.get_stats()
        print("\n📊 知识库统计:")
        print(f"  - 总片段数:    {stats['total_fragments']}")
        print(f"  - 总字符数:    {stats['total_chars']}")
        print(f"  - 平均片段长度: {stats['avg_fragment_len']} 字符")
        print(f"  - 文件路径:    {stats['kb_path']}")
        cache_valid = self.persistence.is_cache_valid()
        print(f"  - 向量缓存:    {'✅ 有效' if cache_valid else '❌ 无缓存'}")
        if cache_valid:
            info = self.persistence.get_cache_info()
            print(f"  - 缓存大小:    {info['total_size_mb']:.2f} MB")
        print()

    def _kb_clear_cache(self):
        """清除知识库向量缓存，下次重建时会重新计算"""
        print("\n🗑️  准备清除向量缓存...")
        if not self.persistence.is_cache_valid():
            print("   当前无有效缓存，无需清除")
            return
        info = self.persistence.get_cache_info()
        print(f"   缓存大小: {info['total_size_mb']:.2f} MB")
        confirm = input("   确认清除？(输入 yes 确认): ").strip().lower()
        if confirm != 'yes':
            print("   已取消")
            return
        success = self.persistence.clear_cache()
        if success:
            print("✅ 向量缓存已清除")
            print("   下次操作或重启时将自动重建缓存")
        else:
            print("❌ 清除失败，请手动删除缓存目录下的文件")
        print()

    # ------------------------------------------------------------------
    # 知识库变更后重新加载并重建索引
    # ------------------------------------------------------------------

    def _reload_kb_and_rebuild(self):
        """
        知识库文件发生变更后，重新读取片段并重建向量索引。
        base_knowledge_fragments 只存知识库原始内容；
        文档片段通过 _build_tagged_doc_chunks() 追加，两者始终隔离。
        """
        print("\n🔄 正在重新加载知识库并重建索引...")

        try:
            with open(self.config['knowledge_base'], 'r', encoding='utf-8') as f:
                content = f.read()

            self.base_knowledge_fragments = self.document_manager.chunk_text(content)
            self.knowledge_fragments = self.base_knowledge_fragments.copy()

            # 修复: 重建时使用打标签的文档片段，保持标签一致性
            if self.has_loaded_documents:
                self.knowledge_fragments.extend(self._build_tagged_doc_chunks())

            print(f"   片段数: {len(self.knowledge_fragments)}")
            print("   正在重新计算向量...")

            start_time = time.time()
            self.fragment_embeddings = self.embedder.encode(
                self.knowledge_fragments,
                batch_size=32,
                show_progress_bar=False
            )
            embed_time = time.time() - start_time
            print(f"   ✅ 向量计算完成 ({embed_time:.2f}s)")

            self.persistence.clear_cache()
            self.persistence.save_embeddings(self.fragment_embeddings)
            self.persistence.save_fragments(self.knowledge_fragments)

            self.build_faiss_index()
            print("   ✅ 知识库已更新完毕\n")

        except Exception as e:
            print(f"❌ 重新加载失败: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # 原有辅助方法
    # ------------------------------------------------------------------

    def _show_help(self):
        """显示完整帮助"""
        print(self.command_parser.get_command_help())
        print()

    def _display_response(self, answer: str, sources: list, response_time: float):
        """显示 LLM 响应"""
        print("\n" + "=" * 80)
        print("💡 回答:")
        print("=" * 80)
        print(answer)
        print("=" * 80)

        print(f"\n📚 参考资料 ({len(sources)} 项):")
        for i, source in enumerate(sources, 1):
            snippet = source[:80] + "..." if len(source) > 80 else source
            print(f"  {i}. {snippet}")

        print(f"\n⏱️  响应时间: {response_time:.2f}s")
        print("-" * 80)

    def _show_document_help(self):
        """显示文档相关命令"""
        print("\n📄 文档相关命令:")
        print("  - 'load <文件路径>': 加载指定文档")
        print("    支持格式: .txt, .md, .docx, .pdf")
        print("    例: load C:\\path\\to\\document.pdf")
        print("  - 'loaddir': 加载文档目录中的所有文档")
        print("  - 'docs'/'cache': 显示已加载文档统计")
        print()

    def _handle_load_document(self, file_path: str):
        """处理单个文档加载"""
        if not file_path:
            print("❌ 请指定文件路径")
            print("   用法: load <文件路径>")
            return

        print(f"\n📥 加载文档: {file_path}")
        success, msg, paragraphs = self.document_manager.load_document(file_path)

        if success:
            print(f"✅ {msg}")
            self.has_loaded_documents = True
            self._rebuild_knowledge_base()
        else:
            print(f"❌ {msg}")

    def _handle_load_directory(self):
        """处理目录加载"""
        doc_dir = self.config['documents_dir']
        print(f"\n📁 加载目录: {doc_dir}")

        results = self.document_manager.load_documents_from_directory(doc_dir)

        if results:
            print(f"\n✅ 加载完成:")
            for doc_name, (success, msg, _) in results.items():
                status = "✅" if success else "❌"
                print(f"  {status} {doc_name}: {msg}")

            self.has_loaded_documents = True
            self._rebuild_knowledge_base()
        else:
            print(f"⚠️  目录中未找到支持的文件")

    def _show_document_stats(self):
        """显示文档统计信息"""
        stats = self.document_manager.get_document_stats()

        print("\n📊 文档统计信息:")
        print(f"  - 已加载文档数: {stats['total_documents']}")
        print(f"  - 总段落数: {stats['total_paragraphs']}")
        print(f"  - 总字符数: {stats['total_characters']}")

        if stats['documents']:
            print(f"\n  📄 文档明细:")
            for doc_name, doc_info in stats['documents'].items():
                print(f"    • {doc_name}")
                print(f"      格式: {doc_info['format']}")
                print(f"      段落: {doc_info['paragraphs']}")
                print(f"      字符: {doc_info['characters']}")
        else:
            print("  ⚠️  还未加载任何文档")

        print()

    def _rebuild_knowledge_base(self):
        """
        加载文档后重建知识库。
        base_knowledge_fragments 只存知识库原始内容，不受影响；
        文档片段带文件名标签追加到 knowledge_fragments。
        """
        print("\n🔄 重建知识库...")

        self.knowledge_fragments = self.base_knowledge_fragments.copy()

        # 修复: 使用统一的打标签函数
        tagged_doc_chunks = self._build_tagged_doc_chunks()
        if tagged_doc_chunks:
            self.knowledge_fragments.extend(tagged_doc_chunks)
            # 计算纯内容片段数（去掉索引片段）
            content_chunks = [c for c in tagged_doc_chunks if "用户已加载文档" not in c]
            print(f"   包含原始知识库片段: {len(self.base_knowledge_fragments)}")
            print(f"   包含文档内容片段: {len(content_chunks)}")
            print(f"   总计片段数: {len(self.knowledge_fragments)}")

        print(f"   正在计算向量 ({len(self.knowledge_fragments)} 个片段)...")
        start_time = time.time()

        self.fragment_embeddings = self.embedder.encode(
            self.knowledge_fragments,
            batch_size=32,
            show_progress_bar=False
        )

        embed_time = time.time() - start_time
        print(f"   ✅ 向量计算完成 ({embed_time:.2f}s)")

        self.build_faiss_index()
        print("   ✅ 知识库重建完成")

    def _list_knowledge_base(self):
        """列出知识库内容（简要）"""
        print(f"\n📚 知识库内容 ({len(self.knowledge_fragments)} 项):")
        display_count = min(20, len(self.knowledge_fragments))
        for i, fragment in enumerate(self.knowledge_fragments[:display_count], 1):
            snippet = fragment[:60] + "..." if len(fragment) > 60 else fragment
            print(f"  {i}. {snippet}")

        if len(self.knowledge_fragments) > display_count:
            print(f"  ... 还有 {len(self.knowledge_fragments) - display_count} 项")
        print(f"\n💡 使用 'kblist' 可分页浏览并管理知识库\n")

    def _cleanup_on_exit(self):
        """
        系统退出时的清理逻辑。
        清除文档后还原为纯知识库片段，base_knowledge_fragments 完全不受影响。
        """
        print("\n🧹 正在清理...")

        if self.has_loaded_documents:
            print("   清除文档产生的向量存储...")
            self.document_manager.clear_documents()
            self.has_loaded_documents = False

            # 直接用 base 还原，知识库向量不受影响
            self.knowledge_fragments = self.base_knowledge_fragments.copy()
            self.fragment_embeddings = self.embedder.encode(
                self.knowledge_fragments,
                batch_size=32,
                show_progress_bar=False
            )

            self.build_faiss_index()
            print("   ✅ 已恢复原始知识库")

        print("   ✅ 清理完成")


def main():
    """主函数"""
    try:
        rag_system = RAGSystem()
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()