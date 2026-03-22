"""
命令解析和执行系统
提供灵活的命令识别、验证和执行能力
"""
import logging
from typing import Dict, Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """命令类型"""
    SYSTEM = "system"       # 系统命令 (help, quit, clear)
    DOCUMENT = "document"   # 文档命令 (load, loaddir, docs)
    KNOWLEDGE = "knowledge" # 知识库命令 (包含查询与管理: kb, kblist, kbadd, kbimport...)
    CACHE = "cache"         # 缓存命令 (cache)
    QUERY = "query"         # 查询命令 (普通问题)


class CommandParser:
    """
    命令解析器
    支持多种命令格式：command, -command, --command, /command
    """

    def __init__(self):
        """初始化命令解析器"""
        # 命令配置：命令名 -> (类型, 别名, 是否需要参数, 参数说明)
        self.commands = {
            'help': {
                'type': CommandType.SYSTEM,
                'aliases': ['h', '帮助', '?'],
                'requires_args': False,
                'description': '显示帮助信息'
            },
            'quit': {
                'type': CommandType.SYSTEM,
                'aliases': ['exit', 'q', '退出'],
                'requires_args': False,
                'description': '退出程序'
            },
            'clear': {
                'type': CommandType.SYSTEM,
                'aliases': ['清屏', 'cls'],
                'requires_args': False,
                'description': '清屏'
            },
            'kb': {
                'type': CommandType.KNOWLEDGE,
                'aliases': ['知识库'],
                'requires_args': False,
                'description': '列出知识库内容'
            },
            'cache': {
                'type': CommandType.CACHE,
                'aliases': ['缓存'],
                'requires_args': False,
                'description': '显示缓存信息'
            },
            'doc': {
                'type': CommandType.DOCUMENT,
                'aliases': ['文档'],
                'requires_args': False,
                'description': '显示文档相关命令'
            },
            'load': {
                'type': CommandType.DOCUMENT,
                'aliases': ['加载'],
                'requires_args': True,
                'arg_description': '<文件路径>',
                'description': '加载指定文件'
            },
            'loaddir': {
                'type': CommandType.DOCUMENT,
                'aliases': ['加载目录'],
                'requires_args': False,
                'description': '加载文档目录中的所有文档'
            },
            'docs': {
                'type': CommandType.DOCUMENT,
                'aliases': ['文档统计'],
                'requires_args': False,
                'description': '显示已加载文档统计信息'
            },
            'kblist': {
                'type': CommandType.KNOWLEDGE,  # 修改此处
                'aliases': [],
                'requires_args': False,
                'arg_description': '[页码] [关键字]',
                'description': '分页浏览知识库片段，可按关键字筛选'
            },
            'kbsearch': {
                'type': CommandType.KNOWLEDGE,  # 修改此处
                'aliases': [],
                'requires_args': True,
                'arg_description': '<关键字>',
                'description': '搜索包含关键字的知识库片段'
            },
            'kbadd': {
                'type': CommandType.KNOWLEDGE,  # 修改此处
                'aliases': [],
                'requires_args': True,
                'arg_description': '<片段内容>',
                'description': '向知识库末尾追加一条新片段'
            },
            'kbdel': {
                'type': CommandType.KNOWLEDGE,  # 修改此处
                'aliases': [],
                'requires_args': True,
                'arg_description': '<序号> 或 keyword <关键字>',
                'description': '按序号或关键字删除知识库片段'
            },
            'kbupdate': {
                'type': CommandType.KNOWLEDGE,  # 修改此处
                'aliases': [],
                'requires_args': True,
                'arg_description': '<序号> <新内容>',
                'description': '更新指定序号的知识库片段'
            },
            'kbimport': {
                'type': CommandType.KNOWLEDGE,  # 修改此处
                'aliases': [],
                'requires_args': True,
                'arg_description': '<文件路径>',
                'description': '从 TXT/MD 文件导入片段到知识库'
            },
            'kbstats': {
                'type': CommandType.KNOWLEDGE,
                'aliases': [],
                'requires_args': False,
                'description': '查看知识库统计信息（片段数、缓存等）'
            },
            'kbclearcache': {
                'type': CommandType.KNOWLEDGE,
                'aliases': [],
                'requires_args': False,
                'description': '清除知识库向量缓存，下次操作时自动重建'
            },
        }

        # 构建别名映射
        self.alias_map = {}
        for cmd, config in self.commands.items():
            self.alias_map[cmd] = cmd
            for alias in config.get('aliases', []):
                self.alias_map[alias] = cmd

    def parse(self, user_input: str) -> Tuple[Optional[str], CommandType, List[str]]:
        """
        解析用户输入

        Args:
            user_input: 用户输入字符串

        Returns:
            (命令名, 命令类型, 参数列表) 或 (None, CommandType.QUERY, [原始输入])
        """
        if not user_input.strip():
            return None, CommandType.QUERY, []

        # 移除前导的 -, --, / 符号
        cleaned_input = self._clean_command_prefix(user_input.strip())

        # 分割命令和参数
        parts = cleaned_input.split(maxsplit=1)
        cmd_part = parts[0].lower()
        args = [parts[1]] if len(parts) > 1 else []

        # 查找命令
        actual_cmd = self.alias_map.get(cmd_part)

        if actual_cmd is None:
            # 不是有效命令，当作查询处理
            return None, CommandType.QUERY, [user_input]

        cmd_config = self.commands[actual_cmd]
        cmd_type = cmd_config['type']

        return actual_cmd, cmd_type, args

    @staticmethod
    def _clean_command_prefix(text: str) -> str:
        """移除命令前缀"""
        # 移除前导的 -, --, /
        while text and text[0] in ['-', '/']:
            text = text[1:]
        return text

    def is_command(self, user_input: str) -> bool:
        """检查输入是否为有效命令"""
        cmd, _, _ = self.parse(user_input)
        return cmd is not None

    def get_command_help(self, cmd: Optional[str] = None) -> str:
        """
        获取命令帮助信息

        Args:
            cmd: 命令名，None则返回所有命令帮助

        Returns:
            帮助文本
        """
        if cmd is None:
            return self._get_all_help()
        else:
            return self._get_single_help(cmd)

    def _get_all_help(self) -> str:
        """获取所有命令帮助"""
        help_text = "\n📖 完整命令列表:\n"
        help_text += "=" * 80 + "\n\n"

        # 按类型分组
        by_type = {}
        for cmd, config in self.commands.items():
            cmd_type = config['type']
            if cmd_type not in by_type:
                by_type[cmd_type] = []
            by_type[cmd_type].append((cmd, config))

        # 显示各类命令
        type_names = {
            CommandType.SYSTEM: "🔧 系统命令",
            CommandType.KNOWLEDGE: "📚 知识库命令 (查询与管理)", # 更新描述
            CommandType.DOCUMENT: "📄 文档命令",
            CommandType.CACHE: "💾 缓存命令",
        }

        # 调整显示顺序，确保知识库在一起
        display_order = [
            CommandType.SYSTEM,
            CommandType.KNOWLEDGE,
            CommandType.DOCUMENT,
            CommandType.CACHE
        ]

        for cmd_type in display_order:
            if cmd_type in by_type:
                help_text += f"{type_names[cmd_type]}:\n"
                for cmd, config in by_type[cmd_type]:
                    aliases = "/".join([cmd] + config.get('aliases', []))
                    arg_part = f" {config.get('arg_description', '')}" if config['requires_args'] else ""
                    help_text += f"  • {aliases}{arg_part}\n"
                    help_text += f"    {config['description']}\n"
                help_text += "\n"

        help_text += "=" * 80 + "\n"
        help_text += "💡 使用提示:\n"
        help_text += "  - 命令支持多种格式: 'help', '-help', '--help', '/help'\n"
        help_text += "  - 直接输入问题开始对话 (不需要加命令前缀)\n"
        help_text += "  - 按 Ctrl+C 快速退出\n"
        help_text += "  - 文档向量仅在会话期间保存，退出时自动清除\n"

        return help_text

    def _get_single_help(self, cmd: str) -> str:
        """获取单个命令的帮助"""
        # 处理别名
        cmd = self.alias_map.get(cmd.lower(), cmd.lower())

        if cmd not in self.commands:
            return f"❌ 未知命令：{cmd}"

        config = self.commands[cmd]
        help_text = f"\n📖 命令：{cmd}\n"
        help_text += f"描述：{config['description']}\n"

        if config.get('aliases'):
            help_text += f"别名：{', '.join(config['aliases'])}\n"

        if config['requires_args']:
            help_text += f"用法：{cmd} {config.get('arg_description', '<参数>')}\n"
        else:
            help_text += f"用法：{cmd}\n"

        help_text += "\n"
        return help_text


class CommandValidator:
    """
    命令验证器
    验证命令的有效性和完整性
    """

    def __init__(self, parser: CommandParser):
        """初始化验证器"""
        self.parser = parser

    def validate(self, cmd: str, args: List[str]) -> Tuple[bool, str]:
        """
        验证命令和参数

        Args:
            cmd: 命令名
            args: 参数列表

        Returns:
            (是否有效, 错误消息)
        """
        if cmd not in self.parser.commands:
            return False, f"❌ 未知命令：{cmd}"

        config = self.parser.commands[cmd]

        # 检查参数要求
        if config['requires_args'] and not args:
            arg_desc = config.get('arg_description', '<参数>')
            return False, f"❌ 命令'{cmd}'需要参数\n用法：{cmd} {arg_desc}"

        return True, ""

    def suggest_command(self, user_input: str) -> Optional[str]:
        """
        为不存在的命令提供建议

        Args:
            user_input: 用户输入

        Returns:
            建议的命令名
        """
        cleaned = self.parser._clean_command_prefix(user_input.strip())
        parts = cleaned.split(maxsplit=1)
        if not parts:
            return None

        input_cmd = parts[0].lower()

        # 简单的相似度匹配（编辑距离）
        best_match = None
        best_distance = 3  # 最大距离

        for cmd in self.parser.commands.keys():
            distance = self._edit_distance(input_cmd, cmd)
            if distance < best_distance:
                best_distance = distance
                best_match = cmd

        return best_match

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return CommandValidator._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
