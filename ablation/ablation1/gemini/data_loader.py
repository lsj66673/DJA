# -*- coding: utf-8 -*-
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional


def get_desktop_translated_path() -> Path:
    """获取桌面上的翻译文件路径"""
    desktop = Path.home() / "Desktop"
    return desktop / "Advbench_translated.csv"


def get_desktop_subquestion_path() -> Path:
    """获取桌面上的子问题文件路径"""
    desktop = Path.home() / "Desktop"
    return desktop / "subquestion.csv"


def _read_translated_csv(file_path: Path) -> List[Dict[str, Any]]:
    """
    读取翻译CSV文件，返回每条记录

    Raises:
        ValueError: 当必需列缺失或数据无效时
    """
    data = []

    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"文件 {file_path} 没有列头")

        # 检测实际列名
        fieldnames_lower = {name.lower(): name for name in reader.fieldnames}

        # 查找关键列
        original_col = None
        translated_col = None

        for col in ['original', 'goal', 'prompt']:
            if col in fieldnames_lower:
                original_col = fieldnames_lower[col]
                break

        for col in ['translated', 'chinese']:
            if col in fieldnames_lower:
                translated_col = fieldnames_lower[col]
                break

        if original_col is None:
            raise ValueError(f"翻译文件 {file_path} 缺少原始问题列 (original/goal/prompt)")
        if translated_col is None:
            raise ValueError(f"翻译文件 {file_path} 缺少翻译列 (translated/chinese)")

        for row_idx, row in enumerate(reader):
            original = row.get(original_col, "").strip()
            translated = row.get(translated_col, "").strip()

            # 检查空值
            if not original:
                raise ValueError(f"翻译文件 {file_path} 第 {row_idx + 1} 行原始问题为空")
            if not translated:
                raise ValueError(f"翻译文件 {file_path} 第 {row_idx + 1} 行翻译为空")

            data.append({
                "index": row_idx,
                "original": original,
                "translated": translated
            })

    return data


def _read_subquestion_csv(file_path: Path) -> List[Dict[str, Any]]:
    """
    读取子问题CSV文件，返回每条记录

    Raises:
        ValueError: 当必需列缺失或数据无效时
    """
    data = []

    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"文件 {file_path} 没有列头")

        # 检测实际列名
        fieldnames_lower = {name.lower(): name for name in reader.fieldnames}

        sub_col = None
        for col in ['sub_questions', 'sub_questions_json']:
            if col in fieldnames_lower:
                sub_col = fieldnames_lower[col]
                break

        if sub_col is None:
            raise ValueError(f"子问题文件 {file_path} 缺少子问题列 (sub_questions/sub_questions_json)")

        for row_idx, row in enumerate(reader):
            sub_questions_str = row.get(sub_col, "").strip()

            if not sub_questions_str:
                # 空字符串视为空列表
                sub_questions = []
            else:
                try:
                    parsed = json.loads(sub_questions_str)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"子问题文件 {file_path} 第 {row_idx + 1} 行 JSON 解析失败: {e}\n"
                        f"内容: {sub_questions_str[:200]}{'...' if len(sub_questions_str) > 200 else ''}"
                    )

                # 类型检查
                if not isinstance(parsed, list):
                    raise ValueError(
                        f"子问题文件 {file_path} 第 {row_idx + 1} 行 JSON 不是数组类型，"
                        f"实际类型: {type(parsed).__name__}"
                    )

                # 检查每个元素是否为字符串
                sub_questions = []
                for q_idx, q in enumerate(parsed):
                    if not isinstance(q, str):
                        raise ValueError(
                            f"子问题文件 {file_path} 第 {row_idx + 1} 行 "
                            f"第 {q_idx + 1} 个子问题不是字符串，实际类型: {type(q).__name__}"
                        )
                    q_clean = q.strip()
                    if q_clean:  # 过滤空字符串
                        sub_questions.append(q_clean)
                    # 空字符串直接跳过

            data.append({
                "index": row_idx,
                "sub_questions": sub_questions
            })

    return data


def load_precomputed_questions(translated_path: str, subquestion_path: str) -> List[Dict[str, Any]]:
    """
    加载预先翻译好的中文问题和预拆解的子问题

    Args:
        translated_path: Advbench_translated.csv 路径
        subquestion_path: subquestion.csv 路径

    Returns:
        List of dict，每个包含 index, original, translated, sub_questions, prompt

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 数据校验失败（行数不匹配、列缺失、数据无效等）
    """
    trans_path = Path(translated_path)
    sub_path = Path(subquestion_path)

    if not trans_path.exists():
        raise FileNotFoundError(f"翻译文件不存在: {trans_path}")
    if not sub_path.exists():
        raise FileNotFoundError(f"子问题文件不存在: {sub_path}")

    # 读取两个文件
    translated_data = _read_translated_csv(trans_path)
    sub_data = _read_subquestion_csv(sub_path)

    # 检查行数是否一致
    if len(translated_data) != len(sub_data):
        raise ValueError(
            f"文件行数不匹配:\n"
            f"  翻译文件 ({trans_path}): {len(translated_data)} 行\n"
            f"  子问题文件 ({sub_path}): {len(sub_data)} 行"
        )

    # 合并数据（按行号，但通过 original 校验确保对应关系正确）
    questions = []
    for i in range(len(translated_data)):
        trans_item = translated_data[i]
        sub_item = sub_data[i]

        # 校验：两个文件同一行的原始问题应该一致
        # 注意：subquestion.csv 可能没有 original 列，这里只做警告
        # 如果子问题文件有 original 列，可以启用以下校验：
        # if trans_item["original"] != sub_item.get("original", trans_item["original"]):
        #     raise ValueError(
        #         f"第 {i + 1} 行原始问题不匹配:\n"
        #         f"  翻译文件: {trans_item['original']}\n"
        #         f"  子问题文件: {sub_item.get('original', 'N/A')}"
        #     )

        questions.append({
            "index": trans_item["index"],
            "original": trans_item["original"],
            "translated": trans_item["translated"],
            "sub_questions": sub_item["sub_questions"],
            "prompt": trans_item["translated"]  # 兼容原有字段
        })

    return questions


def get_batch(questions: List[Dict[str, Any]], batch_size: int, offset: int = 0):
    """获取批次"""
    end = min(offset + batch_size, len(questions))
    return questions[offset:end]


# ===================== 以下为已弃用函数（保留仅为兼容性，建议删除） =====================

def get_desktop_advbench_path() -> Path:
    """已弃用，保留仅为兼容性"""
    desktop = Path.home() / "Desktop"
    return desktop / "Advbench.csv"


def load_advbench(path: str) -> List[Dict[str, Any]]:
    """已弃用，请使用 load_precomputed_questions"""
    raise DeprecationWarning("此函数已弃用，请使用 load_precomputed_questions")


def _normalize_prompt(row: Dict[str, Any]) -> Optional[str]:
    """已弃用"""
    raise DeprecationWarning("此函数已弃用")


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    """已弃用"""
    raise DeprecationWarning("此函数已弃用")


def _load_json(path: Path) -> List[Dict[str, Any]]:
    """已弃用"""
    raise DeprecationWarning("此函数已弃用")