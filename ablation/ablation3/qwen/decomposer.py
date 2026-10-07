# -*- coding: utf-8 -*-
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


class PrecomputedDataNotFoundError(Exception):
    """预计算数据未找到异常"""
    pass


class PrecomputedDataError(Exception):
    """预计算数据错误异常"""
    pass


def _load_csv_with_encoding(file_path: str) -> pd.DataFrame:
    """
    使用 utf-8-sig 编码加载 CSV 文件

    Args:
        file_path: CSV 文件路径

    Returns:
        DataFrame

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 读取失败
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        return pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception as e:
        raise ValueError(f"读取文件失败 {file_path}: {e}")


def _validate_subquestion_data(df: pd.DataFrame) -> None:
    """
    校验子问题数据框的列和内容

    Args:
        df: 子问题数据框

    Raises:
        ValueError: 列缺失或数据无效
    """
    # 检查必需列
    if "sub_questions" not in df.columns:
        raise ValueError(
            f"子问题文件缺少必需列: 'sub_questions'\n"
            f"当前列: {list(df.columns)}"
        )

    if "original" not in df.columns:
        raise ValueError(
            f"子问题文件缺少必需列: 'original'\n"
            f"当前列: {list(df.columns)}"
        )

    # 检查空值
    if df["sub_questions"].isna().any():
        null_rows = df[df["sub_questions"].isna()].index.tolist()
        raise ValueError(f"子问题文件第 {null_rows} 行的 'sub_questions' 列为空")

    if df["original"].isna().any():
        null_rows = df[df["original"].isna()].index.tolist()
        raise ValueError(f"子问题文件第 {null_rows} 行的 'original' 列为空")


def _parse_sub_questions(json_str: str, file_path: str, row_idx: int) -> List[str]:
    """
    解析子问题 JSON 字符串，并验证结构

    Args:
        json_str: JSON 字符串
        file_path: 文件路径（用于错误信息）
        row_idx: 行号（用于错误信息）

    Returns:
        子问题列表

    Raises:
        ValueError: JSON 格式错误或结构不符合要求
    """
    if not json_str or pd.isna(json_str):
        return []  # 空值视为空列表

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"子问题文件 {file_path} 第 {row_idx + 1} 行 JSON 解析失败: {e}\n"
            f"内容: {json_str[:200]}{'...' if len(json_str) > 200 else ''}"
        )

    # 验证结构
    if not isinstance(parsed, list):
        raise ValueError(
            f"子问题文件 {file_path} 第 {row_idx + 1} 行: "
            f"期望 JSON 数组，实际类型: {type(parsed).__name__}\n"
            f"内容: {json_str[:200]}{'...' if len(json_str) > 200 else ''}"
        )

    # 验证每个元素都是字符串
    sub_questions = []
    for q_idx, q in enumerate(parsed):
        if not isinstance(q, str):
            raise ValueError(
                f"子问题文件 {file_path} 第 {row_idx + 1} 行 "
                f"第 {q_idx + 1} 个子问题不是字符串，实际类型: {type(q).__name__}\n"
                f"内容: {q}"
            )
        q_clean = q.strip()
        if q_clean:  # 只保留非空字符串
            sub_questions.append(q_clean)

    return sub_questions


def load_precomputed_data(
        original_question: str,
        subquestion_csv_path: str
) -> Dict[str, Any]:
    """
    从已处理好的 CSV 文件中加载子问题（数据已经是中文，无需翻译）

    Args:
        original_question: 原始问题
        subquestion_csv_path: 子问题文件路径

    Returns:
        包含 original, sub_questions 的字典

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 数据校验失败、列缺失、数据无效
        PrecomputedDataNotFoundError: 未找到匹配的问题
        PrecomputedDataError: 数据异常
    """
    # 加载文件
    df_sub = _load_csv_with_encoding(subquestion_csv_path)

    # 校验数据
    _validate_subquestion_data(df_sub)

    # 去除首尾空白
    original_clean = original_question.strip()
    if not original_clean:
        raise ValueError("原始问题不能为空")

    # 清理原始问题列
    df_sub["original_clean"] = df_sub["original"].astype(str).str.strip()

    # 查找匹配的原始问题
    matches = df_sub[df_sub["original_clean"] == original_clean]

    # 处理匹配结果
    if len(matches) == 0:
        # 未找到匹配
        close_matches = df_sub[
            df_sub["original_clean"].str.lower() == original_clean.lower()
            ]
        if len(close_matches) > 0:
            raise PrecomputedDataNotFoundError(
                f"未找到精确匹配的问题: '{original_question}'\n"
                f"找到大小写不同的近似匹配: {close_matches['original'].tolist()}\n"
                f"请检查原始问题是否完全相同（包括标点和空格）"
            )
        else:
            raise PrecomputedDataNotFoundError(
                f"未找到预计算数据: '{original_question}'\n"
                f"请确认该问题已包含在 {subquestion_csv_path} 中"
            )

    if len(matches) > 1:
        raise PrecomputedDataError(
            f"找到多个相同的问题: '{original_question}'\n"
            f"匹配的行号: {matches.index.tolist()}"
        )

    # 获取匹配的那一行
    match_idx = matches.index[0]
    sub_row = matches.iloc[0]
    sub_json = sub_row["sub_questions"]

    # 解析子问题
    sub_questions = _parse_sub_questions(
        str(sub_json) if not pd.isna(sub_json) else "",
        subquestion_csv_path,
        sub_row.name
    )

    return {
        "original": original_question,
        "sub_questions": sub_questions,
        "source": "precomputed",
        "file": subquestion_csv_path
    }


def load_precomputed_batch(
        questions: List[str],
        subquestion_csv_path: str,
        skip_missing: bool = False
) -> List[Dict[str, Any]]:
    """
    批量加载预计算数据

    Args:
        questions: 原始问题列表
        subquestion_csv_path: 子问题文件路径
        skip_missing: 是否跳过缺失的问题

    Returns:
        加载结果列表
    """
    results = []
    errors = []

    for q in questions:
        try:
            result = load_precomputed_data(q, subquestion_csv_path)
            results.append(result)
        except (PrecomputedDataNotFoundError, PrecomputedDataError) as e:
            if skip_missing:
                errors.append({"question": q, "error": str(e)})
            else:
                raise
        except Exception as e:
            if skip_missing:
                errors.append({"question": q, "error": f"未知错误: {e}"})
            else:
                raise

    if errors and skip_missing:
        print(f"警告: {len(errors)} 个问题加载失败")
        for err in errors:
            print(f"  - {err['question']}: {err['error']}")

    return results