# -*- coding: utf-8 -*-
import json
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Any, Union
from copy import deepcopy


class IOUtilsError(Exception):
    pass


class JSONLParseError(IOUtilsError):
    pass


def prepare_output_path(path: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def save_json(data: Union[List[Dict], Dict], path: str, indent: int = 2) -> None:
    """保存JSON，支持列表和字典"""
    if not isinstance(data, (list, dict)):
        raise TypeError(f"data 必须是列表或字典，实际类型: {type(data).__name__}")

    try:
        text = json.dumps(data, ensure_ascii=False, indent=indent, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise TypeError(f"数据序列化失败: {e}") from e

    output_path = prepare_output_path(path)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(text)
        temp_path.replace(output_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"保存JSON文件失败 {path}: {e}") from e


def save_csv(data: List[Dict], path: str, include_header: bool = True) -> None:
    """
    保存CSV文件，所有记录必须是字典类型
    遇到非字典记录立即报错，不静默跳过
    """
    if not data:
        raise ValueError("无数据可导出")

    # 检查所有记录是否为字典
    non_dict_indices = [i for i, record in enumerate(data) if not isinstance(record, dict)]
    if non_dict_indices:
        raise TypeError(
            f"数据中包含非字典记录，索引: {non_dict_indices[:10]}"
            f"{'...' if len(non_dict_indices) > 10 else ''}"
        )

    # 收集所有字段
    fieldnames = list(
        dict.fromkeys(
            key
            for record in data
            if isinstance(record, dict)
            for key in record.keys()
        )
    )

    if not fieldnames:
        raise ValueError("所有记录都没有字段")

    output_path = prepare_output_path(path)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if include_header:
                writer.writeheader()
            for record in data:
                row = {key: csv_cell(record.get(key)) for key in fieldnames}
                writer.writerow(row)
        temp_path.replace(output_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"保存CSV文件失败 {path}: {e}") from e


def load_existing_jsonl(path: str) -> List[Dict]:
    """加载JSONL文件"""
    file_path = Path(path)
    records = []

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise JSONLParseError(
                        f"{file_path} 第 {line_number} 行 JSON 解析失败: {e}\n"
                        f"错误内容: {line[:200]}{'...' if len(line) > 200 else ''}"
                    ) from e
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{file_path} 第 {line_number} 行必须是 JSON 对象，"
                        f"实际类型: {type(record).__name__}"
                    )
                records.append(record)
    except FileNotFoundError:
        return []

    return records


def safe_append_jsonl(data: List[Dict], path: str) -> None:
    """原子追加数据到JSONL文件"""
    if not data:
        return

    # 读取已有数据
    existing = load_existing_jsonl(path)

    # 合并数据
    all_data = existing + data

    # 序列化所有数据
    try:
        lines = [
            json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
            for item in all_data
        ]
    except (TypeError, ValueError) as e:
        raise TypeError(f"数据序列化失败: {e}") from e

    output_path = prepare_output_path(path)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        temp_path.replace(output_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"原子追加JSONL文件失败 {path}: {e}") from e


def append_jsonl(data: List[Dict], path: str) -> None:
    """普通追加（非原子），保留兼容性"""
    if not data:
        return

    try:
        lines = [
            json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
            for item in data
        ]
    except (TypeError, ValueError) as e:
        raise TypeError(f"数据序列化失败: {e}") from e

    output_path = prepare_output_path(path)

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
            if content and not content.endswith("\n"):
                with open(output_path, "a", encoding="utf-8") as f_append:
                    f_append.write("\n")

    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        raise IOError(f"追加JSONL文件失败 {path}: {e}") from e


def to_output_record(result: Dict) -> Dict:
    if not isinstance(result, dict):
        raise TypeError(f"result 必须是字典，实际类型: {type(result).__name__}")
    return deepcopy(result)