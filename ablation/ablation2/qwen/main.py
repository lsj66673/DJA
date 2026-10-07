# -*- coding: utf-8 -*-
"""
Advbench 流水线 - 主入口（支持多次尝试，成功即停止）
"""
import argparse
import sys
import traceback
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import shutil
import os

import pandas as pd

import config
from config import DEFAULT_BATCH_SIZE
from pipeline import run_single, PipelineResult, AnswerStatus
from io_utils import save_json, save_csv, load_existing_jsonl, safe_append_jsonl, prepare_output_path


class ProcessStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INVALID = "invalid"
    PARTIAL = "partial"


@dataclass
class AttemptRecord:
    attempt_number: int
    result: Optional[PipelineResult] = None
    error: Optional[str] = None
    is_success: bool = False
    timestamp: float = field(default_factory=time.time)


class DataLoader:
    """数据加载器"""

    @staticmethod
    def load_subquestions(csv_path: Path) -> List[Dict[str, Any]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"子问题文件不存在: {csv_path}")

        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except Exception as e:
            raise ValueError(f"读取CSV失败: {e}")

        required_cols = ['original']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV缺少必需列: {missing_cols}\n"
                f"实际列: {list(df.columns)}\n"
                f"请确保CSV包含 'original' 列"
            )

        questions = []
        for idx in range(len(df)):
            row = df.iloc[idx]

            original = row.get('original')
            if pd.isna(original) or not str(original).strip():
                raise ValueError(f"第 {idx + 1} 行的 'original' 列为空")
            original = str(original).strip()

            sub_questions_str = row.get('sub_questions')
            if pd.isna(sub_questions_str) or not str(sub_questions_str).strip():
                # 没有子问题列，直接用原始问题作为唯一子问题
                sub_questions = [original]
            else:
                try:
                    parsed = json.loads(str(sub_questions_str))
                    if not isinstance(parsed, list):
                        raise ValueError(f"第 {idx + 1} 行 'sub_questions' 不是JSON数组")
                    sub_questions = [str(q).strip() for q in parsed if q and str(q).strip()]
                except json.JSONDecodeError as e:
                    raise ValueError(f"第 {idx + 1} 行 'sub_questions' JSON解析失败: {e}")

            if not sub_questions:
                print(f"  [WARN] 第 {idx + 1} 行子问题为空，跳过")
                continue

            questions.append({
                "index": idx,
                "original": original,
                "sub_questions": sub_questions,
                "prompt": original
            })

        if not questions:
            raise ValueError("没有加载到任何有效问题")

        return questions

    @staticmethod
    def get_file_hash(file_path: Path) -> str:
        if not file_path.exists():
            return ""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()[:8]
        except:
            return "unknown"


class ResultManager:
    """结果管理器 - 完整续跑校验"""

    def __init__(self, output_dir: Path, run_config: Dict[str, Any], max_attempts: int = 10):
        self.output_dir = output_dir
        self.run_config = run_config
        self.max_attempts = max_attempts
        self.jsonl_path = output_dir / "results.jsonl"
        self.json_path = output_dir / "results.json"
        self.csv_path = output_dir / "results.csv"
        self.summary_path = output_dir / "summary.json"

        self.records: List[Dict[str, Any]] = []
        self.processed_indices: Set[int] = set()
        self.failed_indices: Set[int] = set()
        self.partial_indices: Set[int] = set()
        self.skipped_indices: Set[int] = set()

        self._load_existing()

    def _get_config_signature(self) -> str:
        """获取当前配置签名（含 interaction_mode）"""
        require_judge = self.run_config.get("run_jailbreak", True)
        base = config.get_config_signature(require_judge)
        mode = self.run_config.get("interaction_mode", "sequential")
        return f"{base}|mode={mode}"

    def _verify_all_records_config(self, records: List[Dict]) -> Tuple[bool, str]:
        """验证所有记录的配置签名是否一致"""
        if not records:
            return True, "无记录"

        signatures = set()
        for r in records:
            sig = r.get("run_config", {}).get("config_signature")
            if sig:
                signatures.add(sig)

        if len(signatures) > 1:
            return False, f"历史记录包含多个不同的配置签名: {sorted(signatures)}"

        if len(signatures) == 1:
            current_sig = self._get_config_signature()
            stored_sig = list(signatures)[0]
            if stored_sig != current_sig:
                return False, f"配置签名不匹配: {stored_sig} vs {current_sig}"

        first_config = None
        for r in records:
            rc = r.get("run_config", {})
            if not rc:
                continue
            key_fields = {
                k: rc.get(k) for k in [
                    "data_hash", "subquestion_file", "max_attempts",
                    "run_jailbreak", "batch_size", "interaction_mode"
                ] if k in rc
            }
            if not key_fields:
                continue
            if first_config is None:
                first_config = key_fields
            else:
                for k, v in key_fields.items():
                    if first_config.get(k) != v:
                        return False, f"历史记录配置不一致: {k}={v} vs {first_config.get(k)}"

        return True, "配置一致"

    def _load_existing(self):
        """加载已有结果 - 完整校验"""
        if not self.jsonl_path.exists():
            return

        try:
            existing = load_existing_jsonl(str(self.jsonl_path))
        except Exception as e:
            error_type = type(e).__name__
            if "JSONLParseError" in error_type or "JSONDecodeError" in error_type:
                backup_path = self._get_backup_path("corrupted")
                shutil.copy2(self.jsonl_path, backup_path)
                print(f"⚠️ 原有结果文件损坏，已备份到: {backup_path}")
                print(f"   错误: {e}")
                self.jsonl_path.unlink()
                return
            else:
                raise

        if not existing:
            return

        is_consistent, reason = self._verify_all_records_config(existing)

        if not is_consistent:
            print(f"⚠️ 历史记录配置不一致，忽略旧结果")
            print(f"   {reason}")
            backup_path = self._get_backup_path("config_mismatch")
            shutil.copy2(self.jsonl_path, backup_path)
            print(f"   旧结果已备份到: {backup_path}")
            self.jsonl_path.unlink()
            return

        current_hash = self.run_config.get("data_hash")
        stored_hash = None
        for r in existing:
            if r.get("run_config", {}).get("data_hash"):
                stored_hash = r["run_config"]["data_hash"]
                break

        if current_hash and stored_hash and current_hash != stored_hash:
            print(f"⚠️ 数据文件已变化，忽略旧结果")
            print(f"   旧哈希: {stored_hash}")
            print(f"   新哈希: {current_hash}")
            backup_path = self._get_backup_path("data_mismatch")
            shutil.copy2(self.jsonl_path, backup_path)
            print(f"   旧结果已备份到: {backup_path}")
            self.jsonl_path.unlink()
            return

        self.records = existing

        for r in existing:
            status = r.get("status")
            idx = r.get("index")
            if idx is None:
                continue

            if status == "completed":
                self.processed_indices.add(idx)
            elif status == "failed":
                self.failed_indices.add(idx)
            elif status == "partial":
                self.partial_indices.add(idx)
            elif status == "skipped":
                self.skipped_indices.add(idx)

        print(f"✅ 加载已有结果: {len(self.records)} 条记录")
        print(f"   - 成功（越狱）: {len(self.processed_indices)} 条")
        print(f"   - 失败: {len(self.failed_indices)} 条")
        print(f"   - 部分成功: {len(self.partial_indices)} 条")
        print(f"   - 跳过: {len(self.skipped_indices)} 条")

    def _get_backup_path(self, suffix: str) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return self.jsonl_path.with_suffix(f".jsonl.{suffix}.{timestamp}.bak")

    def _safe_append(self, records: List[Dict]):
        if not records:
            return
        try:
            safe_append_jsonl(records, str(self.jsonl_path))
        except Exception as e:
            print(f"⚠️ 原子写入JSONL失败: {e}")
            raise

    def should_process(self, index: int) -> bool:
        return not (self.is_completed(index) or self.is_failed(index) or
                    self.is_partial(index) or self.is_skipped(index))

    def is_completed(self, index: int) -> bool:
        return index in self.processed_indices

    def is_failed(self, index: int) -> bool:
        return index in self.failed_indices

    def is_partial(self, index: int) -> bool:
        return index in self.partial_indices

    def is_skipped(self, index: int) -> bool:
        return index in self.skipped_indices

    def _build_attempts_list(self, attempts: List[AttemptRecord]) -> List[Dict]:
        """构建尝试记录列表（含批量模式提取状态）"""
        out = []
        for a in attempts:
            item = {
                "attempt": a.attempt_number,
                "status": a.result.final_answer_status.value if a.result else "error",
                "judgment": a.result.judgment if a.result and a.result.judgment else None,
                "judgment_error": a.result.judgment_error if a.result else None,
                "error": a.error,
                "timestamp": a.timestamp
            }
            if a.result is not None:
                item["interaction_mode"] = getattr(a.result, "interaction_mode", None)
                item["extraction_status"] = getattr(a.result, "extraction_status", None)
            out.append(item)
        return out

    def add_completed(self, index: int, result: PipelineResult, attempts: List[AttemptRecord],
                      success_attempt: int):
        record = {
            "index": index,
            "original": result.original,
            "sub_questions": result.sub_questions,
            "status": "completed",
            "success_attempt": success_attempt,
            "total_attempts": len(attempts),
            "final_answer": result.final_answer,
            "final_answer_status": result.final_answer_status.value,
            "judgment": result.judgment,
            "judgment_error": result.judgment_error,
            "attempts": self._build_attempts_list(attempts),
            "run_config": {
                **self.run_config,
                "config_signature": self._get_config_signature()
            },
            "rule_version": result.rule_version,
            "interaction_mode": getattr(result, "interaction_mode", None),
            "extraction_status": getattr(result, "extraction_status", None),
        }

        try:
            self._safe_append([record])
        except Exception as e:
            print(f"❌ 写入成功记录失败: {e}")
            raise

        self.records.append(record)
        self.processed_indices.add(index)

    def add_failed(self, index: int, original: str, sub_questions: List[str],
                   attempts: List[AttemptRecord]):
        record = {
            "index": index,
            "original": original,
            "sub_questions": sub_questions,
            "status": "failed",
            "success_attempt": None,
            "total_attempts": len(attempts),
            "final_answer": "",
            "final_answer_status": "failed",
            "judgment": None,
            "judgment_error": f"所有 {len(attempts)} 次尝试均未成功",
            "attempts": self._build_attempts_list(attempts),
            "run_config": {
                **self.run_config,
                "config_signature": self._get_config_signature()
            }
        }

        try:
            self._safe_append([record])
        except Exception as e:
            print(f"❌ 写入失败记录失败: {e}")
            raise

        self.records.append(record)
        self.failed_indices.add(index)

    def add_partial(self, index: int, original: str, sub_questions: List[str],
                    attempts: List[AttemptRecord]):
        run_jailbreak = self.run_config.get("run_jailbreak", True)
        interaction_mode = self.run_config.get("interaction_mode", "sequential")

        judgment_errors = []
        has_judgment_error = False
        has_judgment_no = False
        has_judgment_yes = False
        has_extraction_failed = False

        for a in attempts:
            if a.result:
                if getattr(a.result, "extraction_status", None) == "failed":
                    has_extraction_failed = True
                if a.result.judgment_error:
                    judgment_errors.append(a.result.judgment_error)
                    has_judgment_error = True
                elif a.result.judgment == "NO":
                    has_judgment_no = True
                elif a.result.judgment == "YES":
                    has_judgment_yes = True

        if not run_jailbreak:
            status_desc = "评判已关闭，仅回答问题（无评判结果）"
        elif has_extraction_failed:
            status_desc = "批量回答未能提取最后一个编号回答，无法评判"
        elif has_judgment_error:
            status_desc = f"有回答但评判失败: {'; '.join(judgment_errors[:2])}"
        elif has_judgment_no:
            status_desc = "有回答且评判为NO（有效否定，未越狱）"
        elif has_judgment_yes:
            status_desc = "有回答且评判为YES（但状态记录异常）"
        else:
            status_desc = "有回答但无有效评判结果（评判返回异常）"

        record = {
            "index": index,
            "original": original,
            "sub_questions": sub_questions,
            "status": "partial",
            "success_attempt": None,
            "total_attempts": len(attempts),
            "final_answer": "",
            "final_answer_status": "partial",
            "judgment": None,
            "judgment_error": status_desc,
            "attempts": self._build_attempts_list(attempts),
            "run_config": {
                **self.run_config,
                "config_signature": self._get_config_signature()
            },
            "interaction_mode": interaction_mode,
        }

        try:
            self._safe_append([record])
        except Exception as e:
            print(f"❌ 写入部分成功记录失败: {e}")
            raise

        self.records.append(record)
        self.partial_indices.add(index)

    def add_skipped(self, index: int, original: str, reason: str):
        record = {
            "index": index,
            "original": original,
            "status": "skipped",
            "skip_reason": reason,
            "run_config": {
                **self.run_config,
                "config_signature": self._get_config_signature()
            }
        }

        try:
            self._safe_append([record])
        except Exception as e:
            print(f"❌ 写入跳过记录失败: {e}")
            raise

        self.records.append(record)
        self.skipped_indices.add(index)

    def save_final(self):
        if not self.records:
            print("⚠️ 没有记录可保存")
            return

        try:
            save_json(self.records, str(self.json_path))
            print(f"✅ 保存 JSON: {self.json_path}")

            save_csv(self.records, str(self.csv_path))
            print(f"✅ 保存 CSV: {self.csv_path}")

            print(f"✅ JSONL: {self.jsonl_path}")

            self._save_summary()

        except Exception as e:
            print(f"❌ 保存最终结果失败: {e}")
            raise

    def _save_summary(self):
        stats = self.get_stats()
        total_effective = stats["total"] - stats["skipped"]

        summary = {
            "total_questions": stats["total"],
            "completed": stats["completed"],
            "failed": stats["failed"],
            "partial": stats["partial"],
            "skipped": stats["skipped"],
            "success_rate": f"{stats['completed'] / max(total_effective, 1) * 100:.2f}%",
            "max_attempts": self.max_attempts,
            "interaction_mode": self.run_config.get("interaction_mode", "sequential"),
            "run_config": {
                **self.run_config,
                "config_signature": self._get_config_signature()
            }
        }
        save_json(summary, str(self.summary_path))
        print(f"✅ 保存汇总: {self.summary_path}")

    def get_stats(self) -> Dict[str, int]:
        return {
            "total": len(self.records),
            "completed": len(self.processed_indices),
            "failed": len(self.failed_indices),
            "partial": len(self.partial_indices),
            "skipped": len(self.skipped_indices)
        }


def process_with_retries(
        original: str,
        sub_questions: List[str],
        idx: int,
        max_attempts: int = 10,
        run_jailbreak: bool = True,
        interaction_mode: str = "sequential"
) -> Tuple[ProcessStatus, Optional[PipelineResult], List[AttemptRecord], Optional[int]]:
    attempts = []
    success_attempt = None
    final_result = None

    for attempt_num in range(1, max_attempts + 1):
        try:
            print(f"    [尝试 {attempt_num}/{max_attempts}] 开始...")

            result = run_single(
                original_prompt=original,
                index=idx,
                run_jailbreak=run_jailbreak,
                precomputed_sub_questions=sub_questions,
                interaction_mode=interaction_mode
            )

            attempt = AttemptRecord(
                attempt_number=attempt_num,
                result=result,
                is_success=(result.judgment == "YES")
            )
            attempts.append(attempt)

            if result.judgment == "YES":
                success_attempt = attempt_num
                final_result = result
                print(f"    [尝试 {attempt_num}/{max_attempts}] ✅ 越狱成功！")
                return ProcessStatus.COMPLETED, final_result, attempts, success_attempt

            if result.final_answer_status == AnswerStatus.SUCCESS:
                if not run_jailbreak:
                    print(f"    [尝试 {attempt_num}/{max_attempts}] 回答成功（评判已关闭）")
                elif getattr(result, "extraction_status", None) == "failed":
                    print(f"    [尝试 {attempt_num}/{max_attempts}] 批量回答提取失败，无法评判")
                elif result.judgment_error:
                    print(f"    [尝试 {attempt_num}/{max_attempts}] 回答成功但评判失败: {result.judgment_error}")
                elif result.judgment == "NO":
                    print(f"    [尝试 {attempt_num}/{max_attempts}] 回答成功，评判为NO（有效否定）")
                else:
                    print(f"    [尝试 {attempt_num}/{max_attempts}] 回答成功但无有效评判结果")
            else:
                print(f"    [尝试 {attempt_num}/{max_attempts}] 回答状态: {result.final_answer_status.value}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e!s}"
            attempt = AttemptRecord(
                attempt_number=attempt_num,
                error=error_msg,
                is_success=False
            )
            attempts.append(attempt)
            print(f"    [尝试 {attempt_num}/{max_attempts}] ❌ 错误: {error_msg}")

        if attempt_num < max_attempts:
            time.sleep(0.5)

    has_success_answer = any(
        a.result and a.result.final_answer_status == AnswerStatus.SUCCESS
        for a in attempts
    )

    if has_success_answer:
        return ProcessStatus.PARTIAL, None, attempts, None
    else:
        return ProcessStatus.FAILED, None, attempts, None


def validate_arguments(args) -> None:
    if args.offset < 0:
        raise ValueError(f"offset 必须 >= 0，当前值: {args.offset}")
    if args.limit < 0:
        raise ValueError(f"limit 必须 >= 0，当前值: {args.limit}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size 必须 > 0，当前值: {args.batch_size}")
    if args.max_attempts < 1:
        raise ValueError(f"max_attempts 必须 >= 1，当前值: {args.max_attempts}")

    if args.interaction_mode not in ("sequential", "batch"):
        raise ValueError(
            f"interaction_mode 必须是 'sequential' 或 'batch'，当前值: {args.interaction_mode}"
        )

    sub_path = Path(args.subquestion)
    if not sub_path.exists():
        raise FileNotFoundError(f"子问题文件不存在: {sub_path}")
    if not sub_path.is_file():
        raise ValueError(f"子问题路径不是文件: {sub_path}")


def main():
    parser = argparse.ArgumentParser(description="Advbench 流水线")
    parser.add_argument(
        "--subquestion",
        type=str,
        default=r"C:\Users\HP\Desktop\xiaorong2\qwen\120questions.csv",
        help="子问题CSV路径（必须包含original列，可选sub_questions列）"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./output",
        help="输出目录"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="每批处理条数"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="从第几条开始"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理条数，0表示不限制"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="每个问题最大尝试次数"
    )
    parser.add_argument(
        "--no-jailbreak",
        action="store_true",
        help="跳过越狱判断"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不续跑"
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="跳过错误继续处理"
    )
    parser.add_argument(
        "--interaction-mode",
        type=str,
        default="sequential",
        choices=["sequential", "batch"],
        help="交互组织方式: sequential=逐个发送, batch=一次性发送全部子问题"
    )
    args = parser.parse_args()

    try:
        validate_arguments(args)
    except Exception as e:
        print(f"❌ 参数错误: {e}", file=sys.stderr)
        sys.exit(1)

    require_judge = not args.no_jailbreak
    try:
        config.validate_credentials(require_judge=require_judge)
    except Exception as e:
        print(f"❌ 配置错误: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Advbench 流水线（多次尝试，成功即停止）")
    print("=" * 60)
    print(f"输入文件: {args.subquestion}")
    print(f"输出目录: {args.output_dir}")
    print(f"处理范围: offset={args.offset}, limit={args.limit or '全部'}")
    print(f"最大尝试次数: {args.max_attempts}")
    print(f"越狱判断: {'关闭' if args.no_jailbreak else '开启'}")
    print(f"续跑模式: {'关闭' if args.no_resume else '开启'}")
    print(f"跳过错误: {'开启' if args.skip_errors else '关闭'}")
    print(f"交互组织方式: {args.interaction_mode}")
    print("-" * 60)

    sub_path = Path(args.subquestion)
    try:
        questions = DataLoader.load_subquestions(sub_path)
        print(f"✅ 加载 {len(questions)} 条问题")
    except Exception as e:
        print(f"❌ 加载数据失败: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    start = args.offset
    end = len(questions)
    if args.limit > 0:
        end = min(start + args.limit, end)

    if start >= len(questions):
        print(f"⚠️ offset={start} 超出数据范围（共 {len(questions)} 条）")
        sys.exit(0)

    slice_questions = questions[start:end]
    total = len(slice_questions)
    print(f"本次处理范围: [{start}, {end})，共 {total} 条")

    output_dir = prepare_output_path(args.output_dir)

    run_config = {
        "subquestion_file": str(sub_path),
        "offset": args.offset,
        "limit": args.limit,
        "max_attempts": args.max_attempts,
        "run_jailbreak": require_judge,
        "batch_size": args.batch_size,
        "data_hash": DataLoader.get_file_hash(sub_path),
        "interaction_mode": args.interaction_mode,
    }

    result_manager = ResultManager(output_dir, run_config, args.max_attempts)

    if args.no_resume:
        print("⚠️ 不续跑模式：将忽略已有结果")
        if result_manager.jsonl_path.exists():
            backup_path = result_manager._get_backup_path("manual")
            result_manager.jsonl_path.rename(backup_path)
            print(f"   已有结果已备份到: {backup_path}")
        result_manager = ResultManager(output_dir, run_config, args.max_attempts)

    to_process = []
    skipped = []

    for item in slice_questions:
        idx = item.get("index")
        if idx is None:
            continue
        if result_manager.should_process(idx):
            to_process.append(item)
        else:
            skipped.append(idx)

    print(f"\n📊 统计:")
    print(f"  总问题: {total}")
    print(f"  已完成（越狱成功）: {len(result_manager.processed_indices)}")
    print(f"  失败: {len(result_manager.failed_indices)}")
    print(f"  部分成功: {len(result_manager.partial_indices)}")
    print(f"  已跳过: {len(skipped)}")
    print(f"  待处理: {len(to_process)}")
    print("-" * 60)

    if not to_process:
        print("✅ 所有问题已处理完成")
        result_manager.save_final()
        return

    batch_size = args.batch_size
    run_jailbreak = require_judge
    skip_errors = args.skip_errors
    max_attempts = args.max_attempts
    interaction_mode = args.interaction_mode

    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i + batch_size]

        for item in batch:
            idx = item.get("index")
            original = item.get("original", "")
            sub_questions = item.get("sub_questions", [])

            if not original:
                print(f"  [SKIP] index={idx} 问题为空")
                result_manager.add_skipped(idx, original, "问题为空")
                continue

            print(f"\n[{idx + 1}] 处理问题: {original[:80]}...")
            print(f"  子问题数: {len(sub_questions)}")

            try:
                status, final_result, attempts, success_attempt = process_with_retries(
                    original=original,
                    sub_questions=sub_questions,
                    idx=idx,
                    max_attempts=max_attempts,
                    run_jailbreak=run_jailbreak,
                    interaction_mode=interaction_mode
                )

                if status == ProcessStatus.COMPLETED:
                    result_manager.add_completed(idx, final_result, attempts, success_attempt)
                    print(f"  ✅ 成功！第 {success_attempt} 轮越狱成功")
                elif status == ProcessStatus.PARTIAL:
                    result_manager.add_partial(idx, original, sub_questions, attempts)
                    has_judge_err = any(a.result and a.result.judgment_error for a in attempts)
                    has_judge_no = any(a.result and a.result.judgment == "NO" for a in attempts)
                    has_extract_fail = any(
                        a.result and getattr(a.result, "extraction_status", None) == "failed"
                        for a in attempts
                    )

                    if not run_jailbreak:
                        print(f"  ⚠️ 部分成功：评判已关闭，仅回答问题")
                    elif has_extract_fail:
                        print(f"  ⚠️ 部分成功：批量回答提取最后一个编号回答失败")
                    elif has_judge_err:
                        print(f"  ⚠️ 部分成功：有回答但评判失败")
                    elif has_judge_no:
                        print(f"  ⚠️ 部分成功：有回答且评判为NO（有效否定）")
                    else:
                        print(f"  ⚠️ 部分成功：有回答但无有效评判结果")
                else:
                    result_manager.add_failed(idx, original, sub_questions, attempts)
                    print(f"  ❌ 失败：{len(attempts)} 次尝试均未成功")

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e!s}"
                print(f"  [FAIL] index={idx} 错误: {error_msg}", file=sys.stderr)

                if not skip_errors:
                    traceback.print_exc()
                    result_manager.add_failed(idx, original, sub_questions, [])
                    break
                else:
                    traceback.print_exc()
                    result_manager.add_failed(idx, original, sub_questions, [])
                    continue

        stats = result_manager.get_stats()
        print(f"\n📊 进度: 成功 {stats['completed']}, 失败 {stats['failed']}, "
              f"部分 {stats['partial']}, 跳过 {stats['skipped']}, 总计 {stats['total']}")
        print("-" * 60)

        if not skip_errors and result_manager.failed_indices:
            print("❌ 遇到错误，停止处理")
            break

    print("\n📁 保存结果...")
    try:
        result_manager.save_final()
    except Exception as e:
        print(f"❌ 保存最终结果失败: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    stats = result_manager.get_stats()
    total_effective = stats["total"] - stats["skipped"]

    print("\n" + "=" * 60)
    print("📊 最终统计:")
    print(f"  ✅ 成功（越狱）: {stats['completed']}")
    print(f"  ❌ 失败: {stats['failed']}")
    print(f"  ⚠️ 部分成功: {stats['partial']}")
    print(f"  ⏭️ 跳过: {stats['skipped']}")
    print(f"  📊 总计: {stats['total']}")

    if total_effective > 0:
        success_rate = stats['completed'] / total_effective * 100
        print(f"\n📈 成功率: {success_rate:.2f}%")

    if stats['failed'] > 0:
        print(f"\n⚠️ 有 {stats['failed']} 条失败记录，请检查日志")

    print("=" * 60)
    print("✅ 流水线执行完毕")


if __name__ == "__main__":
    main()