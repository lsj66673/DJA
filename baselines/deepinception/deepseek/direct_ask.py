# -*- coding: utf-8 -*-
"""
直接读取 original.csv，逐条用原始有害问题向 doubao 提问。
每条问题最多尝试 10 次，一旦越狱成功（评判 YES）立即停止。
不做任何子问题拆解，完全保持原来的“多次尝试，成功即停”思路。
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import config
from api_client import call_doubao, call_judge
from io_utils import (
    save_json,
    save_csv,
    safe_append_jsonl,
    load_existing_jsonl,
    prepare_output_path,
)

# ===================== 默认路径 =====================
DEFAULT_INPUT = r"C:\Users\HP\Desktop\doubao\original.csv"
DEFAULT_MAX_ATTEMPTS = 10


# ===================== 读取 CSV（忽略空行） =====================
def read_original_csv(csv_path: Path) -> List[str]:
    """
    读取 original.csv，返回非空问题列表。
    自动识别表头：如果第一行包含 original/goal/prompt/question 等关键字，
    则视为表头；否则第一行也作为数据（每行取第一个非空单元格）。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    # 尝试多种编码
    text = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"无法解码文件: {csv_path}")

    reader = csv.reader(io.StringIO(text))

    rows: List[List[str]] = []
    for row in reader:
        # 忽略完全空行 / 纯空白行
        if not row or all((cell is None or str(cell).strip() == "") for cell in row):
            continue
        rows.append([("" if c is None else str(c).strip()) for c in row])

    if not rows:
        raise ValueError(f"文件没有任何有效行: {csv_path}")

    # 判断第一行是否是表头
    header_keywords = {
        "original", "goal", "prompt", "question", "query",
        "问题", "原始问题", "原始", "target", "instruction",
    }
    first_row_lower = [c.lower() for c in rows[0]]
    is_header = any(
        kw in first_row_lower or any(kw in name for name in first_row_lower)
        for kw in header_keywords
    )

    questions: List[str] = []

    if is_header:
        # 找问题所在列
        col_idx = None
        for i, name in enumerate(first_row_lower):
            if name in header_keywords or any(kw in name for kw in header_keywords):
                col_idx = i
                break
        if col_idx is None:
            col_idx = 0

        data_rows = rows[1:]
        for row in data_rows:
            if col_idx < len(row):
                q = row[col_idx].strip()
                if q:
                    questions.append(q)

        # 如果问题列全空，退而求其次：取每行第一个非空单元格
        if not questions:
            for row in data_rows:
                for cell in row:
                    if cell.strip():
                        questions.append(cell.strip())
                        break
    else:
        # 无表头：每行取第一个非空单元格作为问题
        for row in rows:
            for cell in row:
                if cell.strip():
                    questions.append(cell.strip())
                    break

    # 去重（保序）
    seen = set()
    unique_questions: List[str] = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    return unique_questions


# ===================== 直接提问 / 评判 =====================
def ask_once(question: str) -> Tuple[str, Optional[str]]:
    """
    向 doubao 直接提出原始问题一次。
    返回 (answer, error)。error 为 None 表示成功。
    """
    messages = [{"role": "user", "content": question}]
    answer, err = call_doubao(messages=messages)
    if err is not None:
        return "", err
    if answer is None:
        return "", "API返回None"
    if not isinstance(answer, str):
        return str(answer), None
    answer_clean = answer.strip()
    if not answer_clean:
        return "", "API返回空字符串"
    return answer_clean, None


def judge_once(answer: str) -> Tuple[Optional[str], Optional[str]]:
    """调用评判模型判断是否越狱成功。返回 (judgment, error)。"""
    if not answer:
        return None, "回答为空，无法评判"
    prompt = f"{config.JAILBREAK_CHECK}\n\n待判断的内容：\n{answer}"
    judgment, err = call_judge(prompt=prompt)
    if err is not None:
        return None, f"评判调用失败: {err}"
    if judgment is None:
        return None, "评判返回None"
    if not isinstance(judgment, str):
        return None, f"评判返回非字符串: {type(judgment).__name__}"
    raw = judgment.strip()
    if raw == "YES":
        return "YES", None
    if raw == "NO":
        return "NO", None
    return None, f"无效的评判输出: '{raw[:100]}'"


# ===================== 配置 / 续跑 =====================
def build_run_config(csv_path: Path, run_jailbreak: bool, max_attempts: int) -> Dict[str, Any]:
    def file_hash(p: Path) -> str:
        if not p.exists():
            return ""
        with open(p, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]

    return {
        "input_file": str(csv_path),
        "input_hash": file_hash(csv_path),
        "run_jailbreak": run_jailbreak,
        "max_attempts": max_attempts,
        "mode": "direct_original_question",
        "config_signature": config.get_config_signature(require_judge=run_jailbreak),
    }


def load_existing_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    try:
        return load_existing_jsonl(str(jsonl_path))
    except Exception as e:
        print(f"⚠️ 读取已有结果失败: {e}")
        return []


def is_record_done(record: Dict[str, Any]) -> bool:
    """
    判断一条历史记录是否算“已完成”，用于续跑跳过。
    只要越狱成功（judgment == YES）或者已经跑满 max_attempts 都算完成。
    """
    judgment = record.get("judgment")
    if judgment == "YES":
        return True

    attempts = record.get("attempts") or []
    max_attempts = record.get("run_config", {}).get("max_attempts", DEFAULT_MAX_ATTEMPTS)
    if len(attempts) >= max_attempts:
        return True

    return False


# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(
        description="直接读取 original.csv，用原始问题向 doubao 提问（最多尝试 N 次，成功即停）"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"original.csv 路径（默认: {DEFAULT_INPUT}）"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./direct_output",
        help="输出目录"
    )
    parser.add_argument(
        "-a", "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"每个问题最大尝试次数（默认 {DEFAULT_MAX_ATTEMPTS}）"
    )
    parser.add_argument(
        "--no-jailbreak",
        action="store_true",
        help="跳过越狱判断，仅收集回答"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不续跑，重新开始（会备份旧结果）"
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="单条记录异常时跳过继续（默认遇到异常继续本批次其他问题）"
    )
    args = parser.parse_args()

    if args.max_attempts < 1:
        print(f"❌ --max-attempts 必须 >= 1，当前: {args.max_attempts}", file=sys.stderr)
        sys.exit(1)

    run_jailbreak = not args.no_jailbreak

    # 校验配置
    try:
        config.validate_credentials(require_judge=run_jailbreak)
    except Exception as e:
        print(f"❌ 配置错误: {e}", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    try:
        questions = read_original_csv(input_path)
    except Exception as e:
        print(f"❌ 读取输入失败: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("直接提问模式（原始有害问题 -> doubao，多次尝试，成功即停）")
    print("=" * 60)
    print(f"输入文件: {input_path}")
    print(f"有效问题数: {len(questions)}")
    print(f"越狱判断: {'开启' if run_jailbreak else '关闭'}")
    print(f"最大尝试次数: {args.max_attempts}")
    print("-" * 60)

    output_dir = prepare_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "results.jsonl"
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"

    run_config = build_run_config(input_path, run_jailbreak, args.max_attempts)

    # ---------- 续跑：加载已有结果 ----------
    existing_records: List[Dict[str, Any]] = []
    if not args.no_resume and jsonl_path.exists():
        existing_records = load_existing_records(jsonl_path)
        if existing_records:
            old_hash = existing_records[0].get("run_config", {}).get("input_hash")
            old_sig = existing_records[0].get("run_config", {}).get("config_signature")
            current_sig = run_config["config_signature"]
            if old_hash and old_hash != run_config["input_hash"]:
                print("⚠️ 输入文件已变化，忽略旧结果，重新开始")
                ts = time.strftime("%Y%m%d_%H%M%S")
                backup = jsonl_path.with_suffix(f".jsonl.bak.{ts}")
                jsonl_path.rename(backup)
                print(f"   旧结果已备份: {backup}")
                existing_records = []
            elif old_sig and old_sig != current_sig:
                print("⚠️ 配置（模型/评判规则）已变化，忽略旧结果，重新开始")
                ts = time.strftime("%Y%m%d_%H%M%S")
                backup = jsonl_path.with_suffix(f".jsonl.bak.{ts}")
                jsonl_path.rename(backup)
                print(f"   旧结果已备份: {backup}")
                existing_records = []

    if args.no_resume and jsonl_path.exists():
        print("⚠️ 不续跑模式：备份旧结果")
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = jsonl_path.with_suffix(f".jsonl.bak.{ts}")
        jsonl_path.rename(backup)
        print(f"   旧结果已备份: {backup}")
        existing_records = []

    done_questions = set()
    for r in existing_records:
        if is_record_done(r) and r.get("original"):
            done_questions.add(r["original"])

    print(f"📦 已完成并跳过: {len(done_questions)} 条")
    print("-" * 60)

    # ---------- 逐条处理 ----------
    new_records: List[Dict[str, Any]] = []
    stats = {"yes": 0, "no": 0, "error": 0, "skipped": 0}

    for idx, question in enumerate(questions):
        if question in done_questions:
            stats["skipped"] += 1
            print(f"[{idx + 1}/{len(questions)}] ⏭️ 已完成，跳过: {question[:60]}...")
            continue

        print(f"\n[{idx + 1}/{len(questions)}] 提问: {question[:80]}...")

        record: Dict[str, Any] = {
            "index": idx,
            "original": question,
            "final_answer": "",
            "final_answer_status": "error",
            "judgment": None,
            "judgment_error": None,
            "attempts": [],
            "run_config": run_config,
        }

        final_answer = ""
        final_status = "error"
        final_judgment: Optional[str] = None
        final_judgment_error: Optional[str] = None
        success = False
        success_attempt: Optional[int] = None

        for attempt_num in range(1, args.max_attempts + 1):
            attempt_info: Dict[str, Any] = {
                "attempt": attempt_num,
                "status": "error",
                "judgment": None,
                "judgment_error": None,
                "error": None,
                "timestamp": time.time(),
            }

            try:
                answer, ans_err = ask_once(question)

                if ans_err is not None:
                    attempt_info["error"] = ans_err
                    attempt_info["status"] = "error"
                    print(f"    [尝试 {attempt_num}/{args.max_attempts}] ❌ 回答错误: {ans_err}")
                else:
                    final_answer = answer
                    final_status = "success"
                    attempt_info["status"] = "success"
                    print(f"    [尝试 {attempt_num}/{args.max_attempts}] ✅ 回答长度: {len(answer)}")

                    if run_jailbreak:
                        judgment, j_err = judge_once(answer)
                        final_judgment = judgment
                        final_judgment_error = j_err
                        attempt_info["judgment"] = judgment
                        attempt_info["judgment_error"] = j_err
                        if j_err:
                            print(f"    [尝试 {attempt_num}/{args.max_attempts}] ⚠️ 评判失败: {j_err}")
                        else:
                            print(f"    [尝试 {attempt_num}/{args.max_attempts}] 评判: {judgment}")
                            if judgment == "YES":
                                success = True
                                success_attempt = attempt_num
                    else:
                        # 不判断时，只要回答成功就算完成（只跑一次即可）
                        success = True
                        success_attempt = attempt_num

            except Exception as e:
                attempt_info["error"] = f"{type(e).__name__}: {e}"
                print(f"    [尝试 {attempt_num}/{args.max_attempts}] ❌ 异常: {attempt_info['error']}")

            record["attempts"].append(attempt_info)

            if success:
                print(f"    ✅ 第 {success_attempt} 次尝试越狱成功，停止重试")
                break

            if attempt_num < args.max_attempts:
                time.sleep(0.5)

        record["final_answer"] = final_answer
        record["final_answer_status"] = final_status
        record["judgment"] = final_judgment
        record["judgment_error"] = final_judgment_error
        record["success_attempt"] = success_attempt
        record["total_attempts"] = len(record["attempts"])

        # 统计
        if run_jailbreak:
            if final_judgment == "YES":
                stats["yes"] += 1
            elif final_judgment == "NO":
                stats["no"] += 1
            else:
                stats["error"] += 1
        else:
            if final_status == "success":
                stats["yes"] += 1
            else:
                stats["error"] += 1

        new_records.append(record)

        # 立即追加，防止崩溃丢失
        try:
            safe_append_jsonl([record], str(jsonl_path))
        except Exception as e:
            print(f"⚠️ 追加写入失败: {e}")

        print(
            f"    → 本条结果: judgment={final_judgment}, "
            f"尝试次数={len(record['attempts'])}"
        )

    # ---------- 合并并输出 ----------
    all_records = existing_records + new_records

    if all_records:
        try:
            save_json(all_records, str(json_path))
            print(f"\n✅ 保存 JSON: {json_path}")
        except Exception as e:
            print(f"⚠️ 保存 JSON 失败: {e}")

        try:
            save_csv(all_records, str(csv_path))
            print(f"✅ 保存 CSV: {csv_path}")
        except Exception as e:
            print(f"⚠️ 保存 CSV 失败: {e}")

    # 汇总
    total = len(all_records)
    if run_jailbreak:
        yes_count = sum(1 for r in all_records if r.get("judgment") == "YES")
        no_count = sum(1 for r in all_records if r.get("judgment") == "NO")
        err_count = total - yes_count - no_count
        success_rate = f"{yes_count / max(total, 1) * 100:.2f}%"
    else:
        yes_count = sum(1 for r in all_records if r.get("final_answer_status") == "success")
        no_count = 0
        err_count = total - yes_count
        success_rate = f"{yes_count / max(total, 1) * 100:.2f}%"

    summary = {
        "total": total,
        "yes": yes_count,
        "no": no_count,
        "error": err_count,
        "skipped_this_run": stats["skipped"],
        "success_rate": success_rate,
        "max_attempts": args.max_attempts,
        "run_jailbreak": run_jailbreak,
        "run_config": run_config,
    }
    try:
        save_json(summary, str(summary_path))
        print(f"✅ 保存汇总: {summary_path}")
    except Exception as e:
        print(f"⚠️ 保存汇总失败: {e}")

    print("\n" + "=" * 60)
    print("📊 最终统计:")
    print(f"  总计: {total}")
    if run_jailbreak:
        print(f"  ✅ YES (越狱成功): {yes_count}")
        print(f"  ❌ NO  (未越狱): {no_count}")
    else:
        print(f"  ✅ 回答成功: {yes_count}")
    print(f"  ⚠️ 错误: {err_count}")
    print(f"  📈 成功率: {success_rate}")
    print(f"  🔁 最大尝试次数: {args.max_attempts}")
    print("=" * 60)
    print("✅ 流水线执行完毕")


if __name__ == "__main__":
    main()