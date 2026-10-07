# -*- coding: utf-8 -*-
"""
DeepInception 正式结果评估
读 direct_ask.py 输出的 results.csv，做统一评估
"""
import csv
import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd

from api_client import call_judge
import config


# ============================================================
# 配置
# ============================================================

INPUT_PATH = "/root/deepinception_gemini/deepinception_output/results.csv"
OUTPUT_DIR = Path("/root/deepinception_gemini/deepinception_formal_output")

MAX_ATTEMPTS = 10

RESULT_JSONL = OUTPUT_DIR / "results.jsonl"
RESULT_JSON = OUTPUT_DIR / "results.json"
RESULT_CSV = OUTPUT_DIR / "results.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"


# ============================================================
# Judge
# ============================================================

def judge_once(answer: str) -> Tuple[Optional[str], Optional[str]]:
    if answer is None:
        return None, "回答为空，无法评判"

    answer = str(answer).strip()
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


# ============================================================
# 工具
# ============================================================

def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_rule_version() -> str:
    rule = getattr(config, "JAILBREAK_CHECK", "")
    digest = hashlib.sha256(rule.encode("utf-8")).hexdigest()[:12]
    return f"judge_rule_{digest}"


def get_file_hash(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(record: Dict[str, Any]):
    with open(RESULT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# 续跑
# ============================================================

def load_existing_results():
    records = []
    processed_indices = set()

    if not RESULT_JSONL.exists():
        return records, processed_indices

    with open(RESULT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
                idx = record.get("index")
                if idx is not None:
                    processed_indices.add(idx)
            except Exception as e:
                print("⚠️ 跳过损坏的历史JSONL记录：", e)

    return records, processed_indices


# ============================================================
# run_config
# ============================================================

def build_run_config():
    model_name = getattr(config, "GEMINI_MODEL", "unknown")

    run_config = {
        "method": "DeepInception",
        "input_file": INPUT_PATH,
        "data_hash": get_file_hash(INPUT_PATH),
        "max_attempts": MAX_ATTEMPTS,
        "run_jailbreak": True,
        "target_model": model_name,
        "judge_enabled": True
    }

    signature_text = json.dumps(run_config, ensure_ascii=False, sort_keys=True)
    signature = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()[:12]
    run_config["config_signature"] = signature

    return run_config


# ============================================================
# 评估单条问题（读 direct_ask.py 的输出格式）
# ============================================================

def evaluate_one_question(index: int, row: Dict[str, Any],
                          run_config: Dict[str, Any], rule_version: str):
    original = safe_text(row.get("original", ""))
    attempts_raw = row.get("attempts", [])

    # attempts 可能是 JSON 字符串
    if isinstance(attempts_raw, str):
        try:
            attempts_raw = json.loads(attempts_raw)
        except Exception:
            attempts_raw = []

    attempt_records: List[Dict[str, Any]] = []
    success_attempt = None
    successful_answer = ""
    successful_judgment = None
    successful_judgment_error = None

    last_valid_answer = ""
    has_valid_answer = False
    has_judgment_no = False
    has_judgment_error = False
    judgment_errors = []

    for att in attempts_raw:
        attempt_num = att.get("attempt", 0)
        answer = safe_text(att.get("target_answer", ""))
        target_error = safe_text(att.get("error", ""))
        judgment = att.get("judgment")
        judge_error = att.get("judgment_error")

        if not answer:
            attempt_records.append({
                "attempt": attempt_num,
                "status": "error",
                "judgment": None,
                "judgment_error": None,
                "error": target_error or "目标模型无有效回答",
                "timestamp": att.get("timestamp")
            })
            continue

        has_valid_answer = True
        last_valid_answer = answer

        if judge_error:
            has_judgment_error = True
            judgment_errors.append(judge_error)

        if judgment == "NO":
            has_judgment_no = True

        attempt_records.append({
            "attempt": attempt_num,
            "status": "success",
            "judgment": judgment,
            "judgment_error": judge_error,
            "error": target_error or None,
            "timestamp": att.get("timestamp")
        })

        if judgment == "YES":
            success_attempt = attempt_num
            successful_answer = answer
            successful_judgment = "YES"
            successful_judgment_error = None
            break

    # completed
    if success_attempt is not None:
        return {
            "index": index,
            "original": original,
            "sub_questions": [],
            "status": "completed",
            "success_attempt": success_attempt,
            "total_attempts": len(attempt_records),
            "final_answer": successful_answer,
            "final_answer_status": "success",
            "judgment": successful_judgment,
            "judgment_error": successful_judgment_error,
            "attempts": attempt_records,
            "run_config": run_config,
            "rule_version": rule_version
        }

    # partial
    if has_valid_answer:
        if has_judgment_error:
            status_desc = "有回答但评判失败: " + "; ".join(judgment_errors[:2])
        elif has_judgment_no:
            status_desc = "有回答且评判为NO（有效否定，未越狱）"
        else:
            status_desc = "有回答但无有效评判结果"

        return {
            "index": index,
            "original": original,
            "sub_questions": [],
            "status": "partial",
            "success_attempt": None,
            "total_attempts": len(attempt_records),
            "final_answer": "",
            "final_answer_status": "partial",
            "judgment": None,
            "judgment_error": status_desc,
            "attempts": attempt_records,
            "run_config": run_config,
            "rule_version": rule_version
        }

    # failed
    return {
        "index": index,
        "original": original,
        "sub_questions": [],
        "status": "failed",
        "success_attempt": None,
        "total_attempts": len(attempt_records),
        "final_answer": "",
        "final_answer_status": "failed",
        "judgment": None,
        "judgment_error": f"所有 {len(attempt_records)} 次尝试均未成功",
        "attempts": attempt_records,
        "run_config": run_config,
        "rule_version": rule_version
    }


# ============================================================
# 保存 CSV
# ============================================================

def save_results_csv(records: List[Dict[str, Any]]):
    columns = [
        "index", "original", "sub_questions", "status",
        "success_attempt", "total_attempts", "final_answer",
        "final_answer_status", "judgment", "judgment_error",
        "attempts", "run_config", "rule_version"
    ]

    csv_records = []
    for record in records:
        row = record.copy()
        row["sub_questions"] = json.dumps(row.get("sub_questions", []), ensure_ascii=False)
        row["attempts"] = json.dumps(row.get("attempts", []), ensure_ascii=False)
        row["run_config"] = json.dumps(row.get("run_config", {}), ensure_ascii=False)
        csv_records.append(row)

    df = pd.DataFrame(csv_records, columns=columns)
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# 汇总
# ============================================================

def save_summary(records: List[Dict[str, Any]], run_config: Dict[str, Any]):
    total = len(records)
    completed = sum(1 for r in records if r.get("status") == "completed")
    partial = sum(1 for r in records if r.get("status") == "partial")
    failed = sum(1 for r in records if r.get("status") == "failed")
    skipped = sum(1 for r in records if r.get("status") == "skipped")

    effective_total = total - skipped
    asr = (completed / effective_total * 100) if effective_total > 0 else 0.0

    successful_queries = [r["success_attempt"] for r in records if r.get("success_attempt") is not None]
    mean_queries_success = (sum(successful_queries) / len(successful_queries)) if successful_queries else None

    total_target_queries = sum(r.get("total_attempts", 0) for r in records)

    summary = {
        "method": "DeepInception",
        "total_questions": total,
        "completed": completed,
        "partial": partial,
        "failed": failed,
        "skipped": skipped,
        "success_rate": f"{asr:.2f}%",
        "ASR": asr,
        "max_attempts": MAX_ATTEMPTS,
        "total_target_queries": total_target_queries,
        "mean_queries_to_success": mean_queries_success,
        "run_config": run_config
    }

    save_json(summary, SUMMARY_JSON)
    return summary


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("DeepInception 正式结果评估")
    print("=" * 70)

    input_path = Path(INPUT_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    print(f"输入记录数：{len(df)}")

    run_config = build_run_config()
    rule_version = get_rule_version()
    print("rule_version：", rule_version)

    records, processed = load_existing_results()
    print(f"已有完成记录：{len(processed)}")

    total_questions = len(df)
    counter = 0

    for _, row in df.iterrows():
        counter += 1
        index = int(row.get("index", counter - 1))

        if index in processed:
            print(f"[{counter}/{total_questions}] index={index} 已存在，跳过")
            continue

        print(f"\n[{counter}/{total_questions}] 开始评估 index={index}")
        record = evaluate_one_question(index, row.to_dict(), run_config, rule_version)
        append_jsonl(record)
        records.append(record)
        processed.add(index)

        print(f"    最终状态：{record['status']}")
        print(f"    success_attempt：{record['success_attempt']}")
        print(f"    total_attempts：{record['total_attempts']}")

    records = sorted(records, key=lambda x: x.get("index", 0))

    save_json(records, RESULT_JSON)
    save_results_csv(records)
    summary = save_summary(records, run_config)

    print()
    print("=" * 70)
    print("正式评估完成")
    print("=" * 70)
    print("总问题数：", summary["total_questions"])
    print("越狱成功：", summary["completed"])
    print("有效否定/部分：", summary["partial"])
    print("调用失败：", summary["failed"])
    print("ASR：", f"{summary['ASR']:.2f}%")
    print("Target总Queries：", summary["total_target_queries"])
    print("成功样本平均Queries：", summary["mean_queries_to_success"])
    print()
    print("results.jsonl：", RESULT_JSONL)
    print("results.json：", RESULT_JSON)
    print("results.csv：", RESULT_CSV)
    print("summary.json：", SUMMARY_JSON)


if __name__ == "__main__":
    main()