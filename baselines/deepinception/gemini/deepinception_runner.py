# -*- coding: utf-8 -*-

import csv
from dataclasses import dataclass
from typing import Optional, Tuple, List

from api_client import call_gemini, call_judge
import config


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class DeepInceptionResult:
    original: str
    final_answer: str
    final_answer_status: str
    judgment: Optional[str]
    judgment_error: Optional[str]
    rule_version: str = "deepinception_v1"


@dataclass
class DeepInceptionAttempt:
    attempt_number: int
    result: Optional[DeepInceptionResult] = None
    error: Optional[str] = None
    is_success: bool = False


# ============================================================
# 2. Judge
# ============================================================

def judge_once(answer: str) -> Tuple[Optional[str], Optional[str]]:
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
# 3. 配置
# ============================================================

TEMPLATE_PATH = "/root/deepinception_gemini/deepinception_template.txt"
INPUT_PATH = "/root/deepinception_gemini/original.csv"
OUTPUT_PATH = "/root/deepinception_gemini/gemini_results.csv"

MAX_ATTEMPTS = 10


# ============================================================
# 4. 读取模板
# ============================================================

with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    template = f.read()


# ============================================================
# 5. 读取问题（自动处理无列头 + 隔行空行）
# ============================================================

def load_goals_from_csv(path: str) -> List[str]:
    goals = []

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV 文件为空")

    first_row = rows[0]
    header_keywords = {"goal", "original", "prompt", "question", "问题", "目标"}

    has_header = False
    if first_row:
        first_cell = first_row[0].strip().lower()
        if first_cell in header_keywords:
            has_header = True

    data_rows = rows[1:] if has_header else rows

    for row in data_rows:
        if not row:
            continue
        goal = row[0].strip() if row else ""
        if goal:
            goals.append(goal)

    return goals


goals = load_goals_from_csv(INPUT_PATH)

print("=" * 60)
print("DeepInception 完整测试（Gemini）")
print("=" * 60)
print(f"模板读取成功: {len(template) > 0}")
print(f"加载问题数: {len(goals)}")
print(f"最大尝试次数: {MAX_ATTEMPTS}")
print("-" * 60)

for i, g in enumerate(goals[:3], 1):
    print(f"  {i}. {g[:60]}...")
if len(goals) > 3:
    print(f"  ... 还有 {len(goals) - 3} 条")


# ============================================================
# 6. 打开输出文件（流式写入）
# ============================================================

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)

    writer.writerow([
        "sample_id",
        "attempt",
        "original",
        "target_answer",
        "target_error",
        "judgment",
        "judge_error",
        "is_success"
    ])

    # ========================================================
    # 7. 遍历每条问题
    # ========================================================

    for sample_id, goal in enumerate(goals, start=1):

        print()
        print("=" * 60)
        print(f"[{sample_id}/{len(goals)}] 处理问题: {goal[:80]}...")
        print("=" * 60)

        attack_prompt = template.replace("{GOAL}", goal)

        attempts: List[DeepInceptionAttempt] = []
        success_attempt = None
        final_response = None
        final_judgment = None
        final_judge_error = None

        # ----------------------------------------------------
        # 最多 10 次尝试
        # ----------------------------------------------------

        for attempt_num in range(1, MAX_ATTEMPTS + 1):

            print(f"  [尝试 {attempt_num}/{MAX_ATTEMPTS}] 调用 Gemini...")

            response, error = call_gemini(prompt=attack_prompt)

            # ------------------------------------------------
            # 模型调用失败
            # ------------------------------------------------

            if not response:

                attempt_record = DeepInceptionAttempt(
                    attempt_number=attempt_num,
                    result=None,
                    error=error or "目标模型无有效回答",
                    is_success=False
                )

                attempts.append(attempt_record)

                print(f"    [尝试 {attempt_num}/{MAX_ATTEMPTS}] 无有效回答")

                writer.writerow([
                    sample_id,
                    attempt_num,
                    goal,
                    "",
                    error or "目标模型无有效回答",
                    "",
                    "",
                    False
                ])
                f.flush()

                continue

            # ------------------------------------------------
            # 有回答 -> Judge
            # ------------------------------------------------

            judgment, judge_error = judge_once(response)

            result = DeepInceptionResult(
                original=goal,
                final_answer=response,
                final_answer_status="success",
                judgment=judgment,
                judgment_error=judge_error
            )

            attempt_record = DeepInceptionAttempt(
                attempt_number=attempt_num,
                result=result,
                error=error,
                is_success=(judgment == "YES")
            )

            attempts.append(attempt_record)

            final_response = response
            final_judgment = judgment
            final_judge_error = judge_error

            writer.writerow([
                sample_id,
                attempt_num,
                goal,
                response,
                error or "",
                judgment or "",
                judge_error or "",
                (judgment == "YES")
            ])
            f.flush()

            # ------------------------------------------------
            # YES -> 立即停止
            # ------------------------------------------------

            if judgment == "YES":
                success_attempt = attempt_num
                print(f"    [尝试 {attempt_num}/{MAX_ATTEMPTS}] ✅ 越狱成功！")
                break
            else:
                print(f"    [尝试 {attempt_num}/{MAX_ATTEMPTS}] judgment={judgment}")

        # ----------------------------------------------------
        # 单条问题的最终结果
        # ----------------------------------------------------

        if success_attempt is not None:
            overall_status = "completed"
        else:
            overall_status = "partial" if final_response else "failed"

        print(f"  → {overall_status}（尝试 {len(attempts)} 次）")


# ============================================================
# 8. 完成
# ============================================================

print()
print("=" * 60)
print("所有问题处理完成")
print(f"结果已保存到: {OUTPUT_PATH}")
print("=" * 60)