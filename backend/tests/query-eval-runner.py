import json
import time
from collections import defaultdict
from pathlib import Path

from src.pipeline import answer


BASE_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = BASE_DIR / "query_eval_set.json"
RESULT_PATH = BASE_DIR / "query_eval_results.json"


def _check_case(case: dict, result: dict) -> tuple[bool, str]:
    sources = result.get("sources", [])
    answer_text = (result.get("answer") or "").lower()
    # Use chunks (raw retrieval) for keyword check — sources may be empty due to relevance gate
    chunks = result.get("chunks", [])
    source_text = " ".join((s.get("product_name") or "").lower() for s in sources)
    chunk_text = " ".join((s.get("product_name") or "").lower() for s in chunks)

    if "expect_min_sources" in case and len(sources) < case["expect_min_sources"]:
        return False, f"sources<{case['expect_min_sources']}"

    if "expect_max_sources" in case and len(sources) > case["expect_max_sources"]:
        return False, f"sources>{case['expect_max_sources']}"

    expected_keywords = [k.lower() for k in case.get("expect_keyword_any", [])]
    if expected_keywords:
        # Fall back to chunk_text when sources is empty (relevance gate blocked it)
        text_to_check = source_text if sources else chunk_text
        matched = any(k in text_to_check or k in answer_text for k in expected_keywords)
        if not matched:
            return False, "keyword_miss"

    return True, "ok"


def run_eval() -> dict:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    started = time.time()
    results = []
    by_type = defaultdict(lambda: {"total": 0, "passed": 0})

    for case in cases:
        query = case["query"]
        case_type = case.get("type", "unknown")

        out = answer(query)
        passed, reason = _check_case(case, out)

        by_type[case_type]["total"] += 1
        by_type[case_type]["passed"] += int(passed)

        results.append({
            "id": case["id"],
            "type": case_type,
            "query": query,
            "passed": passed,
            "reason": reason,
            "sources_count": len(out.get("sources", [])),
            "first_source": (out.get("sources", [{}])[0].get("product_name", "") if out.get("sources") else ""),
            "answer_preview": (out.get("answer") or "")[:220],
        })

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    elapsed = round(time.time() - started, 2)

    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round((passed / total) * 100, 2) if total else 0,
        "duration_sec": elapsed,
        "by_type": {
            k: {
                "total": v["total"],
                "passed": v["passed"],
                "pass_rate": round((v["passed"] / v["total"]) * 100, 2) if v["total"] else 0,
            }
            for k, v in by_type.items()
        },
    }

    payload = {
        "summary": summary,
        "results": results,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


if __name__ == "__main__":
    out = run_eval()
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
