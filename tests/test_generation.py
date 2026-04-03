"""
test_generation.py — Full RAG generation pipeline tester
Run from project root:
    python test_generation.py
    python test_generation.py "what is the remote work stipend?"
    python test_generation.py --all        # runs all test cases
    python test_generation.py --guardrail  # tests out-of-corpus refusal only
"""

import sys
import time

# ── optional colour support ───────────────────────────────────────────────────
try:
    import colorama
    colorama.init()
    GRN = "\033[32m"; YLW = "\033[33m"; RED = "\033[31m"
    CYN = "\033[36m"; DIM = "\033[2m";  RST = "\033[0m"
    BLD = "\033[1m";  MAG = "\033[35m"
except ImportError:
    GRN = YLW = RED = CYN = DIM = RST = BLD = MAG = ""

# ── test cases ────────────────────────────────────────────────────────────────
# Each entry: (question, expected_keyword, source_doc, category)
# expected_keyword: a word/phrase that MUST appear in a correct answer
# source_doc: the document the answer should be cited from
# Set source_doc to None for out-of-corpus questions

IN_CORPUS_TESTS = [
    (
        "How many PTO days does an employee with 4 years of tenure receive?",
        "20",
        "pto_policy.md",
        "pto"
    ),
    (
        "What is the maximum PTO carryover at year end?",
        "5",
        "pto_policy.md",
        "pto"
    ),
    (
        "What is the mileage reimbursement rate for personal vehicle use?",
        "0.67",
        "expense_reimbursement.md",
        "expense"
    ),
    (
        "How much is the monthly internet stipend for remote employees?",
        "50",
        "remote_work_policy.md",
        "remote_work"
    ),
    (
        "What percentage of medical insurance premiums does the company pay?",
        "85",
        "benefits_policy.md",
        "benefits"
    ),
    (
        "What is the minimum password length required?",
        "12",
        "information_security.md",
        "security"
    ),
    (
        "How many weeks of paid maternity leave are eligible employees entitled to?",
        "16",
        "benefits_policy.md",
        "benefits"
    ),
    (
        "Within how many hours must a data breach be reported to authorities?",
        "72",
        "data_privacy_gdpr.md",
        "data_privacy"
    ),
    (
        "What is the company 401k match policy?",
        "3%",
        "benefits_policy.md",
        "benefits"
    ),
    (
        "How many free EAP counseling sessions does each employee get per year?",
        "5",
        "benefits_policy.md",
        "benefits"
    ),
]

OUT_OF_CORPUS_TESTS = [
    (
        "What is the capital of France?",
        None,
        "out_of_corpus"
    ),
    (
        "Who won the FIFA World Cup in 2022?",
        None,
        "out_of_corpus"
    ),
    (
        "What is the boiling point of water?",
        None,
        "out_of_corpus"
    ),
]

# phrase that must appear in a refusal response
REFUSAL_PHRASES = [
    "i can only answer",
    "not covered in",
    "only answer questions about",
    "cannot answer",
    "don't have information",
    "outside the scope",
]

# ── helpers ───────────────────────────────────────────────────────────────────
def divider(char="=", width=62):
    print(f"{DIM}{char * width}{RST}")

def check_refusal(answer_text: str) -> bool:
    """Returns True if the answer looks like a proper refusal."""
    lower = answer_text.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)

def check_citation(citations: list, expected_doc: str) -> bool:
    """Returns True if the expected source doc appears in the citations."""
    return any(c["source"] == expected_doc for c in citations)

def print_result(label: str, passed: bool, detail: str = ""):
    icon = f"{GRN}PASS{RST}" if passed else f"{RED}FAIL{RST}"
    detail_str = f"  {DIM}{detail}{RST}" if detail else ""
    print(f"  [{icon}] {label}{detail_str}")

def run_single(question: str, verbose: bool = True) -> dict:
    """Run one question through the full pipeline and return the result dict."""
    from src.generation import answer

    if verbose:
        divider()
        print(f"{BLD}Question:{RST} {CYN}{question}{RST}")
        divider()

    start = time.perf_counter()
    result = answer(question)
    elapsed = round((time.perf_counter() - start) * 1000)

    if verbose:
        print(f"\n{BLD}Answer:{RST}")
        print(result["answer"])

        print(f"\n{BLD}Citations ({len(result['citations'])}):{RST}")
        for c in result["citations"]:
            print(f"  {MAG}{c['title']}{RST} — {c['section']}")
            print(f"  {DIM}[{c['source']}]{RST}")
            print(f"  {DIM}{c['snippet'][:120]}...{RST}\n")

        latency_color = GRN if elapsed < 5000 else YLW if elapsed < 10000 else RED
        print(f"{BLD}Latency:{RST} {latency_color}{elapsed} ms{RST}")
        print(f"{BLD}Model  :{RST} {result['model']}\n")

    result["elapsed_ms"] = elapsed
    return result

# ── test runners ──────────────────────────────────────────────────────────────
def run_guardrail_tests():
    """Test that out-of-corpus questions are refused."""
    from src.generation import answer

    print(f"\n{BLD}{'='*62}{RST}")
    print(f"{BLD}GUARDRAIL TESTS — out-of-corpus refusals{RST}")
    print(f"{BLD}{'='*62}{RST}\n")

    passed = 0
    for question, _, category in OUT_OF_CORPUS_TESTS:
        print(f"{DIM}Q: {question}{RST}")
        result = answer(question)
        refused = check_refusal(result["answer"])
        print_result(
            "Refusal triggered",
            refused,
            result["answer"][:80] + "..." if not refused else ""
        )
        if refused:
            passed += 1
        print()

    total = len(OUT_OF_CORPUS_TESTS)
    color = GRN if passed == total else YLW if passed > 0 else RED
    print(f"Guardrail result: {color}{passed}/{total} refused correctly{RST}\n")
    return passed, total

def run_corpus_tests():
    """Test in-corpus questions for answer quality and citation accuracy."""
    from src.generation import answer

    print(f"\n{BLD}{'='*62}{RST}")
    print(f"{BLD}IN-CORPUS TESTS — answer quality + citation accuracy{RST}")
    print(f"{BLD}{'='*62}{RST}\n")

    results = []
    latencies = []

    for question, expected_keyword, source_doc, category in IN_CORPUS_TESTS:
        print(f"{DIM}[{category}] {question}{RST}")

        result = answer(question)
        latencies.append(result["latency_ms"])

        # check 1: expected keyword in answer
        keyword_found = expected_keyword.lower() in result["answer"].lower()

        # check 2: correct source doc cited
        citation_ok = check_citation(result["citations"], source_doc)

        # check 3: answer is not a refusal
        not_refused = not check_refusal(result["answer"])

        all_passed = keyword_found and citation_ok and not_refused
        results.append(all_passed)

        print_result(
            f"Keyword '{expected_keyword}' in answer",
            keyword_found,
            result["answer"][:80] + "..." if not keyword_found else ""
        )
        print_result(
            f"Citation from {source_doc}",
            citation_ok
        )
        print_result(
            "Answer not a refusal",
            not_refused
        )
        latency_color = GRN if result["latency_ms"] < 5000 else YLW
        print(f"  {DIM}Latency: {latency_color}{result['latency_ms']} ms{RST}{DIM}{RST}")
        print()

    # summary stats
    passed = sum(results)
    total  = len(results)

    if latencies:
        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
    else:
        p50 = p95 = 0

    color = GRN if passed == total else YLW if passed >= total * 0.7 else RED
    print(f"In-corpus result : {color}{passed}/{total} tests passed{RST}")
    print(f"Latency p50      : {p50} ms")
    print(f"Latency p95      : {p95} ms\n")

    return passed, total, p50, p95

def run_all_tests():
    """Run the full test suite: corpus + guardrail."""
    print(f"\n{BLD}RAG Generation — Full Test Suite{RST}")
    print(f"Model: checking...")

    corpus_passed, corpus_total, p50, p95 = run_corpus_tests()
    guard_passed, guard_total = run_guardrail_tests()

    divider("=")
    print(f"{BLD}FINAL SUMMARY{RST}")
    divider("=")

    total_passed = corpus_passed + guard_passed
    total_tests  = corpus_total  + guard_total
    pct = round(total_passed / total_tests * 100)

    color = GRN if pct >= 80 else YLW if pct >= 60 else RED
    print(f"Overall          : {color}{total_passed}/{total_tests} ({pct}%){RST}")
    print(f"In-corpus        : {corpus_passed}/{corpus_total}")
    print(f"Guardrail        : {guard_passed}/{guard_total}")
    print(f"Latency p50      : {p50} ms")
    print(f"Latency p95      : {p95} ms")
    divider("=")

# ── entry point ───────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    if not args:
        # default: run one demo question
        run_single("How many PTO days does an employee with 4 years get?")
        print(f"\n{DIM}Tip: run with --all to test all cases, "
              f"--guardrail to test refusals only,\n"
              f"or pass any question as an argument.{RST}\n")
        return

    if args[0] == "--all":
        run_all_tests()
        return

    if args[0] == "--guardrail":
        run_guardrail_tests()
        return

    # treat all args as a custom question
    run_single(" ".join(args))

if __name__ == "__main__":
    main()