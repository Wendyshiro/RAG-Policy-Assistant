"""
Quick retrieval tester — run from project root:
    python test_retrieval.py
    python test_retrieval.py "how many pto days do i get"
"""

import sys
from src.retrieval import retrieve

# ── colour helpers (Windows-safe) ────────────────────────────────────────────
try:
    import colorama; colorama.init()
    GRN = "\033[32m"; YLW = "\033[33m"; CYN = "\033[36m"
    DIM = "\033[2m";  RST = "\033[0m";  BLD = "\033[1m"
except ImportError:
    GRN = YLW = CYN = DIM = RST = BLD = ""

# ── default test queries ──────────────────────────────────────────────────────
DEFAULT_QUERIES = [
    "How many PTO days do I get?",
    "What is the mileage reimbursement rate?",
    "Can I work remotely?",
    "What happens if I fail a drug test?",
    "How do I report harassment?",
]

def show_results(query: str, k: int = 5):
    print(f"\n{BLD}{'='*60}{RST}")
    print(f"{BLD}Query:{RST} {CYN}{query}{RST}")
    print(f"{BLD}{'='*60}{RST}")

    results = retrieve(query, k=k)

    if not results:
        print(f"{YLW}No results returned.{RST}")
        return

    for i, r in enumerate(results, 1):
        score_color = GRN if r["score"] >= 0.5 else YLW
        print(f"\n{BLD}Result {i}{RST}  "
              f"score={score_color}{r['score']:.4f}{RST}  "
              f"{DIM}{r['source']}{RST}")
        print(f"  {BLD}{r['title']}{RST} — {r['section']}")
        # show first 200 chars of the chunk text
        snippet = r["text"].replace("\n", " ")[:200]
        print(f"  {DIM}{snippet}...{RST}")

def main():
    # If a query was passed on the command line, test just that one
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        show_results(query)
        return

    # Otherwise run all default queries
    print(f"\n{BLD}Running {len(DEFAULT_QUERIES)} test queries against ChromaDB...{RST}")
    for q in DEFAULT_QUERIES:
        show_results(q)

    print(f"\n{GRN}{BLD}Done.{RST} Retrieval is working.\n")
    print("Tip: pass your own query as an argument:")
    print("  python test_retrieval.py \"what is the password policy?\"")

if __name__ == "__main__":
    main()