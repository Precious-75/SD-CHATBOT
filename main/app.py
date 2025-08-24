"""
app.py
Purpose: Non-UI module/CLI for the chatbot.
 - Auto-loads a CSV at startup from the project folder when possible
 - Allows answering a single question via CLI or interactive REPL
 - Uses HFSmartCSVChatbot (embeddings + TF-IDF + classic hybrid)
"""

from __future__ import annotations
import os
import re
import sys
import argparse
from glob import glob

from hf_retrieval import HFSmartCSVChatbot


# ---------- helpers ----------
_BOT_SINGLETON: HFSmartCSVChatbot | None = None


def ensure_bot() -> HFSmartCSVChatbot:
    """Create/return a process-wide singleton HFSmartCSVChatbot."""
    global _BOT_SINGLETON
    if _BOT_SINGLETON is None:
        bot = HFSmartCSVChatbot()
        # sensible defaults
        bot.use_hybrid = True
        bot.use_embeddings = True
        bot.use_cross_encoder = False
        bot._top_k = 5
        bot._emb_weight = 0.6      # embeddings weight
        bot._tfidf_weight = 0.25   # TF-IDF weight
        bot._min_conf = 0.25
        _BOT_SINGLETON = bot
    return _BOT_SINGLETON


def csv_has_qa_columns(path: str) -> bool:
    """Quick header check for Question/Answer columns (case-insensitive)."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            header = f.readline()
        cols = [c.strip().lower() for c in re.split(r"[,\t;|]", header)]
        has_q = any(c in ("question", "questions", "q", "query") for c in cols)
        has_a = any(c in ("answer", "answers", "a", "response", "solution") for c in cols)
        return has_q and has_a
    except Exception:
        return False


def find_default_csv() -> str | None:
    """
    Find a CSV to auto-load in the project folder:
      1) common names next to this file
      2) any *.csv in this folder that has Question/Answer headers
    """
    base = os.path.dirname(__file__)
    for name in ("school_it_qa.csv", "it_qa.csv", "questions.csv", "data.csv"):
        p = os.path.join(base, name)
        if os.path.exists(p) and csv_has_qa_columns(p):
            return p
    for p in sorted(glob(os.path.join(base, "*.csv"))):
        if csv_has_qa_columns(p):
            return p
    return None


def bootstrap_kb(bot: HFSmartCSVChatbot, override_csv_path: str | None = None) -> tuple[bool, str | None]:
    """
    Load a CSV automatically in this order:
      1) If override_csv_path is provided, load it.
      2) If a previously saved config path exists, load it.
      3) Try to auto-load from project folder (common names, then any CSV w/ Q&A headers).

    Returns: (kb_ready, loaded_csv_path or None)
    """
    # 1) Use explicit override if provided
    if override_csv_path and os.path.exists(override_csv_path) and csv_has_qa_columns(override_csv_path):
        if bot.load_csv_file(override_csv_path, save_to_config=True):
            return True, override_csv_path

    # 2) Try last-used path (saved by bot.save_config)
    try:
        last_path = bot.load_config()
        if last_path and os.path.exists(last_path) and csv_has_qa_columns(last_path):
            if bot.load_csv_file(last_path, save_to_config=False):
                return True, last_path
    except Exception:
        pass

    # 3) Auto-discover from project folder
    auto = find_default_csv()
    if auto and bot.load_csv_file(auto, save_to_config=True):
        return True, auto

    return False, None


def ask(question: str) -> str:
    """Return chatbot response for the given question."""
    bot = ensure_bot()
    return bot.get_response(question)


def _interactive_loop(bot: HFSmartCSVChatbot) -> None:
    print("💬 IT Support Chatbot (type 'exit' to quit)\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            break
        reply = bot.get_response(user_text)
        print(f"Bot: {reply}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IT Support Chatbot (CLI)")
    parser.add_argument("--csv", help="Path to Q&A CSV to load", default=None)
    parser.add_argument("--ask", help="Ask a single question and print the answer", default=None)
    parser.add_argument("--no-embeddings", help="Disable embedding-based retrieval", action="store_true")
    parser.add_argument("--hybrid", help="Enable hybrid retrieval (default: on)", action="store_true")
    parser.add_argument("--no-hybrid", help="Disable hybrid retrieval", action="store_true")
    parser.add_argument("--cross-encoder", help="Enable cross-encoder re-ranking", action="store_true")
    args = parser.parse_args(argv)

    bot = ensure_bot()
    if args.no_embeddings:
        bot.use_embeddings = False
    if args.no_hybrid:
        bot.use_hybrid = False
    if args.hybrid:
        bot.use_hybrid = True
    if args.cross_encoder:
        bot.use_cross_encoder = True

    ready, loaded = bootstrap_kb(bot, args.csv)
    if not ready:
        print("No valid CSV knowledge base found. Place a Q&A CSV next to this script or pass --csv.")
        return 2
    else:
        print(f"Using KB: {os.path.basename(loaded)}")

    if args.ask:
        print(ask(args.ask))
        return 0

    _interactive_loop(bot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
