#!/usr/bin/env python3
"""
Main entry point for the CSIT5970 Custom Agent.

Usage examples:
  # Interactive REPL
  python main.py

  # Single question with default config
  python main.py --question "How do I find all SUID files in Linux?"

  # Use a specific persona
  python main.py --persona linux_expert --question "Tune kernel for PostgreSQL"

  # Run benchmark evaluation
  python main.py --benchmark

  # Index the knowledge base
  python main.py --index-kb
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure repo root is on the path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()  # Load .env if present


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_repl(agent) -> None:
    """Interactive question-answer REPL."""
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    console.print("\n[bold green]CSIT5970 Custom Agent — Interactive Mode[/bold green]")
    console.print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if question.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break
        if not question:
            continue

        console.print("[dim]Thinking...[/dim]")
        context_docs = agent.memory_mgr.retrieve(question)
        result = agent.run(question, context_docs=context_docs)

        console.print(f"\n[bold blue]Agent:[/bold blue]")
        try:
            console.print(Markdown(result.answer))
        except Exception:
            console.print(result.answer)
        console.print(
            f"\n[dim]({len(result.steps)} step(s), {result.total_tokens} tokens)[/dim]\n"
        )


def run_single_question(agent, question: str, persona: str) -> None:
    """Answer a single question and print the result."""
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    context_docs = agent.memory_mgr.retrieve(question)
    result = agent.run(question, persona=persona, context_docs=context_docs)

    console.print(f"\n[bold]Question:[/bold] {question}\n")
    console.print("[bold]Answer:[/bold]")
    try:
        console.print(Markdown(result.answer))
    except Exception:
        console.print(result.answer)

    if result.steps:
        console.print(f"\n[dim]Reasoning steps: {len(result.steps)} | Tokens: {result.total_tokens}[/dim]")


def run_benchmark(config_path: str) -> None:
    """Run the full benchmark evaluation."""
    from rich.console import Console
    console = Console()
    console.print("\n[bold yellow]Starting benchmark evaluation...[/bold yellow]")

    from benchmark.evaluator import BenchmarkEvaluator
    evaluator = BenchmarkEvaluator(config_path=config_path)
    reports = evaluator.run()
    evaluator.print_summary(reports)
    console.print(f"\n[green]Reports saved to: {evaluator._output_dir}[/green]")


def index_knowledge_base(config_path: str) -> None:
    """Index all documents in data/knowledge_base/ into the vector store."""
    import yaml
    from rich.console import Console
    from rag.vectorstore import VectorStore
    from rag.indexer import DocumentIndexer

    console = Console()

    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    rag_cfg = config.get("rag", {})
    store = VectorStore(
        collection_name=rag_cfg.get("collection_name", "knowledge_base"),
        persist_directory=rag_cfg.get("persist_directory", "./data/chroma_db"),
        embedding_model=rag_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
    )
    indexer = DocumentIndexer(store)

    kb_dir = "data/knowledge_base"
    console.print(f"[bold]Indexing documents from:[/bold] {kb_dir}")
    total = indexer.index_directory(kb_dir)
    console.print(f"[green]✓ Indexed {total} chunks. Total in store: {store.count()}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSIT5970 Custom Agent – deploy, run, and evaluate an autonomous agent."
    )
    parser.add_argument("--question", "-q", help="Single question to answer")
    parser.add_argument(
        "--persona", "-p",
        default=None,
        choices=["general", "linux_expert", "security_expert"],
        help="Agent persona to use (default: from config)",
    )
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark evaluation")
    parser.add_argument("--index-kb", action="store_true", help="Index the knowledge base")
    parser.add_argument(
        "--config", "-c",
        default="config/agent_config.yaml",
        help="Path to agent config YAML",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.benchmark:
        run_benchmark(args.config)
        return

    if args.index_kb:
        index_knowledge_base(args.config)
        return

    # Initialise the agent
    from agent.core import Agent
    agent = Agent(config_path=args.config)

    if args.question:
        persona = args.persona or agent.config.get("persona", {}).get("default", "general")
        run_single_question(agent, args.question, persona)
    else:
        run_repl(agent)


if __name__ == "__main__":
    main()
