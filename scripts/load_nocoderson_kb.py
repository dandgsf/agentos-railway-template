"""
NoCoderSon KB Loader
--------------------

Carrega a knowledge base do NoCoderSon a partir do `knowledge-base.jsonl`
exportado pelo pipeline `collect_kb.py` (ver `knowledge-base/exports/`).

Filtros (beginner-first):
- exclui `status in (deprecated, community)`
- inclui apenas docs que tem pelo menos 1 dos temas didaticos
  (educational, prompting, framework, cookbook, sdk-agents)
- alem disso, sempre carrega os arquivos de `knowledge-base/concepts/` como
  prioridade maxima (resumos curados a mao do curso)

Env:
- KNOWLEDGE_BASE_JSONL  caminho do knowledge-base.jsonl
- KNOWLEDGE_BASE_CONCEPTS_DIR  caminho opcional da pasta concepts/

Uso:
    python -m scripts.load_nocoderson_kb --dry     # conta sem carregar
    python -m scripts.load_nocoderson_kb           # carga real
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from agents.nocoderson_agent import knowledge

BEGINNER_THEMES = {
    "educational",
    "prompting",
    "framework",
    "cookbook",
    "sdk-agents",
}

EXCLUDED_STATUS = {"deprecated", "community"}

# Content > 500KB geralmente é notebook com screenshots base64 embutidos, nao texto util.
# Exemplo real no corpus: openai-cookbook/computer_use_with_daytona (8MB, vira 8000+ chunks).
MAX_CONTENT_SIZE = 500_000


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] linha {line_num} invalida: {exc}", file=sys.stderr)


def _should_include(row: dict[str, Any]) -> bool:
    if row.get("status") in EXCLUDED_STATUS:
        return False
    themes = set(row.get("themes") or [])
    if not (themes & BEGINNER_THEMES):
        return False
    content = row.get("content") or ""
    size = len(content.strip())
    if size < 120:
        return False
    if size > MAX_CONTENT_SIZE:
        return False
    return True


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "vendor": row.get("vendor"),
        "repo": row.get("repo"),
        "path": row.get("path"),
        "themes": list(row.get("themes") or []),
        "source_url": row.get("source_url"),
        "license": row.get("license_spdx"),
        "status": row.get("status"),
    }


def _load_concepts(concepts_dir: Path, dry: bool) -> int:
    count = 0
    if not concepts_dir.is_dir():
        print(f"[info] concepts dir nao encontrado: {concepts_dir} (pulando)")
        return 0
    for md_path in sorted(concepts_dir.glob("*.md")):
        if dry:
            print(f"[dry] concept: {md_path.name}")
        else:
            knowledge.insert(
                name=f"concept: {md_path.stem}",
                path=str(md_path),
                metadata={
                    "source": "curated_concepts",
                    "path": str(md_path.name),
                    "priority": "high",
                },
                skip_if_exists=True,
            )
        count += 1
    return count


def _load_jsonl(jsonl_path: Path, dry: bool, limit: int | None) -> Counter:
    counts: Counter = Counter()
    total = 0
    for row in _iter_jsonl(jsonl_path):
        total += 1
        if not _should_include(row):
            counts["skipped"] += 1
            continue
        counts[row.get("vendor") or "unknown"] += 1
        counts["included"] += 1
        if limit and counts["included"] > limit:
            counts["included"] -= 1
            break
        if dry:
            continue
        try:
            knowledge.insert(
                name=row.get("title") or row.get("path") or f"doc_{total}",
                text_content=row["content"],
                metadata=_metadata(row),
                skip_if_exists=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] falha ao inserir {row.get('id')}: {exc}",
                file=sys.stderr,
            )
            counts["errors"] += 1
    counts["total"] = total
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Load NoCoderSon knowledge base")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="apenas conta sem inserir (nao gasta embeddings)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limite maximo de docs carregados do JSONL (debug)",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=os.getenv("KNOWLEDGE_BASE_JSONL"),
        help="caminho do knowledge-base.jsonl",
    )
    parser.add_argument(
        "--concepts-dir",
        type=str,
        default=os.getenv("KNOWLEDGE_BASE_CONCEPTS_DIR"),
        help="caminho da pasta concepts/ (opcional)",
    )
    args = parser.parse_args()

    if not args.jsonl:
        print(
            "erro: defina --jsonl ou a env var KNOWLEDGE_BASE_JSONL",
            file=sys.stderr,
        )
        return 2

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_file():
        print(f"erro: arquivo nao encontrado: {jsonl_path}", file=sys.stderr)
        return 2

    print(f"[info] modo: {'DRY RUN' if args.dry else 'LOAD'}")
    print(f"[info] jsonl: {jsonl_path}")

    concepts_loaded = 0
    if args.concepts_dir:
        concepts_loaded = _load_concepts(Path(args.concepts_dir), args.dry)
        print(f"[info] concepts processados: {concepts_loaded}")

    counts = _load_jsonl(jsonl_path, args.dry, args.limit)

    print("\n=== Resumo ===")
    print(f"total lidos:        {counts.get('total', 0)}")
    print(f"incluidos:          {counts.get('included', 0)}")
    print(f"pulados (filtros):  {counts.get('skipped', 0)}")
    print(f"erros:              {counts.get('errors', 0)}")
    print(f"concepts:           {concepts_loaded}")
    print("\n--- por vendor (incluidos) ---")
    vendor_counts = {
        k: v
        for k, v in counts.items()
        if k not in {"total", "included", "skipped", "errors"}
    }
    for vendor, n in sorted(vendor_counts.items(), key=lambda x: -x[1]):
        print(f"  {vendor:20s} {n:6d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
