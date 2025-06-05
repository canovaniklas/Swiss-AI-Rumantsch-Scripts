from __future__ import annotations
import argparse
import gzip
import json
import pathlib
import sys
from typing import Any, Iterable, Optional, List
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

TOKENIZER_NAME = "alehc/swissai-tokenizer"
hf_logging.set_verbosity_error()

def _iter_lines(path: pathlib.Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from fh
    else:
        with path.open(encoding="utf-8") as fh:
            yield from fh

def _iter_records(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    for i, raw in enumerate(_iter_lines(path), 1):
        if not raw.strip():
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"{path}: line {i}: JSON error ({exc}) – skipping", file=sys.stderr)

def _all_strings(obj: Any) -> List[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out: List[str] = []
        for v in obj.values():
            out.extend(_all_strings(v))
        return out
    if isinstance(obj, list):
        out: List[str] = []
        for v in obj:
            out.extend(_all_strings(v))
        return out
    return []

def _strings_for_field(obj: dict[str, Any], field: str) -> List[str]:
    if field == "*":
        return _all_strings(obj)
    val = obj.get(field, "")
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return [x for x in val if isinstance(x, str)]
    return []

def _count_tokens_batch(strings: List[str], tokenizer) -> int:
    if not strings:
        return 0
    enc = tokenizer(strings, add_special_tokens=False, truncation=False, return_length=True)
    return int(sum(int(x) for x in enc["length"]))

def process_file(
    path: pathlib.Path,
    tokenizer,
    field: str,
    head_len: Optional[int],
    quiet: bool,
    no_metadata: bool,
) -> tuple[int, int]:
    total_lines = 0
    total_tokens = 0
    for lineno, obj in enumerate(_iter_records(path), 1):
        total_lines = lineno
        strings = _strings_for_field(obj, field)
        total_tokens += _count_tokens_batch(strings, tokenizer)
        if not quiet:
            if field == "*":
                preview_source = obj.get("text", strings[0] if strings else "")
            else:
                preview_source = strings[0] if strings else ""
            snippet = str(preview_source).replace("\n", " ")
            if head_len is not None:
                snippet = snippet[:head_len]
            meta_str = ""
            if not no_metadata:
                keys = ("language", "idiom", "language_script", "filename", "source", "url")
                md = {k: v for k, v in obj.items() if k in keys and v}
                if isinstance(obj.get("metadata"), dict):
                    for k in keys:
                        v = obj["metadata"].get(k)
                        if v:
                            md[k] = v
                if md.get("language_script") == "Latn":
                    md.pop("language_script")
                if md:
                    meta_str = json.dumps(md, ensure_ascii=False) + "  |  "
            print(f"{lineno:>6}: {meta_str}{snippet}")
    print(f"\n{path.name}: {total_lines} lines  |  {total_tokens:,} tokens\n")
    return total_lines, total_tokens

def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect jsonl / jsonl.gz datasets and count tokens.")
    p.add_argument("input", help="File or folder containing *.jsonl(.gz) files", type=pathlib.Path)
    p.add_argument("--field", default="text", help='JSON key to read (use "*" to sum every string value)')
    p.add_argument("--head", default=10, type=int, metavar="N", help="Characters to show from each record for the preview (default: 100). Ignored if --no-truncate is used.")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress per‑record preview output")
    p.add_argument("--no-truncate", action="store_true", help="Do not truncate the text snippet in the preview; show entire text. Overrides --head.")
    p.add_argument("--no-metadata", action="store_true", help="Do not print metadata in the per-record preview.")
    return p

def gather_files(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        name = root.name
        if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
            return [root]
        raise SystemExit(f"Unsupported file type: {root}. Expected .jsonl or .jsonl.gz")
    files = list(root.rglob("*.jsonl")) + list(root.rglob("*.jsonl.gz"))
    if not files:
        raise SystemExit(f"No *.jsonl(.gz) files found in {root}")
    return sorted(files)

def main() -> None:
    args = make_argparser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    grand_lines = grand_tokens = 0
    files = gather_files(args.input)
    effective_head_len = None if args.no_truncate else args.head
    for fp in files:
        lines, tokens = process_file(
            fp,
            tokenizer,
            args.field,
            effective_head_len,
            args.quiet,
            args.no_metadata,
        )
        grand_lines += lines
        grand_tokens += tokens
    if len(files) > 1:
        print("=" * 60)
        print(f"TOTAL: {grand_lines} lines  |  {grand_tokens:,} tokens")

if __name__ == "__main__":
    main()
