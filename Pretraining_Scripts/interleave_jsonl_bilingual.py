"""
This script reads JSON-Lines (optionally gzipped) produced by universal_jsonl_builder.py,
finds aligned translations for the same item, and creates a synthetic interleaving
paragraph-by-paragraph with a short prelude at the top.

Pairing:
- Uses language metadata to find roh/de pairs for the same key.
- Keys can be derived from id, filename, url, or file_path.
- Default order is roh then de.

Output:
- Same JSON-Lines schema as before, with source="interleaving" (or "interleaving_orphan").
- "text" contains a prelude and then alternating paragraphs.
- If paragraph counts differ, a warning is printed and the last paragraph of the longer side is appended.

Usage:

  # Auto pairing, roh first then de
  python interleave_jsonl_bilingual.py \
      --src /path/to/in1.jsonl.gz /path/to/in2.jsonl \
      --out /path/to/out_interleaved.jsonl.gz

  # Force pairing by 'id' and split on blank lines
  python interleave_jsonl_bilingual.py \
      --src /dir \
      --out /out.jsonl.gz \
      --pair-by id \
      --paragraph-split blank

  # No paragraph labels
  python interleave_jsonl_bilingual.py \
      --src /dir \
      --out /out.jsonl.gz \
      --no-labels
"""
import argparse
import gzip
import json
import os
import pathlib
import re
import sys
import uuid
from typing import Final, Iterable, Dict, List, Tuple

DEFAULT_LANGUAGE_SCRIPT: Final[str] = "Latn"
DEFAULT_PII_COUNT: Final[int] = 0
DEFAULT_LANGUAGE_SCORE: Final[float] = 0.0

LANG_HINT_PATTERNS: Final[Tuple[re.Pattern, ...]] = (
    re.compile(r"(_deu)(?=\.|$)", re.IGNORECASE),
    re.compile(r"(_de)(?=\.|$)",  re.IGNORECASE),
    re.compile(r"(_rom)(?=\.|$)", re.IGNORECASE),
    re.compile(r"(_rm)(?=\.|$)",  re.IGNORECASE),
    re.compile(r"(\.de)(?=\.|$)", re.IGNORECASE),
    re.compile(r"(\.roh)(?=\.|$)", re.IGNORECASE),
)

PARA_SPLIT_BLANK_RE: Final[re.Pattern] = re.compile(r"\n\s*\n+")
NEWLINES_RE: Final[re.Pattern] = re.compile(r"\r\n|\r")


def create_base_record(
    text: str,
    filename: str = "",
    file_path: str = "",
    language: str = "unknown",
    language_script: str = DEFAULT_LANGUAGE_SCRIPT,
    source: str = "",
    url: str = "",
    idiom: str = "",
    dump: str = "",
    language_score: float = DEFAULT_LANGUAGE_SCORE,
    pii_count: int = DEFAULT_PII_COUNT,
    id: str = "",
) -> dict:
    return {
        "text": text,
        "id": id if id else str(uuid.uuid4()),
        "filename": filename,
        "language": language,
        "language_script": language_script,
        "source": source,
        "file_path": file_path,
        "dump": dump,
        "language_score": language_score,
        "pii_count": pii_count,
        "url": url,
        "idiom": idiom,
    }


def open_jsonl_stream(p: pathlib.Path) -> Iterable[str]:
    if p.suffix.lower() == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield line
    else:
        with p.open("rt", encoding="utf-8") as fh:
            for line in fh:
                yield line


def iter_input_records(srcs: List[str]) -> Iterable[dict]:
    paths: List[pathlib.Path] = []
    for s in srcs:
        sp = pathlib.Path(s).expanduser().resolve()
        if sp.is_dir():
            paths.extend(sorted(list(sp.glob("*.jsonl")) + list(sp.glob("*.jsonl.gz"))))
        elif sp.is_file():
            paths.append(sp)
        else:
            print(f"[skip] Not found: {sp}", file=sys.stderr)

    for p in paths:
        for line in open_jsonl_stream(p):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[skip] {p.name}: JSON decode error: {e}", file=sys.stderr)


def strip_lang_hints(name_wo_ext: str) -> str:
    out = name_wo_ext
    for pat in LANG_HINT_PATTERNS:
        out = pat.sub("", out)
    out = re.sub(r"[._]+", "_", out).strip("_").lower()
    return out


def canon_from_filename(val: str) -> str:
    if not val:
        return ""
    base = os.path.basename(val)
    stem, _ext = os.path.splitext(base)
    return strip_lang_hints(stem)


def canon_from_url(val: str) -> str:
    if not val:
        return ""
    s = val.strip().rstrip("/")
    parts = s.split("/")
    if not parts:
        return ""
    last = parts[-1]
    if last in {"json", "html"} and len(parts) > 1:
        last = parts[-2]
    stem, _ext = os.path.splitext(last)
    return strip_lang_hints(stem)


def canon_from_path(val: str) -> str:
    if not val:
        return ""
    base = os.path.basename(val)
    stem, _ext = os.path.splitext(base)
    return strip_lang_hints(stem)


def choose_pairing_strategy(groups_by, l1: str, l2: str) -> str:
    # prefer a strategy that yields at least one key containing both l1 and l2
    for strategy in ("id", "filename", "url", "file_path"):
        bucket = groups_by[strategy]
        for _key, langs in bucket.items():
            if l1 in langs and l2 in langs:
                return strategy
    return "filename"


def split_paragraphs(text: str, mode: str) -> List[str]:
    if not text:
        return []
    text = NEWLINES_RE.sub("\n", text).strip()
    if not text:
        return []
    if mode == "line":
        return [p.strip() for p in text.split("\n") if p.strip()]
    return [p.strip() for p in PARA_SPLIT_BLANK_RE.split(text) if p.strip()]


def interleave_with_last_append(p1: List[str], p2: List[str], l1: str, l2: str, with_labels: bool) -> str:
    out: List[str] = []
    mn = min(len(p1), len(p2))
    for i in range(mn):
        out.append((f"[{l1}] " if with_labels else "") + p1[i])
        out.append((f"[{l2}] " if with_labels else "") + p2[i])
    if len(p1) != len(p2):
        # append only the last paragraph of the longer side
        if len(p1) > len(p2):
            out.append((f"[{l1}] " if with_labels else "") + p1[-1])
        else:
            out.append((f"[{l2}] " if with_labels else "") + p2[-1])
    return "\n\n".join(out)


def choose_longest(rec_list: List[dict]) -> dict:
    if not rec_list:
        return {}
    return max(rec_list, key=lambda r: len(r.get("text") or ""))


def main():
    parser = argparse.ArgumentParser(
        description="Interleave aligned translations from JSON-Lines into bilingual synthetic records."
    )
    parser.add_argument(
        "--src",
        required=True,
        nargs="+",
        help="Input files or directories (.jsonl / .jsonl.gz)."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output file (.jsonl.gz)."
    )
    parser.add_argument(
        "--pair-by",
        choices=["auto", "id", "filename", "url", "file_path"],
        default="auto",
        help="How to pair aligned translations (default: auto)."
    )
    parser.add_argument(
        "--l1",
        default="roh",
        help="First language code (default: roh)."
    )
    parser.add_argument(
        "--l2",
        default="de",
        help="Second language code (default: de)."
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not prefix paragraphs with language tags."
    )
    parser.add_argument(
        "--paragraph-split",
        choices=["blank", "line"],
        default="blank",
        help="Paragraph detection: 'blank' splits on blank lines (default), 'line' splits on every line."
    )
    parser.add_argument(
        "--prelude",
        default="Prelude: Interleaved edition of '{name}'. This is the same text translated from {L1} \u2192 {L2}, alternating paragraph by paragraph.",
        help="Prelude template. Variables: {name}, {L1}, {L2}."
    )
    parser.add_argument(
        "--keep-orphans",
        action="store_true",
        help="Also output items that have no counterpart (will still include the prelude)."
    )

    args = parser.parse_args()

    out_path = pathlib.Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # strategy -> key -> lang -> [records]
    groups_by: Dict[str, Dict[str, Dict[str, List[dict]]]] = {k: {} for k in ("id", "filename", "url", "file_path")}
    total_in = 0

    for rec in iter_input_records(args.src):
        total_in += 1
        lang = (rec.get("language") or "").strip() or "unknown"
        if not rec.get("text"):
            continue

        key_id = (rec.get("id") or "").strip()
        if key_id:
            groups_by["id"].setdefault(key_id, {}).setdefault(lang, []).append(rec)

        key_fn = canon_from_filename(rec.get("filename") or "")
        if key_fn:
            groups_by["filename"].setdefault(key_fn, {}).setdefault(lang, []).append(rec)

        key_url = canon_from_url(rec.get("url") or "")
        if key_url:
            groups_by["url"].setdefault(key_url, {}).setdefault(lang, []).append(rec)

        key_fp = canon_from_path(rec.get("file_path") or "")
        if key_fp:
            groups_by["file_path"].setdefault(key_fp, {}).setdefault(lang, []).append(rec)

    strategy = args.pair_by
    if strategy == "auto":
        strategy = choose_pairing_strategy(groups_by, args.l1, args.l2)
        print(f"[info] pairing strategy: {strategy}", file=sys.stderr)

    groups = groups_by[strategy]

    pairs_built = 0
    orphans = 0
    mismatches = 0
    skipped = 0

    with gzip.open(out_path, "wt", encoding="utf-8") as gz:
        for key, lang_bucket in sorted(groups.items()):
            has_l1 = args.l1 in lang_bucket
            has_l2 = args.l2 in lang_bucket

            if has_l1 and has_l2:
                r1 = choose_longest(lang_bucket[args.l1])
                r2 = choose_longest(lang_bucket[args.l2])

                p1 = split_paragraphs(r1.get("text") or "", args.paragraph_split)
                p2 = split_paragraphs(r2.get("text") or "", args.paragraph_split)

                if len(p1) != len(p2):
                    print(
                        f"[warn] paragraph count mismatch for key='{key}' ({args.l1}:{len(p1)} vs {args.l2}:{len(p2)}). Appending only the last paragraph of the longer side.",
                        file=sys.stderr,
                    )
                    mismatches += 1

                inter = interleave_with_last_append(p1, p2, args.l1, args.l2, not args.no_labels)
                name_for_prelude = key or (r1.get("filename") or r2.get("filename") or "")
                prelude = (args.prelude or "").format(name=name_for_prelude, L1=args.l1, L2=args.l2).strip()
                full_text = (prelude + "\n\n" + inter) if prelude else inter

                idiom = r1.get("idiom") if args.l1 == "roh" else (r2.get("idiom") if args.l2 == "roh" else "")

                out_filename = f"{key}.{args.l1}-{args.l2}.interleaved.txt" if key else f"{r1.get('filename','')}.{args.l1}-{args.l2}.interleaved.txt"

                dump_meta = {
                    "pair_key": key,
                    "left": {"filename": r1.get("filename"), "id": r1.get("id"), "language": args.l1},
                    "right": {"filename": r2.get("filename"), "id": r2.get("id"), "language": args.l2},
                    "pairing_strategy": strategy,
                }

                out_rec = create_base_record(
                    text=full_text,
                    filename=out_filename,
                    file_path="",
                    language="unknown",
                    language_script=DEFAULT_LANGUAGE_SCRIPT,
                    source="interleaving",
                    url="",
                    idiom=idiom or "",
                    dump=json.dumps(dump_meta, ensure_ascii=False),
                )
                gz.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                pairs_built += 1
            else:
                if not args.keep_orphans:
                    skipped += 1
                    continue

                side_lang = args.l1 if has_l1 else args.l2 if has_l2 else "unknown"
                recs = lang_bucket.get(side_lang, [])
                if not recs:
                    skipped += 1
                    continue
                r = choose_longest(recs)

                ps = split_paragraphs(r.get("text") or "", args.paragraph_split)
                body = "\n\n".join(((f"[{side_lang}] " + p) if not args.no_labels else p) for p in ps)

                name_for_prelude = key or (r.get("filename") or "")
                prelude = (args.prelude or "").format(name=name_for_prelude, L1=args.l1, L2=args.l2).strip()
                full_text = (prelude + "\n\n" + body) if prelude else body

                out_filename = f"{key}.{side_lang}.orphan.interleaved.txt" if key else f"{r.get('filename','')}.{side_lang}.orphan.interleaved.txt"

                out_rec = create_base_record(
                    text=full_text,
                    filename=out_filename,
                    file_path="",
                    language="unknown",
                    language_script=DEFAULT_LANGUAGE_SCRIPT,
                    source="interleaving_orphan",
                    url="",
                    idiom=r.get("idiom") or "",
                    dump=json.dumps({"pair_key": key, "orphan_language": side_lang, "id": r.get("id"), "pairing_strategy": strategy}, ensure_ascii=False),
                )
                gz.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                orphans += 1

    print(
        f"Wrote {out_path}  (input: {total_in}, pairs: {pairs_built}, orphans: {orphans}, mismatches: {mismatches}, skipped: {skipped})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
