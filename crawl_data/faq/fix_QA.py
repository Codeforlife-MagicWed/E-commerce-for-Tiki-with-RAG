import json, re
from pathlib import Path
from urllib.parse import urlparse

def is_generic_title(title: str) -> bool:
    t = (title or "").strip().lower()
    return t in {
        "tiki knowledge base", "knowledge base", "tiki kb",
        "tiki help center", "tiki support", "tiki hỗ trợ"
    }

def title_from_slug(url: str) -> str:
    try:
        path = urlparse(url).path
        last = path.rsplit("/", 1)[-1]
        parts = last.split("-")
        if parts and parts[0].isdigit():
            parts = parts[1:]
        slug = " ".join(parts).strip().replace("-", " ")
        slug = re.sub(r"\s+", " ", slug)
        slug = slug.strip(" /_-")
        return slug.capitalize() if slug else ""
    except Exception:
        return ""

META_PATTERNS = [
    r"^cập nhật lần cuối", r"^last updated", r"^luợt xem", r"^lượt xem", r"^views?:?",
    r"^hotline", r"^chat với trợ lý", r"^gửi yêu cầu", r"^gửi email",
    r"^website .* dấu", r"^kênh zalo", r"^⚠️? *lưu ý", r"^\(?faq\)?", r"^related articles?",
]

def _normalize_line(ln: str) -> str:
    s = ln.strip()
    s = re.sub(r"^[•\-–—●▶️👉✅\(\)\[\]★☆✓\s]+", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def is_meta(ln: str) -> bool:
    s = _normalize_line(ln).lower()
    return any(re.search(p, s, flags=re.I) for p in META_PATTERNS)

def pick_question_from_body(raw_text: str) -> tuple[str, str]:
    if not raw_text:
        return "", ""

    lines = [ln.rstrip() for ln in raw_text.splitlines()]
    lines_norm = [_normalize_line(ln) for ln in lines]

    q_idx, question = None, ""
    for i, ln in enumerate(lines_norm[:5]):
        if is_meta(ln):
            continue
        if ln.endswith("?") and 5 <= len(ln) <= 160:
            q_idx, question = i, ln
            break

    if not question and lines_norm:
        first = lines_norm[0]
        m = re.split(r"\b(Cập nhật lần cuối|Last updated)\b", first, flags=re.I)
        if len(m) >= 2:
            maybe_q = m[0].strip(" :-|")
            if 5 <= len(maybe_q) <= 200 and not is_meta(maybe_q):
                question = maybe_q

    if not question:
        for ln in lines_norm[:6]:
            if not ln or is_meta(ln):
                continue
            if 5 <= len(ln) <= 200:
                question = ln.strip(" :-|")
                break

    cleaned = []
    for ln_raw, ln in zip(lines, lines_norm):
        if not ln:
            continue
        if question and ln.strip() == question.strip():
            continue
        if is_meta(ln):
            continue
        if re.fullmatch(r"#{1,6}\s*", ln):
            continue
        cleaned.append(_normalize_line(ln_raw))
    answer = "\n".join([s for s in cleaned if s]).strip()

    return question, answer

in_file = Path("tiki_help_center.json")
out_file = Path("tiki_help_center_fixed_2.json")

data = json.loads(in_file.read_text(encoding="utf-8"))

fixed = []
for rec in data:
    title = rec.get("title", "")
    answer = rec.get("answer", "") or ""
    url = rec.get("url", "") or ""

    if is_generic_title(title) or "/knowledge-base/post/" in url:
        q, ans = pick_question_from_body(answer)
        if not q:
            q = title_from_slug(url)
        if q:
            rec["question"] = q
        if ans:
            rec["answer"] = ans
    else:
        _, ans = pick_question_from_body(answer)
        if ans:
            rec["answer"] = ans
        if is_generic_title(title) and not rec.get("question"):
            rec["question"] = title_from_slug(url) or title

    words = (rec.get("answer") or "").split()
    chunks, size, overlap = [], 1200, 200
    if words:
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+size]).strip()
            if chunk:
                chunks.append(chunk)
            if i + size >= len(words):
                break
            i += max(1, size - overlap)
    rec["chunks"] = chunks

    fixed.append(rec)

out_file.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Đã xử lý xong, ghi ra {out_file} với {len(fixed)} record.")
