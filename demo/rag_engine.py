
import os
import json
import math
import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, SearchParams
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


try:
    from qdrant_client.http.exceptions import UnexpectedResponse
except Exception:
    class UnexpectedResponse(Exception):
        pass


STRICT_CATEGORY = False
BRAND_STRICT = False
USE_RERANK = True

P_HINT = re.compile(
    r"(giá|mua|đặt|kích\s*thước|dung\s*tích|màu|model|thương\s?hiệu|bảo\s*hành|ml|cm|mm|inch|w|hz|gb|ram|ssd"
    r"|đồ\s*chơi|thú\s*cưng|pet|mèo|chó)",
    re.I,
)
F_HINT = re.compile(
    r"(cách|hướng\s*dẫn|làm\s*sao|làm\s*thế\s*nào|quy\s*định|chính\s*sách|điều\s*kiện|đổi\s*trả|hủy|đăng\s?k|đăng\s?nhập|trả\s*góp|hóa\s*đơn|vat|bồi\s*thường|khiếu\s*nại|xử\s*lý|vấn\s*đề|mong\s*muốn|giải\s*quyết|bị\s*lỗi)",
    re.I,
)
RATING_HINT = re.compile(r"\b(\d(?:\.\d)?)\s*sao\b|đánh\s*giá", re.I)

VN_NUM = {
    "k": 1_000,
    "nghin": 1_000, "nghìn": 1_000,
    "tr": 1_000_000, "trieu": 1_000_000, "triệu": 1_000_000,
    "m": 1_000_000,
    "ty": 1_000_000_000, "tỷ": 1_000_000_000,
}

RE_PRICE_RANGE = re.compile(
    r"(?:từ|tu)\s*([0-9\.\,]+(?:\s*\S+)?)\s*(?:đến|-|tới)\s*([0-9\.\,]+(?:\s*\S+)?)",
    re.I,
)
RE_PRICE_GTE = re.compile(
    r"(?:>=|>\s*=?|ít\s*nhất|tối\s*thiểu|trên|từ)\s*([0-9\.\,]+(?:\s*\S+)?)",
    re.I,
)
RE_PRICE_LTE = re.compile(
    r"(?:<=|<\s*=?|tối\s*đa|max|dưới|không\s*vượt)\s*([0-9\.\,]+(?:\s*\S+)?)",
    re.I,
)
RE_PRICE_ANY = re.compile(
    r"(?:giá\s*~?\s*|≈)?\s*([0-9\.\,]+(?:\s*\S+)?)",
    re.I,
)

RE_RATE_RANGE = re.compile(r"(\d(?:\.\d)?)\s*-\s*(\d(?:\.\d)?)\s*sao", re.I)
RE_RATE_GTE = re.compile(r"(?:>=|trở\s*lên|từ)\s*(\d(?:\.\d)?)\s*sao", re.I)
RE_RATE_LTE = re.compile(r"(?:<=|tối\s*đa|đến|tới|không\s*quá)\s*(\d(?:\.\d)?)\s*sao", re.I)
RE_RATE_ANY = re.compile(r"(\d(?:\.\d)?)\s*sao", re.I)

RE_REV_GTE = re.compile(r"(?:>=|từ|ít\s*nhất)\s*([0-9]{2,})\s*(?:đánh\s*giá|reviews?)", re.I)

RE_FAQ_HINTS = re.compile(
    r"(thanh\s*to[aá]n|payment|momo|zalopay|atm|visa|master\s*card|"
    r"tr[aả]\s*g[oó]p|installment|paylater|tr[aả]\s*sau|"
    r"ho[aà]n\s*ti[eề]n|refund|"
    r"\b(?:[đd]ổi|doi)\s*tr[aả]|return|exchange|"
    r"b[ảa]o\s*h[aà]nh|warranty|"
    r"giao\s*h[aà]ng|v[ậa]n\s*chuy[eể]n|ph[ií]\s*ship|"
    r"h[ủu]y\s*đ[ơo]n|cancel\s*order|"
    r"tr[aạ]ng\s*th[aá]i\s*đ[ơo]n|order\s*status|"
    r"v[ơô]ucher|m[aã]\s*gi[ảa]m\s*gi[aá]|coupon|xu\s*tiki|"
    r"l[aắ]p\s*đ[ặa]t|installation|"
    r"t[aà]i\s*kho[aả]n|login|[đd][ăă]ng\s*nh[aâ]p|"
    r"li[êe]n\s*h[eệ]|ch[aă]m\s*s[oóc]c\s*k[hà]ch\s*h[aà]ng)",
    re.I
)

RE_TECH_UNITS = re.compile(r"(ml|l|mm|cm|inch|\"|hz|w|gb|tb|mah)\b", re.I)

TAIL_NUM = re.compile(r"(\d+)$")
ID_RE_HASH_ONLY = re.compile(r"^product::([^:]+)")
ID_RE_MIDDLE = re.compile(r"^product::([^:]+)(?:::\d+)?$")

CATEGORIES: Dict[str, List[str]] = {
    "Bình giữ nhiệt": ["bình giữ nhiệt", "binh giu nhiet", "thermos", "flask", "chai giữ nhiệt"],
    "Balo": ["balo", "ba lô", "backpack", "túi đeo lưng"],
}

DEDUP_PER_PARENT = 2
DEDUP_BY_URL = True
DEDUP_FAQ_NEAR = True
DEDUP_FAQ_PER_PARENT = 2

DEF_COLS = [
    "id", "point_id", "score", "title", "category", "url", "type", "parent_uid",
    "brand", "thumbnail", "price", "rating", "reviews", "source", "text"
]

DETAIL_TERMS = [
    "thông số", "chi tiết", "cấu hình", "bao lâu", "giữ nóng",
    "giữ lạnh", "dung tích", "kích thước", "bao nhiêu w", "bao nhiêu watt",
    "bảo hành", "xuất xứ", "chống nước", "có tốt không", "hợp không",
]

BROWSE_TERMS = [
    "loại nào", "nên mua", "gợi ý", "tư vấn", "tầm", "khoảng",
    "dưới", "trên", "giữa", "so sánh", "top", "phù hợp",
]

MODEL_PATTERN = re.compile(r"\b[A-Z0-9]{2,}[-_ ]?[A-Z0-9]{2,}\b")

DETAIL_PATTERNS = [
    r"mấy\s+sim",
    r"may\s+sim",
    r"bao\s+nhiêu\s+sim",
    r"bao\s+nhiu\s+sim",
    r"esim",
    r"e\s*sim",
    r"có\s+chống\s+nước\s+không",
    r"co\s+chong\s+nuoc\s+khong",
    r"có\s+sạc\s+nhanh\s+không",
    r"co\s+sac\s+nhanh\s+khong",
    r"bao\s+nhiêu\s+gb",
    r"bao\s+nhiu\s+gb",
    r"ram\s+mấy\s*gb",
    r"rom\s+mấy\s*gb",
]

INFO_DETAIL_RE = re.compile(
    r"thông\s*tin\s*(chi\s*tiết\s*)?(sản\s*phẩm|ve)\b",
    re.I
)

INFO_TERMS = [
    "thông tin",
    "thong tin",
    "giới thiệu",
    "gioi thieu",
    "review",
    "đánh giá sản phẩm",
    "danh gia san pham",
]


def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _norm(s: str) -> str:
    """Normalize string: remove diacritics, non-alnum, lowercase"""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _norm_sig(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s[:180]

def _norm_sparse_query(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^\w\s\-\%\./]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_text_for_sparse(s: str) -> str:
    s = strip_accents(str(s)).lower()
    s = re.sub(r"[^\w\s\-\%\./]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _safe(x, default=""):
    return default if x is None else str(x)

def tokenize(s: str) -> List[str]:
    s = normalize_text_for_sparse(s)
    toks = s.split()
    out = []
    for t in toks:
        if len(t) == 1 and not t.isdigit():
            continue
        out.append(t)
    return out


def _parse_vn_money(s: str) -> Optional[int]:
    """Parse Vietnamese money format"""
    if not s:
        return None

    s = s.strip().lower().replace(" ", "")
    s = re.sub(r"\b(vnđ|vnd|đồng)\b", "", s).strip()
    s_ascii = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s_ascii = s_ascii.replace("đ", "")
    s_ascii = s_ascii.replace(" ", "")
    s_ascii = s_ascii.replace(",", "").replace(".", "")

    if not s_ascii:
        return None

    VN_NUM_ASCII = {
        "k": 1_000,
        "nghin": 1_000,
        "ngan": 1_000,
        "tr": 1_000_000,
        "trieu": 1_000_000,
        "m": 1_000_000,
        "ty": 1_000_000_000,
    }

    for suf, mul in VN_NUM_ASCII.items():
        if s_ascii.endswith(suf):
            num_part = s_ascii[:-len(suf)] or "0"
            try:
                return int(float(num_part) * mul)
            except Exception:
                return None

    try:
        return int(float(s_ascii))
    except Exception:
        return None

def _parse_price_slots(ql: str) -> Tuple[Optional[int], Optional[int]]:
    gte = lte = None

    if m := RE_PRICE_RANGE.search(ql):
        gte, lte = _parse_vn_money(m.group(1)), _parse_vn_money(m.group(2))
    if m := RE_PRICE_GTE.search(ql):
        gte = _parse_vn_money(m.group(1)) or gte
    if m := RE_PRICE_LTE.search(ql):
        lte = _parse_vn_money(m.group(1)) or lte

    if gte is None and lte is None:
        has_price_hint = bool(re.search(
            r"(giá|khoảng|tầm|dưới|trên|<=|>=|tối\s*đa|tối\s*thiểu|rẻ|đắt|triệu|tr|nghìn|ngàn|tỷ|vnđ|vnd|đ\b)",
            ql,
            re.I
        ))
        if has_price_hint:
            if m := RE_PRICE_ANY.search(ql):
                val = _parse_vn_money(m.group(1))
                if val:
                    lte = val

    return gte, lte

def _parse_rating_slots(ql: str) -> Tuple[Optional[float], Optional[float]]:
    gte = lte = None
    if m := RE_RATE_RANGE.search(ql):
        gte, lte = float(m.group(1)), float(m.group(2))
    if m := RE_RATE_GTE.search(ql):
        gte = float(m.group(1))
    if m := RE_RATE_LTE.search(ql):
        lte = float(m.group(1))
    if gte is None and lte is None:
        if m := RE_RATE_ANY.search(ql):
            gte = float(m.group(1))
    if gte is not None:
        gte = max(0.0, min(5.0, gte))
    if lte is not None:
        lte = max(0.0, min(5.0, lte))
    return gte, lte


def resolve_category_syn(q: str) -> Optional[str]:
    ql = q.lower()
    for cat, syns in CATEGORIES.items():
        for s in syns:
            if s in ql:
                return cat
    return None

def is_faqish_query(q: str) -> bool:
    return bool(RE_FAQ_HINTS.search(q))

def is_productish_query(q: str) -> bool:
    ql = q.lower()
    try:
        if resolve_category_syn(ql):
            return True
    except Exception:
        pass
    if RE_TECH_UNITS.search(ql):
        return True
    if re.search(r"\b[A-Z0-9]{3,}[-_\.]?[A-Z0-9]{2,}\b", unicodedata.normalize("NFKD", q), re.I):
        return True
    return False

def route_strength(q: str) -> tuple[int, int]:
    ql = q.lower()
    p = len(P_HINT.findall(ql))
    f = len(F_HINT.findall(ql))
    if RATING_HINT.search(ql):
        p += 1
    if RE_PRICE_GTE.search(ql) or RE_PRICE_LTE.search(ql) or RE_PRICE_ANY.search(ql):
        p += 1
    if re.search(r"\bđánh\s*giá\b|reviews?", ql, re.I):
        p += 1
    return p, f

def route(query: str) -> str:
    q = query or ""
    try:
        p_sig, f_sig = route_strength(q)
    except Exception:
        p_sig, f_sig = 0, 0

    if is_faqish_query(q) or (f_sig > p_sig):
        return "faq"
    if is_productish_query(q) and (p_sig >= f_sig):
        return "product"
    if p_sig >= f_sig + 1:
        return "product"
    if f_sig >= p_sig + 1:
        return "faq"
    return "both"

def parse_slots(query: str, known_brands: Optional[List[str]] = None) -> Dict[str, Optional[object]]:
    out = {
        "category": None,
        "brand": None,
        "price_gte": None,
        "price_lte": None,
        "rating_gte": None,
        "rating_lte": None,
        "reviews_gte": None,
    }
    ql = query.lower()
    qn = _norm(query)
    out["category"] = resolve_category_syn(ql)
    if known_brands:
        nb = {b: _norm(b) for b in known_brands}
        for b, bn in nb.items():
            if bn and bn in qn:
                out["brand"] = b
                break

    pg, pl = _parse_price_slots(ql)
    out["price_gte"], out["price_lte"] = pg, pl

    rg, rl = _parse_rating_slots(ql)
    out["rating_gte"], out["rating_lte"] = rg, rl

    if m := RE_REV_GTE.search(ql):
        try:
            out["reviews_gte"] = int(m.group(1))
        except Exception:
            pass

    return out


def dedup_product(df: pd.DataFrame, per_parent: int = DEDUP_PER_PARENT) -> pd.DataFrame:
    if df.empty:
        return df
    base = df.sort_values(["score"], ascending=False)
    out = (base.groupby("parent_uid", as_index=False, sort=False).head(per_parent)
           .reset_index(drop=True))
    if DEDUP_BY_URL and "url" in out.columns:
        out = out.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return out

def dedup_faq(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if "parent_uid" in out.columns:
        out = (
            out.sort_values("score", ascending=False)
            .groupby("parent_uid", as_index=False, sort=False)
            .head(DEDUP_FAQ_PER_PARENT)
        )

    if DEDUP_BY_URL and "url" in out.columns:
        out = out.drop_duplicates(subset=["url"])

    if DEDUP_FAQ_NEAR and "text" in out.columns:
        out["_sig"] = out["text"].fillna("").map(_norm_sig)
        out = out.drop_duplicates(subset=["_sig"]).drop(columns=["_sig"])

    return out.reset_index(drop=True)

def dedup_merged(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df

def build_filter(
    category: Optional[str] = None,
    brands: Optional[List[str]] = None,
    parent_uid: Optional[str] = None,
    dtype: Optional[str] = None,
    price_gte: Optional[float] = None,
    price_lte: Optional[float] = None,
    rating_gte: Optional[float] = None,
    rating_lte: Optional[float] = None,
    review_count_gte: Optional[int] = None,
    review_count_lte: Optional[int] = None,
) -> Optional[Filter]:
    must, should, must_not = [], [], []

    if category:
        should.append(FieldCondition(key="category", match=MatchValue(value=category)))

    if brands:
        brands = [b for b in brands if b]
        if brands:
            should.extend([FieldCondition(key="brand", match=MatchValue(value=b)) for b in brands])

    if parent_uid:
        must.append(FieldCondition(key="parent_uid", match=MatchValue(value=parent_uid)))
    if dtype:
        must.append(FieldCondition(key="type", match=MatchValue(value=dtype)))

    if price_gte is not None or price_lte is not None:
        must.append(FieldCondition(key="price_numeric", range=Range(gte=price_gte, lte=price_lte)))
    if rating_gte is not None or rating_lte is not None:
        must.append(FieldCondition(key="rating", range=Range(gte=rating_gte, lte=rating_lte)))
    if review_count_gte is not None or review_count_lte is not None:
        must.append(FieldCondition(key="review_count", range=Range(gte=review_count_gte, lte=review_count_lte)))

    if not (must or should or must_not):
        return None
    return Filter(must=must, should=should, must_not=must_not)

def qdrant_search(client, collection: str, qv: np.ndarray, topk: int = 8,
                  flt: Optional[Filter] = None, hnsw_ef: int = 128) -> pd.DataFrame:
    res = client.query_points(
        collection_name=collection,
        query=qv,
        limit=topk,
        query_filter=flt,
        search_params=SearchParams(hnsw_ef=hnsw_ef),
        with_payload=True,
        with_vectors=False,
    )
    points = getattr(res, "points", res)

    rows = []
    for p in points or []:
        pl = getattr(p, "payload", None) or {}
        rows.append({
            "id": pl.get("id") or "",
            "point_id": str(getattr(p, "id", "")),
            "score": float(getattr(p, "score", 0.0)),
            "title": pl.get("title"),
            "category": pl.get("category"),
            "url": pl.get("url"),
            "type": pl.get("type"),
            "parent_uid": pl.get("parent_uid"),
            "brand": pl.get("brand"),
            "thumbnail": pl.get("thumbnail"),
            "price": pl.get("price_numeric"),
            "rating": pl.get("rating"),
            "reviews": pl.get("review_count"),
            "source": collection,
            "text": pl.get("text"),
        })

    df = pd.DataFrame(rows, columns=DEF_COLS)
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).reset_index(drop=True)

def _rerank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or ("rating" not in df.columns) or ("reviews" not in df.columns):
        return df
    r = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    v = pd.to_numeric(df["reviews"], errors="coerce").fillna(0.0)
    s = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    rerank = s * (1.0 + 0.02 * r) * (1.0 + 0.001 * np.log1p(v))
    df = df.assign(_rerank=rerank)
    return df.sort_values(["_rerank", "score"], ascending=False).drop(columns=["_rerank"]).reset_index(drop=True)


def read_jsonl_strict(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không thấy file: {path}")
    try:
        return pd.read_json(p, lines=True)
    except Exception as e:
        print(f"[read_jsonl_strict] thất bại, chuyển parser thủ công. Lý do: {e}")

    rows, bad = [], 0
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as ex:
                bad += 1
    if not rows:
        raise ValueError(f"File '{path}' không hợp lệ.")
    print(f"[read_jsonl_strict] Done: {len(rows)} dòng hợp lệ, {bad} dòng lỗi.")
    return pd.DataFrame(rows)

def _safe_meta(obj):
    m = obj.get("meta", {})
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except Exception:
            m = {}
    return m if isinstance(m, dict) else {}

def parent_from_chunk_id(chunk_id: str):
    if not isinstance(chunk_id, str):
        return None
    m = ID_RE_HASH_ONLY.match(chunk_id)
    return f"product::{m.group(1)}" if m else None

def extract_source_id(row):
    m = _safe_meta(row)
    sid = m.get("source_id")
    if sid:
        return str(sid)
    pu = str(row.get("parent_uid", ""))
    t = TAIL_NUM.search(pu)
    return t.group(1) if t else None

def build_parent_lookup(jsonl_path: str) -> dict:
    df = read_jsonl_strict(jsonl_path)
    for c in ["id", "parent_uid", "meta"]:
        if c not in df.columns:
            df[c] = ""

    df["_source_id"] = df.apply(extract_source_id, axis=1)
    df["_parent_qdrant"] = df["id"].map(parent_from_chunk_id)

    lk = (
        df.dropna(subset=["_source_id", "_parent_qdrant"])
        .drop_duplicates(subset=["_source_id"])
        .set_index("_source_id")["_parent_qdrant"]
        .to_dict()
    )

    df["_tail_num"] = df["parent_uid"].apply(
        lambda s: (TAIL_NUM.search(str(s)).group(1) if s else None)
    )
    tmp = (
        df.dropna(subset=["_tail_num", "_parent_qdrant"])
        .drop_duplicates(subset=["_tail_num"])
        .set_index("_tail_num")["_parent_qdrant"]
        .to_dict()
    )
    for k, v in tmp.items():
        lk.setdefault(k, v)

    print(f"[build_parent_lookup] entries: {len(lk)}")
    return lk

def normalize_parent_for_qdrant(pid: str, parent_lookup: dict) -> str:
    if not pid:
        return ""
    m = ID_RE_MIDDLE.match(pid)
    if m:
        return f"product::{m.group(1)}::0"
    m2 = TAIL_NUM.search(str(pid))
    if not m2:
        return pid
    sid = m2.group(1)
    return parent_lookup.get(sid, pid)

def _parent_key_for_sparse(pid: str) -> str:
    m = TAIL_NUM.search(str(pid))
    return m.group(1) if m else str(pid)

def _parent_key_for_dense(pid: str, parent_lookup_rev: dict) -> str:
    if isinstance(pid, str):
        m = ID_RE_MIDDLE.match(pid)
        if m:
            pid0 = f"product::{m.group(1)}::0"
            if pid0 in parent_lookup_rev:
                return parent_lookup_rev[pid0]
    if pid in parent_lookup_rev:
        return parent_lookup_rev[pid]
    m = TAIL_NUM.search(str(pid))
    return m.group(1) if m else str(pid)


class BM25Okapi:
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.corpus_size = len(corpus_tokens)
        self.doc_len = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.corpus_size else 0.0

        self.term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}
        for doc in corpus_tokens:
            tf: Dict[str, int] = {}
            for w in doc:
                tf[w] = tf.get(w, 0) + 1
            self.term_freqs.append(tf)
            for w in tf.keys():
                self.doc_freqs[w] = self.doc_freqs.get(w, 0) + 1

        self.idf: Dict[str, float] = {}
        for w, df in self.doc_freqs.items():
            val = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1e-9)
            if val < 0:
                val *= self.epsilon
            self.idf[w] = val

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if self.corpus_size == 0:
            return scores
        for w in query_tokens:
            if w not in self.idf:
                continue
            idf = self.idf[w]
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(w, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / (self.avgdl or 1.0)))
                scores[i] += idf * (f * (self.k1 + 1)) / (denom + 1e-9)
        return scores


class ParentBM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.parents: List[str] = []
        self.meta: Dict[str, Dict[str, Any]] = {}
        self.docs_tokens: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def fit(self, parent_rows: Dict[str, Dict[str, Any]]):
        self.parents.clear()
        self.meta = {}
        self.docs_tokens.clear()

        for pid, info in parent_rows.items():
            ts = (info.get("text_sparse_parent") or "").strip()
            if not ts:
                continue
            self.parents.append(pid)
            self.meta[pid] = info
            self.docs_tokens.append(tokenize(ts))

        self._bm25 = BM25Okapi(self.docs_tokens, k1=self.k1, b=self.b, epsilon=self.epsilon)

    def is_ready(self) -> bool:
        return (self._bm25 is not None) and (len(self.parents) == len(self.docs_tokens) > 0)

    def search(self, query: str, topk: int = 20) -> pd.DataFrame:
        if not self.is_ready():
            return pd.DataFrame(columns=["parent_uid", "score", "brand", "category_norm", "kv_compact"])

        q_tokens = tokenize(query)
        scores = self._bm25.get_scores(q_tokens)

        k = min(topk, len(scores))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        rows = []
        for i in idx:
            pid = self.parents[i]
            m = self.meta.get(pid, {})
            rows.append({
                "parent_uid": pid,
                "score": float(scores[i]),
                "brand": m.get("brand"),
                "category_norm": m.get("category_norm"),
                "kv_compact": m.get("kv_compact"),
            })
        return pd.DataFrame(rows)

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "parents.pkl"), "wb") as f:
            pickle.dump(self.parents, f)
        with open(os.path.join(folder, "docs_tokens.pkl"), "wb") as f:
            pickle.dump(self.docs_tokens, f)
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(folder, "params.json"), "w", encoding="utf-8") as f:
            json.dump({"k1": self.k1, "b": self.b, "epsilon": self.epsilon}, f)

    @classmethod
    def load(cls, folder: str) -> "ParentBM25Index":
        with open(os.path.join(folder, "params.json"), "r", encoding="utf-8") as f:
            p = json.load(f)
        obj = cls(k1=p.get("k1", 1.5), b=p.get("b", 0.75), epsilon=p.get("epsilon", 0.25))
        with open(os.path.join(folder, "parents.pkl"), "rb") as f:
            obj.parents = pickle.load(f)
        with open(os.path.join(folder, "docs_tokens.pkl"), "rb") as f:
            obj.docs_tokens = pickle.load(f)
        with open(os.path.join(folder, "meta.json"), "r", encoding="utf-8") as f:
            obj.meta = json.load(f)
        obj._bm25 = BM25Okapi(obj.docs_tokens, k1=obj.k1, b=obj.b, epsilon=obj.epsilon)
        return obj


_q_model = None

def get_embedding_model():
    """Get or initialize the embedding model"""
    global _q_model
    if _q_model is None:
        _q_model = SentenceTransformer("BAAI/bge-m3")
    return _q_model

def embed_query(q: str) -> np.ndarray:
    model = get_embedding_model()
    return model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")


def _aggregate_dense_by_parent(df_chunks: pd.DataFrame) -> pd.DataFrame:
    if df_chunks.empty or "parent_uid" not in df_chunks.columns or "score" not in df_chunks.columns:
        return pd.DataFrame(columns=["parent_uid", "score_dense_parent"])
    d = df_chunks[["parent_uid", "score"]].copy()
    d["score"] = pd.to_numeric(d["score"], errors="coerce").fillna(0.0)
    top2 = (
        d.sort_values(["parent_uid", "score"], ascending=[True, False])
        .groupby("parent_uid")["score"]
        .apply(lambda s: list(s.head(2)))
        .reset_index(name="_scores")
    )

    def _combine(scores):
        if not scores:
            return 0.0
        return float(scores[0]) + (0.05 * float(scores[1]) if len(scores) > 1 else 0.0)

    top2["score_dense_parent"] = top2["_scores"].apply(_combine)
    return top2[["parent_uid", "score_dense_parent"]]

def _fuse_parent_rrf(
    df_dense_parent: pd.DataFrame,
    df_sparse_parent: pd.DataFrame,
    parent_lookup_rev: dict,
    topk: int = 12,
    k: int = 60,
    lam: float = 0.8,
) -> pd.DataFrame:
    if df_dense_parent is None or df_dense_parent.empty:
        D = pd.DataFrame(columns=["parent_uid", "score_dense_parent"])
    else:
        D = df_dense_parent.copy()
        D = D[["parent_uid", "score_dense_parent"]].copy()
        D["score_dense_parent"] = pd.to_numeric(D["score_dense_parent"], errors="coerce").fillna(0.0)
        D = D.sort_values("score_dense_parent", ascending=False).reset_index(drop=True)
        D["_pid_key"] = D["parent_uid"].map(lambda x: _parent_key_for_dense(x, parent_lookup_rev))
        D["_rank_dense"] = np.arange(len(D), dtype=float)

    if df_sparse_parent is None or df_sparse_parent.empty:
        S = pd.DataFrame(columns=["parent_uid", "score"])
    else:
        S = df_sparse_parent.copy()
        S = S[["parent_uid", "score"]].copy()
        S["score"] = pd.to_numeric(S["score"], errors="coerce").fillna(0.0)
        S = S.sort_values("score", ascending=False).reset_index(drop=True)
        S["_pid_key"] = S["parent_uid"].map(_parent_key_for_sparse)
        S["_rank_sparse"] = np.arange(len(S), dtype=float)

    if D.empty and S.empty:
        return pd.DataFrame(columns=["parent_uid", "_rrf", "rank", "_pid_key"])

    if D.empty and not S.empty:
        M = S.copy()
        M["_rank_dense"] = np.inf
        M["parent_uid"] = M["parent_uid"]
    else:
        if S.empty:
            M = D.copy()
            M["_rank_sparse"] = np.inf
        else:
            M = pd.merge(
                D[["_pid_key", "_rank_dense", "parent_uid"]].rename(
                    columns={"parent_uid": "parent_uid_dense"}
                ),
                S[["_pid_key", "_rank_sparse"]],
                on="_pid_key",
                how="left",
            )
            M["parent_uid"] = M["parent_uid_dense"]
            M.drop(columns=["parent_uid_dense"], inplace=True)

    M["_rank_dense"] = M.get("_rank_dense", np.inf).astype(float)
    M["_rank_sparse"] = M.get("_rank_sparse", np.inf).astype(float)

    M["_rrf"] = (1.0 / (k + M["_rank_dense"])) + lam * (1.0 / (k + M["_rank_sparse"]))
    M = M.sort_values("_rrf", ascending=False).head(topk).reset_index(drop=True)
    M["rank"] = np.arange(len(M))
    M["_pid_key"] = M["_pid_key"].astype(str)

    return M[["parent_uid", "_rrf", "rank", "_pid_key"]]

def _fetch_best_chunks_for_parent(
    client,
    qv: np.ndarray,
    parent_uid: str,
    take: int = 2,
    price_gte: Optional[float] = None,
    price_lte: Optional[float] = None
) -> pd.DataFrame:
    if not parent_uid:
        return pd.DataFrame()
    flt = build_filter(parent_uid=parent_uid, price_gte=price_gte, price_lte=price_lte)
    try:
        return qdrant_search(client, "product_bge", qv, topk=take, flt=flt)
    except Exception:
        return pd.DataFrame()


"""
RAG Engine - Missing parts to complete the implementation
Append this to the end of your rag_engine.py (document 5)
"""

def search_auto_hybrid(
        client,
        bm25_index,
        query: str,
        topk: int = 8,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        brands: Optional[List[str]] = None,
        price_gte: Optional[float] = None,
        price_lte: Optional[float] = None,
        rating_gte: Optional[float] = None,
        rating_lte: Optional[float] = None,
        review_count_gte: Optional[int] = None,
        parent_uid: Optional[str] = None,
        dense_topk_chunks: int = 48,
        bm25_topk_parents: int = 50,
        rrf_k: int = 60,
        rrf_lambda: float = 0.8,
        per_parent_chunks: int = 2,
        parent_lookup: Optional[dict] = None,
        parent_lookup_rev: Optional[dict] = None,
        verbose: bool = True,
) -> Tuple[str, pd.DataFrame]:
    """
    Hybrid search combining dense (vector) and sparse (BM25) retrieval
    """
    if parent_lookup is None or parent_lookup_rev is None:
        raise ValueError("parent_lookup and parent_lookup_rev are required")

    def _dbg(msg: str):
        if verbose:
            print(msg)

    def _peek(df: pd.DataFrame, cols: List[str], n: int = 3, title: str = ""):
        if not verbose:
            return
        if title:
            print(title)
        if df is None or df.empty:
            print("  (empty)")
        else:
            cols_eff = [c for c in cols if c in df.columns]
            print(df[cols_eff].head(n).to_string(index=False))

    # Route
    target = route(query)
    p_sig, f_sig = route_strength(query)
    _dbg(f"Route={target} | p={p_sig}, f={f_sig}")

    # Parse slots
    slots_q = parse_slots(query)
    if price_gte is None and slots_q.get("price_gte") is not None:
        price_gte = slots_q["price_gte"]
    if price_lte is None and slots_q.get("price_lte") is not None:
        price_lte = slots_q["price_lte"]
    if rating_gte is None and slots_q.get("rating_gte") is not None:
        rating_gte = slots_q["rating_gte"]
    if rating_lte is None and slots_q.get("rating_lte") is not None:
        rating_lte = slots_q["rating_lte"]

    _dbg(f"Params: topk={topk}, price=[{price_gte}, {price_lte}], rating=[{rating_gte}, {rating_lte}]")

    qv = embed_query(query)
    brand_list = brands if brands else ([brand] if brand else None)

    cond_product = build_filter(
        category=None,
        brands=None,
        parent_uid=parent_uid,
        dtype=None,
        price_gte=price_gte,
        price_lte=price_lte,
        rating_gte=rating_gte,
        rating_lte=rating_lte,
        review_count_gte=review_count_gte,
    )
    cond_faq = build_filter(category=None, brands=None, parent_uid=None, dtype=None)

    def search_with_fallback(collection: str, cond, _topk: int) -> pd.DataFrame:
        try:
            df = qdrant_search(client, collection, qv, _topk, cond)
            _dbg(f"Qdrant[{collection}]: {len(df)} results")
        except Exception:
            _dbg(f"Qdrant[{collection}] error → retry no-filter")
            df = qdrant_search(client, collection, qv, _topk, flt=None)

        if df.empty:
            if collection == "product_bge":
                cond_relaxed = build_filter(
                    parent_uid=parent_uid,
                    price_gte=price_gte,
                    price_lte=price_lte,
                )
            else:
                cond_relaxed = None
            df = qdrant_search(client, collection, qv, _topk, cond_relaxed)
            if df.empty:
                df = qdrant_search(client, collection, qv, _topk, flt=None)

        if collection == "product_bge":
            if USE_RERANK and not df.empty:
                df = _rerank(df)
            df = dedup_product(df, per_parent=per_parent_chunks)
        elif collection == "faq_bge":
            if not df.empty and f_sig > 0:
                df["score"] = df["score"] * 1.07
            df = dedup_faq(df)
        return df

    # FAQ only
    if target == "faq":
        _dbg("→ FAQ only")
        df_faq = search_with_fallback("faq_bge", cond_faq, topk)
        return "faq", df_faq

    if target == "both" and f_sig >= p_sig + 1:
        _dbg("→ FAQ (strong signal)")
        a = search_with_fallback("faq_bge", cond_faq, topk)
        return "faq", a

    # Product/Both → Hybrid
    df_dense_chunks = search_with_fallback("product_bge", cond_product, max(dense_topk_chunks, topk * 4))
    _dbg(f"Dense chunks: {len(df_dense_chunks)}")

    df_dense_parent = _aggregate_dense_by_parent(df_dense_chunks)[["parent_uid", "score_dense_parent"]]
    _dbg(f"Dense→Parent: {len(df_dense_parent)}")

    if bm25_index is not None:
        q_sparse = _norm_sparse_query(query)
        df_sparse_parent = bm25_index.search(q_sparse, topk=bm25_topk_parents)
        df_sparse_parent = df_sparse_parent.rename(columns={"score": "score"})[["parent_uid", "score"]]
        _dbg(f"BM25-parent: {len(df_sparse_parent)}")
    else:
        df_sparse_parent = pd.DataFrame(columns=["parent_uid", "score"])

    fused_parents = _fuse_parent_rrf(
        df_dense_parent=df_dense_parent,
        df_sparse_parent=df_sparse_parent,
        parent_lookup_rev=parent_lookup_rev,
        topk=max(topk * 3, 12),
        k=rrf_k,
        lam=rrf_lambda
    )
    _dbg(f"Fused parents: {len(fused_parents)}")

    # Collect chunks
    out_chunks = []
    have = set()
    if not df_dense_chunks.empty:
        pool = (
            df_dense_chunks.sort_values(["parent_uid", "score"], ascending=[True, False])
            .groupby("parent_uid")
            .head(per_parent_chunks)
        )
        out_chunks.append(pool)
        have = set(pool["parent_uid"].unique().tolist())

    need_fetch_keys = []
    for _, row in fused_parents.iterrows():
        pu = row["parent_uid"]
        if pu not in have:
            need_fetch_keys.append(row["_pid_key"])

    for key in need_fetch_keys:
        mapped = parent_lookup.get(str(key), "")
        if not mapped:
            continue
        fetched = _fetch_best_chunks_for_parent(
            client, qv, mapped, take=per_parent_chunks,
            price_gte=price_gte, price_lte=price_lte
        )
        if not fetched.empty:
            out_chunks.append(fetched)

    # Merge chunks
    out_chunks = [df for df in out_chunks if df is not None and not df.empty]
    if out_chunks:
        prod = pd.concat(out_chunks, ignore_index=True)
        top_parent_keys = fused_parents["_pid_key"].astype(str).tolist()
        order_map = {k: i for i, k in enumerate(top_parent_keys)}

        def _prod_pid_key(pid: str) -> str:
            return _parent_key_for_dense(pid, parent_lookup_rev)

        prod["_pid_key"] = prod["parent_uid"].map(_prod_pid_key).astype(str)
        prod["_ord"] = prod["_pid_key"].map(order_map).fillna(1e9)
        prod = prod.sort_values(["_ord", "score"], ascending=[True, False]).drop(columns=["_ord"])
        prod = prod.groupby("parent_uid", as_index=False, sort=False).head(per_parent_chunks)
        prod = prod.head(topk).reset_index(drop=True)
    else:
        prod = pd.DataFrame(columns=DEF_COLS)

    # Merge scores
    if not fused_parents.empty and not prod.empty:
        parent_rrf = dict(zip(fused_parents["_pid_key"].astype(str), fused_parents["_rrf"]))
        prod["_rrf_parent"] = prod["_pid_key"].map(parent_rrf).fillna(0.0)

        alpha = 0.8
        if RE_TECH_UNITS.search(query) or re.search(r'\b[A-Z0-9]{3,}[-_\.]?[A-Z0-9]{2,}\b', query, re.I):
            alpha = 0.65

        prod["score_dense_tmp"] = pd.to_numeric(prod["score"], errors="coerce").fillna(0.0)
        prod["score"] = alpha * prod["score_dense_tmp"] + (1.0 - alpha) * pd.to_numeric(prod["_rrf_parent"],
                                                                                        errors="coerce").fillna(0.0)
        prod = prod.drop(columns=["_rrf_parent", "score_dense_tmp", "_pid_key"])

    if prod.empty and (bm25_index is not None):
        _dbg("BM25 rescue")
        q_sparse = _norm_sparse_query(query)
        df_sparse_full = bm25_index.search(q_sparse, topk=max(topk * 6, 60))
        out = []
        for pid in df_sparse_full["parent_uid"].head(topk * 3):
            mapped = normalize_parent_for_qdrant(pid, parent_lookup)
            fetched = _fetch_best_chunks_for_parent(
                client, qv, mapped, take=per_parent_chunks,
                price_gte=price_gte, price_lte=price_lte
            )
            if not fetched.empty:
                out.append(fetched)
        if out:
            prod = (
                pd.concat(out, ignore_index=True)
                .groupby("parent_uid", as_index=False, sort=False)
                .head(per_parent_chunks)
                .head(topk)
                .reset_index(drop=True)
            )

    # Dense fallback
    if prod.empty and not df_dense_chunks.empty:
        prod = (
            df_dense_chunks
            .sort_values("score", ascending=False)
            .groupby("parent_uid", as_index=False, sort=False)
            .head(per_parent_chunks)
            .head(topk)
            .reset_index(drop=True)
        )

    if not prod.empty and (price_gte is not None or price_lte is not None):
        if price_gte is not None:
            prod = prod[pd.to_numeric(prod["price"], errors="coerce").fillna(np.inf) >= float(price_gte)]
        if price_lte is not None:
            prod = prod[pd.to_numeric(prod["price"], errors="coerce").fillna(-np.inf) <= float(price_lte)]
        prod = prod.reset_index(drop=True)

    _dbg(f"Final product: {len(prod)}")

    if target == "product":
        return "product", prod

    a = search_with_fallback("faq_bge", cond_faq, topk)
    merged = pd.concat([a, prod], ignore_index=True) if not a.empty else prod.copy()
    merged = dedup_merged(merged)
    merged = merged.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    return "both", merged



@dataclass
class LLMConfig:
    model_name: str = "songthienll/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    compute_dtype = torch.bfloat16
    double_quant: bool = True
    device_map: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05


@dataclass
class LLMWrapper:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    cfg: LLMConfig


def load_llm(cfg: Optional[LLMConfig] = None) -> LLMWrapper:
    """Load LLM with quantization"""
    cfg = cfg or LLMConfig()
    if cfg.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.quant_type,
            bnb_4bit_compute_dtype=cfg.compute_dtype,
            bnb_4bit_use_double_quant=cfg.double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb,
            device_map=cfg.device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=cfg.compute_dtype,
            device_map=cfg.device_map,
            trust_remote_code=True,
        )
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return LLMWrapper(tokenizer=tok, model=model, cfg=cfg)


def pick_topk_parents(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Select top K unique parents"""
    if df is None or df.empty:
        return df
    base = df.sort_values(["score"], ascending=False)
    return (
        base.groupby("parent_uid", as_index=False, sort=False)
        .head(1)
        .head(k)
        .reset_index(drop=True)
    )


def format_context_for_llm_product_browse(df: pd.DataFrame, max_items: int = 5) -> str:
    """Format product context for browse mode"""
    if df is None or df.empty:
        return ""
    rows = []
    for i, row in df.head(max_items).iterrows():
        title = _safe(row.get("title")).strip()
        price = _safe(row.get("price", ""))
        url = _safe(row.get("url", "")).strip()
        rows.append(
            f"[{i + 1}]\n"
            f"Tên: {title}\n"
            f"Giá_raw: {price}\n"
            f"Link: {url}\n"
        )
    return "\n".join(rows)


def format_context_for_llm_product_detail(df: pd.DataFrame) -> str:
    """Format product context for detail mode"""
    if df is None or df.empty:
        return ""
    rows = []
    for i, row in df.iterrows():
        snippet = _safe(row.get("text", ""))[:900]
        rows.append(
            f"[{i + 1}] {(_safe(row.get('title'))).strip()}\n"
            f"Giá_raw: {_safe(row.get('price', ''))}\n"
            f"URL: {_safe(row.get('url', ''))}\n"
            f"{snippet}"
        )
    return "\n".join(rows)


def format_context_for_llm_faq(df: pd.DataFrame, max_items: int = 5) -> str:
    """Format FAQ context"""
    if df is None or df.empty:
        return ""
    rows = []
    for i, row in df.head(max_items).iterrows():
        snippet = _safe(row.get("text", ""))[:1200]
        rows.append(
            f"[{i + 1}] {(_safe(row.get('title'))).strip()}\n"
            f"URL: {_safe(row.get('url', ''))}\n{snippet}"
        )
    return "\n".join(rows)


def build_prompt_product(query: str, context: str, no_result: bool = False, mode: str = "browse") -> List[
    Dict[str, str]]:
    """Build prompt for product queries"""
    if mode == "detail":
        sys = (
            "Bạn là trợ lý AI tư vấn CHI TIẾT cho MỘT sản phẩm cụ thể.\n"
            "- Chỉ dùng dữ liệu đã cung cấp.\n"
            "- Không suy đoán.\n"
            "- Nếu không đủ dữ liệu: trả lời 'không tìm thấy thông tin đủ chi tiết'."
        )
        if no_result:
            user = f"Câu hỏi: {query}\n\nDữ liệu: (không có)"
        else:
            user = (
                f"Câu hỏi: {query}\n\n"
                f"Dữ liệu:\n{context}\n\n"
                "Trả lời ngắn gọn, dễ hiểu."
            )
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    # Browse mode
    sys = (
        "Bạn là trợ lý tư vấn sản phẩm Tiki.\n"
        "- Chỉ dùng dữ liệu đã cung cấp.\n"
        "- PHẢI LIỆT KÊ ĐỦ CẢ N sản phẩm trong dữ liệu.\n"
        "- Trả lời dạng danh sách gạch đầu dòng."
    )
    if no_result:
        user = f"Câu hỏi: {query}\n\nDữ liệu: (không có)"
    else:
        user = (
            f"Câu hỏi: {query}\n\n"
            f"Dữ liệu:\n{context}\n\n"
            "Liệt kê TẤT CẢ sản phẩm. Mỗi gạch đầu dòng: Tên – Giá – Link."
        )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def build_prompt_faq(query: str, context: str, no_result: bool = False) -> List[Dict[str, str]]:
    """Build prompt for FAQ queries"""
    sys = (
        "Bạn là trợ lý hỗ trợ Tiki.\n"
        "- CHỈ dùng nội dung trong 'Dữ liệu'.\n"
        "- KHÔNG thêm thông tin mới.\n"
        "- Nếu thiếu thông tin: 'Không tìm thấy trong Dữ liệu, vui lòng liên hệ Tiki'."
    )
    if no_result:
        user = f"Câu hỏi: {query}\n\nDữ liệu: (không có)"
    else:
        user = (
            f"Câu hỏi: {query}\n\n"
            f"Dữ liệu:\n{context}\n\n"
            "Tóm tắt từ Dữ liệu. Trả lời ngắn gọn, dạng gạch đầu dòng."
        )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def generate_chat(
        llm: LLMWrapper,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
) -> str:
    """Generate response from LLM"""
    cfg = llm.cfg
    max_new_tokens = max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens
    temperature = temperature if temperature is not None else 0.05
    top_p = top_p if top_p is not None else 0.8

    inputs = llm.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(llm.model.device)

    out = llm.model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=cfg.repetition_penalty,
        pad_token_id=llm.tokenizer.pad_token_id,
        eos_token_id=llm.tokenizer.eos_token_id,
    )
    text = llm.tokenizer.decode(out[0], skip_special_tokens=True)
    return text.strip()


@dataclass
class AnswerPolicy:
    min_results_product: int = 1
    min_results_faq: int = 1
    min_score_product: float = 0.0
    min_score_faq: float = 0.0
    allow_expand_chunks_detail: bool = True
    expand_chunks_per_parent_detail: int = 3
    token_budget_est_product_browse: int = 1800
    token_budget_est_product_detail: int = 2000
    token_budget_est_faq: int = 1500
    per_item_char_budget_product_browse: int = 360
    per_item_char_budget_product_detail: int = 700
    per_item_char_budget_faq: int = 420


def _should_no_result(df: pd.DataFrame, min_results: int, min_score: float) -> bool:
    if df is None or df.empty:
        return True
    if len(df) < min_results:
        return True
    if min_score > 0.0 and "score" in df.columns:
        tops = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
        if float(tops.max()) < min_score:
            return True
    return False


def _trim_budget(df: pd.DataFrame, token_budget_est: int, per_item_char_budget: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    max_items = max(1, math.floor(token_budget_est / max(1, per_item_char_budget)))
    return df.head(max_items).reset_index(drop=True)


def detect_product_mode(query: str, df_top: pd.DataFrame) -> str:
    q = query.lower()

    if any(t in q for t in DETAIL_TERMS):
        return "detail"
    if any(t in q for t in BROWSE_TERMS):
        return "browse"
    if MODEL_PATTERN.search(query):
        return "detail"
    if df_top is not None and not df_top.empty:
        n_parent = df_top["parent_uid"].nunique()
        if n_parent == 1:
            return "detail"
    return "browse"


def expand_parents_with_more_chunks(
        df_prod_one_chunk: pd.DataFrame,
        client,
        embed_query_fn,
        query: str,
        per_parent_chunks: int = 2,
        price_gte: Optional[float] = None,
        price_lte: Optional[float] = None,
) -> pd.DataFrame:
    if df_prod_one_chunk is None or df_prod_one_chunk.empty:
        return df_prod_one_chunk
    if per_parent_chunks <= 1:
        return df_prod_one_chunk

    qv = embed_query_fn(query)
    parents = df_prod_one_chunk["parent_uid"].dropna().unique().tolist()
    rows = []

    for pu in parents:
        try:
            flt = build_filter(parent_uid=pu, price_gte=price_gte, price_lte=price_lte)
            more = qdrant_search(client, "product_bge", qv, topk=per_parent_chunks, flt=flt)
            if more is not None and not more.empty:
                rows.append(more)
        except:
            pass

    if not rows:
        return df_prod_one_chunk

    expanded = pd.concat([df_prod_one_chunk] + rows, ignore_index=True)
    expanded = (
        expanded.sort_values(["parent_uid", "score"], ascending=[True, False])
        .groupby("parent_uid", as_index=False, sort=False)
        .head(per_parent_chunks)
        .reset_index(drop=True)
    )
    return expanded


def answer_with_rag(
        llm: LLMWrapper,
        client,
        bm25_index,
        query: str,
        search_fn,
        parent_lookup: Dict[str, str],
        parent_lookup_rev: Dict[str, str],
        policy: Optional[AnswerPolicy] = None,
        topk: int = 5,
        rrf_lambda: float = 0.8,
        per_parent_chunks: int = 1,
        verbose: bool = False,
) -> Tuple[str, pd.DataFrame, str]:

    policy = policy or AnswerPolicy()

    # Search
    target, df_all = search_fn(
        client,
        bm25_index,
        query=query,
        topk=topk,
        rrf_lambda=rrf_lambda,
        per_parent_chunks=per_parent_chunks,
        verbose=verbose,
        parent_lookup=parent_lookup,
        parent_lookup_rev=parent_lookup_rev
    )

    if df_all is None:
        df_all = pd.DataFrame()

    # FAQ path
    if target == "faq":
        df_top = pick_topk_parents(df_all, k=topk)
        none_found = _should_no_result(df_top, policy.min_results_faq, policy.min_score_faq)

        if none_found or df_top.empty:
            messages = build_prompt_faq(query, context="", no_result=True)
            answer = generate_chat(llm, messages)
            return "faq", df_all, answer

        df_ctx = _trim_budget(df_top, policy.token_budget_est_faq, policy.per_item_char_budget_faq)
        ctx = format_context_for_llm_faq(df_ctx, max_items=len(df_ctx))
        messages = build_prompt_faq(query, ctx, no_result=False)
        answer = generate_chat(llm, messages)
        return "faq", df_ctx, answer

    # Product path
    df_top = pick_topk_parents(df_all, k=topk)
    mode = detect_product_mode(query, df_top)

    # Browse mode
    if mode == "browse":
        none_found = _should_no_result(df_top, policy.min_results_product, policy.min_score_product)

        if none_found or df_top.empty:
            messages = build_prompt_product(query, context="", no_result=True, mode="browse")
            answer = generate_chat(llm, messages)
            return target, df_all, answer

        df_ctx = _trim_budget(
            df_top,
            policy.token_budget_est_product_browse,
            policy.per_item_char_budget_product_browse
        )
        ctx = format_context_for_llm_product_browse(df_ctx, max_items=len(df_ctx))
        messages = build_prompt_product(query, ctx, no_result=False, mode="browse")
        answer = generate_chat(llm, messages)
        return target, df_ctx, answer

    # Detail mode
    if df_top is None or df_top.empty:
        messages = build_prompt_product(query, context="", no_result=True, mode="detail")
        answer = generate_chat(llm, messages)
        return target, df_all, answer

    main_row = df_top.sort_values("score", ascending=False).iloc[0]
    main_parent = main_row.get("parent_uid")
    df_detail_base = df_all[df_all["parent_uid"] == main_parent].copy()

    if df_detail_base.empty:
        df_detail_base = df_top[df_top["parent_uid"] == main_parent].copy()

    none_found = _should_no_result(df_detail_base, policy.min_results_product, policy.min_score_product)

    if none_found or df_detail_base.empty:
        messages = build_prompt_product(query, context="", no_result=True, mode="detail")
        answer = generate_chat(llm, messages)
        return target, df_all, answer

    if policy.allow_expand_chunks_detail and policy.expand_chunks_per_parent_detail > 1:
        df_detail = expand_parents_with_more_chunks(
            df_prod_one_chunk=df_detail_base.head(1),
            client=client,
            embed_query_fn=embed_query,
            query=query,
            per_parent_chunks=policy.expand_chunks_per_parent_detail,
            price_gte=None,
            price_lte=None,
        )
    else:
        df_detail = df_detail_base

    df_ctx = _trim_budget(
        df_detail,
        policy.token_budget_est_product_detail,
        policy.per_item_char_budget_product_detail
    )
    ctx = format_context_for_llm_product_detail(df_ctx)
    messages = build_prompt_product(query, ctx, no_result=False, mode="detail")
    answer = generate_chat(llm, messages)
    return target, df_ctx, answer
