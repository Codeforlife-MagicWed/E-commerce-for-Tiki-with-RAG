import argparse, json, time, re, logging, sys, hashlib, unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urlunparse
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

BASE = "https://hotro.tiki.vn"


def setup_logger():
    logger = logging.getLogger("tiki-help-selenium-json")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(ch)
    return logger


def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def doc_id_from_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def split_into_chunks(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    words = normalize_ws(text).split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunk_words = words[i:i + size]
        chunks.append(" ".join(chunk_words))
        if i + size >= len(words):
            break
        i += max(1, size - overlap)
    return chunks


def load_state(p: Path) -> Dict:
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"articles": {}, "discovered": []}


def save_state(p: Path, s: Dict):
    p.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_accents(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


CATEGORY_MAP = [
    ("Tài khoản", [
        r"tai khoan", r"dang nhap", r"dang ky", r"mat khau", r"doi mat khau",
        r"bao mat", r"quyen rieng tu", r"ho so", r"xac thuc", r"otp", r"2fa"
    ]),
    ("Đặt hàng & Thanh toán", [
        r"dat hang", r"gio hang", r"thanh toan", r"the", r"visa", r"master",
        r"momo", r"zalopay", r"cod", r"hoa don", r"vat", r"ma giam", r"voucher",
        r"coupon", r"uu dai", r"ap dung ma"
    ]),
    ("Giao & Nhận hàng", [
        r"giao hang", r"nhan hang", r"van chuyen", r"ship", r"phi ship",
        r"don vi van chuyen", r"thoi gian giao", r"tra cuu( |)don", r"giao khong thanh cong",
        r"giao cham", r"giao nhanh"
    ]),
    ("Đổi trả - Bảo hành & Bồi thường", [
        r"doi tra", r"tra hang", r"hoan tien", r"refund", r"bao hanh", r"boi thuong",
        r"doi hang", r"chinh sach doi tra", r"chinh sach bao hanh"
    ]),
    ("Thông tin & Chính sách", [
        r"chinh sach", r"quy dinh", r"dieu khoan", r"phap ly", r"quy tac",
        r"bao mat thong tin", r"quy che", r"van ban", r"thong tin chung"
    ]),
    ("Dịch vụ & Chương trình", [
        r"dich vu", r"chuong trinh", r"khuyen mai", r"tiki xu", r"tiki s\+", r"prime",
        r"subs?cription", r"goi dich vu", r"uu dai thanh vien", r"tien ich"
    ]),
]


def normalize_category_text(text: str) -> str:
    return strip_accents((text or "").lower())


def normalize_category(src_text: str) -> str:
    t = normalize_category_text(src_text)
    if not t:
        return "Khác"
    for label, patterns in CATEGORY_MAP:
        for p in patterns:
            if re.search(p, t):
                return label
    return "Khác"


def pick_category(title: str, breadcrumb: Optional[str], answer: str) -> str:
    for src in (breadcrumb, title, (answer or "")[:1500]):
        if src:
            cat = normalize_category(src)
            if cat != "Khác":
                return cat
    return "Khác"


def _deep_query_all_with_slots(driver, selector):
    js = r"""
    const sel = arguments[0];
    function getShadowRootsIn(node) {
      const out = [];
      const tw = document.createTreeWalker(node, NodeFilter.SHOW_ELEMENT);
      let cur = tw.currentNode;
      while (cur) {
        if (cur.shadowRoot) out.push(cur.shadowRoot);
        cur = tw.nextNode();
      }
      return out;
    }
    function getAllRoots(){
      const roots = [document];
      for (let i=0; i<roots.length; i++){
        const r = roots[i];
        const subs = getShadowRootsIn(r);
        for (const s of subs) roots.push(s);
      }
      return roots;
    }
    function queryDeepAll(selector){
      const roots = getAllRoots();
      const results = [];
      for (const root of roots){
        let nodes = [];
        try { nodes = root.querySelectorAll(selector) || []; } catch(e){}
        for (const n of nodes) results.push(n);
        const slots = root.querySelectorAll ? root.querySelectorAll('slot') : [];
        for (const sl of slots){
          const assigned = sl.assignedNodes ? sl.assignedNodes({flatten:true}) : [];
          for (const an of assigned){
            if (an.nodeType === Node.ELEMENT_NODE){
              try {
                const subRoots = getShadowRootsIn(an);
                for (const s of subRoots) {
                  let subNodes = [];
                  try { subNodes = s.querySelectorAll(selector) || []; } catch(e){}
                  for (const sn of subNodes) results.push(sn);
                }
                let subNodes2 = [];
                try { subNodes2 = an.querySelectorAll(selector) || []; } catch(e){}
                for (const sn2 of subNodes2) results.push(sn2);
              } catch(e){}
            }
          }
        }
      }
      return Array.from(new Set(results));
    }
    return queryDeepAll(sel);
    """
    return driver.execute_script(js, selector)


def query_deep_text(driver, selectors, many=False):
    nodes = _deep_query_all_with_slots(driver, selectors)
    if not nodes:
        return None if not many else []
    if not many:
        n = nodes[0]
        txt = (n.get_attribute("innerText") or n.get_attribute("textContent") or n.text or "")
        return txt.strip()
    out = []
    for n in nodes:
        t = (n.get_attribute("innerText") or n.get_attribute("textContent") or n.text or "")
        t = t.strip()
        if t:
            out.append(t)
    return out


def query_deep_html(driver, selectors):
    nodes = _deep_query_all_with_slots(driver, selectors)
    if not nodes:
        return None
    return nodes[0].get_attribute("innerHTML")


def make_driver(headless: bool = True):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1366,768")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--lang=vi-VN")
    opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36")
    drv = webdriver.Chrome(options=opts)
    drv.set_page_load_timeout(40)
    return drv


def safe_get(drv, url: str, logger, retries=2, delay=0.8) -> bool:
    for i in range(retries + 1):
        try:
            drv.get(url)
            return True
        except Exception as e:
            logger.warning(f"GET {url} error {type(e).__name__}: {e}")
            time.sleep(delay * (i + 1))
    return False


def _has_real_text(d, min_chars=200):
    sel = (
        ".article-content, article, "
        ".slds-rich-text-editor__output, lightning-formatted-rich-text, "
        "c-article-output, c-knowledge-article, "
        "div.forceCommunityKnowledgeArticle, "
        "div[data-region-name='article']"
    )
    try:
        nodes = _deep_query_all_with_slots(d, sel)
        for n in nodes:
            txt = (n.get_attribute("innerText") or n.get_attribute("textContent") or n.text or "").strip()
            if txt and len(txt) >= min_chars:
                return True
    except Exception:
        return False
    return False


def wait_render(drv, timeout=35, min_chars=200):
    WebDriverWait(drv, timeout).until(lambda d: _has_real_text(d, min_chars=min_chars) or d.find_elements(By.CSS_SELECTOR, "h1"))


def scroll_lazy(drv, steps=6, pause=0.6):
    last_h = 0
    for _ in range(steps):
        drv.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        h = drv.execute_script("return document.body.scrollHeight")
        if h == last_h:
            break
        last_h = h


def discover_articles(drv, start_url: str, limit: int, delay: float, logger) -> List[str]:
    queue: List[str] = [start_url]
    seen: Set[str] = set(queue)
    articles: List[str] = []

    while queue and len(articles) < limit:
        url = queue.pop(0)
        if not safe_get(drv, url, logger):
            continue
        try:
            wait_render(drv, timeout=25, min_chars=50)
        except Exception:
            pass

        scroll_lazy(drv, steps=8, pause=delay)

        anchors = drv.find_elements(By.CSS_SELECTOR, "a[href]")
        for a in anchors:
            href = a.get_attribute("href") or ""
            if not href.startswith("http"):
                continue
            if ("/s/article/" in href) or ("/knowledge-base/post/" in href):
                if href not in articles:
                    articles.append(href)
                    if len(articles) >= limit:
                        break

        for a in anchors:
            href = a.get_attribute("href") or ""
            if ("/s/topic" in href) or ("/knowledge-base" in href):
                if href not in seen:
                    seen.add(href)
                    queue.append(href)

        logger.info(f"Discovered so far: articles={len(articles)} | queue={len(queue)} | at={url}")
        time.sleep(delay)
    return articles


LOGIN_TITLES = {"login | tiki help center", "đăng nhập | tiki help center", "login | tiki"}
FALLBACK_NETLOCS = ["hotro.tiki.vn", "tiki.my.site.com", "tiki.force.com"]


def swap_domain_keep_path(u, new_netloc):
    pu = urlparse(u)
    return urlunparse((pu.scheme, new_netloc, pu.path, pu.params, pu.query, pu.fragment))


def is_login_or_404(drv):
    try:
        ttl = (drv.title or "").lower().strip()
    except Exception:
        ttl = ""
    try:
        body_txt = (drv.find_element(By.TAG_NAME, "body").text or "").lower()
    except Exception:
        body_txt = ""
    if ttl in LOGIN_TITLES or "login" in ttl or "đăng nhập" in ttl:
        return True
    if (
        "not found" in ttl
        or "article not available" in body_txt
        or "không tìm thấy" in body_txt
        or "không tồn tại" in body_txt
        or "page not found" in body_txt
    ):
        return True
    return False


def fetch_with_fallback(drv, url, logger, delay):
    if safe_get(drv, url, logger):
        time.sleep(delay)
        if not is_login_or_404(drv):
            return drv.current_url
    pu = urlparse(url)
    for host in FALLBACK_NETLOCS:
        if host == pu.netloc:
            continue
        cand = swap_domain_keep_path(url, host)
        if not safe_get(drv, cand, logger):
            time.sleep(delay)
            continue
        time.sleep(delay)
        if not is_login_or_404(drv):
            return drv.current_url
        logger.info(f"Fallback NG at {cand}, try next...")
    return None


def html_to_markdown(html: str) -> str:
    if not html:
        return ""
    txt = html
    txt = re.sub(r"<br\s*/?>", "\n", txt, flags=re.I)
    txt = re.sub(r"</p\s*>", "\n\n", txt, flags=re.I)
    txt = re.sub(r"<li\s*>", "- ", txt, flags=re.I)
    txt = re.sub(r"</li\s*>", "\n", txt, flags=re.I)
    txt = re.sub(r"</(ul|ol)\s*>", "\n", txt, flags=re.I)
    txt = re.sub(r"<h1[^>]*>(.*?)</h1\s*>", r"# \1\n\n", txt, flags=re.I | re.S)
    txt = re.sub(r"<h2[^>]*>(.*?)</h2\s*>", r"## \1\n\n", txt, flags=re.I | re.S)
    txt = re.sub(r"<h3[^>]*>(.*?)</h3\s*>", r"### \1\n\n", txt, flags=re.I | re.S)
    txt = re.sub(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a\s*>', r"[\2](\1)", txt, flags=re.I | re.S)
    txt = re.sub(r"<[^>]+>", "", txt)
    txt = re.sub(r"[ \t]+\n", "\n", txt)
    return txt.strip()


def parse_article(drv) -> Optional[dict]:
    title = query_deep_text(drv, "h1", many=False) or ""
    if not title:
        try:
            ogt = drv.execute_script(
                "return (document.querySelector('meta[property=\"og:title\"]')||{}).content || null;"
            )
            if ogt:
                title = ogt
        except Exception:
            pass
    if not title:
        try:
            title = (drv.title or "").strip()
        except Exception:
            title = ""

    raw_text = extract_main_text(drv)

    bc_nodes = query_deep_text(drv, ".slds-breadcrumb__item a", many=True)
    breadcrumb = " > ".join(bc_nodes) if isinstance(bc_nodes, list) and bc_nodes else None

    if not title and not raw_text:
        return None

    answer = raw_text or ""
    if answer:
        lines = [ln for ln in answer.splitlines()]
        cleaned = []
        for ln in lines:
            ln_stripped = ln.strip()
            if re.search(r"\b\d{1,2}\s+tháng\s+\d{1,2}(,)?\s+\d{4}\b", ln_stripped, flags=re.I):
                continue
            if ln_stripped.lower() in {"knowledge", "bài viết", "article"}:
                continue
            if re.fullmatch(r"#{1,6}\s*", ln_stripped):
                continue
            cleaned.append(ln)
        if cleaned and title and cleaned[0].lstrip("# ").strip().lower() == title.strip().lower():
            cleaned = cleaned[1:]
        answer = "\n".join([c for c in cleaned if c.strip()]).strip()

    content = (title.strip() + "\n\n" + answer.strip()).strip() if title else answer.strip()

    return {
        "question": title.strip(),
        "answer": answer.strip(),
        "content": content,
        "breadcrumb": breadcrumb,
    }


def extract_main_text(drv) -> str:
    candidate_selectors = [
        ".slds-rich-text-editor__output",
        "lightning-formatted-rich-text",
        "c-article-output",
        "c-knowledge-article",
        "div.forceCommunityKnowledgeArticle",
        "div[data-region-name='article']",
        "article",
    ]
    best = ""
    for sel in candidate_selectors:
        try:
            nodes = _deep_query_all_with_slots(drv, sel) or []
        except Exception:
            nodes = []
        for n in nodes:
            try:
                t = (n.get_attribute("innerText") or n.get_attribute("textContent") or n.text or "").strip()
            except Exception:
                t = ""
            if t and len(t) > len(best):
                best = t
    return best.strip()


def crawl(
    start: str,
    state_path: Path,
    out_path: Path,
    limit: int,
    delay: float,
    include: List[str],
    exclude: List[str],
    headless: bool,
    chunk_size: int,
    chunk_overlap: int,
    force: bool,
    logger,
):
    state = load_state(state_path)
    drv = make_driver(headless=headless)

    results: List[dict] = []

    try:
        logger.info("Discovering article links from hub (JS-rendered)...")
        links = discover_articles(drv, start_url=start, limit=limit, delay=delay, logger=logger)
        logger.info(f"Total discovered articles: {len(links)}")

        ok = 0
        for i, url in enumerate(links, 1):
            meta = state["articles"].get(url)
            if meta and meta.get("status") == "ok" and not force:
                logger.info(f"[{i}/{len(links)}] SKIP ok {url}")
                continue

            cand_url = fetch_with_fallback(drv, url, logger, delay)
            if not cand_url:
                state["articles"][url] = {"status": "fail", "note": "login_required_or_no_response"}
                save_state(state_path, state)
                logger.info(f"FAIL login/404/no_response: {url}")
                continue

            try:
                wait_render(drv, timeout=25, min_chars=200)
            except Exception:
                pass

            time.sleep(delay)
            final_url = drv.current_url
            art = parse_article(drv)

            if not art:
                state["articles"][url] = {"status": "fail", "note": "parse_returned_none"}
                save_state(state_path, state)
                logger.info(f"FAIL parse none: {final_url}")
                continue

            chunks = split_into_chunks(art["answer"], size=chunk_size, overlap=chunk_overlap)

            cat_norm = pick_category(art["question"], art.get("breadcrumb"), art["answer"])

            if include and (cat_norm not in include):
                state["articles"][url] = {"status": "ok", "note": "filtered_out"}
                save_state(state_path, state)
                logger.info(f"[{i}/{len(links)}] FILTER-OUT ({cat_norm}) {final_url}")
                continue

            if exclude and (cat_norm in exclude):
                state["articles"][url] = {"status": "ok", "note": "filtered_out"}
                save_state(state_path, state)
                logger.info(f"[{i}/{len(links)}] FILTER-OUT ({cat_norm}) {final_url}")
                continue

            rec = {
                "id": doc_id_from_url(final_url),
                "url": final_url,
                "title": art["question"],
                "breadcrumb": art.get("breadcrumb"),
                "category_norm": cat_norm,
                "source": "tiki_help_center",
                "language": "vi",
                "crawled_at": now_iso(),
                "chunks": chunks if chunks else [],
                "question": art["question"],
                "answer": art["answer"] or "",
            }
            results.append(rec)

            state["articles"][url] = {"status": "ok", "note": ""}
            save_state(state_path, state)

            ok += 1
            logger.info(f"[{i}/{len(links)}] OK ({cat_norm}) {final_url}")
            time.sleep(delay)

        logger.info(f"Done. OK={ok}, total={len(links)}; output: {out_path}, state: {state_path}")
    finally:
        drv.quit()

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"Wrote JSON: {out_path} (records={len(results)})")


def main():
    ap = argparse.ArgumentParser(description="Crawl Tiki Help Center (JS-rendered) -> JSON array")
    ap.add_argument("--start", default="https://hotro.tiki.vn/knowledge-base")
    ap.add_argument("--out", default="tiki_help_center.json")
    ap.add_argument("--state", default="state.json")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--delay", type=float, default=0.6)
    ap.add_argument("--include", default="")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--no-headless", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    include = [s.strip() for s in args.include.split(",") if s.strip()]
    exclude = [s.strip() for s in args.exclude.split(",") if s.strip()]
    headless = not args.no_headless

    logger = setup_logger()
    crawl(
        start=args.start,
        state_path=Path(args.state),
        out_path=Path(args.out),
        limit=args.limit,
        delay=args.delay,
        include=include,
        exclude=exclude,
        headless=headless,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force=args.force,
        logger=logger,
    )


if __name__ == "__main__":
    main()
