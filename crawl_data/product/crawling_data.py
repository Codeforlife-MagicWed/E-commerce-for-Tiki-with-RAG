import requests
import json
import time
import re
from datetime import datetime
from html import unescape


def clean_html(raw_html):

    if not raw_html or raw_html == 'N/A':
        return 'N/A'

    text = unescape(raw_html)

    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?p>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?div[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</h[1-6]>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)

    text = re.sub(r'<[^>]+>', '', text)

    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)

    text = re.sub(r'\n\s*\n+', '\n', text)

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = ' '.join(lines)  # Join thành một đoạn văn liền

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def get_product_detail_info(product_id, session, headers):

    try:
        detail_url = f"https://tiki.vn/api/v2/products/{product_id}"
        response = session.get(detail_url, headers=headers, timeout=10)

        if response.status_code == 200:
            detail_data = response.json()

            description = detail_data.get('description') or detail_data.get('short_description', 'N/A')
            if description and description != 'N/A':
                description = clean_html(description)

            specifications = detail_data.get("specifications", [])
            detail_info = {}
            for spec in specifications:
                attrs = spec.get("attributes", [])
                for attr in attrs:
                    name = attr.get("name")
                    value = attr.get("value")
                    if name and value:
                        detail_info[name] = value

            return description, detail_info
        else:
            return 'N/A', {}
    except Exception as e:
        return 'N/A', {}


def crawl_tiki_products(category_url, total_products=80, fetch_detail=True):


    match = re.search(r'/([^/]+)/c(\d+)', category_url)
    if not match:
        print("✗ URL không hợp lệ! Định dạng đúng: https://tiki.vn/category-name/c1234")
        return []

    url_key = match.group(1)
    category_id = match.group(2)

    print("Đang lấy guest token...")
    session = requests.Session()

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Referer': category_url,
        'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
    }

    try:
        init_response = session.get('https://tiki.vn', headers=headers, timeout=10)
        guest_token = None

        for cookie in session.cookies:
            if 'TOKENS' in cookie.name or 'guest_token' in cookie.name:
                cookie_value = cookie.value
                # Parse JSON từ cookie nếu có
                if '{' in cookie_value:
                    try:
                        token_data = json.loads(cookie_value.replace('%22', '"').replace('%2C', ','))
                        guest_token = token_data.get('guest_token') or token_data.get('access_token')
                    except:
                        pass

        if guest_token:
            headers['x-guest-token'] = guest_token
            print(f" Đã lấy guest token")
    except Exception as e:
        print(f" Không lấy được token, thử crawl mà không có token...")

    all_products = []
    page = 1
    limit = 40

    print(f"\n{'=' * 60}")
    print(f"Category: {url_key}")
    print(f"Category ID: {category_id}")
    print(f"Target: {total_products} sản phẩm")
    print(f"{'=' * 60}\n")

    while len(all_products) < total_products:
        print(f" Đang crawl trang {page} (đã có {len(all_products)} sản phẩm)...")

        # API endpoint chính xác từ Network tab
        api_url = (
            f"https://tiki.vn/api/personalish/v1/blocks/listings"
            f"?limit={limit}"
            f"&include=advertisement"
            f"&aggregations=2"
            f"&version=home-persionalized"
            f"&urlKey={url_key}"
            f"&category={category_id}"
            f"&page={page}"
        )

        try:
            response = session.get(api_url, headers=headers, timeout=15)

            if response.status_code == 200:
                data = response.json()

                print(f"\n   DEBUG - Response keys: {list(data.keys())}")

                products = data.get('data', [])

                if products:
                    print(f"   DEBUG - Products type: {type(products)}")
                    print(f"   DEBUG - First item type: {type(products[0]) if len(products) > 0 else 'N/A'}")
                    if len(products) > 0 and products[0]:
                        print(
                            f"   DEBUG - First item keys: {list(products[0].keys()) if isinstance(products[0], dict) else 'Not a dict'}")
                        print(f"   DEBUG - First item sample: {str(products[0])[:200]}")

                if not products:
                    print("  ℹ Không còn sản phẩm nào!")
                    print(f"   DEBUG - Full response: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")
                    break

                print(f"  Tìm thấy {len(products)} items")

                for idx, product in enumerate(products, 1):
                    try:
                        if not product or not isinstance(product, dict):
                            continue
                        if 'id' not in product or 'name' not in product:
                            continue

                        product_id = product.get('id')
                        product_name = product.get('name', 'N/A')

                        price = product.get('price')
                        if not price:
                            price = product.get('list_price') or product.get('original_price', 0)

                        price_formatted = f"{price:,} đ" if isinstance(price, (int, float)) else str(price)
                        discount = product.get('discount_rate', 0)

                        description = product.get('short_description', 'N/A')

                        if fetch_detail and product_id:
                            description, detail_info = get_product_detail_info(product_id, session, headers)
                        else:
                            detail_info = {}

                        url_path = product.get('url_path') or product.get('url_key', '')
                        product_url = f"https://tiki.vn/{url_path}" if url_path else 'N/A'

                        rating = product.get('rating_average', 0)
                        review_count = product.get('review_count', 0)
                        thumbnail = product.get('thumbnail_url') or product.get('thumbnail', '')

                        product_data = {
                            'id': product_id,
                            'name': product_name,
                            'price': price_formatted,
                            'original_price': product.get('original_price'),
                            'discount': f"{discount}%" if discount else "0%",
                            'description': description,
                            'detailed_info': detail_info,
                            'rating': rating,
                            'review_count': review_count,
                            'url': product_url,
                            'thumbnail': thumbnail
                        }

                        all_products.append(product_data)

                        # Hiển thị progress
                        if idx <= 3 or (fetch_detail and idx <= 2):  # Hiển thị ít hơn khi fetch detail
                            print(f"    {idx}. {product_name[:60]}")
                            if fetch_detail and description != 'N/A':
                                print(f" Description: {description[:80]}...")

                    except Exception as e:
                        print(f"Lỗi khi xử lý sản phẩm: {e}")
                        continue

                if len(products) < limit:
                    print(f"\n  Đã crawl hết sản phẩm (trang cuối trả về {len(products)} sản phẩm)")
                    break

            elif response.status_code == 404:
                print("  API không tồn tại (404)")
                break
            elif response.status_code == 403:
                print("   Bị chặn (403) - cần thêm thời gian delay")
                time.sleep(5)
            else:
                print(f"   Lỗi {response.status_code}: {response.text[:100]}")
                break

        except requests.exceptions.Timeout:
            print("   Timeout - thử lại...")
            time.sleep(3)
            continue
        except Exception as e:
            print(f"   Lỗi: {str(e)[:100]}")
            break

        page += 1

        time.sleep(2)

    if all_products:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'tiki_{url_key}_{timestamp}.json'

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_products, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print(f" HOÀN THÀNH!")
        print(f" Đã crawl {len(all_products)} sản phẩm")
        print(f" Dữ liệu đã lưu vào: {filename}")
        print(f"{'=' * 60}\n")

        # Hiển thị preview
        print(" Preview 3 sản phẩm đầu tiên:")
        for i, p in enumerate(all_products[:3], 1):
            print(f"\n{i}. {p['name']}")
            print(f"    Giá: {p['price']} (Giảm {p['discount']})")
            print(f"    Rating: {p['rating']} ({p['review_count']} đánh giá)")
            print(f"    Description: {p['description'][:150]}..." if len(
                str(p['description'])) > 150 else f"    Description: {p['description']}")
            print(f"    {p['url']}")
    else:
        print("\n✗ Không crawl được sản phẩm nào!")

    return all_products


# SỬ DỤNG
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" " * 20 + "TIKI CRAWLER")
    print("=" * 60 + "\n")

    examples = {
        'Điện thoại': 'https://tiki.vn/dien-thoai-may-tinh-bang/c1789',
        'Laptop': 'https://tiki.vn/laptop-may-vi-tinh-linh-kien/c1846',
        'Sách': 'https://tiki.vn/nha-sach-tiki/c8322',
        'Đồ Chơi - Mẹ & Bé': 'https://tiki.vn/do-choi-me-be/c2549',
        "Thời trang nam": "https://tiki.vn/thoi-trang-nam/c915",
        "Thời trang nữ": "https://tiki.vn/thoi-trang-nu/c931",
        "Nhà Cửa - Đời Sống": "https://tiki.vn/nha-cua-doi-song/c1883",
        "Thể thao": "https://tiki.vn/the-thao-da-ngoai/c1975",
        'Điện gia dụng': 'https://tiki.vn/dien-gia-dung/c1882',
        'Làm Đẹp - Sức Khoẻ': 'https://tiki.vn/lam-dep-suc-khoe/c1520',
        'Đồng hồ - Trang sức': 'https://tiki.vn/dong-ho-va-trang-suc/c8371',
        'Balo - Vali': 'https://tiki.vn/balo-va-vali/c6000'
    }

    print(" Một số danh mục phổ biến:")
    for name, url in examples.items():
        print(f"   • {name}: {url}")
    print()

    total_products = 5000
    for _, url in examples.items():
        products = crawl_tiki_products(url, total_products, fetch_detail=True)
