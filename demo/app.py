import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
import time

import config
try:
    from rag_engine import (
        build_parent_lookup,
        route,
        answer_with_rag,
        search_auto_hybrid,
        AnswerPolicy,
        ParentBM25Index,
        load_llm,
        LLMConfig
    )

    print("Imported from rag_engine.py")

except ImportError as e:
    print(f" Cannot import from rag_engine.py: {e}")


st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)


st.markdown(f"""
<style>
    /* Main styling */
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {config.PRIMARY_COLOR};
        text-align: center;
        margin-bottom: 0.5rem;
    }}

    .sub-header {{
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }}

    /* Result cards */
    .result-card {{
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid {config.PRIMARY_COLOR};
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    .product-item {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }}

    .product-item:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}

    /* Badges */
    .badge {{
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0.5rem 0.5rem 0;
    }}

    .badge-product {{
        background-color: {config.SUCCESS_COLOR};
        color: white;
    }}

    .badge-faq {{
        background-color: #17a2b8;
        color: white;
    }}

    .badge-both {{
        background-color: {config.WARNING_COLOR};
        color: white;
    }}

    /* Metrics */
    .metric-card {{
        background: linear-gradient(135deg, {config.SECONDARY_COLOR} 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }}

    .metric-label {{
        font-size: 0.9rem;
        opacity: 0.9;
    }}

    /* Buttons */
    .stButton>button {{
        width: 100%;
        background-color: {config.PRIMARY_COLOR};
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s;
    }}

    .stButton>button:hover {{
        background-color: #1557b0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}

    /* Sidebar */
    .css-1d391kg {{
        background-color: #f8f9fa;
    }}

    /* Price display */
    .price {{
        color: {config.DANGER_COLOR};
        font-weight: bold;
        font-size: 1.2rem;
    }}

    /* Rating */
    .rating {{
        color: #ffc107;
        font-weight: bold;
    }}

    /* Loading animation */
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    .loading {{
        animation: pulse 1.5s ease-in-out infinite;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_models():
    """Load and cache all models"""
    with st.spinner("Model loading..."):
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from rag_engine import ParentBM25Index, load_llm, LLMConfig

        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY
        )

        q_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        parent_lookup = build_parent_lookup(str(config.PRODUCTS_JSONL))
        parent_lookup_rev = {v: k for k, v in parent_lookup.items()}

        bm25_index = ParentBM25Index.load(str(config.BM25_INDEX_DIR))

        llm_config = LLMConfig(
            model_name=config.LLM_MODEL_NAME,
            load_in_4bit=config.LLM_LOAD_IN_4BIT,
            max_new_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE
        )
        llm = load_llm(llm_config)

        policy = AnswerPolicy(
            min_results_product=config.MIN_RESULTS_PRODUCT,
            min_results_faq=config.MIN_RESULTS_FAQ,
            allow_expand_chunks_detail=config.ALLOW_EXPAND_CHUNKS_DETAIL,
            expand_chunks_per_parent_detail=config.EXPAND_CHUNKS_PER_PARENT_DETAIL
        )

        return {
            'client': client,
            'q_model': q_model,
            'parent_lookup': parent_lookup,
            'parent_lookup_rev': parent_lookup_rev,
            'bm25_index': bm25_index,
            'llm': llm,
            'policy': policy
        }


if 'models' not in st.session_state:
    st.session_state.models = initialize_models()
    st.success(" Models loaded successfully!")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>Tiki RAG Bot</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Xóa lịch sử", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.search_results = None
        st.rerun()


st.markdown(f'<div class="main-header">{config.PAGE_TITLE} {config.PAGE_ICON}</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Trợ lý mua sắm thông minh với Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Chat", "Kết quả chi tiết", "Hướng dẫn"])


# TAB 1: CHAT
with tab1:
    st.markdown("### Đặt câu hỏi của bạn")

    # User input
    query = st.text_area(
        "Nhập câu hỏi:",
        height=120,
        placeholder="VD: Gợi ý balo laptop, hoặc Chính sách đổi trả như thế nào?",
        key="query_input",
        value=st.session_state.get('query_input', '')
    )

    # Action buttons
    col_search, col_clear = st.columns([4, 1])
    with col_search:
        search_button = st.button(" Tìm kiếm", type="primary", use_container_width=True)
    price_min = 0
    price_max = 0
    rating_min = 0.0
    # Process query
    if search_button and query:
        start_time = time.time()

        with st.spinner(" Đang tìm kiếm..."):
            try:
                # Get models
                models = st.session_state.models

                # Apply filters
                price_gte = price_min if price_min > 0 else None
                price_lte = price_max if price_max > 0 else None
                rating_gte = rating_min if rating_min > 0 else None

                # Call RAG
                target, df_ctx, answer = answer_with_rag(
                    llm=models['llm'],
                    client=models['client'],
                    bm25_index=models['bm25_index'],
                    query=query,
                    search_fn=search_auto_hybrid,
                    parent_lookup=models['parent_lookup'],
                    parent_lookup_rev=models['parent_lookup_rev'],
                    policy=models['policy'],
                    topk=config.DEFAULT_TOPK,
                    rrf_lambda=config.DEFAULT_TOPK,
                    per_parent_chunks=config.DEFAULT_PER_PARENT_CHUNKS,
                    verbose=False
                )

                elapsed = time.time() - start_time

                # Store results
                st.session_state.search_results = {
                    'query': query,
                    'target': target,
                    'df_ctx': df_ctx,
                    'answer': answer,
                    'elapsed': elapsed
                }

                # Add to history
                st.session_state.chat_history.append({
                    'query': query,
                    'answer': answer,
                    'target': target,
                    'num_results': len(df_ctx) if df_ctx is not None else 0,
                    'timestamp': time.strftime("%H:%M:%S")
                })

                st.success(f" Hoàn thành trong {elapsed:.2f}s!")

            except Exception as e:
                st.error(f" Lỗi: {str(e)}")
                st.exception(e)

    # Display current result
    if st.session_state.search_results:
        result = st.session_state.search_results

        st.markdown("---")
        st.markdown("###  Câu trả lời")

        # Display target badge
        badge_class = f"badge-{result['target']}"
        emoji_map = {"product": "product", "faq": "faq", "both": "both"}
        emoji = emoji_map.get(result['target'], "target")

        st.markdown(f"""
        <div class="badge {badge_class}">
            {emoji} {result['target'].upper()}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="result-card">{result["answer"]}</div>', unsafe_allow_html=True)

        if result['df_ctx'] is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Kết quả", len(result['df_ctx']))
            with col2:
                if 'parent_uid' in result['df_ctx'].columns:
                    st.metric("Sản phẩm", result['df_ctx']['parent_uid'].nunique())
            with col3:
                st.metric("Thời gian", f"{result['elapsed']:.2f}s")

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("###  Lịch sử chat")

        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(
                    f"[{chat['timestamp']}] {chat['query'][:80]}...",
                    expanded=(i == 0)
            ):
                st.markdown(f"**Loại:** {chat['target']} | **Kết quả:** {chat['num_results']}")
                st.markdown("---")
                st.markdown(chat['answer'])


# TAB 2: DETAILED RESULTS
with tab2:
    st.markdown("### Chi tiết kết quả tìm kiếm")

    if st.session_state.search_results and st.session_state.search_results['df_ctx'] is not None:
        df_display = st.session_state.search_results['df_ctx']

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tổng kết quả</div>
                <div class="metric-value">{len(df_display)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if 'parent_uid' in df_display.columns:
                n_products = df_display['parent_uid'].nunique()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sản phẩm</div>
                    <div class="metric-value">{n_products}</div>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            if 'score' in df_display.columns:
                avg_score = df_display['score'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Score TB</div>
                    <div class="metric-value">{avg_score:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

        with col4:
            if 'price' in df_display.columns:
                prices = pd.to_numeric(df_display['price'], errors='coerce').dropna()
                if len(prices) > 0:
                    avg_price = prices.mean() / 1000
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Giá TB</div>
                        <div class="metric-value">{avg_price:.0f}k</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Display results as cards
        for idx, row in df_display.iterrows():
            st.markdown('<div class="product-item">', unsafe_allow_html=True)

            # Title
            st.markdown(f"### {idx + 1}. {row.get('title', 'N/A')}")

            # Main info
            col_info, col_meta = st.columns([3, 1])

            with col_info:
                # Price
                if row.get('price'):
                    price_vnd = int(row['price'])
                    st.markdown(f"<p class='price'> {price_vnd:,} VNĐ</p>", unsafe_allow_html=True)

                # Rating
                if row.get('rating'):
                    rating = row['rating']
                    reviews = row.get('reviews', 0)
                    st.markdown(f"<p class='rating'> {rating}/5 ({reviews:,} đánh giá)</p>", unsafe_allow_html=True)

                # Brand & Category
                info_parts = []
                if row.get('brand'):
                    info_parts.append(f" {row['brand']}")
                if row.get('category'):
                    info_parts.append(f" {row['category']}")
                if info_parts:
                    st.markdown(" | ".join(info_parts))

                # URL
                if row.get('url'):
                    st.markdown(f" [Xem sản phẩm trên Tiki]({row['url']})")

            with col_meta:
                if 'score' in row:
                    st.metric("Score", f"{row['score']:.3f}")
                if row.get('type'):
                    st.caption(f"Type: {row['type']}")

            # Text snippet
            if row.get('text'):
                with st.expander(" Xem mô tả chi tiết"):
                    text = row['text']
                    if len(text) > 800:
                        st.text(text[:800] + "...")
                    else:
                        st.text(text)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Download button
        csv = df_display.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label=" Download kết quả (CSV)",
            data=csv,
            file_name=f"tiki_search_results_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.info(" Chưa có kết quả. Vui lòng thực hiện tìm kiếm trong tab **Chat**.")

# TAB 3: GUIDE
with tab3:
    st.markdown("###  Hướng dẫn sử dụng")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ####  Các loại câu hỏi

        **1. Tìm kiếm sản phẩm (Product)**
        - Duyệt danh sách sản phẩm theo nhu cầu
        - VD: "Gợi ý balo dưới 500k"
        - VD: "Điện thoại Samsung tầm 5 triệu"

        **2. Chi tiết sản phẩm (Detail)**
        - Xem thông tin chi tiết 1 sản phẩm
        - VD: "Thông số bình Lock&Lock"
        - VD: "Balo Lian có chống nước không"

        **3. FAQ - Chính sách**
        - Hỏi về chính sách, hướng dẫn
        - VD: "Chính sách đổi trả"
        - VD: "Thanh toán Momo được không"
        """)

    with col2:
        st.markdown("""
        #### Cấu hình nâng cao

        **Top-K**
        - Số lượng kết quả trả về (1-20)

        **RRF Lambda** (0-1)
        - 0: Chỉ dùng BM25 (keyword)
        - 1: Chỉ dùng Dense (semantic)
        - 0.8: Cân bằng (khuyến nghị)

        **Chunks per Parent**
        - Số đoạn text cho mỗi sản phẩm
        - Tăng để có thông tin chi tiết hơn

        **Temperature** (0-1)
        - 0: Câu trả lời chính xác, cứng nhắc
        - 1: Sáng tạo hơn, ít chính xác
        """)

    st.markdown("---")

    st.markdown("""
    #### Tips sử dụng hiệu quả

    1. **Đặt câu hỏi cụ thể**: "Gợi ý balo laptop dưới 500k có chống nước" tốt hơn "balo"
    2. **Sử dụng filters**: Lọc theo giá và rating để thu hẹp kết quả
    3. **Xem tab "Kết quả chi tiết"**: Để kiểm tra các sản phẩm tìm được
    4. **Download CSV**: Để phân tích offline hoặc so sánh
    5. **RAG Parameters**: Điều chỉnh RRF Lambda và Chunks per Parent để tối ưu kết quả
    #### Lưu ý

    - Hệ thống CHỈ trả lời dựa trên dữ liệu có sẵn
    - Không suy đoán hay bịa đặt thông tin
    - Nếu không tìm thấy, hãy thử câu hỏi khác hoặc thay đổi filters
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Using <strong>Streamlit</strong> | 
    Powered by <strong>Qwen2.5-7B</strong> & <strong>BGE-M3</strong> | 
    Data from <strong>Tiki</strong> 
</div>
""", unsafe_allow_html=True)