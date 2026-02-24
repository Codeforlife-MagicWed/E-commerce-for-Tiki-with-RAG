# E-commerce Chatbot for Tiki using RAG
---

## Overview
---

This project builds a Vietnamese E-commerce Chatbot for the Tiki platform using a Retrieval-Augmented Generation (RAG) framework.  
The system combines hybrid retrieval techniques with instruction-tuned Large Language Models (LLMs) to provide accurate, grounded, and context-aware responses.

The chatbot supports:

- Product search and filtering  
- Specification and price queries  
- Brand comparison  
- FAQ and policy lookup  
- General shopping assistance in Vietnamese  


## Dataset
---

Data was collected directly from the Tiki platform:

- 22,102 products  
- 116 FAQ entries  

**Product data includes:**

- Title, brand, category  
- Normalized price  
- Rating and review count  
- Description and detailed specifications  
- Metadata for filtering and ranking  

**FAQ data covers:**

- Account  
- Ordering & Payment  
- Delivery  
- Returns – Warranty – Compensation  
- Information & Policy  
- Services & Programs  

All data is stored in structured JSON format and processed before indexing.

## System Architecture
---

The RAG pipeline consists of five main components:

1. Data preprocessing and chunking  
2. Embedding and indexing  
3. Query routing  
4. Hybrid retrieval  
5. LLM-based grounded answer generation  

### Retrieval Strategy

- Dense embeddings: **BGE-M3**  
- Sparse retrieval: **BM25**  
- Rank fusion: **Reciprocal Rank Fusion (RRF)**  
- Vector database: **Qdrant**

This hybrid design improves both semantic understanding and exact keyword matching.



## Language Models
---

Three LLMs were evaluated and fine-tuned using LoRA:

- PhoGPT-4B  
- Qwen1.8-2B  
- Qwen2.5-7B-Instruct  

Training setup:

- ~9,000 instruction–response samples  
- 90/10 train–test split  
- Causal language modeling objective  
- Mixed precision training (fp16 / bfloat16)  



## Results
---

**PhoGPT**
- Fluent Vietnamese responses  
- Stable convergence  
- Limited reasoning depth  

**Qwen1.8**
- Strong instruction-following  
- Stable generalization  
- 99/100 correct responses on real-world test queries  

**Qwen2.5-7B-Instruct**
- Best overall reasoning capability  
- Strong contextual accuracy  
- Most suitable for production deployment  

Qwen-based models significantly outperformed PhoGPT in complex product queries.



## Hallucination Control
---

To ensure reliability and factual consistency:

- Generation is grounded strictly on retrieved evidence  
- No-information fallback responses are enforced  
- Context is filtered and sanitized before prompting  
- Product chunk expansion is applied for detailed queries  




## Conclusion
---

This project demonstrates that combining hybrid retrieval (BGE-M3 + BM25 + RRF) with instruction-tuned LLMs provides a scalable and production-ready solution for Vietnamese e-commerce chatbots.

The final system achieves:

- High retrieval accuracy  
- Strong instruction-following behavior  
- Reliable, grounded responses suitable for deployment  
