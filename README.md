# AromaLatent: Project Synthesis & Learning Outcomes

## Overview
This document summarizes the technical work, methodologies, and learning outcomes from the **AromaLatent** project. It is intended to provide structured context for drafting a professional CV, highlighting practical experience in Data Engineering, Machine Learning, Natural Language Processing (NLP), and Information Retrieval.

## Project Description
AromaLatent is a dual-track Artificial Intelligence project focused on the high-dimensional latent space of perfumery. Moving beyond basic classification and "black-box" APIs, the project maps mathematical relationships between chemical notes and finished fragrances, and builds an explainable Semantic Retrieval-Augmented Generation (RAG) system for text-to-perfume recommendations.

## Core Technical Contributions & Learnings

### 1. Advanced Data Engineering & Scraping
- **Challenges Overcome:** Successfully bypassed aggressive anti-scraping measures (Cloudflare JS challenges) on Fragrantica by migrating from basic `requests`/`BeautifulSoup` to `undetected-chromedriver` and Selenium.
- **Data Integration:** Built a custom data pipeline (`pipeline_v3.py`) to ingest and merge canonical fragrance data (notes, accords) with unstructured, noisy user reviews scraped dynamically from Reddit (`r/fragrance`).
- **Database Management:** Designed and managed a local SQLite database (`perfumes.db`) to efficiently store and query complex multi-modal data (static CSV facts and dynamic LLM profiles).

### 2. Natural Language Processing (NLP) & LLM Integration
- **Aspect-Based Sentiment Analysis (ABSA):** Leveraged the DeepSeek API to parse unstructured Reddit text into structured JSON profiles, extracting critical variables such as vibes, longevity, sillage, occasion, and sentiment.
- **Explainable Generation (XAI):** Focused on generation explainability by treating LLMs as probabilistic engines. Explored extracting token probabilities (logprobs) and cross-attention matrices (using HuggingFace Transformers) to mathematically prove why specific recommendations were generated, ensuring transparency.

### 3. Information Retrieval & Similarity Engines
Designed and implemented a progressive three-tier recommendation engine to identify "Mathematical Dupes" and semantic matches:
- **V1 (Lexical):** Implemented Jaccard Index for discrete sets (notes/accords) and Cosine Similarity for one-hot encoded LLM vibes.
- **V2 (Semantic Hybrid):** Utilized `SentenceTransformers` (`all-MiniLM-L6-v2`) to compress full string profiles into high-dimensional semantic vectors to compare user intent.
- **V3 (Hybrid BM25 + Vector RRF):** Built an advanced hybrid search engine. Combined lexical rarity (`rank_bm25`) with semantic vector search, fusing the results via **Reciprocal Rank Fusion (RRF)** and applying custom structural "dupe bonuses."

### 4. Machine Learning & Unsupervised Taxonomy 
- **Latent Space Clustering:** Conceptualized and explored dimensionality reduction (PCA/UMAP) and density-based clustering (GMM/HDBSCAN) to discover "hidden families" of fragrances, moving beyond traditional taxonomies.
- **Network Analysis:** Designed Note Co-occurrence Graphs using `NetworkX` and Adjacency Matrices to visualize the foundational rules of perfumery and identify structurally important chemical notes.

## Tech Stack
- **Languages:** Python
- **Data Collection:** BeautifulSoup, Selenium (`undetected-chromedriver`), Reddit API
- **NLP & AI:** HuggingFace Transformers (`SentenceTransformers`), DeepSeek API, LangChain
- **Machine Learning & IR:** Scikit-learn (PCA, GMM, TF-IDF), SciPy, `rank_bm25` (Lexical Search)
- **Data Storage & Graphing:** SQLite, Pandas, NetworkX
- **Visualization:** Plotly (3D Interactive mapping), Matplotlib, HTML/CSS (UI generation)

## Key Takeaway for CV
Through AromaLatent, I demonstrated end-to-end AI application development: from complex web scraping and data engineering to building sophisticated hybrid search engines (RRF, BM25, Semantic Embeddings). My focus on **Explainable AI (XAI)** and measurable probabilistic outputs highlights a commitment to building transparent, production-ready machine learning systems.
