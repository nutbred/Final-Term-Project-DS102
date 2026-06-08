# Reddit Crawled Data

Thu muc nay chua du lieu review da crawl tu Reddit va cac bang phai sinh cho hoc may.

## File

- `perfumes_with_reviews.csv`: 636 nuoc hoa co it nhat mot review Reddit duoc gan voi perfume id.
- `reddit_reviews_flat.csv`: 2.159 bai viet/binh luan Reddit o dang bang phang.
- `reddit_reviews.jsonl`: review Reddit o dang JSONL, thuan tien cho doc tung dong.
- `llm_profiles.csv`: 636 profile da duoc DeepSeek rut trich tu review.
- `llm_profiles.jsonl`: profile DeepSeek o dang JSONL.
- `similarity_hybrid_rrf_v3.csv`: 201.930 cap similarity theo method chinh trong bao cao.

## Ghi chu

Review Reddit la nguon van ban danh gia chinh. Pipeline chi chay similarity, PCA va clustering tren 636 nuoc hoa co review Reddit, khong chay tren toan bo catalog Fragrantica/Kaggle.
