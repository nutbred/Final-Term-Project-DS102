# Dữ liệu crawl từ Reddit

Thư mục này chứa dữ liệu review đã crawl từ Reddit và các bảng phái sinh cho học máy.

## File

- `perfumes_with_reviews.csv`: 636 nước hoa có ít nhất một review Reddit được gắn với perfume id.
- `reddit_reviews_flat.csv`: 2.159 bài viết/bình luận Reddit ở dạng bảng phẳng.
- `reddit_reviews.jsonl`: review Reddit ở dạng JSONL, thuận tiện cho đọc từng dòng.
- `llm_profiles.csv`: 636 profile đã được DeepSeek rút trích từ review.
- `llm_profiles.jsonl`: profile DeepSeek ở dạng JSONL.
- `similarity_hybrid_rrf_v3.csv`: 201.930 cặp similarity theo method chính trong báo cáo.

## Ghi chú

Review Reddit là nguồn văn bản đánh giá chính. Pipeline chỉ chạy similarity, PCA và clustering trên 636 nước hoa có review Reddit, không chạy trên toàn bộ catalog Fragrantica/Kaggle.
