# Danh mục Fragrantica/Kaggle

Thư mục này chứa dữ liệu catalog ban đầu, không phải dữ liệu review được crawl trực tiếp từ Fragrantica.

## File

- `fra_cleaned.csv`: bản đã làm sạch, dùng chính trong đồ án.
- `fra_perfumes.csv`: bản dữ liệu gốc được giữ kèm để đối chiếu.

## Vai trò trong pipeline

- Lấy tên nước hoa và hãng để chọn seed.
- Lấy note/accord làm tín hiệu thành phần mùi.
- Lấy rating/votes để ưu tiên các mẫu phổ biến khi crawl Reddit.
- Ánh xạ từ catalog sang review Reddit thông qua tên nước hoa và hãng.
