# DeepLearning Project (CO3133)

Repository cho học phần CO3133 (Học sâu và ứng dụng), nhóm DL123.

- Trường: Đại học Bách khoa
- Khoa: Khoa Khoa học và Kỹ thuật Máy tính
- Giảng viên: Lê Thành Sách
- Năm học: 2025-2026, Học kỳ 2
- Thành viên:
  - Hồ Anh Dũng (2310543)
  - Huỳnh Đức Nhân (2312420)

## Giới Thiệu Dự Án

Dự án tổng hợp bài tập lớn của môn học, tập trung vào các bài toán phân loại trong học sâu.

Phạm vi công khai hiện tại là Bài tập lớn số 1, gồm 3 bài toán:

1. Phân loại văn bản:
   Jigsaw Toxic Comment đa nhãn (6 nhãn), so sánh BERT và LSTM.
2. Phân loại đa phương thức:
   N24News, so sánh CLIP (zero-shot/few-shot) và VisualBERT, kèm các biến thể deep head và LoRA.
3. Phân loại ảnh:
   Weather dataset, so sánh ResNet50 và ViT-Base.

BTL2 và BTL3 đang trong giai đoạn chuẩn bị nội dung.

## Cấu Trúc Thư Mục Chính

- index.html, assignments/, assets/: landing page và dữ liệu điều hướng
- btl1/: toàn bộ nội dung Bài tập lớn số 1
  - data/: dữ liệu sử dụng cho thực nghiệm
  - notebooks/: notebook huấn luyện và đánh giá
  - artifacts/: bảng kết quả, summary, checkpoint metadata
  - reports/: trang báo cáo chi tiết
  - demo/, scripts/: ứng dụng demo và script hỗ trợ
- btl2/, btl3/: khung thư mục cho các bài tập lớn tiếp theo
- reports/: các trang tổng hợp/mirror để điều hướng nhanh
- docs/, legacy/, logs/: tài liệu tham khảo, phiên bản cũ, log chạy

## Truy Cập Nhanh

- Landing page: index.html
- Trang Bài tập lớn số 1: assignments/assignment-1.html
- Tổng hợp kết quả cuối: btl1/reports/final-results.html

Báo cáo theo bài toán:

- Văn bản: btl1/reports/text-report.html
- Đa phương thức: btl1/reports/multimodal-report.html
- Ảnh: btl1/reports/image-report.html

Notebook chính:

- btl1/notebooks/text_classification.ipynb
- btl1/notebooks/text_image_classification.ipynb
- btl1/notebooks/image_classification.ipynb

Artifact tiêu biểu:

- btl1/artifacts/text/text_model_comparison.csv
- btl1/artifacts/multimodal/n24news_model_comparison_full_finetune.csv
- btl1/artifacts/multimodal/n24news_model_comparison_all_variants.csv
- btl1/artifacts/image/training_summary.json

## Trạng Thái Hiện Tại

- Hướng nghiên cứu và triển khai chính: BTL1
- Tổng kết mô hình tốt nhất (theo landing page):
  - Văn bản: BERT
  - Đa phương thức: VisualBERT
  - Ảnh: ViT-Base
- BTL2 và BTL3 chưa công bố đầy đủ kết quả thực nghiệm
