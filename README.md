# CO3133 Project Workspace

Repo này được tổ chức theo từng bài tập lớn để tránh trộn lẫn tài nguyên giữa các giai đoạn làm việc.

## Cấu trúc chính

- `index.html`, `assignments/`, `assets/`: landing page chung của môn học.
- `btl1/`: notebook, dữ liệu, artifact và report của Bài tập lớn 1.
- `btl2/`: chỗ trống cho Bài tập lớn 2.
- `btl3/`: chỗ trống cho Bài tập lớn 3.
- `docs/`: tài liệu chung như đề bài.
- `legacy/`: hướng triển khai cũ không còn dùng chính.
- `logs/`: log cục bộ.
- `reports/`: lớp redirect tương thích, chuyển về report thật trong `btl1/reports/`.
- `GPT.md`: trạng thái và quy ước hiện tại của repo.
- `tracking_progress.txt`: nhật ký mốc triển khai ở mức repo.

## Trạng thái hiện tại

- Phần đã hoàn thành thực tế hiện chỉ thuộc `BTL1`.
- `BTL2` và `BTL3` chưa có pipeline chính.
- Hai notebook chính của `BTL1` là:
  - `btl1/notebooks/text_classification.ipynb`
  - `btl1/notebooks/text_image_classification.ipynb`

## Quy ước giữ repo sạch

- Không đặt dữ liệu raw, zip dataset hay artifact ở root.
- Mọi file thực thi của BTL1 phải nằm trong `btl1/`.
- Nếu cần giữ đường dẫn cũ cho trình duyệt hoặc bookmark, dùng wrapper trong `reports/` thay vì nhân bản report.
- `legacy/visual_genome/` chỉ còn là tham chiếu lịch sử, không thuộc nhánh triển khai chính.
