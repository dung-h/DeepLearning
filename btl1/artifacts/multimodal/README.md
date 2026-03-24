# Multimodal Artifacts

Thư mục này lưu artifact của nhánh đa phương thức trong `BTL1`.

## Dataset hiện tại

- Dataset: `N24News`
- Task: `image + news text -> category classification`
- Models: `CLIP` và `VisualBERT`

## Artifact chính

- `n24news_model_comparison.csv`: bảng so sánh metric cuối giữa hai mô hình
- `n24news_metrics_summary.json`: tóm tắt dataset, split, label names và metric
- `n24news_*_history.csv/png`: lịch sử huấn luyện và figure learning curve
- `n24news_*_confusion_matrix.png`: confusion matrix trên tập test
- `n24news_per_label_f1.csv/png`: F1 theo từng lớp
- `n24news_*_best.pt`: checkpoint tốt nhất của từng mô hình
