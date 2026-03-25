# Tóm tắt kết quả cuối BTL1

## Nhánh văn bản

- Dataset: `Jigsaw Toxic Comment Classification Challenge`
- Bài toán: `multi-label text classification` với 6 nhãn
- So sánh: `BERT` vs `LSTM`
- Kết quả tốt nhất:
  - `BERT`
  - Exact-match accuracy: `0.9306`
  - Micro F1: `0.8030`
  - Macro F1: `0.6765`

## Nhánh đa phương thức

- Dataset: `N24News`
- Bài toán: `image + news text -> category classification`
- So sánh: `CLIP` vs `VisualBERT`
- Split official đã dùng:
  - `train = 48,988`
  - `val = 6,123`
  - `test = 6,124`
- Kết quả tốt nhất:
  - `VisualBERT`
  - Accuracy: `0.8751`
  - Macro F1: `0.8575`
  - Weighted F1: `0.8732`

## Nhận xét nhanh

- Ở nhánh văn bản, `BERT` vượt `LSTM` trên toàn bộ chỉ số chính.
- Ở nhánh đa phương thức, `VisualBERT` vượt `CLIP` trên cả accuracy, macro F1 và weighted F1.
- Toàn bộ số liệu trên đều được lấy từ artifact thật sau khi chạy notebook trong thư mục `btl1/artifacts/`.
