# Tóm tắt kết quả cuối BTL1

## Nhánh văn bản

- Dataset: `Jigsaw Toxic Comment Classification Challenge`
- Bài toán: `multi-label text classification` với 6 nhãn
- So sánh: `BERT` vs `LSTM`
- Kết quả tốt nhất:
  - `BERT`
  - Exact-match accuracy: `0.8545`
  - Micro F1: `0.6385`
  - Macro F1: `0.5787`

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
  - Accuracy: `0.8752`
  - Macro F1: `0.8641`
  - Weighted F1: `0.8746`

## Nhánh ảnh

- Dataset: `Weather Dataset`
- Bài toán: `image classification`
- So sánh: `ResNet50` vs `ViT-Base`
- Kết quả tốt nhất:
  - `ViT-Base`
  - Accuracy: `0.9049`
  - Weighted F1: `0.9047`

## Nhận xét nhanh

- Ở nhánh văn bản, `BERT` vượt `LSTM` trên toàn bộ chỉ số chính trong protocol stratified hiện tại.
- Ở nhánh đa phương thức, `VisualBERT` vượt `CLIP` trên cả accuracy, macro F1 và weighted F1.
- Ở nhánh ảnh, `ViT-Base` vượt `ResNet50` khá rõ trên cả accuracy và weighted F1.
- Toàn bộ số liệu trên đều được lấy từ artifact thật sau khi chạy notebook trong thư mục `btl1/artifacts/`.
