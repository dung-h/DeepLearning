# BTL1 Demo

Demo Gradio cho Bài tập lớn số 1.

## Thành phần hiện có

- `Text Classification`
  - dataset: Jigsaw Toxic Comment
  - models: BERT, LSTM
  - hỗ trợ chạy một mô hình hoặc so sánh trực tiếp hai mô hình

- `Text-Image Classification`
  - dataset: N24News
  - models: CLIP, VisualBERT
  - hỗ trợ chạy một mô hình hoặc so sánh trực tiếp hai mô hình

## Yêu cầu local

Demo này phụ thuộc vào checkpoint local trong:

- `btl1/artifacts/text/`
- `btl1/artifacts/multimodal/`

Vì các checkpoint `.pt` không được đẩy lên remote, demo đầy đủ chỉ chạy được trên máy có artifact local.
