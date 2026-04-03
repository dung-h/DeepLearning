# BTL1 Demo

Gradio demo cho BTL1, dùng lại các checkpoint đã huấn luyện để minh họa ba nhánh bài toán:

- `Text classification`
  - dataset: Jigsaw Toxic Comment
  - models: `BERT`, `LSTM`
- `Multimodal classification`
  - dataset: N24News
  - official live demo models: `CLIP full_finetune`, `VisualBERT full_finetune`
  - benchmark extension: `deep_head`, `lora`
- `Image classification`
  - weather image dataset
  - models: `ResNet50`, `ViT-Base`

## Demo này chứng minh gì

- chạy suy luận trực tiếp từ checkpoint thật
- hiển thị benchmark đã chốt từ artifact
- cho thấy so sánh chiến lược huấn luyện ở nhánh multimodal
- hỗ trợ khôi phục checkpoint từ bundle zip nếu file local bị thiếu

## Yêu cầu local

Demo ưu tiên dùng checkpoint local trong:

- `btl1/artifacts/text/`
- `btl1/artifacts/multimodal/`
- `btl1/artifacts/image/`

Nếu thiếu checkpoint, app sẽ thử khôi phục từ các bundle zip trong:

- `btl1/artifacts/text/downloads/`
- `btl1/artifacts/multimodal/downloads/`
- `btl1/artifacts/image/downloads/`

## Chạy demo

```powershell
cd D:\DeepLearning\btl1\demo
python app.py
```

Mặc định app mở ở:

- `http://127.0.0.1:43881`
