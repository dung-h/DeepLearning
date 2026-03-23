# Tổng hợp kết quả cuối cùng - CO3133

Tài liệu này được viết để đưa thẳng vào slide báo cáo hoặc phần tổng hợp trên landing page.

## 1. Nhánh văn bản

Dataset: `Jigsaw Toxic Comment Classification Challenge`

| Model | Exact-match accuracy | Micro F1 | Macro F1 |
| --- | ---: | ---: | ---: |
| BERT | 0.9306 | 0.8030 | 0.6765 |
| LSTM | 0.9224 | 0.7386 | 0.5489 |

Nhận xét:
- BERT là mô hình tốt nhất ở nhánh văn bản.
- BERT vượt LSTM trên cả exact-match accuracy, micro F1 và macro F1.

## 2. Nhánh đa phương thức

Dataset: `CrisisMMD v2.0 - Humanitarian Categories (5 classes, agreed-label split)`

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| CLIP | 0.8806 | 0.7965 |
| VisualBERT | 0.8115 | 0.7290 |

Nhận xét:
- CLIP là mô hình tốt nhất ở nhánh đa phương thức.
- CLIP vượt VisualBERT trên cả accuracy và macro F1.

## 3. Kết luận tổng hợp

- Ở hai nhánh đã hoàn tất trong repo hiện tại, các mô hình transformer-based đều cho kết quả tốt hơn baseline còn lại.
- Nhánh văn bản: `BERT > LSTM`
- Nhánh đa phương thức: `CLIP > VisualBERT`

## 4. Nguồn và tài nguyên

- Official CrisisMMD: https://crisisnlp.qcri.org/crisismmd
- CrisisMMD paper: https://arxiv.org/abs/1805.00713
- Tệp notebook văn bản: `btl1/notebooks/text_classification.ipynb`
- Tệp notebook đa phương thức: `btl1/notebooks/text_image_classification.ipynb`
