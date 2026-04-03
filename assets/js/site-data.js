window.SITE_DATA = {
  course: {
    code: "CO3133",
    name: "Học sâu và ứng dụng",
    university: "Đại học Bách khoa",
    faculty: "Khoa Khoa học và Kỹ thuật Máy tính",
    instructor: "Lê Thành Sách",
    term: "Năm học 2025-2026, Học kỳ 2",
    summary:
      "Bài tập lớn số 1 gồm ba bài toán phân loại: văn bản, đa phương thức và ảnh.",
    overview: {
      title: "Bài tập lớn số 1",
      points: [
        "Bài toán phân loại văn bản dùng Jigsaw Toxic Comment để so sánh BERT và LSTM trên bài toán đa nhãn 6 lớp.",
        "Bài toán phân loại đa phương thức dùng N24News để so sánh CLIP, VisualBERT và các biến thể deep head, LoRA.",
        "Bài toán phân loại ảnh dùng weather dataset để so sánh ResNet50 và ViT-Base.",
      ],
    },
  },
  group: {
    name: "DL123",
    summary: "Hồ Anh Dũng · 2310543 | Huỳnh Đức Nhân · 2312420",
    members: [
      {
        name: "Hồ Anh Dũng",
        role: "MSSV 2310543",
      },
      {
        name: "Huỳnh Đức Nhân",
        role: "MSSV 2312420",
      },
    ],
  },
  assignments: [
    {
      id: "assignment-1",
      numberLabel: "Bài tập lớn số 1",
      title: "Ba bài toán phân loại: văn bản, đa phương thức và ảnh",
      page: "assignments/assignment-1.html",
      cardSummary:
        "Jigsaw Toxic Comment, N24News và weather dataset.",
      overview:
        "Trang bài tập tổng hợp báo cáo, notebook, artifact và checkpoint cho cả ba bài toán.",
      summary:
        "Toàn bộ đường dẫn chính của Bài tập lớn số 1 được gom tại đây.",
      resources: [],
      resourceGroups: [
        {
          title: "Điểm truy cập chính",
          copy: "Các trang nên mở trước để xem nhanh toàn bộ bài làm.",
          featured: true,
          items: [
            {
              type: "Tổng hợp",
              label: "Mở bảng kết quả",
              title: "Bảng kết quả cuối",
              note: "Tổng hợp kết quả của cả ba bài toán và dẫn sang từng báo cáo chi tiết.",
              url: "../btl1/reports/final-results.html",
              featured: true,
            },
            {
              type: "Mã nguồn",
              label: "Mở GitHub",
              title: "Repository của nhóm",
              note: "Landing page, notebook, artifact, report và mã nguồn của Bài tập lớn số 1.",
              url: "https://github.com/dung-h/DeepLearning",
            },
          ],
        },
        {
          title: "Báo cáo theo từng bài toán",
          copy: "Mỗi bài toán có một trang báo cáo riêng.",
          items: [
            {
              type: "Bài toán phân loại văn bản",
              label: "Mở báo cáo",
              title: "Jigsaw Toxic Comment",
              note: "EDA, mô hình, kết quả và error analysis cho BERT và LSTM.",
              url: "../btl1/reports/text-report.html",
            },
            {
              type: "Bài toán phân loại đa phương thức",
              label: "Mở báo cáo",
              title: "N24News",
              note: "EDA, CLIP, VisualBERT, PEFT variants và phân tích kết quả.",
              url: "../btl1/reports/multimodal-report.html",
            },
            {
              type: "Bài toán phân loại ảnh",
              label: "Mở báo cáo",
              title: "Weather Dataset",
              note: "EDA, split 70/15/15, augmentation và so sánh ResNet50 với ViT-Base.",
              url: "../btl1/reports/image-report.html",
            },
          ],
        },
        {
          title: "Notebook và artifact",
          copy: "Các nguồn gốc để đối chiếu lại pipeline và số liệu.",
          items: [
            {
              type: "Notebook",
              label: "Notebook văn bản",
              title: "Notebook Jigsaw",
              note: "Huấn luyện và đánh giá BERT và LSTM trên dữ liệu toxic comment đa nhãn.",
              url: "../btl1/notebooks/text_classification.ipynb",
            },
            {
              type: "Notebook",
              label: "Notebook đa phương thức",
              title: "Notebook N24News",
              note: "EDA, split, CLIP, VisualBERT, PEFT variants và artifact của bài toán đa phương thức.",
              url: "../btl1/notebooks/text_image_classification.ipynb",
            },
            {
              type: "Notebook",
              label: "Notebook ảnh",
              title: "Notebook weather dataset",
              note: "EDA, augmentation, training và evaluation của ResNet50 và ViT-Base.",
              url: "../btl1/notebooks/image_classification.ipynb",
            },
            {
              type: "Artifact",
              label: "Mở summary ảnh",
              title: "Training summary của bài toán ảnh",
              note: "Accuracy, weighted F1, best epoch và thời gian train của hai mô hình ảnh.",
              url: "../btl1/artifacts/image/training_summary.json",
            },
          ],
        },
        {
          title: "Các biến thể đã huấn luyện",
          copy: "Các cấu hình chính được giữ lại để so sánh trong phạm vi môn học.",
          items: [
            {
              type: "Bài toán phân loại văn bản",
              label: "Xem bảng so sánh",
              title: "BERT và LSTM",
              note: "BERT là mô hình tốt nhất, LSTM là baseline tuần tự.",
              url: "../btl1/artifacts/text/text_model_comparison.csv",
            },
            {
              type: "Bài toán phân loại đa phương thức",
              label: "Xem full finetune",
              title: "CLIP và VisualBERT",
              note: "So sánh chính thức giữa hai mô hình sau khi fine-tune đầy đủ backbone.",
              url: "../btl1/artifacts/multimodal/n24news_model_comparison_full_finetune.csv",
            },
            {
              type: "Bài toán phân loại đa phương thức",
              label: "Xem deep head và LoRA",
              title: "Các biến thể PEFT trên N24News",
              note: "Deep head và LoRA cho cả CLIP lẫn VisualBERT.",
              url: "../btl1/artifacts/multimodal/n24news_model_comparison_all_variants.csv",
            },
            {
              type: "Bài toán phân loại ảnh",
              label: "Xem training summary",
              title: "ResNet50 và ViT-Base",
              note: "ViT-Base vượt ResNet50 trên test set.",
              url: "../btl1/artifacts/image/training_summary.json",
            },
          ],
        },
        {
          title: "Bundle checkpoint",
          copy: "Các gói checkpoint tốt nhất để chạy lại mô hình mà không cần huấn luyện từ đầu.",
          items: [
            {
              type: "Bài toán phân loại văn bản",
              label: "Tải bundle",
              title: "BERT và LSTM",
              note: "Hai best checkpoint của bài toán văn bản.",
              url: "https://drive.google.com/file/d/1PhIMgu-1unj7Yt0dMTGkX2H9473lJycj/view?usp=sharing",
            },
            {
              type: "Bài toán phân loại đa phương thức",
              label: "Tải bundle",
              title: "Sáu checkpoint tốt nhất của N24News",
              note: "Full finetune, deep head và LoRA cho CLIP và VisualBERT.",
              url: "https://drive.google.com/file/d/1sZBUPxE-LtUDARN0PRzPZI7yi4ARUm22/view?usp=sharing",
            },
            {
              type: "Bài toán phân loại ảnh",
              label: "Tải bundle",
              title: "ResNet50 và ViT-Base",
              note: "Hai best checkpoint của bài toán ảnh.",
              url: "https://drive.google.com/file/d/1wkPuWUMKkm0K2N5l00Kk4Jij-xJKF7u8/view?usp=sharing",
            },
          ],
        },
      ],
      sections: [
        {
          title: "Dữ liệu",
          copy:
            "Ba bài toán dùng ba bộ dữ liệu khác nhau: Jigsaw Toxic Comment, N24News và weather dataset.",
        },
        {
          title: "Mô hình",
          copy:
            "Bài toán văn bản dùng BERT và LSTM. Bài toán đa phương thức dùng CLIP, VisualBERT, deep head và LoRA. Bài toán ảnh dùng ResNet50 và ViT-Base.",
        },
        {
          title: "Kết quả tốt nhất",
          copy:
            "BERT là mô hình tốt nhất của bài toán văn bản, VisualBERT là mô hình tốt nhất của bài toán đa phương thức, còn ViT-Base là mô hình tốt nhất của bài toán ảnh.",
        },
        {
          title: "Tái sử dụng",
          copy:
            "Cả ba bài toán đều có bundle checkpoint để tải về và chạy lại mô hình.",
        },
      ],
    },
    {
      id: "assignment-2",
      numberLabel: "Bài tập lớn số 2",
      title: "Trang đang được chuẩn bị",
      page: "assignments/assignment-2.html",
      cardSummary: "Nội dung sẽ được bổ sung khi phần thực nghiệm hoàn tất.",
      overview: "Trang dành cho Bài tập lớn số 2 hiện chưa có nội dung công khai.",
      summary: "Nhóm sẽ bổ sung báo cáo và tài nguyên khi có kết quả chính thức.",
      resources: [],
      sections: [],
    },
    {
      id: "assignment-3",
      numberLabel: "Bài tập lớn số 3",
      title: "Trang đang được chuẩn bị",
      page: "assignments/assignment-3.html",
      cardSummary: "Nội dung sẽ được bổ sung khi phần thực nghiệm hoàn tất.",
      overview: "Trang dành cho Bài tập lớn số 3 hiện chưa có nội dung công khai.",
      summary: "Nhóm sẽ bổ sung báo cáo và tài nguyên khi có kết quả chính thức.",
      resources: [],
      sections: [],
    },
  ],
};
