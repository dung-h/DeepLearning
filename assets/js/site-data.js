window.SITE_DATA = {
  course: {
    code: "CO3133",
    name: "Học sâu và ứng dụng",
    university: "Đại học Bách khoa",
    faculty: "Khoa Khoa học và Kỹ thuật Máy tính",
    instructor: "Lê Thành Sách",
    term: "Năm học 2025-2026, Học kỳ 2",
    summary:
      "Trang công khai của nhóm DL123 cho Bài tập lớn số 1, gồm ba nhánh thực nghiệm: văn bản, đa phương thức và ảnh.",
    overview: {
      title: "Bài tập lớn số 1",
      copy:
        "Mỗi nhánh đều có notebook, artifact kết quả và điểm truy cập công khai để giảng viên hoặc bạn học đối chiếu nhanh.",
      points: [
        "Nhánh văn bản dùng Jigsaw Toxic Comment để so sánh BERT và LSTM trên bài toán multi-label 6 nhãn.",
        "Nhánh đa phương thức dùng N24News để so sánh CLIP, VisualBERT và các biến thể deep_head, lora.",
        "Nhánh ảnh dùng weather dataset để so sánh ResNet50 và ViT-Base, kèm bundle checkpoint để chạy lại model.",
      ],
    },
  },
  group: {
    name: "DL123",
    summary:
      "Nhóm thực hiện Bài tập lớn số 1 theo hướng notebook-first: số liệu công khai đều bám theo notebook và artifact đang có trong repo.",
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
      title: "Phân loại văn bản, đa phương thức và ảnh",
      page: "assignments/assignment-1.html",
      cardSummary:
        "Ba nhánh thực nghiệm của nhóm: Jigsaw Toxic Comment, N24News và weather dataset.",
      overview:
        "Bài tập lớn số 1 tổng hợp ba bài toán phân loại khác nhau để trình bày dữ liệu, mô hình, kết quả và khả năng tái sử dụng checkpoint.",
      summary:
        "Trang này gom báo cáo, notebook, bảng kết quả và bundle checkpoint cho toàn bộ Bài tập lớn số 1.",
      resources: [],
      resourceGroups: [
        {
          title: "Lối vào chính",
          copy:
            "Các điểm truy cập phù hợp nhất để xem nhanh bài làm trước khi đi vào từng nhánh.",
          featured: true,
          items: [
            {
              type: "Tổng quan",
              label: "Xem bảng kết quả",
              title: "Bảng kết quả cuối",
              note: "Trang tổng hợp ba nhánh với bảng metric cuối, biểu đồ so sánh và liên kết sang từng báo cáo chi tiết.",
              url: "../btl1/reports/final-results.html",
              featured: true,
            },
            {
              type: "Mã nguồn",
              label: "Mở GitHub",
              title: "Repository của nhóm",
              note: "Kho mã nguồn chứa landing page, notebook, artifact, report web và các script phục vụ Bài tập lớn số 1.",
              url: "https://github.com/dung-h/DeepLearning",
            },
          ],
        },
        {
          title: "Báo cáo theo từng nhánh",
          copy:
            "Ba nhánh hiện đã có điểm truy cập công khai rõ ràng để theo dõi riêng phần dữ liệu, mô hình và kết quả.",
          items: [
            {
              type: "Văn bản",
              label: "Mở report văn bản",
              title: "Jigsaw Toxic Comment",
              note: "Trang trình bày nhánh văn bản với metric test, learning curves, error analysis và liên kết notebook.",
              url: "../btl1/reports/text-report.html",
            },
            {
              type: "Đa phương thức",
              label: "Mở report đa phương thức",
              title: "N24News",
              note: "Trang trình bày nhánh đa phương thức với CLIP, VisualBERT, các chart tương tác và phần PEFT comparison.",
              url: "../btl1/reports/multimodal-report.html",
            },
            {
              type: "Ảnh",
              label: "Mở report ảnh",
              title: "Weather Dataset",
              note: "Trang trình bày nhánh ảnh với EDA, split 70/15/15, augmentation, so sánh ResNet50 và ViT-Base.",
              url: "../btl1/reports/image-report.html",
            },
          ],
        },
        {
          title: "Notebook và artifact",
          copy:
            "Các notebook gốc và file tóm tắt chính để đối chiếu lại số liệu hoặc xem pipeline đầy đủ.",
          items: [
            {
              type: "Notebook",
              label: "Notebook văn bản",
              title: "Notebook Jigsaw",
              note: "Notebook huấn luyện và đánh giá cho BERT và LSTM trên dữ liệu toxic comment đa nhãn.",
              url: "../btl1/notebooks/text_classification.ipynb",
            },
            {
              type: "Notebook",
              label: "Notebook đa phương thức",
              title: "Notebook N24News",
              note: "Notebook đầy đủ cho EDA, split, CLIP, VisualBERT, PEFT variants và các artifact của nhánh đa phương thức.",
              url: "../btl1/notebooks/text_image_classification.ipynb",
            },
            {
              type: "Notebook",
              label: "Notebook ảnh",
              title: "Notebook weather dataset",
              note: "Notebook cho EDA, augmentation, training và evaluation của ResNet50 và ViT-Base.",
              url: "../btl1/notebooks/image_classification.ipynb",
            },
            {
              type: "Artifact",
              label: "Mở summary ảnh",
              title: "Training summary của nhánh ảnh",
              note: "File JSON tổng hợp accuracy, weighted F1, best epoch và thời gian train của hai mô hình ảnh.",
              url: "../btl1/artifacts/image/training_summary.json",
            },
          ],
        },
        {
          title: "Biến thể mô hình đã huấn luyện",
          copy:
            "Ngoài model tốt nhất, trang này cũng chỉ ra những biến thể đã được huấn luyện để phục vụ đối chiếu trong bối cảnh môn học.",
          items: [
            {
              type: "Văn bản",
              label: "Xem bảng so sánh",
              title: "BERT và LSTM",
              note: "Nhánh văn bản giữ cả hai mốc so sánh chính: BERT là model tốt nhất, LSTM là baseline tuần tự.",
              url: "../btl1/artifacts/text/text_model_comparison.csv",
            },
            {
              type: "Đa phương thức",
              label: "Xem full finetune",
              title: "CLIP và VisualBERT (full finetune)",
              note: "Bảng so sánh official cho hai model chính trên N24News sau khi fine-tune đầy đủ backbone.",
              url: "../btl1/artifacts/multimodal/n24news_model_comparison_full_finetune.csv",
            },
            {
              type: "Đa phương thức",
              label: "Xem deep_head và lora",
              title: "PEFT variants trên N24News",
              note: "Tổng hợp kết quả deep_head và lora cho cả CLIP lẫn VisualBERT, dùng để đối chiếu với full finetune.",
              url: "../btl1/artifacts/multimodal/n24news_model_comparison_all_variants.csv",
            },
            {
              type: "Ảnh",
              label: "Xem training summary",
              title: "ResNet50 và ViT-Base",
              note: "Bảng tóm tắt nhánh ảnh cho thấy ViT-Base vượt ResNet50 trên test set và cần ít diễn giải hơn khi đối chiếu.",
              url: "../btl1/artifacts/image/training_summary.json",
            },
          ],
        },
        {
          title: "Bundle checkpoint",
          copy:
            "Các gói checkpoint tốt nhất để chạy lại model mà không cần huấn luyện từ đầu. Mỗi bundle chỉ chứa best checkpoint đã chốt.",
          items: [
            {
              type: "Văn bản",
              label: "Tải bundle văn bản",
              title: "BERT + LSTM best checkpoints",
              note: "Bundle checkpoint của nhánh văn bản gồm hai file best cho BERT và LSTM.",
              url: "https://drive.google.com/file/d/1PhIMgu-1unj7Yt0dMTGkX2H9473lJycj/view?usp=sharing",
            },
            {
              type: "Đa phương thức",
              label: "Tải bundle đa phương thức",
              title: "6 best checkpoints của N24News",
              note: "Bundle checkpoint của nhánh đa phương thức gồm CLIP và VisualBERT cho full finetune, deep_head và lora.",
              url: "https://drive.google.com/file/d/1sZBUPxE-LtUDARN0PRzPZI7yi4ARUm22/view?usp=sharing",
            },
            {
              type: "Ảnh",
              label: "Tải bundle ảnh",
              title: "ResNet50 + ViT-Base best checkpoints",
              note: "Bundle checkpoint của nhánh ảnh gồm hai mô hình đã chốt trong training summary.",
              url: "https://drive.google.com/file/d/1wkPuWUMKkm0K2N5l00Kk4Jij-xJKF7u8/view?usp=sharing",
            },
          ],
        },
      ],
      sections: [
        {
          title: "Dữ liệu",
          copy:
            "Ba nhánh dùng ba bộ dữ liệu khác nhau: Jigsaw Toxic Comment cho văn bản đa nhãn, N24News cho bài toán ảnh + văn bản tin tức, và weather dataset cho phân loại ảnh thời tiết.",
        },
        {
          title: "Mô hình đã thử",
          copy:
            "Nhánh văn bản so sánh BERT với LSTM. Nhánh đa phương thức có CLIP, VisualBERT và thêm các biến thể deep_head, lora. Nhánh ảnh so sánh ResNet50 với ViT-Base.",
        },
        {
          title: "Kết quả tốt nhất",
          copy:
            "BERT hiện là model tốt nhất của nhánh văn bản, VisualBERT là model tốt nhất của nhánh đa phương thức, còn ViT-Base là model tốt nhất của nhánh ảnh.",
        },
        {
          title: "Khả năng tái sử dụng",
          copy:
            "Cả ba nhánh đều đã có bundle checkpoint để tải về và chạy lại model; nhánh đa phương thức còn kèm inventory cho các biến thể đã huấn luyện.",
        },
      ],
    },
    {
      id: "assignment-2",
      numberLabel: "Bài tập lớn số 2",
      title: "Trang đang được chuẩn bị",
      page: "assignments/assignment-2.html",
      cardSummary:
        "Nội dung cho Bài tập lớn số 2 sẽ được bổ sung khi phần thực nghiệm hoàn tất.",
      overview:
        "Trang dành cho Bài tập lớn số 2 hiện chưa có nội dung công khai.",
      summary:
        "Nhóm sẽ bổ sung báo cáo và tài nguyên khi có kết quả chính thức.",
      resources: [],
      sections: [],
    },
    {
      id: "assignment-3",
      numberLabel: "Bài tập lớn số 3",
      title: "Trang đang được chuẩn bị",
      page: "assignments/assignment-3.html",
      cardSummary:
        "Nội dung cho Bài tập lớn số 3 sẽ được bổ sung khi phần thực nghiệm hoàn tất.",
      overview:
        "Trang dành cho Bài tập lớn số 3 hiện chưa có nội dung công khai.",
      summary:
        "Nhóm sẽ bổ sung báo cáo và tài nguyên khi có kết quả chính thức.",
      resources: [],
      sections: [],
    },
  ],
};
