window.SITE_DATA = {
  course: {
    code: "CO3133",
    name: "Học sâu và ứng dụng",
    university: "Đại học Bách khoa",
    faculty: "Khoa Khoa học và Kỹ thuật Máy tính",
    instructor: "Lê Thành Sách",
    term: "Năm học 2025-2026, Học kỳ 2",
    summary:
      "Landing page này giới thiệu các bài tập lớn của nhóm trong môn CO3133, đồng thời đóng vai trò là điểm truy cập tập trung tới báo cáo, bảng kết quả, notebook và mã nguồn công khai.",
    overview: {
      title: "Một cổng truy cập thống nhất",
      copy:
        "Mỗi trang bài tập lớn được tổ chức như một trang giới thiệu ngắn gọn, ưu tiên khả năng xem nhanh, điều hướng rõ ràng và trình bày phù hợp với bối cảnh học thuật.",
      points: [
        "Trang chủ giới thiệu nhóm và tập hợp toàn bộ assignment trong một cấu trúc nhất quán.",
        "Mỗi assignment có trang riêng để gắn báo cáo, notebook, bảng kết quả và các tài liệu công khai.",
        "Các kết quả công khai chỉ phản ánh những phần vẫn còn hợp lệ trong nhánh chính của repo.",
      ],
    },
  },
  group: {
    name: "DL123",
    summary:
      "Nhóm DL123 thực hiện Bài tập lớn số 1 của môn CO3133 với hai nhánh chính: phân loại văn bản và phân loại đa phương thức.",
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
      title: "Phân loại văn bản và đa phương thức",
      page: "assignments/assignment-1.html",
      cardSummary:
        "Trang công khai cho Bài tập lớn 1. Nhánh văn bản đã hoàn tất với BERT vượt LSTM, còn nhánh đa phương thức dùng N24News và cho thấy VisualBERT vượt CLIP trên tập test 24 lớp.",
      overview:
        "Bài tập lớn số 1 tập trung vào các bài toán phân loại trên dữ liệu văn bản và đa phương thức. Nhánh văn bản dùng Jigsaw Toxic Comment, còn nhánh đa phương thức dùng N24News cho bài toán image + news text classification.",
      summary:
        "Trang này đóng vai trò là điểm truy cập công khai tới báo cáo, notebook, bảng kết quả tổng hợp và các tài liệu minh họa cho quá trình thực nghiệm của Bài tập lớn số 1.",
      resources: [],
      resourceGroups: [
        {
          title: "Bắt đầu từ đây",
          copy:
            "Dành cho người xem lần đầu. Nhóm tài nguyên này đi thẳng vào phần kết quả và diễn giải tổng quan, không buộc người xem phải mở notebook hay tệp phụ trợ ngay từ đầu.",
          featured: true,
          items: [
            {
              type: "Tổng quan",
              label: "Mở trang kết quả tổng hợp",
              title: "Bảng kết quả tổng hợp",
              note: "Trang tổng hợp ngắn gọn hai nhánh thực nghiệm, gồm bảng metric cuối và biểu đồ so sánh trực quan.",
              url: "../btl1/reports/final-results.html",
              featured: true,
            },
            {
              type: "Mã nguồn",
              label: "Mở repo GitHub",
              title: "Kho mã nguồn công khai",
              note: "Repository GitHub của nhóm, chứa landing page, notebook, báo cáo web và toàn bộ mã nguồn công khai của Bài tập lớn số 1.",
              url: "https://github.com/dung-h/DeepLearning",
            },
          ],
        },
        {
          title: "Báo cáo theo từng nhánh",
          copy:
            "Dành cho giảng viên hoặc peers muốn xem sâu hơn từng phần thực nghiệm mà vẫn ở định dạng web công khai, có số liệu, hình minh họa và phần đọc mã nguồn trực tiếp.",
          items: [
            {
              type: "Văn bản",
              label: "Mở báo cáo văn bản",
              title: "Trang trình bày nhánh văn bản",
              note: "Bản trình bày cho Jigsaw Toxic Comment với số liệu chính, learning curves, per-label F1 và phần đọc mã nguồn trực tiếp trên web.",
              url: "../btl1/reports/text-report.html",
            },
            {
              type: "Đa phương thức",
              label: "Mở báo cáo đa phương thức",
              title: "Trang trình bày nhánh đa phương thức",
              note: "Bản trình bày cho N24News với mô tả dataset, bảng kết quả, chart tương tác, figure tham chiếu và mẫu preview.",
              url: "../btl1/reports/multimodal-report.html",
            },
          ],
        },
        {
          title: "Phụ lục kỹ thuật",
          copy:
            "Dành cho người cần đối chiếu notebook gốc hoặc tái sử dụng phần tóm tắt cho slide. Các tài nguyên này được đặt ở lớp phụ để không lấn át phần trình bày công khai chính.",
          items: [
            {
              type: "Trình xem",
              label: "Mở notebook văn bản",
              title: "Trình xem notebook nhánh văn bản",
              note: "Trình xem notebook theo giao diện trình soạn thảo chỉ đọc, phù hợp cho người muốn theo dõi mã nguồn trực tiếp trên web.",
              url: "../btl1/reports/text-notebook-viewer.html",
            },
            {
              type: "Trình xem",
              label: "Mở notebook đa phương thức",
              title: "Trình xem notebook nhánh đa phương thức",
              note: "Trình xem notebook cho pipeline N24News với các ô mã chính về dữ liệu, mô hình và huấn luyện.",
              url: "../btl1/reports/multimodal-notebook-viewer.html",
            },
            {
              type: "Tóm tắt",
              label: "Mở bản tóm tắt cho slide",
              title: "Bản tóm tắt Markdown",
              note: "Bản tóm tắt ngắn gọn để tái sử dụng trong slide hoặc phần tổng kết của báo cáo.",
              url: "../btl1/reports/final-results-summary.md",
            },
          ],
        },
      ],
      sections: [
        {
          title: "Thiết lập thực nghiệm",
          copy:
            "Nhánh văn bản dùng Jigsaw Toxic Comment Classification Challenge với bài toán multi-label 6 nhãn. Nhánh đa phương thức dùng N24News cho bài toán image + news text → category classification với 24 lớp.",
        },
        {
          title: "Kết quả nhánh văn bản",
          copy:
            "BERT đạt exact-match accuracy 0.9306, micro F1 0.8030 và macro F1 0.6765. LSTM đạt exact-match accuracy 0.9224, micro F1 0.7386 và macro F1 0.5489. BERT là mô hình tốt hơn trên cả ba chỉ số chính.",
        },
        {
          title: "Kết quả nhánh đa phương thức",
          copy:
            "VisualBERT đạt accuracy 0.8751, macro F1 0.8575 và weighted F1 0.8732 trên tập test N24News. CLIP đạt accuracy 0.8488, macro F1 0.8064 và weighted F1 0.8346. VisualBERT là mô hình tốt nhất của nhánh đa phương thức hiện tại.",
        },
        {
          title: "Trình bày kết quả",
          copy:
            "Các kết quả công khai được tổ chức thành các trang trình bày riêng cho từng nhánh, có hình minh họa, bảng số liệu, chart tương tác và phần đọc mã nguồn trực tiếp trên web thay vì chỉ đưa liên kết tới tệp thô.",
        },
        {
          title: "Nhận xét tổng hợp",
          copy:
            "Trong trạng thái hiện tại, BERT là mô hình tốt nhất cho nhánh văn bản và VisualBERT là mô hình tốt nhất cho nhánh đa phương thức. Cả hai nhánh đều đã có artifact thật, report web và demo local để phục vụ báo cáo.",
        },
      ],
    },
    {
      id: "assignment-2",
      numberLabel: "Bài tập lớn số 2",
      title: "Trang đang được chuẩn bị",
      page: "assignments/assignment-2.html",
      cardSummary:
        "Trang dành cho Bài tập lớn số 2 đã được chuẩn bị sẵn trong cùng cấu trúc trình bày để bảo đảm tính nhất quán trên toàn landing page.",
      overview:
        "Trang assignment này sẽ được bổ sung nội dung công khai ngay khi nhóm hoàn thiện phần giới thiệu và các tài nguyên cần công bố.",
      summary:
        "Cấu trúc trang được giữ tối giản để tập trung vào khả năng truy cập nhanh tới báo cáo, demo và mã nguồn khi có dữ liệu chính thức.",
      resources: [],
      sections: [],
    },
    {
      id: "assignment-3",
      numberLabel: "Bài tập lớn số 3",
      title: "Trang đang được chuẩn bị",
      page: "assignments/assignment-3.html",
      cardSummary:
        "Trang dành cho Bài tập lớn số 3 được bố trí cùng ngôn ngữ thiết kế để về sau có thể cập nhật liền mạch với các assignment còn lại.",
      overview:
        "Trang này được giữ ở trạng thái công khai, tối giản và sẽ được mở rộng bằng các tài nguyên phù hợp khi assignment có nội dung chính thức.",
      summary:
        "Mọi thành phần nội bộ đã được loại bỏ để trang chỉ giữ vai trò là trang giới thiệu công khai, gọn gàng và chuyên nghiệp.",
      resources: [],
      sections: [],
    },
  ],
};
