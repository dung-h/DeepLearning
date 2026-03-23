window.REPORT_CODE_DATA = {
  text: {
    title: "text_classification.ipynb",
    footer: "Trích các ô mã chính từ notebook text_classification.ipynb",
    files: [
      {
        name: "text_classification.ipynb",
        tabLabel: "Cấu hình dữ liệu",
        source: "text_classification.ipynb · Ô mã cấu hình dữ liệu",
        language: "python",
        summary:
          "Xác định thư mục BTL1, cấu hình đường dẫn cho Jigsaw và tách riêng phần dữ liệu khỏi phần tham số huấn luyện.",
        badges: ["Ô notebook", "Python", "Dữ liệu", "Đường dẫn"],
        code: `def find_btl1_root() -> Path:
    # Resolve the BTL1 folder from the current working directory.
    for base in [Path.cwd(), *Path.cwd().parents]:
        if base.name == "btl1" and (base / "data").exists() and (base / "artifacts").exists():
            return base
        candidate = base / "btl1"
        if candidate.exists() and (candidate / "data").exists() and (candidate / "artifacts").exists():
            return candidate
    raise FileNotFoundError("Could not locate the btl1 directory.")

BTL1_ROOT = find_btl1_root()
TEXT_DIR = BTL1_ROOT / "data" / "text" / "jigsaw"
SOURCE_DIR = TEXT_DIR / "source"
RAW_DIR = TEXT_DIR / "raw"
PROCESSED_DIR = TEXT_DIR / "processed"
ARTIFACT_DIR = BTL1_ROOT / "artifacts" / "text"

ZIP_PATH = SOURCE_DIR / "jigsaw-toxic-comment-classification-challenge.zip"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]`,
      },
      {
        name: "text_classification.ipynb",
        tabLabel: "Tinh chỉnh BERT",
        source: "text_classification.ipynb · Ô mã tinh chỉnh BERT",
        language: "python",
        summary:
          "Tinh chỉnh BERT với tên biến ngắn gọn hơn, giữ tokenizer và dataloader ở mức dễ đọc cho bài toán multi-label.",
        badges: ["Ô notebook", "Python", "BERT", "Huấn luyện"],
        code: `tokenizer = AutoTokenizer.from_pretrained(BERT_CFG["model_name"])

def collate_batch(batch):
    texts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=BERT_CFG["max_length"],
        return_tensors="pt",
    )
    encoded["labels"] = labels
    return encoded

model = AutoModelForSequenceClassification.from_pretrained(
    BERT_CFG["model_name"],
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
).to(DEVICE)`,
      },
      {
        name: "text_classification.ipynb",
        tabLabel: "Baseline LSTM",
        source: "text_classification.ipynb · Ô mã baseline LSTM",
        language: "python",
        summary:
          "Baseline tuần tự dùng vocabulary tự xây và BiLSTM. Các hàm tokenization, vocab và collate được tách rõ ràng hơn.",
        badges: ["Ô notebook", "Python", "LSTM", "Baseline"],
        code: `token_pattern = re.compile(r"[a-z']+")

def tokenize(text: str) -> list[str]:
    return token_pattern.findall(text.lower())

def build_vocab(texts, max_size=50000, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)`,
      },
    ],
  },
  multimodal: {
    title: "text_image_classification.ipynb",
    footer: "Trích các ô mã chính từ notebook text_image_classification.ipynb",
    files: [
      {
        name: "text_image_classification.ipynb",
        tabLabel: "Cấu hình dữ liệu",
        source: "text_image_classification.ipynb · Ô mã cấu hình CrisisMMD",
        language: "python",
        summary:
          "Tách riêng cấu hình dữ liệu, giải thích rõ agreed-label split và giữ phần tải, giải nén độc lập với phần mô hình.",
        badges: ["Ô notebook", "Python", "CrisisMMD", "Dữ liệu"],
        code: `MM_DIR = BTL1_ROOT / "data" / "multimodal" / "crisismmd"
EXT_DIR = MM_DIR / "external"
RAW_DIR = MM_DIR / "raw"
PROCESSED_DIR = MM_DIR / "processed"
ARTIFACT_DIR = BTL1_ROOT / "artifacts" / "multimodal"

LABELS = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian",
]
label2id = {label: idx for idx, label in enumerate(LABELS)}

SPLIT_FILES = {
    "train": find_one(AGREED_DIR, "task_humanitarian_text_img_agreed_lab_train.tsv"),
    "dev": find_one(AGREED_DIR, "task_humanitarian_text_img_agreed_lab_dev.tsv"),
    "test": find_one(AGREED_DIR, "task_humanitarian_text_img_agreed_lab_test.tsv"),
}`,
      },
      {
        name: "text_image_classification.ipynb",
        tabLabel: "Huấn luyện CLIP",
        source: "text_image_classification.ipynb · Ô mã huấn luyện CLIP",
        language: "python",
        summary:
          "CLIP dùng tên biến ngắn gọn hơn như processor, model, train_loader, val_loader và giữ luồng huấn luyện dễ đọc hơn.",
        badges: ["Ô notebook", "Python", "CLIP", "Huấn luyện"],
        code: `processor = AutoProcessor.from_pretrained(CLIP_CFG["model_name"])

def collate_batch(batch):
    images, texts, labels = zip(*batch)
    encoded = processor(
        text=list(texts),
        images=list(images),
        padding=True,
        truncation=True,
        max_length=CLIP_CFG["max_length"],
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    return encoded

class ClipClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.backbone = CLIPModel.from_pretrained(model_name)
        dim = self.backbone.config.projection_dim
        self.head = nn.Linear(dim * 2, num_classes)`,
      },
      {
        name: "text_image_classification.ipynb",
        tabLabel: "Huấn luyện VisualBERT",
        source: "text_image_classification.ipynb · Ô mã huấn luyện VisualBERT",
        language: "python",
        summary:
          "VisualBERT dùng tokenizer và processor tách riêng, kết hợp CLIP vision encoder để sinh visual embedding cho backbone.",
        badges: ["Ô notebook", "Python", "VisualBERT", "Bộ mã hóa ảnh"],
        code: `processor = AutoProcessor.from_pretrained(VB_CFG["vision_name"])
tokenizer = AutoTokenizer.from_pretrained(VB_CFG["model_name"])

def collate_batch(batch):
    images, texts, labels = zip(*batch)
    image_inputs = processor(images=list(images), return_tensors="pt")
    text_inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=VB_CFG["max_length"],
        return_tensors="pt",
    )
    text_inputs["pixel_values"] = image_inputs["pixel_values"]
    text_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    return text_inputs

class VisualBertClassifier(nn.Module):
    def __init__(self, model_name: str, vision_name: str, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(vision_name)
        self.backbone = VisualBertModel.from_pretrained(model_name)`,
      },
      {
        name: "text_image_classification.ipynb",
        tabLabel: "Đánh giá",
        source: "text_image_classification.ipynb · Ô mã đánh giá và xuất kết quả",
        language: "python",
        summary:
          "Xuất bảng so sánh, per-label F1 và confusion matrix theo đúng tên artifact mà report web đang sử dụng.",
        badges: ["Ô notebook", "Python", "Đánh giá", "Kết quả"],
        code: `comparison_df = pd.DataFrame([
    {"model": "CLIP", "accuracy": clip_metrics["accuracy"], "macro_f1": clip_metrics["macro_f1"]},
    {"model": "VisualBERT", "accuracy": visualbert_metrics["accuracy"], "macro_f1": visualbert_metrics["macro_f1"]},
])
comparison_df.to_csv(ARTIFACT_DIR / "crisismmd_model_comparison.csv", index=False)

clip_cm = confusion_matrix(clip_y_true, clip_y_pred, labels=list(range(len(LABELS))))
visualbert_cm = confusion_matrix(
    visualbert_y_true,
    visualbert_y_pred,
    labels=list(range(len(LABELS))),
)
(ARTIFACT_DIR / "crisismmd_confusion_matrices.json").write_text(
    json.dumps({"labels": LABELS, "clip": clip_cm.tolist(), "visualbert": visualbert_cm.tolist()}, indent=2),
    encoding="utf-8",
)`,
      },
    ],
  },
};
