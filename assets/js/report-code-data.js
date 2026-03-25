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
          "Tinh chỉnh BERT với tokenizer và dataloader gọn hơn, phù hợp cho bài toán multi-label.",
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
          "Baseline tuần tự dùng vocabulary tự xây và BiLSTM. Các hàm tokenization, vocab và collate được tách rõ ràng.",
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
        tabLabel: "Cấu hình N24News",
        source: "text_image_classification.ipynb · Ô mã cấu hình dữ liệu",
        language: "python",
        summary:
          "Thiết lập thư mục source/raw/processed, link Google Drive và các siêu tham số chính cho CLIP và VisualBERT.",
        badges: ["Ô notebook", "Python", "N24News", "Cấu hình"],
        code: `N24_DIR = BTL1_ROOT / "data" / "multimodal" / "n24news"
SOURCE_DIR = N24_DIR / "source"
RAW_DIR = N24_DIR / "raw"
PROCESSED_DIR = N24_DIR / "processed"
ARTIFACT_DIR = BTL1_ROOT / "artifacts" / "multimodal"

N24NEWS_DRIVE_URL = "https://drive.google.com/file/d/1OS1fXwZ1Vsj70lEQajccyssxQRYp5X9D/view?usp=share_link"
SEED = 42
DOWNLOAD_IF_MISSING = True

CLIP_CFG = {
    "backbone": "openai/clip-vit-base-patch32",
    "batch_size": 8,
    "epochs": 2,
    "lr": 1e-5,
    "dropout": 0.2,
    "max_text_length": 77,
}

VB_CFG = {
    "visualbert_backbone": "uclanlp/visualbert-vqa-coco-pre",
    "vision_backbone": "openai/clip-vit-base-patch32",
    "tokenizer_name": "bert-base-uncased",
    "batch_size": 4,
    "epochs": 2,
    "lr": 1e-5,
    "dropout": 0.2,
    "max_text_length": 256,
    "max_visual_tokens": 32,
}`,
      },
      {
        name: "text_image_classification.ipynb",
        tabLabel: "Chuẩn bị split",
        source: "text_image_classification.ipynb · Ô mã build train/val/test",
        language: "python",
        summary:
          "Notebook ưu tiên dùng split official của N24News, ghép ảnh với văn bản bài báo và lưu ra train/val/test đã xử lý.",
        badges: ["Ô notebook", "Python", "Split", "Preprocessing"],
        code: `official_split_files = {}
for root in candidate_data_roots():
    for internal_name, official_name in official_split_map.items():
        split_file = next(root.rglob(f"nytimes_{official_name}.json"), None)
        if split_file is not None:
            official_split_files[internal_name] = split_file

if len(official_split_files) == 3:
    train_records = load_manifest_records(official_split_files["train"])
    val_records = load_manifest_records(official_split_files["val"])
    test_records = load_manifest_records(official_split_files["test"])

    train_df = pd.DataFrame(build_rows_from_records(train_records, image_index, "train"))
    val_df = pd.DataFrame(build_rows_from_records(val_records, image_index, "val"))
    test_df = pd.DataFrame(build_rows_from_records(test_records, image_index, "test"))

label_names = combined_df["category"].value_counts().index.tolist()
label_to_id = {label: idx for idx, label in enumerate(label_names)}`,
      },
      {
        name: "text_image_classification.ipynb",
        tabLabel: "Định nghĩa mô hình",
        source: "text_image_classification.ipynb · Ô mã CLIP và VisualBERT",
        language: "python",
        summary:
          "Hai backbone pretrained được bọc bằng classification head để giải cùng một bài toán news category classification.",
        badges: ["Ô notebook", "Python", "CLIP", "VisualBERT"],
        code: `class CLIPClassifier(nn.Module):
    def __init__(self, backbone: str, num_labels: int, dropout: float):
        super().__init__()
        self.backbone = CLIPModel.from_pretrained(backbone)
        hidden_size = self.backbone.config.projection_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

class VisualBERTClassifier(nn.Module):
    def __init__(self, visualbert_backbone: str, vision_backbone: str, num_labels: int, dropout: float, max_visual_tokens: int):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_backbone)
        config = VisualBertConfig.from_pretrained(visualbert_backbone)
        config.visual_embedding_dim = self.vision_encoder.config.hidden_size
        self.backbone = VisualBertModel.from_pretrained(
            visualbert_backbone,
            config=config,
            ignore_mismatched_sizes=True,
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)`,
      },
      {
        name: "text_image_classification.ipynb",
        tabLabel: "Huấn luyện và eval",
        source: "text_image_classification.ipynb · Ô mã train/eval",
        language: "python",
        summary:
          "Cell này gom logic huấn luyện, early stopping, lưu checkpoint và đánh giá cuối trên tập test.",
        badges: ["Ô notebook", "Python", "Train", "Evaluation"],
        code: `def train_model(model_name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: dict) -> dict:
    checkpoint_path = ARTIFACT_DIR / f"n24news_{model_name.lower()}_best.pt"
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_macro_f1 = -1.0
    patience_left = cfg["patience"]

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        ...
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            patience_left = cfg["patience"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_names": label_names,
                    "config": cfg,
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break`,
      },
    ],
  },
};
