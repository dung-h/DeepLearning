from __future__ import annotations

import json
import os
import re
import warnings
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPVisionModel,
    VisualBertConfig,
    VisualBertModel,
)
from transformers.utils import logging as hf_logging

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning,
)
hf_logging.set_verbosity_error()


TEXT_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

TEXT_LABEL_TITLES = {
    "toxic": "Độc hại",
    "severe_toxic": "Độc hại nghiêm trọng",
    "obscene": "Tục tĩu",
    "threat": "Đe dọa",
    "insult": "Xúc phạm",
    "identity_hate": "Thù ghét danh tính",
}

TEXT_LABEL_DESCRIPTIONS = {
    "toxic": "Bình luận độc hại hoặc công kích nói chung.",
    "severe_toxic": "Ngôn ngữ độc hại ở mức nghiêm trọng hơn.",
    "obscene": "Câu chữ tục tĩu hoặc phản cảm.",
    "threat": "Nội dung mang tính đe dọa hoặc bạo lực.",
    "insult": "Lời lẽ xúc phạm trực tiếp cá nhân hoặc nhóm.",
    "identity_hate": "Ngôn từ thù ghét nhắm vào đặc điểm danh tính.",
}

TEXT_EXAMPLES = [
    ["You are such a disgusting person and nobody wants to hear your nonsense.", "Một mô hình", "BERT"],
    ["I will find you and make you pay for this.", "So sánh hai mô hình", "BERT"],
    ["This is the dumbest thing I have ever read.", "Một mô hình", "LSTM"],
]

token_pattern = re.compile(r"[a-z']+")

DEMO_CSS = """
.demo-shell {max-width: 1220px; margin: 0 auto;}
.demo-hero {
  padding: 24px 26px;
  border-radius: 26px;
  border: 1px solid rgba(19, 54, 95, 0.14);
  background:
    radial-gradient(circle at top right, rgba(208,175,103,0.16), transparent 28%),
    linear-gradient(135deg, rgba(12,37,66,0.98), rgba(20,54,95,0.94));
  color: #fffdfa;
  margin-bottom: 18px;
  box-shadow: 0 24px 48px rgba(19,54,95,0.16);
}
.demo-hero h1 {margin: 0 0 10px; color: #fffdfa; letter-spacing: -0.03em;}
.demo-hero p, .demo-hero strong, .demo-hero code, .demo-hero a {color: #fffdfa !important;}
.demo-hero code {
  background: rgba(255,253,250,0.14);
  border: 1px solid rgba(255,253,250,0.16);
  padding: 2px 6px;
  border-radius: 8px;
}
.demo-chip-row {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px;}
.demo-chip {
  display: inline-flex; align-items: center; min-height: 30px;
  padding: 0 12px; border-radius: 999px; background: rgba(255,253,250,0.12);
  color: #fffdfa; font-size: 13px; font-weight: 700;
}
.demo-card, .result-card {
  border-radius: 22px;
  border: 1px solid rgba(19, 54, 95, 0.12);
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(249,245,237,0.92));
  padding: 20px;
}
.section-eyebrow {
  display: inline-flex;
  margin-bottom: 8px;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #866526;
  font-weight: 800;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}
.metric-tile {
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid rgba(19,54,95,0.10);
  background: rgba(255,255,255,0.82);
}
.metric-value {font-size: 1.5rem; font-weight: 800; color: #14365f;}
.metric-label {margin-top: 6px; font-size: 12px; color: #6b7788; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 800;}
.compare-card {
  border-radius: 20px;
  border: 1px solid rgba(19,54,95,0.12);
  background: linear-gradient(180deg, rgba(245,248,252,0.96), rgba(255,255,255,0.96));
  padding: 18px;
}
.compare-row {display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px;}
.compare-model {
  padding: 16px;
  border-radius: 18px;
  border: 1px solid rgba(19,54,95,0.10);
  background: rgba(255,255,255,0.84);
}
.compare-model h4 {margin: 0 0 8px; color: #14365f;}
.winner-chip {
  display:inline-flex; align-items:center; min-height:34px; padding:0 12px; border-radius:999px;
  background: rgba(182,146,71,0.18); color:#7a5a14; font-size:12px; font-weight:800;
}
.demo-shell .tab-nav {
  background: rgba(245,247,250,0.96);
  border: 1px solid rgba(19,54,95,0.10);
  border-radius: 999px;
  padding: 6px;
  gap: 8px;
}
.demo-shell .tab-nav button {
  border-radius: 999px !important;
  border: 0 !important;
  min-height: 42px;
  padding: 0 18px !important;
  font-weight: 700;
  color: #425469 !important;
}
.demo-shell .tab-nav button.selected {
  background: linear-gradient(135deg, #14365f, #1d4d88) !important;
  color: #fffdfa !important;
  box-shadow: 0 12px 24px rgba(20,54,95,0.18);
}
.demo-shell .tabitem {
  border-top: 0 !important;
  padding-top: 18px !important;
}
"""


def find_btl1_root() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        if base.name == "btl1" and (base / "artifacts").exists():
            return base
        candidate = base / "btl1"
        if candidate.exists() and (candidate / "artifacts").exists():
            return candidate
    raise FileNotFoundError("Could not locate the btl1 directory.")


BTL1_ROOT = find_btl1_root()
REPO_ROOT = BTL1_ROOT.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_ARTIFACT_DIR = BTL1_ROOT / "artifacts" / "text"
MM_ARTIFACT_DIR = BTL1_ROOT / "artifacts" / "multimodal"
N24_PROCESSED_DIR = BTL1_ROOT / "data" / "multimodal" / "n24news" / "processed"
DEMO_PORT = int(os.getenv("DEMO_PORT", "43881"))


def ensure_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_labels: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        mask = (input_ids != 0).unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(self.dropout(pooled))


class CLIPClassifier(nn.Module):
    def __init__(self, backbone: str, num_labels: int, dropout: float):
        super().__init__()
        self.backbone = CLIPModel.from_pretrained(backbone)
        hidden_size = self.backbone.config.projection_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, pixel_values):
        image_features = self.backbone.get_image_features(pixel_values=pixel_values)
        text_features = self.backbone.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)
        fused = torch.cat([image_features, text_features], dim=-1)
        return self.classifier(self.dropout(fused))


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
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.max_visual_tokens = max_visual_tokens

    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values):
        visual_outputs = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        visual_embeds = visual_outputs.last_hidden_state[:, 1 : 1 + self.max_visual_tokens, :]
        batch_size, visual_tokens, _ = visual_embeds.shape
        visual_attention_mask = torch.ones((batch_size, visual_tokens), dtype=torch.long, device=visual_embeds.device)
        visual_token_type_ids = torch.ones((batch_size, visual_tokens), dtype=torch.long, device=visual_embeds.device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            return_dict=True,
        )
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))


def tokenize(text: str) -> list[str]:
    return token_pattern.findall(text.lower())


def encode_text(text: str, vocab: dict[str, int], max_length: int = 300) -> list[int]:
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)[:max_length]]
    return ids if ids else [vocab["<unk>"]]


def load_image(image_input) -> Image.Image:
    if image_input is None or str(image_input).strip() == "":
        raise gr.Error("Vui lòng chọn một ảnh đầu vào.")
    return Image.open(image_input).convert("RGB")


def shorten_text(text: str, limit: int = 480) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@lru_cache(maxsize=1)
def load_text_metrics() -> dict:
    return json.loads(ensure_file(TEXT_ARTIFACT_DIR / "text_metrics_summary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_multimodal_metrics() -> dict:
    return json.loads(ensure_file(MM_ARTIFACT_DIR / "n24news_metrics_summary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_bert_bundle():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(TEXT_LABELS),
        problem_type="multi_label_classification",
    )
    state_dict = torch.load(ensure_file(TEXT_ARTIFACT_DIR / "bert_multilabel_best.pt"), map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def load_lstm_bundle():
    vocab = json.loads(ensure_file(TEXT_ARTIFACT_DIR / "lstm_vocab.json").read_text(encoding="utf-8"))
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=192,
        num_labels=len(TEXT_LABELS),
        dropout=0.3,
    )
    state_dict = torch.load(ensure_file(TEXT_ARTIFACT_DIR / "lstm_multilabel_best.pt"), map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return vocab, model


@lru_cache(maxsize=1)
def load_clip_bundle():
    checkpoint = torch.load(ensure_file(MM_ARTIFACT_DIR / "n24news_clip_best.pt"), map_location=DEVICE)
    cfg = checkpoint["config"]
    labels = checkpoint["label_names"]
    processor = AutoProcessor.from_pretrained(cfg["backbone"])
    model = CLIPClassifier(cfg["backbone"], len(labels), cfg["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return processor, model, labels, cfg


@lru_cache(maxsize=1)
def load_visualbert_bundle():
    checkpoint = torch.load(ensure_file(MM_ARTIFACT_DIR / "n24news_visualbert_best.pt"), map_location=DEVICE)
    cfg = checkpoint["config"]
    labels = checkpoint["label_names"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    image_processor = AutoImageProcessor.from_pretrained(cfg["vision_backbone"])
    model = VisualBERTClassifier(
        visualbert_backbone=cfg["visualbert_backbone"],
        vision_backbone=cfg["vision_backbone"],
        num_labels=len(labels),
        dropout=cfg["dropout"],
        max_visual_tokens=cfg["max_visual_tokens"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return tokenizer, image_processor, model, labels, cfg

def predict_with_bert(text: str) -> dict[str, float]:
    tokenizer, model = load_bert_bundle()
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors="pt",
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits[0]
    probs = torch.sigmoid(logits).detach().cpu().tolist()
    return dict(zip(TEXT_LABELS, probs))


def predict_with_lstm(text: str) -> dict[str, float]:
    vocab, model = load_lstm_bundle()
    token_ids = encode_text(text, vocab)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(input_ids, lengths)[0]
    probs = torch.sigmoid(logits).detach().cpu().tolist()
    return dict(zip(TEXT_LABELS, probs))


def predict_with_clip(image: Image.Image, text: str) -> dict[str, float]:
    processor, model, labels, cfg = load_clip_bundle()
    encoded = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg["max_text_length"],
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            pixel_values=encoded["pixel_values"],
        )[0]
    probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    return dict(zip(labels, probs))


def predict_with_visualbert(image: Image.Image, text: str) -> dict[str, float]:
    tokenizer, image_processor, model, labels, cfg = load_visualbert_bundle()
    text_inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg["max_text_length"],
    )
    pixel_values = image_processor(images=[image], return_tensors="pt")["pixel_values"]
    batch_inputs = {
        "input_ids": text_inputs["input_ids"].to(DEVICE),
        "attention_mask": text_inputs["attention_mask"].to(DEVICE),
        "token_type_ids": text_inputs.get("token_type_ids", torch.zeros_like(text_inputs["input_ids"])).to(DEVICE),
        "pixel_values": pixel_values.to(DEVICE),
    }
    with torch.no_grad():
        logits = model(**batch_inputs)[0]
    probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    return dict(zip(labels, probs))


def top_label(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda item: item[1])[0]


def build_score_table(
    model_a: str,
    scores_a: dict[str, float],
    model_b: str | None = None,
    scores_b: dict[str, float] | None = None,
    title_map: dict[str, str] | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    rows = []
    for label, score_a in scores_a.items():
        display = title_map.get(label, label) if title_map else label
        row = {"Nhãn": display, model_a: round(score_a, 4)}
        if model_b and scores_b:
            score_b = scores_b[label]
            row[model_b] = round(score_b, 4)
            row["Chênh lệch"] = round(score_a - score_b, 4)
            row["_sort"] = max(score_a, score_b)
        else:
            row["_sort"] = score_a
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
    if top_n is not None:
        frame = frame.head(top_n)
    return frame.reset_index(drop=True)


def render_single_text_result(model_name: str, scores: dict[str, float]) -> str:
    label = top_label(scores)
    tags = "".join(
        f'<span class="demo-chip">{TEXT_LABEL_TITLES[name]}: {score:.3f}</span>'
        for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
    )
    return f"""
    <div class="compare-card">
      <span class="section-eyebrow">Kết quả suy luận</span>
      <h3>{model_name}</h3>
      <p><strong>Nhãn nổi bật:</strong> {TEXT_LABEL_TITLES[label]}</p>
      <p>{TEXT_LABEL_DESCRIPTIONS[label]}</p>
      <div class="demo-chip-row">{tags}</div>
    </div>
    """


def render_compare_text_result(bert_scores: dict[str, float], lstm_scores: dict[str, float]) -> str:
    metrics = load_text_metrics()
    winner = "BERT" if metrics["bert"]["macro_f1"] >= metrics["lstm"]["macro_f1"] else "LSTM"
    bert_label = TEXT_LABEL_TITLES[top_label(bert_scores)]
    lstm_label = TEXT_LABEL_TITLES[top_label(lstm_scores)]
    delta_macro = metrics["bert"]["macro_f1"] - metrics["lstm"]["macro_f1"]

    return f"""
    <div class="compare-card">
      <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start;">
        <div>
          <span class="section-eyebrow">Compare mode</span>
          <h3>BERT vs LSTM</h3>
          <p>So sánh trực tiếp hai mô hình trên cùng một bình luận đầu vào.</p>
        </div>
        <span class="winner-chip">{winner} dẫn đầu benchmark</span>
      </div>
      <div class="metric-grid">
        <div class="metric-tile"><div class="metric-value">{metrics['bert']['macro_f1']:.4f}</div><div class="metric-label">BERT macro F1</div></div>
        <div class="metric-tile"><div class="metric-value">{metrics['lstm']['macro_f1']:.4f}</div><div class="metric-label">LSTM macro F1</div></div>
        <div class="metric-tile"><div class="metric-value">{metrics['bert']['micro_f1']:.4f}</div><div class="metric-label">BERT micro F1</div></div>
        <div class="metric-tile"><div class="metric-value">{delta_macro:.4f}</div><div class="metric-label">Delta macro F1</div></div>
      </div>
      <div class="compare-row">
        <div class="compare-model"><h4>BERT</h4><p>Nhãn nổi bật: <strong>{bert_label}</strong></p></div>
        <div class="compare-model"><h4>LSTM</h4><p>Nhãn nổi bật: <strong>{lstm_label}</strong></p></div>
      </div>
    </div>
    """


def render_single_multimodal_result(model_name: str, scores: dict[str, float]) -> str:
    label = top_label(scores)
    tags = "".join(
        f'<span class="demo-chip">{name}: {score:.3f}</span>'
        for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:4]
    )
    return f"""
    <div class="compare-card">
      <span class="section-eyebrow">Kết quả suy luận</span>
      <h3>{model_name}</h3>
      <p><strong>Chuyên mục dự đoán:</strong> {label}</p>
      <p>Mô hình trả về phân phối xác suất trên 24 lớp của N24News từ cùng một cặp ảnh-văn bản.</p>
      <div class="demo-chip-row">{tags}</div>
    </div>
    """


def render_compare_multimodal_result(clip_scores: dict[str, float], visualbert_scores: dict[str, float]) -> str:
    metrics = load_multimodal_metrics()["models"]
    winner = "VisualBERT" if metrics["VisualBERT"]["macro_f1"] >= metrics["CLIP"]["macro_f1"] else "CLIP"
    clip_label = top_label(clip_scores)
    vb_label = top_label(visualbert_scores)
    delta_macro = metrics["VisualBERT"]["macro_f1"] - metrics["CLIP"]["macro_f1"]

    return f"""
    <div class="compare-card">
      <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start;">
        <div>
          <span class="section-eyebrow">Compare mode</span>
          <h3>CLIP vs VisualBERT</h3>
          <p>So sánh trực tiếp hai mô hình trên cùng một cặp ảnh bài báo và văn bản bài báo.</p>
        </div>
        <span class="winner-chip">{winner} dẫn đầu benchmark</span>
      </div>
      <div class="metric-grid">
        <div class="metric-tile"><div class="metric-value">{metrics['VisualBERT']['macro_f1']:.4f}</div><div class="metric-label">VisualBERT macro F1</div></div>
        <div class="metric-tile"><div class="metric-value">{metrics['CLIP']['macro_f1']:.4f}</div><div class="metric-label">CLIP macro F1</div></div>
        <div class="metric-tile"><div class="metric-value">{metrics['VisualBERT']['accuracy']:.4f}</div><div class="metric-label">VisualBERT accuracy</div></div>
        <div class="metric-tile"><div class="metric-value">{delta_macro:.4f}</div><div class="metric-label">Delta macro F1</div></div>
      </div>
      <div class="compare-row">
        <div class="compare-model"><h4>CLIP</h4><p>Nhãn nổi bật: <strong>{clip_label}</strong></p></div>
        <div class="compare-model"><h4>VisualBERT</h4><p>Nhãn nổi bật: <strong>{vb_label}</strong></p></div>
      </div>
    </div>
    """


def run_text_demo(text: str, mode: str, model_name: str) -> tuple[str, pd.DataFrame]:
    text = str(text or "").strip()
    if not text:
        raise gr.Error("Vui lòng nhập một bình luận để chạy suy luận.")

    if mode == "Một mô hình":
        scores = predict_with_bert(text) if model_name == "BERT" else predict_with_lstm(text)
        table = build_score_table(model_name, scores, title_map=TEXT_LABEL_TITLES)
        return render_single_text_result(model_name, scores), table

    bert_scores = predict_with_bert(text)
    lstm_scores = predict_with_lstm(text)
    table = build_score_table("BERT", bert_scores, "LSTM", lstm_scores, title_map=TEXT_LABEL_TITLES)
    return render_compare_text_result(bert_scores, lstm_scores), table


def run_multimodal_demo(image_input, text: str, mode: str, model_name: str) -> tuple[str, pd.DataFrame]:
    text = str(text or "").strip()
    if not text:
        raise gr.Error("Vui lòng nhập văn bản bài báo để chạy suy luận.")

    image = load_image(image_input)

    if mode == "Một mô hình":
        scores = predict_with_clip(image, text) if model_name == "CLIP" else predict_with_visualbert(image, text)
        table = build_score_table(model_name, scores, top_n=10)
        return render_single_multimodal_result(model_name, scores), table

    clip_scores = predict_with_clip(image, text)
    visualbert_scores = predict_with_visualbert(image, text)
    table = build_score_table("CLIP", clip_scores, "VisualBERT", visualbert_scores, top_n=10)
    return render_compare_multimodal_result(clip_scores, visualbert_scores), table


def checkpoint_status() -> str:
    items = [
        ("BERT checkpoint", TEXT_ARTIFACT_DIR / "bert_multilabel_best.pt"),
        ("LSTM checkpoint", TEXT_ARTIFACT_DIR / "lstm_multilabel_best.pt"),
        ("CLIP checkpoint", MM_ARTIFACT_DIR / "n24news_clip_best.pt"),
        ("VisualBERT checkpoint", MM_ARTIFACT_DIR / "n24news_visualbert_best.pt"),
    ]
    rows = "".join(
        f"<tr><td>{label}</td><td>{'Có' if path.exists() else 'Thiếu'}</td><td><code>{path.name}</code></td></tr>"
        for label, path in items
    )
    return f"""
    <div class="demo-card">
      <p class="section-eyebrow">Checkpoint</p>
      <h3 style="margin-top:0; color:#14365f;">Trạng thái checkpoint local</h3>
      <table class="results-table">
        <thead><tr><th>Thành phần</th><th>Trạng thái</th><th>Tệp</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="margin-top:12px; color:#425469;">
        Demo local sẽ hoạt động đầy đủ khi bốn checkpoint trên đều hiện trạng thái <strong>Có</strong>.
      </p>
    </div>
    """


def multimodal_examples() -> list[list[str]]:
    path = N24_PROCESSED_DIR / "test.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, usecols=["category", "text_input", "image_relpath"]).dropna()
    examples = []
    seen = set()
    for _, row in frame.iterrows():
        category = row["category"]
        if category in seen:
            continue
        image_path = REPO_ROOT / row["image_relpath"]
        if not image_path.exists():
            continue
        seen.add(category)
        examples.append([str(image_path), shorten_text(row["text_input"], 700), "Một mô hình", "VisualBERT"])
        if len(examples) == 2:
            break
    if examples:
        examples.append([examples[0][0], examples[0][1], "So sánh hai mô hình", "VisualBERT"])
    return examples


def toggle_model_dropdown(mode: str):
    return gr.update(visible=mode == "Một mô hình")


TEXT_METRICS = load_text_metrics()
MM_METRICS = load_multimodal_metrics()
MM_EXAMPLES = multimodal_examples()

with gr.Blocks(title="CO3133 Demo Hub") as demo:
    gr.HTML(
        f"""
        <div class="demo-shell">
          <section class="demo-hero">
            <h1>CO3133 Demo Hub</h1>
            <p>
              Demo này load checkpoint đã huấn luyện cho <strong>Bài tập lớn 1</strong> và cho phép thử nhanh hai task:
              <strong>text classification</strong> với Jigsaw và
              <strong>text-image classification</strong> với N24News.
            </p>
            <div class="demo-chip-row">
              <span class="demo-chip">Device: {DEVICE}</span>
              <span class="demo-chip">BERT vs LSTM</span>
              <span class="demo-chip">CLIP vs VisualBERT</span>
              <span class="demo-chip">Lazy checkpoint loading</span>
            </div>
          </section>
        </div>
        """
    )

    with gr.Accordion("Trạng thái checkpoint và lưu ý chạy demo", open=False):
        gr.HTML(checkpoint_status())

    with gr.Tabs(elem_classes=["demo-shell"]):
        with gr.Tab("Văn bản"):
            gr.HTML(
                f"""
                <div class="demo-card">
                  <p class="section-eyebrow">Text classification</p>
                  <h2 style="margin-top:0;">Jigsaw Toxic Comment</h2>
                  <p>Nhập một bình luận để xem mô hình dự đoán các nhãn độc hại theo bài toán multi-label 6 nhãn. Bạn có thể chạy một mô hình riêng lẻ hoặc bật compare mode để đối chiếu BERT với LSTM trên cùng một prompt.</p>
                  <div class="metric-grid">
                    <div class="metric-tile"><div class="metric-value">{TEXT_METRICS['bert']['exact_match_accuracy']:.4f}</div><div class="metric-label">BERT exact-match</div></div>
                    <div class="metric-tile"><div class="metric-value">{TEXT_METRICS['bert']['micro_f1']:.4f}</div><div class="metric-label">BERT micro F1</div></div>
                    <div class="metric-tile"><div class="metric-value">{TEXT_METRICS['bert']['macro_f1']:.4f}</div><div class="metric-label">BERT macro F1</div></div>
                    <div class="metric-tile"><div class="metric-value">{TEXT_METRICS['bert']['macro_f1'] - TEXT_METRICS['lstm']['macro_f1']:.4f}</div><div class="metric-label">Delta macro F1</div></div>
                  </div>
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=11):
                    text_input = gr.Textbox(
                        label="Bình luận đầu vào",
                        lines=8,
                        placeholder="Nhập một comment để chạy suy luận...",
                    )
                    text_mode_input = gr.Radio(
                        choices=["Một mô hình", "So sánh hai mô hình"],
                        value="Một mô hình",
                        label="Chế độ suy luận",
                    )
                    text_model_input = gr.Dropdown(
                        choices=["BERT", "LSTM"],
                        value="BERT",
                        label="Mô hình",
                    )
                    text_run_button = gr.Button("Chạy demo", variant="primary")
                with gr.Column(scale=9):
                    text_html_output = gr.HTML(
                        "<div class='result-card'><h3>Kết quả sẽ hiển thị ở đây</h3><p>Demo sẽ trả về phần tóm tắt nổi bật và bảng xác suất theo nhãn. Khi bật compare mode, bảng sẽ hiển thị cả hai mô hình trên cùng input.</p></div>"
                    )
                    text_table_output = gr.Dataframe(label="Bảng xác suất", interactive=False)

            gr.Examples(examples=TEXT_EXAMPLES, inputs=[text_input, text_mode_input, text_model_input], label="Ví dụ nhanh")

            text_mode_input.change(fn=toggle_model_dropdown, inputs=text_mode_input, outputs=text_model_input)
            text_run_button.click(
                fn=run_text_demo,
                inputs=[text_input, text_mode_input, text_model_input],
                outputs=[text_html_output, text_table_output],
            )

        with gr.Tab("Đa phương thức"):
            gr.HTML(
                f"""
                <div class="demo-card">
                  <p class="section-eyebrow">Text-image classification</p>
                  <h2 style="margin-top:0;">N24News</h2>
                  <p>Nhập ảnh bài báo và văn bản bài báo để mô hình dự đoán chuyên mục tin tức. Bạn có thể chạy một mô hình riêng lẻ hoặc bật compare mode để đối chiếu CLIP với VisualBERT trên cùng một cặp đầu vào.</p>
                  <div class="metric-grid">
                    <div class="metric-tile"><div class="metric-value">{MM_METRICS['models']['VisualBERT']['accuracy']:.4f}</div><div class="metric-label">VisualBERT accuracy</div></div>
                    <div class="metric-tile"><div class="metric-value">{MM_METRICS['models']['VisualBERT']['macro_f1']:.4f}</div><div class="metric-label">VisualBERT macro F1</div></div>
                    <div class="metric-tile"><div class="metric-value">{MM_METRICS['models']['CLIP']['macro_f1']:.4f}</div><div class="metric-label">CLIP macro F1</div></div>
                    <div class="metric-tile"><div class="metric-value">{MM_METRICS['split_sizes']['test']}</div><div class="metric-label">Số mẫu test</div></div>
                  </div>
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=11):
                    image_input = gr.Image(label="Ảnh bài báo", type="filepath")
                    news_text_input = gr.Textbox(
                        label="Văn bản bài báo",
                        lines=10,
                        placeholder="Nhập headline, abstract hoặc nội dung bài báo để chạy suy luận...",
                    )
                    mm_mode_input = gr.Radio(
                        choices=["Một mô hình", "So sánh hai mô hình"],
                        value="Một mô hình",
                        label="Chế độ suy luận",
                    )
                    mm_model_input = gr.Dropdown(
                        choices=["CLIP", "VisualBERT"],
                        value="VisualBERT",
                        label="Mô hình",
                    )
                    mm_run_button = gr.Button("Chạy demo", variant="primary")
                with gr.Column(scale=9):
                    mm_html_output = gr.HTML(
                        "<div class='result-card'><h3>Kết quả sẽ hiển thị ở đây</h3><p>Demo sẽ trả về chuyên mục dự đoán và bảng xác suất trên 24 lớp của N24News. Khi bật compare mode, bảng sẽ hiển thị đồng thời CLIP và VisualBERT.</p></div>"
                    )
                    mm_table_output = gr.Dataframe(label="Bảng xác suất", interactive=False)

            if MM_EXAMPLES:
                gr.Examples(
                    examples=MM_EXAMPLES,
                    inputs=[image_input, news_text_input, mm_mode_input, mm_model_input],
                    label="Ví dụ nhanh",
                )

            mm_mode_input.change(fn=toggle_model_dropdown, inputs=mm_mode_input, outputs=mm_model_input)
            mm_run_button.click(
                fn=run_multimodal_demo,
                inputs=[image_input, news_text_input, mm_mode_input, mm_model_input],
                outputs=[mm_html_output, mm_table_output],
            )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=DEMO_PORT, share=False, css=DEMO_CSS)
