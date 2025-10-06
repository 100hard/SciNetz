from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "tests" / "fixtures" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

PAGE_WIDTH, PAGE_HEIGHT = LETTER
MARGIN = 72
LINE_SPACING = 18

def _draw_lines(c: canvas.Canvas, lines: list[tuple[str, str]]) -> None:
    y = PAGE_HEIGHT - MARGIN
    for text, font in lines:
        if text == "<PAGE_BREAK>":
            c.showPage()
            y = PAGE_HEIGHT - MARGIN
            continue
        family, size_str = font.split(":")
        size = int(size_str)
        c.setFont(family, size)
        c.drawString(MARGIN, y, text)
        y -= LINE_SPACING if size <= 16 else LINE_SPACING * 1.4

def build_sample_transformer() -> None:
    output_path = PDF_DIR / "sample_transformer.pdf"
    c = canvas.Canvas(str(output_path), pagesize=LETTER)
    lines = [
        ("Exploring Transformer Architectures for NLP", "Helvetica-Bold:24"),
        ("Jane Doe, John Smith", "Helvetica:14"),
        ("Proceedings of NeurIPS 2024", "Helvetica:14"),
        ("Published: 2024", "Helvetica:14"),
        ("DOI: 10.1234/transformers.2024.001", "Helvetica:14"),
        ("Abstract", "Helvetica-Bold:18"),
        (
            "We explore transformer models for natural language processing tasks and analyze their performance across benchmarks.",
            "Helvetica:12",
        ),
        ("Intro", "Helvetica-Bold:18"),
        (
            "Transformers have reshaped NLP by enabling parallel training and long-range dependencies. In this study we review architectural improvements.",
            "Helvetica:12",
        ),
        ("Methodology", "Helvetica-Bold:18"),
        (
            "We evaluate encoder-only and encoder-decoder variants on translation and question answering. Experiments include ablation on attention heads.",
            "Helvetica:12",
        ),
        ("<PAGE_BREAK>", "Helvetica:12"),
        ("Results", "Helvetica-Bold:18"),
        (
            "Transformer Large outperforms baselines by 5% BLEU on WMT14 and achieves 90% accuracy on SQuAD.",
            "Helvetica:12",
        ),
        ("Discussion", "Helvetica-Bold:18"),
        (
            "We discuss limitations including compute costs and data bias.",
            "Helvetica:12",
        ),
    ]
    _draw_lines(c, lines)
    c.save()

def build_sample_graph() -> None:
    output_path = PDF_DIR / "sample_graph.pdf"
    c = canvas.Canvas(str(output_path), pagesize=LETTER)
    lines = [
        ("Graph Embeddings for Scientific Literature", "Helvetica-Bold:24"),
        ("Alice Johnson, Bob Lee", "Helvetica:14"),
        ("ACL 2023", "Helvetica:14"),
        ("Published: 2023", "Helvetica:14"),
        ("DOI: 10.5678/graph.2023.010", "Helvetica:14"),
        ("Abstract", "Helvetica-Bold:18"),
        (
            "We present a graph embedding approach for linking scientific papers using metadata and citation graphs.",
            "Helvetica:12",
        ),
        ("Introduction", "Helvetica-Bold:18"),
        (
            "Graph embeddings capture structural information useful for recommendation.",
            "Helvetica:12",
        ),
        ("Methods", "Helvetica-Bold:18"),
        (
            "We extend DeepWalk with contextual metadata features and evaluate on arXiv.",
            "Helvetica:12",
        ),
        ("Conclusion", "Helvetica-Bold:18"),
        (
            "Embedding-based similarity improves retrieval precision by 10%.",
            "Helvetica:12",
        ),
    ]
    _draw_lines(c, lines)
    c.save()

def main() -> None:
    build_sample_transformer()
    build_sample_graph()


if __name__ == "__main__":
    main()
