"""Five-page UIR report based exclusively on recorded translation experiments."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from data import ROOT

FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("ReportArial", str(FONT_DIR / "arial.ttf")))
pdfmetrics.registerFont(TTFont("ReportArialBold", str(FONT_DIR / "arialbd.ttf")))
pdfmetrics.registerFontFamily(
    "ReportArial",
    normal="ReportArial",
    bold="ReportArialBold",
    italic="ReportArial",
    boldItalic="ReportArialBold",
)
INK, TEAL, MUTED = "#18364B", "#007F7A", "#526878"


def main():
    results_dir = ROOT / "results"

    def load(name):
        return json.loads((results_dir / f"{name}.json").read_text(encoding="utf-8"))

    data, runs, results = (
        load("data_summary"),
        load("experiments"),
        load("final_results"),
    )
    selection, examples, baseline = (
        load("selection"),
        load("examples"),
        load("copy_baseline"),
    )
    chosen = max(selection, key=lambda kind: selection[kind]["best_validation_chrf"])
    body = ParagraphStyle(
        "Body",
        fontName="ReportArial",
        fontSize=9.7,
        leading=13.5,
        spaceAfter=9,
        textColor=colors.HexColor(INK),
    )
    heading = ParagraphStyle(
        "Heading",
        fontName="ReportArialBold",
        fontSize=18,
        leading=23,
        spaceAfter=12,
        textColor=colors.HexColor(TEAL),
    )
    small = ParagraphStyle(
        "Small", parent=body, fontSize=8.2, leading=10.7, spaceAfter=0
    )
    title = ParagraphStyle(
        "Title",
        fontName="ReportArialBold",
        fontSize=31,
        leading=37,
        spaceAfter=18,
        textColor=colors.HexColor(INK),
    )
    meta = ParagraphStyle(
        "Meta", parent=body, fontSize=11, leading=17, textColor=colors.HexColor(MUTED)
    )
    story = []

    def p(text):
        story.append(Paragraph(text, body))

    def h(text):
        story.append(Paragraph(text, heading))

    def page():
        story.append(PageBreak())

    def image(name, width=480):
        path = results_dir / name
        with PILImage.open(path) as im:
            ratio = im.height / im.width
        story.append(Image(str(path), width=width, height=width * ratio))

    def table(rows, widths):
        formatted = [
            [Paragraph(escape(str(value)), small) for value in row] for row in rows
        ]
        # Header styling is independent of body Paragraph defaults.
        header_style = ParagraphStyle(
            "TableHeader",
            parent=small,
            fontName="ReportArialBold",
            textColor=colors.white,
        )
        formatted[0] = [
            Paragraph(escape(str(value)), header_style) for value in rows[0]
        ]
        t = Table(formatted, colWidths=widths, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102C40")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#EFF3EF"), colors.white],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.extend([t, Spacer(1, 11)])

    logo = ROOT / "assets/UIR_logo.png"
    with PILImage.open(logo) as im:
        ratio = im.height / im.width
    story.extend(
        [Image(str(logo), width=225, height=225 * ratio, hAlign="LEFT"), Spacer(1, 36)]
    )
    story.append(
        Paragraph(
            "LAB PROJECT / RECURRENT NEURAL NETWORKS",
            ParagraphStyle(
                "Kicker",
                parent=body,
                fontName="ReportArialBold",
                fontSize=10,
                textColor=colors.HexColor(TEAL),
                spaceAfter=16,
            ),
        )
    )
    story.append(
        Paragraph("English-to-Darija Translation<br/>with Manual LSTM Cells", title)
    )
    story.append(
        Paragraph(
            "Advanced Natural Language Processing<br/>University year 2026/2027", meta
        )
    )
    story.append(Spacer(1, 24))
    story.append(
        Paragraph(
            "Zouhair Eouichak / Aymane El Youssi<br/>Malak Bartal / Sara Baalla<br/>Group TP B / 22 October 2026",
            meta,
        )
    )
    story.append(Spacer(1, 28))
    h("Task and study design")
    p(
        "This project translates English sentences into Latin-script Moroccan Darija using the dataset specified in the lab brief. It compares a vanilla LSTM baseline with Coupled Input-Forget Gate (CIFG) and Working Memory Connections (WMC) variants. Both encoder and decoder use manually implemented cells based on nn.Linear."
    )
    p(
        "All models share training, validation and test partitions, a training-only subword tokenizer, attention and decoding rules. Nine runs compare learning rate and batch size over twelve epochs each. Validation chrF selects checkpoints and configurations before final test evaluation. Corpus BLEU and chrF, parameter counts and actual generated examples support the comparison."
    )
    p(
        "The objective is a controlled study of recurrent-cell design under a limited parallel-data budget. The outputs are experimental translations rather than a production translation service."
    )

    page()
    h("1. Dataset exploration and preparation")
    p(
        f"The pinned Hugging Face sentences.csv file contains {data['raw_rows']:,} rows, but {data['missing_english']:,} have no English value. Only {data['complete_pairs_before_deduplication']:,} complete English-Darija pairs remain before deduplication. Removing {data['duplicate_pairs_removed']:,} normalized duplicate pairs leaves {data['usable_pairs']:,} usable pairs. The source is eng and the target is darija; darija_ar is excluded from this experiment."
    )
    p(
        "Unicode normalization and whitespace cleanup preserve Darija capitalization and digits, which can encode pronunciation. Connected components join repeated normalized English sources and repeated exact Darija targets. Splitting these groups prevents source or target overlap across partitions. Seed 42 gives approximately 80/10/10 proportions, with no downsampling of the usable corpus."
    )
    table(
        [["Partition", "Pairs", "Mean source tokens", "Mean target tokens"]]
        + [
            [
                name,
                info["pairs"],
                f"{info['source_mean_tokens']:.2f}",
                f"{info['target_mean_tokens']:.2f}",
            ]
            for name, info in data["splits"].items()
        ],
        [110, 65, 130, 175],
    )
    image("data_exploration.png")
    p(
        "A joint byte-level BPE tokenizer with 2,000 entries is trained only on training pairs. PAD, UNK, BOS and EOS mark padding, unknown input and sequence boundaries. Batches pad to their longest sequence, and padding does not update recurrent states or receive loss. The source and generated-target limits are 48 tokens including EOS. Training truncation affects approximately "
        + f"{data['splits']['train']['source_truncation_fraction']:.2%} of sources and {data['splits']['train']['target_truncation_fraction']:.2%} of targets. Near-duplicate wording can still survive exact grouping."
    )

    page()
    h("2. Models and implementation")
    p(
        "Each system uses a shared 96-dimensional embedding, a one-layer encoder and decoder with 128-dimensional states, scaled dot-product attention and dropout 0.2 before the output projection. The encoder's final hidden and memory states initialize the decoder. Attention lets each decoder state consult valid encoder outputs. It is identical across all architectures, isolating the recurrent-cell comparison."
    )
    table(
        [["Cell", "Main update", "Total trainable parameters"]]
        + [
            [
                r["model"].upper(),
                {
                    "vanilla": "Independent input, forget and output gates",
                    "cifg": "Input gate i_t = 1 - f_t",
                    "wmc": "Memory-to-gate connections",
                }[r["model"]],
                f"{r['parameters']:,}",
            ]
            for r in results
        ],
        [65, 260, 155],
    )
    p(
        "<b>Vanilla:</b> sigmoid input, forget and output gates combine with a tanh candidate. The memory update is c_t = f_t * c_(t-1) + i_t * g_t, and h_t = o_t * tanh(c_t). nn.Linear computes input and recurrent affine terms. The model contains no built-in nn.LSTM or nn.LSTMCell."
    )
    p(
        "<b>CIFG:</b> coupling i_t = 1 - f_t removes one learned gate in each recurrent cell. This reduces parameters while constraining the balance between retaining existing memory and inserting new information."
    )
    p(
        "<b>WMC:</b> input and forget gates add tanh-transformed connections from c_(t-1). After updating memory, the output gate adds a connection from c_t. This ordering follows the appendix and introduces additional memory matrices."
    )
    p(
        "Training uses full teacher forcing: decoder input begins with BOS and previous reference tokens, while shifted labels include EOS. Cross entropy averages over non-padding target positions. At validation and test time, greedy generation receives only the source sentence and its own previously generated tokens. It stops at EOS or the token limit, and never receives reference targets."
    )
    p(
        "Equation tests check all cells, finite gradients, padding invariance, decoder causality and EOS termination. An independent verifier checks source/target/group separation, saved metric calculations, validation-only selection and parameter counts. Counts include both recurrent cells, embeddings, attention projection and vocabulary output layer."
    )

    page()
    h("3. Hyperparameter experiments")
    p(
        "All nine runs use Adam, gradient clipping at norm 1.0, seed 42 and twelve epochs. Compare learning rates 0.001 and 0.003 at batch 128, then batches 128 and 256 at learning rate 0.003. The design tests each factor under a fixed counterpart, without claiming a full interaction study. Smaller batches perform more optimizer updates per epoch."
    )
    table(
        [["Cell", "Learning rate", "Batch", "Best epoch", "Validation chrF"]]
        + [
            [
                r["model"],
                r["learning_rate"],
                r["batch_size"],
                r["best_epoch"],
                f"{r['best_validation_chrf']:.2f}",
            ]
            for r in runs
        ],
        [70, 85, 65, 85, 175],
    )
    image("learning_curves.png")
    p(
        "Each run retains the epoch with the highest generated validation chrF. The best validation configuration for each architecture is frozen before any final test comparison. Both BLEU and chrF use the same single reference, tokenizer settings and case policy for all models. Recorded SacreBLEU signatures specify the exact scoring setup."
    )
    p(
        "chrF uses character n-grams up to order 6 and beta 2, accommodating some subword variation. BLEU uses 13a tokenization. Neither metric fully captures translation adequacy, and both are sensitive to the limited reference coverage and inconsistent Darija spelling. The experiment reports one seed and does not estimate statistical significance."
    )

    page()
    h("4. Test evaluation and discussion")
    table(
        [["Cell", "BLEU", "chrF", "Exact match", "Selected LR / batch"]]
        + [
            [
                r["model"].upper(),
                f"{r['bleu']:.2f}",
                f"{r['chrf']:.2f}",
                f"{r['exact_match']:.2%}",
                f"{r['learning_rate']} / {r['batch_size']}",
            ]
            for r in results
        ],
        [80, 65, 65, 90, 180],
    )
    p(
        f"The configuration with the highest validation chrF belongs to {chosen.upper()}. Final scores above come from reference-free generation on {data['splits']['test']['pairs']:,} held-out pairs. Copying the English source without translating yields BLEU {baseline['bleu']:.2f} and chrF {baseline['chrf']:.2f}, providing a simple non-neural baseline. Architecture rankings are observations for this protocol rather than universal conclusions."
    )
    image("test_comparison.png", 460)
    p(
        "<b>Actual generated examples:</b> the cases below come from the validation-selected architecture, at the 10th and 50th percentiles of test sentence chrF. They expose weak and typical behavior rather than displaying only successes. The 90th-percentile case and all predictions are saved separately."
    )
    sample_rows = [["English source", "Reference Darija", "Generated Darija"]]
    for row in examples["examples"][:2]:
        sample_rows.append(
            [row["source"], row["reference"], row["prediction"] or "[empty output]"]
        )
    table(sample_rows, [160, 160, 160])
    p(
        "<b>Improvements:</b> repeat across seeds, extend training and assess BPE vocabulary size, larger states, beam search and teacher-forcing schedules. More verified parallel pairs and multiple Darija references would strengthen both learning and evaluation. Test paraphrase robustness and inspect omissions, repetition and spelling errors before stronger claims."
    )
    p(
        "<b>References:</b> Guidelines_RNN_2627, appendix equations. imomayiz/darija-english, Hugging Face, pinned revision "
        + load("dataset_provenance")["revision"][:12]
        + ". SacreBLEU (Post, 2018), github.com/mjpost/sacrebleu. CIFG reading: arxiv.org/abs/1503.04069. WMC reading: S0893608021003439."
    )

    def decorate(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#C7D5D2"))
        canvas.setLineWidth(0.5)
        canvas.line(48, 39, A4[0] - 48, 39)
        canvas.setFont("ReportArial", 8)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(48, 25, "UIR / ADVANCED NLP / TP B")
        canvas.drawRightString(A4[0] - 48, 25, f"{doc.page:02d} / 05")
        if doc.page > 1:
            canvas.setFont("ReportArialBold", 8)
            canvas.setFillColor(colors.HexColor(TEAL))
            canvas.drawString(
                48, A4[1] - 30, "ENGLISH-TO-DARIJA / MANUAL LSTM COMPARISON"
            )

    target = ROOT / "output/Darija_Translation_Report.pdf"
    SimpleDocTemplate(
        str(target),
        pagesize=A4,
        leftMargin=48,
        rightMargin=48,
        topMargin=54,
        bottomMargin=50,
        title="English-to-Darija Translation with Manual LSTM Cells",
        author="Zouhair Eouichak; Aymane El Youssi; Malak Bartal; Sara Baalla",
    ).build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(target)


if __name__ == "__main__":
    main()
