"""Capture browser views of real experiment records for the project evidence."""

import argparse
import html
import json

from playwright.sync_api import sync_playwright

from data import ROOT


def read(name):
    return (ROOT / name).read_text(encoding="utf-8", errors="replace")


def block(title, value):
    return f"<h2>{html.escape(title)}</h2><pre>{html.escape(value)}</pre>"


def run_page(record):
    columns = (
        "epoch",
        "train_token_loss",
        "validation_bleu",
        "validation_chrf",
        "seconds",
    )
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{row[c]:.4f}</td>" if c != "epoch" else f"<td>{row[c]}</td>"
            for c in columns
        )
        + "</tr>"
        for row in record["history"]
    )
    return (
        f"<h2>{html.escape(record['run_id'])}</h2><p>Parameters: {record['parameters']:,}. "
        f"Learning rate: {record['learning_rate']}. Batch: {record['batch_size']}. "
        f"Best validation chrF: {record['best_validation_chrf']:.4f}. Selected epoch: {record['best_epoch']}.</p>"
        "<table><tr><th>Epoch</th><th>Token loss</th><th>Validation BLEU</th><th>Validation chrF</th><th>Seconds</th></tr>"
        + rows
        + "</table>"
    )


def capture(stage, record=None):
    directory = ROOT / "screenshots"
    directory.mkdir(exist_ok=True)
    if record is not None:
        content = run_page(record)
    elif stage == "01_environment":
        content = block(
            "Recorded training runtime", read("results/environment.json")
        ) + block("Equation and model checks", read("logs/model_checks.txt"))
    elif stage == "02_data":
        content = block(
            "Parallel-data audit and fixed splits", read("results/data_summary.json")
        )
        content += '<img src="../results/data_exploration.png">'
    elif stage == "03_model_checks":
        content = block(
            "Equations, gradients, padding, causality and EOS",
            read("logs/model_checks.txt"),
        )
        content += block("Manual recurrent-cell implementation", read("cells.py"))
    elif stage == "04_training":
        content = block("Actual training log", read("logs/training.txt")[-16000:])
    elif stage == "05_evaluation":
        content = block("Final held-out test scores", read("results/comparison.csv"))
        content += block("Independent verification", read("results/verification.json"))
        content += '<img src="../results/test_comparison.png"><img src="../results/learning_curves.png">'
    else:
        raise ValueError(stage)
    document = (
        """<!doctype html><meta charset="utf-8"><style>
    body{font:18px Arial;background:#f7f5ee;color:#18364b;margin:42px;max-width:1160px}
    h1{font-size:34px}h2{color:#007f7a;font-size:25px}pre{white-space:pre-wrap;background:white;padding:24px;font:15px Consolas;line-height:1.45}
    table{border-collapse:collapse;width:100%;background:white}th{background:#102c40;color:white}td,th{padding:12px;text-align:left;border-bottom:1px solid #c7d5d2}img{max-width:100%;margin-top:24px}footer{font-size:14px;color:#526878;margin-top:30px}
    </style>"""
        + f"<h1>English-to-Darija LSTM / {html.escape(stage.replace('_', ' '))}</h1><p>UIR / Advanced NLP / TP B / 2026-2027</p>"
        + content
        + "<footer>Browser capture of saved experiment data, code and logs.</footer>"
    )
    page_file = directory / f"{stage}.html"
    page_file.write_text(document, encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 960})
        page.goto(page_file.as_uri())
        page.screenshot(path=str(directory / f"{stage}.png"), full_page=True)
        browser.close()
    print("Captured", stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stages",
        nargs="*",
        default=[
            "01_environment",
            "02_data",
            "03_model_checks",
            "04_training",
            "05_evaluation",
        ],
    )
    parser.add_argument("--runs", action="store_true")
    args = parser.parse_args()
    for stage in args.stages:
        capture(stage)
    if args.runs:
        for index, record in enumerate(json.loads(read("results/experiments.json")), 1):
            capture(f"run_{index:02d}", record)


if __name__ == "__main__":
    main()
