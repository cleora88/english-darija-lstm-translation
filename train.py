"""Controlled comparison of manual LSTM encoder-decoder translation models."""

import argparse
import json
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import DataLoader

from data import EOS, PAD, ROOT, ParallelDataset, collate, prepare, save_json
from metrics import score_translations
from model import Translator

CONFIGURATIONS = ((0.001, 128), (0.003, 128), (0.003, 256))


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_context(device: str):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )


def decode_rows(tokenizer: Tokenizer, tokens: torch.Tensor) -> list[str]:
    decoded = []
    for row in tokens.cpu().tolist():
        if EOS in row:
            row = row[: row.index(EOS)]
        decoded.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return decoded


@torch.inference_mode()
def evaluate(model, loader, tokenizer, frame, device):
    model.eval()
    predictions = []
    for source, _ in loader:
        with precision_context(device):
            generated = model.generate(source.to(device))
        predictions.extend(decode_rows(tokenizer, generated))
    return score_translations(predictions, frame.target.tolist()), predictions


def train(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "precision": "bfloat16 autocast with float32 parameters"
            if device == "cuda"
            else "float32",
        },
    )
    if args.reuse_data and (output / "tokenizer.json").exists():
        frames = {
            n: pd.read_csv(output / f"{n}_pairs.csv", keep_default_na=False)
            for n in ("train", "validation", "test")
        }
        tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
    else:
        frames, tokenizer, _ = prepare(output)
    if args.smoke:
        frames = {n: f.head(64).copy() for n, f in frames.items()}
    datasets = {n: ParallelDataset(f, tokenizer) for n, f in frames.items()}
    evaluation_loaders = {
        n: DataLoader(datasets[n], batch_size=128, collate_fn=collate)
        for n in ("validation", "test")
    }
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    experiments = []
    selected = {}
    kinds = ["vanilla", "cifg", "wmc"]
    configurations = CONFIGURATIONS[:1] if args.smoke else CONFIGURATIONS
    epochs = 1 if args.smoke else args.epochs

    for kind in kinds:
        best_architecture_score = -1.0
        for learning_rate, batch_size in configurations:
            run_id = f"{kind}_lr{learning_rate}_batch{batch_size}"
            run_file = output / f"{run_id}.json"
            checkpoint = output / f"{run_id}.pt"
            if args.resume and run_file.exists() and checkpoint.exists():
                record = json.loads(run_file.read_text())
                if record["epochs"] != epochs:
                    raise ValueError(
                        f"Epoch budget differs for completed run {run_id}; use a separate output directory."
                    )
                experiments.append(record)
                if record["best_validation_chrf"] > best_architecture_score:
                    best_architecture_score = record["best_validation_chrf"]
                    selected[kind] = record
                print("RESUMED", run_id, flush=True)
                continue
            seed_everything()
            model = Translator(tokenizer.get_vocab_size(), kind).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loader = DataLoader(
                datasets["train"],
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate,
            )
            history = []
            best_score = -1.0
            best_state = None
            started = time.perf_counter()
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss, total_tokens = 0.0, 0
                epoch_started = time.perf_counter()
                for source, target in loader:
                    source, target = source.to(device), target.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    labels = target[:, 1:]
                    valid_tokens = int((labels != PAD).sum())
                    with precision_context(device):
                        logits = model(source, target[:, :-1])
                        summed_loss = criterion(
                            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                        )
                        loss = summed_loss / valid_tokens
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += float(summed_loss.detach())
                    total_tokens += valid_tokens
                metrics, _ = evaluate(
                    model,
                    evaluation_loaders["validation"],
                    tokenizer,
                    frames["validation"],
                    device,
                )
                step = {
                    "epoch": epoch,
                    "train_token_loss": total_loss / total_tokens,
                    "validation_bleu": metrics["bleu"],
                    "validation_chrf": metrics["chrf"],
                    "seconds": time.perf_counter() - epoch_started,
                }
                history.append(step)
                print(run_id, json.dumps(step), flush=True)
                save_json(
                    output / "progress.json", {"run_id": run_id, "history": history}
                )
                if metrics["chrf"] > best_score:
                    best_score = metrics["chrf"]
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
            if best_state is None:
                raise RuntimeError(f"No valid checkpoint produced for {run_id}")
            torch.save(best_state, checkpoint)
            record = {
                "run_id": run_id,
                "model": kind,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "seed": 42,
                "optimizer": "Adam",
                "gradient_clip": 1.0,
                "embedding_size": 96,
                "hidden_size": 128,
                "dropout": 0.2,
                "teacher_forcing": 1.0,
                "parameters": sum(p.numel() for p in model.parameters()),
                "best_validation_chrf": best_score,
                "best_epoch": max(history, key=lambda h: h["validation_chrf"])["epoch"],
                "seconds": time.perf_counter() - started,
                "history": history,
            }
            save_json(run_file, record)
            experiments.append(record)
            save_json(output / "experiments.json", experiments)
            if best_score > best_architecture_score:
                best_architecture_score = best_score
                selected[kind] = record

    # Freeze selection for all architectures before calculating any test score.
    save_json(output / "selection.json", selected)
    final = []
    for kind, record in selected.items():
        model = Translator(tokenizer.get_vocab_size(), kind).to(device)
        state = torch.load(
            output / f"{record['run_id']}.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(state)
        torch.save({k: v.cpu() for k, v in state.items()}, output / f"{kind}.pt")
        metrics, predictions = evaluate(
            model, evaluation_loaders["test"], tokenizer, frames["test"], device
        )
        row = {
            "model": kind,
            "run_id": record["run_id"],
            "learning_rate": record["learning_rate"],
            "batch_size": record["batch_size"],
            "best_epoch": record["best_epoch"],
            "parameters": record["parameters"],
            **metrics,
        }
        final.append(row)
        pd.DataFrame(
            {
                "source": frames["test"].source.tolist(),
                "reference": frames["test"].target.tolist(),
                "prediction": predictions,
            }
        ).to_csv(output / f"{kind}_predictions.csv", index=False)
        print("TEST", json.dumps(row), flush=True)
    save_json(output / "final_results.json", final)
    pd.DataFrame(final).to_csv(output / "comparison.csv", index=False)
    copy_baseline = score_translations(
        frames["test"].source.tolist(), frames["test"].target.tolist()
    )
    save_json(output / "copy_baseline.json", copy_baseline)
    print("COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs.")
    parser.add_argument(
        "--reuse-data", action="store_true", help="Use the saved splits and tokenizer."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Exercise training and generation on 64 pairs.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epoch count must be positive")
    train(args)


if __name__ == "__main__":
    main()
