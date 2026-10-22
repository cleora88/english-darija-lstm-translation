"""Generate Latin Darija from English using a selected checkpoint."""

import argparse
import json
from contextlib import nullcontext

import torch
from tokenizers import Tokenizer

from data import EOS, MAX_LENGTH, ROOT, normalize
from model import Translator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="English source sentence")
    parser.add_argument(
        "--model",
        choices=("vanilla", "cifg", "wmc"),
        help="Default: best validation-selected architecture",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.model is None:
        selection_file = ROOT / "results/selection.json"
        if not selection_file.exists():
            parser.error("Run train.py before translation.")
        selection = json.loads(selection_file.read_text(encoding="utf-8"))
        args.model = max(
            selection, key=lambda kind: selection[kind]["best_validation_chrf"]
        )
    tokenizer_file = ROOT / "results/tokenizer.json"
    checkpoint = ROOT / f"results/{args.model}.pt"
    if not checkpoint.exists():
        parser.error(
            "Checkpoint missing. Run train.py or use the local project archive."
        )
    text = normalize(args.text)
    if not text:
        parser.error("Enter a nonempty English sentence")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    model = Translator(tokenizer.get_vocab_size(), args.model).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable on this machine")
    model.to(device)
    source = torch.tensor([tokenizer.encode(text).ids[: MAX_LENGTH - 1] + [EOS]])
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), context:
        tokens = model.generate(source.to(device))[0].tolist()
    if EOS in tokens:
        tokens = tokens[: tokens.index(EOS)]
    print(tokenizer.decode(tokens, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
