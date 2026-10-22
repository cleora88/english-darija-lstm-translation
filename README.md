# English-to-Darija translation with manual LSTM cells

Advanced Natural Language Processing, UIR, 2026/2027. Group TP B.

Team: Zouhair Eouichak, Aymane El Youssi, Malak Bartal and Sara Baalla. The assignment requires five or six students from the same lab group; the remaining membership or an exception must be confirmed before submission.

Public PyTorch repository: https://github.com/cleora88/english-darija-lstm-translation

## Task and data

Translate English into Latin-script Moroccan Darija using the dataset explicitly listed in the project brief: [imomayiz/darija-english](https://huggingface.co/datasets/imomayiz/darija-english). Source column: `eng`. Target column: `darija`. The Arabic-script `darija_ar` column is not a model input or target in this experiment.

The pinned `sentences.csv` snapshot has 87,785 rows, of which 75,042 lack an English value. There are 12,743 complete pairs and 12,622 after removing normalized duplicate pairs. These are not 87,785 usable parallel examples. The exact revision, URL and SHA-256 hash are recorded in `results/dataset_provenance.json`.

Connected components join repeated normalized English sources and repeated Darija targets. Groups, rather than individual rows, are randomly assigned to approximately 80/10/10 partitions with seed 42. The resulting splits contain 10,110 training, 1,245 validation and 1,267 test pairs. No normalized source key or exact normalized target occurs in more than one partition. Near-duplicates can still remain.

## Models and controlled comparison

All models share a 2,000-entry byte-level BPE tokenizer learned only from the training pairs, 96-dimensional shared embeddings, 128-dimensional hidden states, a single recurrent encoder and decoder, scaled dot-product attention and dropout 0.2. Latin Darija case and digits are preserved. Attention is common to every architecture, so the comparison changes the recurrent cell rather than the attention design.

The three cell types are vanilla LSTM, Coupled Input-Forget Gate (CIFG), and Working Memory Connections (WMC), implemented directly with `nn.Linear`. No built-in `nn.LSTM` or `nn.LSTMCell` implements the model. WMC's output gate reads the updated memory state, following the appendix.

Each architecture uses twelve epochs under three Adam configurations: `(learning rate 0.001, batch 128)`, `(0.003, 128)` and `(0.003, 256)`. The loss averages cross entropy over non-padding target tokens. Training uses full teacher forcing and gradient clipping at norm 1.0. Validation and test predictions are generated autoregressively without reference target input, using greedy decoding and a maximum of 48 generated tokens. Source sequences also have a 48-token limit including EOS. Truncation rates are documented in the data summary.

Select each run's checkpoint and each architecture's configuration by **validation chrF**, then freeze all selections before test evaluation. Report corpus BLEU and chrF (0-100), exact match, metric signatures and total trainable parameter counts. BLEU uses SacreBLEU's 13a tokenizer; chrF uses character order 6 and beta 2 without word n-grams. English-copy predictions supply a simple non-neural reference baseline. BLEU and chrF are imperfect with a single reference and variable Darija spellings.

## Reproduction

Python 3.12 or a compatible newer version is recommended. A CUDA GPU is useful. The local `.venv` reuses installed CUDA libraries to avoid a duplicate download; recreate dependencies independently on another computer.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python test_models.py
.venv\Scripts\python train.py --epochs 12
.venv\Scripts\python verify_results.py
.venv\Scripts\python make_figures.py
```

For CPU-only installation, install `requirements.txt` from the default PyPI index instead of the CUDA lock file. Use `--smoke --output-dir .build/smoke` to exercise the pipeline on small subsets without replacing benchmark records. `--resume --reuse-data` reuses complete runs from the same output directory. Do not mix a resumed run with different epochs, data or tokenizer settings.

## Project files

| File | Purpose |
|---|---|
| `data.py` | Pinned download, normalization, grouped splitting, BPE and batch padding |
| `cells.py` | The three manual recurrent cells |
| `model.py` | Encoder-decoder, shared attention and greedy generation |
| `train.py` | Controlled experiments, validation selection and final testing |
| `metrics.py` | Reproducible corpus scores |
| `translate.py` | Translation with a selected local checkpoint |
| `verify_results.py` | Recompute metrics, inspect splits and check parameter totals |
| `make_figures.py` | Scientific figures based on measured data |
| `REQUIREMENTS.md` | Assignment checklist and pending team-size requirement |

`results/` records the tokenizer, frozen splits, histories, checkpoint selections, predictions and scores. `screenshots/` contains actual browser captures of these saved records and rendered deliverable previews. Raw data, environments and checkpoint files are excluded from Git; local archives include selected weights and split files. The previous Drug Review project remains separate and is not the topic for this submission.

## Recorded test results

| Model | BLEU | chrF | Exact match | Parameters |
|---|---:|---:|---:|---:|
| Vanilla | 0.65 | 15.88 | 0.16% | 713,296 |
| CIFG | 0.68 | 15.86 | 0.24% | 655,696 |
| WMC | 1.33 | 16.05 | 0.08% | 811,600 |
| English copy | 0.39 | 9.97 | 0.08% | - |

CIFG has the highest validation chrF, so it supplies the default translation demonstration and qualitative examples. WMC has the highest observed test BLEU and chrF, but the chrF differences among trained cells are small. Scores and actual outputs show that translation quality remains limited. All selected configurations use learning rate 0.003, batch size 128 and the epoch-12 checkpoint.

## Limitations

This is a single-seed study on a small parallel corpus. Spelling variation, limited context, reference coverage and exposure bias limit translation quality. Compare more seeds, longer training, subword settings and beam search before making general architecture claims. Keep the target script explicit when interpreting results. Review actual model outputs before the oral defense.

## Submission

The English report must remain 4-5 pages. The English slides cover goals, data exploration, modeling, hyperparameter choices, final evaluation and improvements in ten minutes. Deadline: 25 October 2026, including that day. Defense: lab 6. Add the remaining team members, review the results and submit the public repository link through the professor's form.
