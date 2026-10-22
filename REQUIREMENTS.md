# Professor's requirements and project evidence

Source: Guidelines_RNN_2627.pdf, Advanced Natural Language Processing, 2026/2027.

| Requirement | Translation implementation / evidence |
|---|---|
| Choose one listed task and dataset | English to Darija, `imomayiz/darija-english`, sentence pairs |
| Vanilla LSTM baseline | Manual vanilla cells in both encoder and decoder |
| Two different non-vanilla appendix cells | CIFG and WMC cells in encoder and decoder |
| Use `nn.Linear` for cell affine operations, no `nn.LSTM` | `cells.py`, checked by `test_models.py` |
| Compare hyperparameters | Learning rate 0.001 versus 0.003 at batch 128; batch 128 versus 256 at learning rate 0.003 |
| Same train/validation/test partitions and evaluation protocol | Frozen grouped splits, one shared tokenizer, same greedy decoding and SacreBLEU metrics |
| Report trainable parameter counts | Experiment records, report and slides |
| Explain dataset exploration and design choices | Missing-English audit, lengths, deduplication, BPE, attention, training settings |
| Evaluate final performance and discuss improvements | Test BLEU/chrF, copy baseline, actual examples and limitations |
| Short report of 4-5 pages | Five-page English report in `output/` |
| Ten-minute oral presentation in English, with slides | English slide deck and timed speaker notes in `output/` |
| Public GitHub repository containing PyTorch code | Published repository linked in README after verification |
| Five or six students from the same lab group | **Pending: four supplied names currently listed, TP B. Add one or two members or obtain the professor's exception.** |
| Submission by 25 October 2026, including that day | Student submits through https://forms.cloud.microsoft/r/78AEQ7pj2d |
| Defense during lab 6 | Team prepares and rehearses the presentation |

The submission form is not sent automatically. Training completion does not resolve the team-size requirement or replace the team's review and oral preparation.
