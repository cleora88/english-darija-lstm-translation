"""Corpus-level translation scores with reproducible SacreBLEU signatures."""

from sacrebleu.metrics import BLEU, CHRF


def score_translations(hypotheses: list[str], references: list[str]) -> dict:
    bleu = BLEU(tokenize="13a", lowercase=False)
    chrf = CHRF(char_order=6, word_order=0, beta=2)
    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])
    return {
        "bleu": bleu_score.score,
        "chrf": chrf_score.score,
        "exact_match": sum(
            h.strip() == r.strip() for h, r in zip(hypotheses, references)
        )
        / len(references),
        "bleu_signature": str(bleu.get_signature()),
        "chrf_signature": str(chrf.get_signature()),
    }
