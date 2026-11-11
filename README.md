# Crossword Surprisal

**Surprisal as an objective difficulty metric for Italian crossword clues**

[![Paper](https://img.shields.io/badge/CLiC--it_2025-paper-b31b1b)](https://aclanthology.org/2025.clicit-1.50/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97_Dataset-Crossword--Surprisal--IT-yellow)](https://huggingface.co/datasets/Kamyar-zeinalipour/Crossword-Surprisal-IT)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97_Collection-Crossword_Surprisal-orange)]({{COLLECTION_URL}})
[![License](https://img.shields.io/badge/code-Apache_2.0-blue)](LICENSE)

Crossword difficulty has traditionally been judged by human setters, which leaves automated puzzle
generators without an objective yardstick. This work models difficulty as the **Surprisal** of the
answer given the clue, estimated from the token probabilities of a causal language model, and shows
that it tracks how hard human solvers actually find a clue.

This repository accompanies [*Surprisal and Crossword Clues Difficulty: Evaluating Linguistic
Processing between LLMs and Humans*](https://aclanthology.org/2025.clicit-1.50/) (CLiC-it 2025, [PDF](https://ceur-ws.org/Vol-4112/49_main_long.pdf)).

> **Note on the human study.** The paper also reports a solving experiment with 60 native Italian
> speakers (response times and accuracy per clue). Those human measurements are **not** included
> here; this release covers the clue set and the language-model Surprisal scores only. For the
> human data, please contact the authors at IUSS Pavia.

## What is here

- **`data/`** — Surprisal scores for 159 Italian clues under 21 concatenation rules and 3 surprisal
  variants, for three models: `llama3_8b`, `llama2_7b` and `gpt2_small_ita` (Ita-GPT-2-121M).
- **`scripts/compute_surprisal.py`** — compute Surprisal for your own clue–answer pairs.

```python
import pandas as pd

d = pd.read_csv("data/surprisal_llama3_8b.csv")
print(d[["clue", "solution", "macro_category", "concatenation_type", "surprisal"]].head())
```

| Column | Description |
|---|---|
| `model` | Model the score comes from |
| `item` | Clue id |
| `clue`, `solution` | The Italian clue and its answer |
| `category`, `macro_category` | Syntactic category (20) and macro-category (7): nominal, copular, infinitive, verb_pred, prepositional, adjectival, metalinguistic |
| `n_of_words`, `answer_length` | Length of the clue in words, of the answer in characters |
| `concatenation`, `concatenation_type` | The sentence built from clue and answer, and the rule used |
| `surprisal_type` | `surprisal_lw`, `lw_surprisal` or `surprisal_difference` |
| `surprisal` | The score |

## Measuring Surprisal

For a clue–answer pair, a concatenation rule turns the two into a well-formed Italian sentence.
When the answer comes last, the **answer Surprisal** is the summed negative log-probability of its
tokens given everything before them. When a rule puts the answer first — a topicalized structure —
a causal model cannot condition on later tokens, so the paper uses the **Surprisal difference**
between the full sentence and the same sentence without the answer.

```bash
pip install -r requirements.txt
python scripts/compute_surprisal.py clues.csv out.csv \
  --model meta-llama/Meta-Llama-3-8B --concatenation subj
```

## Findings

- Surprisal correlates negatively with human solving accuracy, most strongly for nominal clues
  (*r* = −0.62).
- The Italian-trained GPT-2 and Llama-3 outperform larger, non-specialised models: language-specific
  training and model scale both matter.
- Predictive strength depends on the syntactic category; choosing the best concatenation rule per
  macro-category adds up to 0.15 to *r*.
- Category-specific Surprisal thresholds separate easy, medium and hard clues, which lets a
  generator target a chosen solver level.

## Citation

```bibtex
@inproceedings{iaquinta2025surprisal,
  title     = {Surprisal and Crossword Clues Difficulty: Evaluating Linguistic Processing between LLMs and Humans},
  author    = {Tommaso Iaquinta and Asya Zanollo and Achille Fusco and Kamyar Zeinalipour and Cristiano Chesi},
  booktitle = {Proceedings of the 11th Italian Conference on Computational Linguistics (CLiC-it 2025)},
  series    = {CEUR Workshop Proceedings},
  volume    = {4112},
  pages     = {512--525},
  year      = {2025},
  url       = {https://aclanthology.org/2025.clicit-1.50/}
}
```

## License

Code: [Apache License 2.0](LICENSE). The clue set and Surprisal scores are released for research use.
