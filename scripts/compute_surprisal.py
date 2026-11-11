"""Compute the Surprisal of a crossword answer given its clue with a causal language model.

    python scripts/compute_surprisal.py clues.csv out.csv --model meta-llama/Meta-Llama-3-8B

The input CSV needs a `clue` and a `solution` column. Each pair is turned into a well-formed
sentence by a concatenation rule (`--concatenation`), and the script reports:

  answer_surprisal      −Σ log P(answer token | everything before it), usable when the answer
                        comes last in the sentence;
  surprisal_difference  the sentence's total surprisal minus the surprisal of the same sentence
                        without the answer, for rules that place the answer first (a causal model
                        cannot condition on tokens that come later).

Scores are in bits (log base 2) by default; pass --natural-log for nats.
"""
import argparse
import math

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# clue/answer -> sentence. "answer_last" rules allow the plain answer surprisal.
CONCATENATIONS = {
    "topic": ("{answer}, {clue}", False),
    "topic_art": ("{answer}, il {clue}", False),
    "subj": ("{clue} è {answer}", True),
    "cop": ("{answer} è {clue}", False),
    "cioe": ("{clue}, cioè {answer}", True),
    "plain": ("{clue} {answer}", True),
}


def token_logprobs(model, tokenizer, text):
    """Log-probability of every token of `text` after the first."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits
    logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = ids[0, 1:]
    return logprobs[range(targets.shape[0]), targets], tokenizer.convert_ids_to_tokens(targets)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="CSV with `clue` and `solution` columns.")
    p.add_argument("output")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--concatenation", default="subj", choices=sorted(CONCATENATIONS))
    p.add_argument("--natural-log", action="store_true", help="Report nats instead of bits.")
    args = p.parse_args()

    template, answer_last = CONCATENATIONS[args.concatenation]
    scale = 1.0 if args.natural_log else 1.0 / math.log(2)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    df = pd.read_csv(args.input)
    rows = []
    for clue, answer in tqdm(zip(df["clue"], df["solution"]), total=len(df), desc=args.concatenation):
        sentence = template.format(clue=clue, answer=answer)
        lp, _ = token_logprobs(model, tokenizer, sentence)
        total = -lp.sum().item() * scale

        without = template.format(clue=clue, answer="").replace("  ", " ").strip(" ,")
        lp_without, _ = token_logprobs(model, tokenizer, without)
        difference = total - (-lp_without.sum().item() * scale)

        answer_surprisal = None
        if answer_last:
            n_answer = len(tokenizer(" " + str(answer), add_special_tokens=False).input_ids)
            answer_surprisal = -lp[-n_answer:].sum().item() * scale

        rows.append({"clue": clue, "solution": answer, "concatenation": sentence,
                     "concatenation_type": args.concatenation,
                     "answer_surprisal": answer_surprisal, "surprisal_difference": difference})

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(out.describe(include="all").loc[["count", "mean"]].to_string())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
