"""Isolated worker: load the AI detector once, score stdin JSON, print JSON.

Kept as a separate process so the ~0.9GB model RSS never overlaps with the
embedding or ASR models on small machines. Contract:
  stdin : {"model": str, "texts": [str, ...]}
  stdout: {"scores": [0..1, ...]}   (probability of the 'AI-generated' label)
"""
import json
import sys


def main() -> int:
    payload = json.loads(sys.stdin.read())
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(payload["model"])
    model = AutoModelForSequenceClassification.from_pretrained(payload["model"])
    model.eval()

    scores = []
    for text in payload["texts"]:
        enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        scores.append(float(probs[-1]))  # last label = AI-generated
    print(json.dumps({"scores": scores}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
