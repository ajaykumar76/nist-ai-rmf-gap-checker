# NIST AI RMF Governance Gap Checker (and why a small local LLM should not certify compliance)

Given an organisation's AI governance document, this tool checks 24 NIST AI RMF 1.0 subcategories
and reports, for each one, whether the document **addresses** it, covers it only **partially**, or
leaves it **absent**, with a quoted piece of evidence. It runs fully locally (sentence-transformers +
Ollama `llama3.2`).

> **Short version of the result:** on this benchmark a 3B local LLM was **no better than a plain
> similarity threshold** at judging whether a policy is concrete, and it called real gaps
> "addressed" most of the time. The project's value is the evaluation and the design lesson:
> the safe use of this tool is **evidence retrieval with a human confirming every "addressed"
> claim**, not automatic compliance scoring.

> **Not an official NIST product and not compliance advice.** The subcategory summaries in
> `data/subcategories.json` are my paraphrases of a 24-item subset of NIST AI RMF 1.0 (NIST AI 100-1).
> Verify against the NIST AI Resource Center. Ids map as GV = GOVERN, MP = MAP, MS = MEASURE, MG = MANAGE
> (for example `MS-2.11` is MEASURE 2.11, fairness and bias).

## The question

The dangerous error in a governance review is **false assurance**: calling a real gap "addressed".
So the evaluation reports that error separately from overall accuracy.

## How it works

1. **Retrieve** the 3 most similar sentences in the document for each subcategory (`all-MiniLM-L6-v2`).
2. **Judge** with a local LLM: addressed / partial / absent, with a verbatim quote as evidence.
3. **Check** each answer. Flags that send an item to a human instead of auto-deciding:
   - `evidence_not_verbatim`: the claimed evidence is not an exact quote from the document
   - `absent_but_strong_hit`: LLM says absent although a sentence matches strongly
   - `addressed_weak_hit`: LLM says addressed although the best match is weak
   - `addressed_needs_confirmation` (policy, added after the first dev run): **every "addressed" claim goes to a human**

## Data (synthetic, and what that means)

`src/00_build_dataset.py` generates 12 fictional company governance documents (8 dev, 4 test; 288 labelled
subcategory items). **Gold labels are assigned by construction**: I choose the status first and then
insert a matching sentence (or none), plus unrelated distractor sentences. Labels are therefore certain,
but the task is cleaner than real policies. Test documents use **held-out sentence variants and
distractors** that never appear in dev (I found and fixed a leak in my first draft where 91 of 114 test
sentences also appeared in dev). Everything is reproducible with the fixed seed.

## Results

CIs are 95% Wilson intervals. Items within one document are correlated, so these intervals are
**optimistically narrow**.

### Dev (8 documents, 192 items), three rounds of tuning

| | A. Retrieval only | LLM prompt v1 | v1 + human confirms "addressed" | LLM prompt v3 |
|---|---|---|---|---|
| LLM alone: accuracy | 60% (53-67%) | 54% (47-61%) | 54% | 40% (33-47%) |
| Real gaps called "addressed" | 55% | 60% | 60% | 30% |
| Addressed items called a gap | 6% | 11% | 11% | 57% |
| Verbatim quotes | n/a | 90% | 90% | 64% |
| Auto-decided | n/a | 64% | 15% | 41% |
| Accuracy on auto-decided | n/a | 67% | 71% | 37% |
| False assurance among auto-decided | n/a | 54% | 0 (by policy) | 0 (by policy) |

- **v1**: one holistic label per item. Over-credits vague statements.
- **v3**: four yes/no questions (relevant, owner named, concrete process, schedule or tool), with the status
  derived by code. It halved false assurance (60% to 30%) but over-corrected: it called 57% of real,
  concrete practices gaps, and copied fewer exact quotes (cause not investigated). Kept as a failed experiment.
- **Frozen for the test run:** v1 prompt + the "addressed needs a human" policy, chosen *before* seeing test
  results by this rule: auto-decided accuracy above the retrieval baseline, and no silent false assurance.

### Test (4 documents, 96 items), run once

| | A. Retrieval only | B. LLM alone | C. LLM + checks |
|---|---|---|---|
| Accuracy | 54% (44-64%) | 53% (43-63%) | 5/8 = 62% on auto-decided (31-86%) |
| Real gaps called "addressed" | 33/46 = 72% (57-83%) | 34/46 = 74% (60-84%) | 0/7 auto-decided (0-35%) |
| Addressed items called a gap | 10% | 10% | n/a |
| Exact quotes | n/a | 72/86 = 84% (75-90%) | n/a |
| Auto-decided | n/a | n/a | **8/96 (8%)**, 88 sent to a human |
| LLM errors caught by a flag | n/a | n/a | 42/45 (93%) |

## What I take from this

- **The LLM did not beat retrieval.** 53% vs 54% on test, 54% vs 60% on dev. A 3B model cannot reliably tell
  "Legal keeps a register reviewed every quarter" from "we aim to stay aware of regulation".
- **Most real gaps were reported as covered** (72-74% on test, by both systems). A tool that auto-scored
  compliance this way would give false comfort.
- **The 0% false assurance is a policy result, not a model result.** Sending every "addressed" claim to a human
  makes silent false assurance impossible by construction; the cost is that 92% of items need review.
  I report it that way on purpose.
- **Some genuine value remains:** it surfaces the best evidence sentence per subcategory, and the verbatim-quote
  check catches fabricated evidence (16% of the LLM's claims on test were not exact quotes).
- **Next experiment (not done):** a larger model, and real policy documents. I ran on a laptop with limited disk.

## Limitations

- Synthetic documents with labels set by construction; real policies are messier and ambiguous.
- Partial statements in my data are usually marked by hedging language, which may make "partial" easier to
  spot than in real documents (I kept hedge-word lists out of the prompt for this reason).
- Only 24 of the framework's 72 subcategories; paraphrased wording, written by a non-expert.
- Small test set (4 documents); correlated items; one model, one embedding model.
- Tuned over three rounds on dev; the final version was fixed before the single test run.

## Run it

```powershell
pip install -r requirements.txt
ollama pull llama3.2
python src\00_build_dataset.py                                   # writes data\*.json
python src\01_gapcheck_eval.py --split dev --no-llm              # retrieval baseline only
python src\01_gapcheck_eval.py --split dev --tag v1 --prompt v1  # LLM, dev
python src\01_gapcheck_eval.py --split test --tag final --prompt v1   # run once, at the end
```

LLM answers are cached in `reports/llm_cache_*.jsonl`, so an interrupted run resumes.

## Files

```
src/00_build_dataset.py   generates the synthetic documents and labels (seeded)
src/01_gapcheck_eval.py   retrieval, LLM judging, checks, evaluation
data/                     subcategories.json (24, paraphrased), docs.json (12 documents)
reports/                  saved results for each run
```
