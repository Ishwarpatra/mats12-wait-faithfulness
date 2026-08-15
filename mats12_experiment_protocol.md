# Experimental Protocol: Does “Wait” Signal Genuine Self-Correction?

## Research question

When an open reasoning model emits a marker such as “Wait”, does the marker identify a genuine algorithmic correction of an earlier reasoning state, or is it mainly a discourse convention that rationalizes a decision already made?

The project is intentionally narrower than a general study of chain-of-thought faithfulness. It treats **Wait** as a candidate observable, not as evidence of faithfulness by default.

## Claims and evidence levels

| Claim | Evidence required | What the pilot can establish |
|---|---|---|
| Wait is associated with local reversals or corrections | Trace-level annotation and structural metrics | Yes, observationally, on public traces |
| Wait distinguishes correction from generic discourse | Matched lexical controls and hand-coded examples | Partly, if controls are available |
| Wait reflects computation that matters for the final answer | Counterfactual intervention, resampling, or paired generations | Not from static traces alone |
| CoT is faithful in a monitoring-relevant sense | Cue intervention plus articulation/causal tests | Requires target-model generation and careful controls |

## Dataset and model scope

The first empirical pass uses the public OpenThoughts-114k dataset, whose examples contain a problem, DeepSeek-generated reasoning, and a final solution. This is a real dataset rather than a simulated corpus. The pass is explicitly observational. It does not claim to identify hidden activations or causal influence.

The target-model extension is designed for `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, with `Qwen/Qwen2.5-7B-Instruct` as a non-reasoning comparison where available. Because the present sandbox has no GPU and the built-in model catalog does not expose those models, target-specific counterfactual results must be run in a GPU notebook or supplied as rollouts. The write-up must not silently substitute another model.

## Observational analyses

1. **Marker prevalence and position.** Count case-insensitive occurrences of `wait`, `but`, `however`, `actually`, and `let me check`. Report rates per trace and normalized positions within the reasoning text.
2. **Local reversal heuristic.** For every Wait occurrence, inspect a window before and after the marker. Detect whether the text contains explicit reversal language, a changed candidate answer, a corrected calculation, or a newly introduced verification step. This is a weak heuristic and must be labeled as such.
3. **Answer-state consistency.** Compare the first explicit candidate answer, the last candidate before the final answer, and the final answer where extractable. Report whether Wait co-occurs with a state transition and whether the final answer is correct against the dataset’s provided solution.
4. **Lexical-control comparison.** Compare Wait windows with matched windows containing generic discourse markers. Match approximately on trace length, domain, marker position, and local punctuation before interpreting differences.
5. **Random qualitative audit.** Sample examples using a fixed seed from each key category. Manually inspect the raw reasoning and record whether the heuristic is correct, incorrect, or ambiguous. Include representative examples in the write-up, not only successes.

## Target-model causal extension

For a fixed prompt set, sample multiple generations at fixed decoding settings under three conditions: ordinary prompt; a prompt containing an answer cue that should induce a switch; and a prompt with a semantically irrelevant lexical cue. Keep all generation settings and prompt order fixed.

For each prompt, record whether the final answer changes, whether the trace explicitly identifies the cue, whether Wait appears before or after the cue-related switch, and whether deleting or replacing the Wait segment changes the final answer under a continuation/resampling protocol. The key comparison is not Wait frequency but the conditional likelihood that Wait predicts a validated answer-state correction beyond matched lexical controls.

## Baselines and red-team checks

The minimum baselines are: no marker; generic discourse marker; random token-position windows; answer-switch without Wait; and a simple length/position classifier. A stronger result must beat these baselines. Red-team checks include: duplicated or templated phrase effects; math-domain leakage; final-answer extraction errors; traces where “Wait” is quoted or appears in the problem; and cases where the answer is already correct before Wait.

## Reproducibility and time accounting

Save the exact dataset query, row indices, random seed, regexes, extraction rules, and analysis version. Save raw sampled rows separately from derived metrics. The final write-up should state which analyses were completed within the 16-hour task window and which are proposed follow-ups. All headline numbers should be recomputed independently from saved outputs, and at least 30 raw traces should be hand-checked.

## Falsifiable interpretation

Evidence for genuine correction would require a higher rate of independently validated answer-state transitions after Wait than after matched controls, robust across domains and trace lengths, with examples where replacing or removing the Wait segment alters the downstream answer. Evidence for rationalization would include high Wait prevalence without validated state change, similar performance for generic lexical controls, and cue-induced answer switches that are not acknowledged or are explained only after the fact.

## References

1. [Chua, Evans et al., Are DeepSeek R1 and other reasoning models more faithful?](https://arxiv.org/html/2501.08156v4)
2. [DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/html/2501.12948)
3. [OpenThoughts-114k dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
4. [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
