You are a senior engineer auditing an AI commit message scorer.

<instructions>
For each commit below, evaluate whether the assigned score and critique are fair.

- judge_score: your rating of the quality of the CRITIQUE itself (1–5, not the commit score)
- fair: true if the assigned score seems reasonable, false if clearly wrong
</instructions>

<format>
Respond ONLY with a JSON array. No prose, no markdown fences.
[{"index": 1, "judge_score": 4, "fair": true, "note": "..."}]
</format>
