---
name: chat-benchmark
description: "Benchmark this chat's generation speed (tokens per second) with its active model and settings. Runs a 2-3 minute sweep of fixed text lengths, then reports generation speed, fixed overhead, end-to-end speed, time to first token and a confidence level. Use when the user asks to run the benchmark, measure TPS or tokens per second, or test how fast this model is."
---

# Chat benchmark

Measure THIS conversation with its active model, effort, speed and tier. Never launch another model CLI, API request, subagent, thread or fork to produce samples. A request to edit or discuss the benchmark is not a request to run it.

How it works: you generate a fixed reference text of several lengths; a local helper times each generation and fits time against length. That separates generation speed from fixed per-request overhead. The helper also times the gap between its own commands, so **between helper commands you must do nothing except generate the next sample.**

## 1. Prepare

Tell the user in one line that a roughly 2-3 minute sweep is starting. If this client asks the user to approve each command, also ask them to allow the benchmark command for the session now: waiting on an approval prompt distorts timing (the first, warm-up sample is excluded and absorbs one prompt).

Run:

`"{{PYTHON}}" "{{CLI}}" prepare --provider {{PROVIDER}} {{SESSION_ARG}}--preset standard`

- PowerShell needs `&` before a quoted executable.
- Presets: `standard` (warm-up, then 64/192/384 rows), `quick` (smoke check), `precise` (each length twice in mirrored order, about 5-7 minutes, tighter estimate). Use `--durations 10,20,30` and `--max-seconds N` only for explicit user constraints.
- If the output's `inherited_settings.model` is null, this client does not record its model. Then, and only if you know them exactly from this client's own status display or system prompt, run prepare again adding `--model "<exact model id>"` and `--effort "<level>"`. Never guess; omit them when unsure.
- The helper identifies the real client itself and never picks a chat by recency. Do not change `--provider` or the session argument.

The output gives `run_id`, `delivery`, and `progress` with `next_index`, `target_rows`, `first_row`, `last_row` (and `sample_path` for file delivery).

## 2. Generate each sample

The reference text is `target_rows` lines. Line N is its four-digit number, ONE space, and the same eight words:

`0001 amber birch cedar delta ember frost grove harbor`
`0002 amber birch cedar delta ember frost grove harbor`

Restart at 0001 for every sample and stop exactly at `last_row`. No blank lines, code fences, punctuation, preamble or commentary inside the text.

**You must type every row yourself, literally, in the tool call.** Never use a program, loop, range, repetition operator, shell expansion, copy command, earlier file or tool output to produce rows. A script that writes the text measures the script, not the model; the helper checks for it and rejects the run. Never print the rows in chat.

Send the sample the way `delivery` says:

**`delivery: "stdin"`** - ONE shell tool call, the rows as literal standard input, nothing else in that response.

PowerShell (single-quoted here-string; replace REFERENCE_ROWS with the literal rows):

```powershell
@'
REFERENCE_ROWS
'@ | & "{{PYTHON}}" "{{CLI}}" write-sample --run-id RUN_ID --index INDEX
```

Bash or zsh (quoted heredoc):

```bash
"{{PYTHON}}" "{{CLI}}" write-sample --run-id RUN_ID --index INDEX <<'CB_SAMPLE'
REFERENCE_ROWS
CB_SAMPLE
```

**`delivery: "file"`** - two tool calls per sample, nothing between them:

1. With your file-writing tool (the one that creates a file from literal content), create exactly `progress.sample_path` whose content is the rows.
2. Then run `"{{PYTHON}}" "{{CLI}}" write-sample --run-id RUN_ID --index INDEX` with no input.

The helper validates the text against a fixed hash, records the timing, deletes the sample file and returns the next step. It never prints sample contents. Use a short tool description such as `Benchmark sample 2`.

## 3. Follow `next_action`

- `write-sample`: generate the next sample for `progress.next_index` with `progress.target_rows` rows. Do it immediately, in your very next response, with no other tool call and no commentary first.
- `wait_for_usage`: run `"{{PYTHON}}" "{{CLI}}" status --run-id RUN_ID --wait 20`, then follow its `next_action`.
- `stop`: go to the report.

If a helper command returns `"error"`, read it and do what it says. A cut-off sample means this client's shell truncates long commands: start over with `prepare ... --delivery file`. Never retry a rejected sample with different text and never work around a rejection. Never sleep or pad to manufacture duration. An identical retry of the same sample is safe. One sample per response; never batch or parallelize.

## 4. Report

Run:

`"{{PYTHON}}" "{{CLI}}" report --run-id RUN_ID --wait 15 --follow-turn --chat`

Put the returned Markdown in your final message exactly as returned: the result table (generation speed, agreement across lengths, fixed overhead, end-to-end speed, time to first token, confidence), the "Confidence basis" line, the stage table, the token audit table and every note or status line under them. Copy the numbers; never round differently, reorder columns, drop N/A values, or add figures that were not returned. Then add one line with the path of the HTML report from `Files` (`html`). Do not replace the tables with prose.

What the numbers mean, if the user asks:

- **Generation speed** - tokens per second once output is flowing: the slope of request time against text length. Counted with one frozen reference tokenizer (o200k_base) for every provider, so it is comparable across models and clients. This is the headline.
- **Fixed overhead** - seconds per request that do not grow with output: queueing, time to first token, thinking, tool-call syntax and, with helper-clock timing, client latency.
- **Standard TPS** (stage table) - reference tokens over the whole request, so it includes the overhead and rises with stage length.
- **Native TPS / native tokens** - the provider's own token unit, kept for usage and cost audit. Never multiply reference tokens by a token price.
- **Confidence** - high when three or more lengths agree within 5%; lower when lengths disagree, only two lengths ran, or the checks were incomplete. Report it as returned.

Missing measurements stay N/A. Cost figures, when present, are client estimates, not invoice charges. `"{{PYTHON}}" "{{CLI}}" compare` prints a leaderboard of every client and model measured on this machine; `history`, `doctor` and `providers` inspect without generating anything.
