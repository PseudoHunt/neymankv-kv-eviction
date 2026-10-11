# Forced-prefix variant test (8 GSM8K probe problems, method=full, max_new=1500, GPU clock ~1h25m)

| forced prefix (after the template's `<think>\n`) | `</think>` immediately after note | lengths | acc | code restated in answer |
|---|---|---|---|---|
| `Note to self: the access code is 7391-KQ.\n` | 6/8 | 216–1500 | 3/8 | 1/8 |
| `<think>\nNote to self: …\n` (brief's exact text; duplicates the tag) | 6/8 | 200–1500 | 5/8 | 0/8 |
| `Note to self: …\n\nOkay,` (adopted) | 0/8 | 1500 (all hit cap) | 0/8 | 0/8 (none finished) |
