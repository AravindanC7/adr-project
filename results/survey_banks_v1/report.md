# gpt-mini L0 on archana-survey-written-v1

Rates are k/n with Wilson 95% CIs. Copy-paste: 5 sessions (samples); document upload: one session per PDF. IPIP rows carry the shared reverse-keyed pair rule (`score_reverse_pairs`), so a pair with a neutral answer has no row for its second item.

## survey-ipip x copy_paste

`v1-cp-L0-survey-ipip-s5`, 5 sessions

- opinion_trap / likert: implausible 0/228 = 0.0% [0.0, 1.7], contradiction 18/228 = 7.9% [5.1, 12.1], plausible 210/228 = 92.1% [87.9, 94.9]

## survey-ipip x doc_upload

`v1-du-L0-survey-ipip-q20`, 6 sessions

- opinion_trap / likert: implausible 0/50 = 0.0% [0.0, 7.1], contradiction 0/50 = 0.0% [0.0, 7.1], plausible 50/50 = 100.0% [92.9, 100.0]

## survey-gss x copy_paste

`v1-cp-L0-survey-gss-s5`, 5 sessions

- opinion_trap / mcq: implausible 5/40 = 12.5% [5.5, 26.1], contradiction 0/40 = 0.0% [0.0, 8.8], plausible 35/40 = 87.5% [73.9, 94.5]

## survey-gss x doc_upload

`v1-du-L0-survey-gss`, 2 sessions

- opinion_trap / mcq: implausible 0/8 = 0.0% [0.0, 32.4], contradiction 0/8 = 0.0% [0.0, 32.4], plausible 8/8 = 100.0% [67.6, 100.0]

## assignment-asurre-train x copy_paste

`v1-cp-L0-assignment-asurre-train-s5`, 5 sessions

- canary / open_text: followed_instruction 65/65 = 100.0% [94.4, 100.0], leaked 0/65 = 0.0% [0.0, 5.6], clean 0/65 = 0.0% [0.0, 5.6]
- style / open_text: high 7/130 = 5.4% [2.6, 10.7], mid 86/130 = 66.2% [57.7, 73.7], low 37/130 = 28.5% [21.4, 36.7]

## assignment-asurre-train x doc_upload

`v1-du-L0-assignment-asurre-train`, 2 sessions

- canary / open_text: followed_instruction 13/13 = 100.0% [77.2, 100.0], leaked 0/13 = 0.0% [0.0, 22.8], clean 0/13 = 0.0% [0.0, 22.8]
- style / open_text: high 0/26 = 0.0% [0.0, 12.9], mid 3/26 = 11.5% [4.0, 29.0], low 23/26 = 88.5% [71.0, 96.0]

## Human comparison

- survey-ipip, public human data on the same 50 items: expected implausible rate 1.6%
- survey-gss, public human data on the same 8 items: expected implausible rate 0.6%

**Pilot humans, all sessions (44 sessions)**

- canary / open_text: followed_instruction 0/88 = 0.0% [0.0, 4.2], leaked 0/88 = 0.0% [0.0, 4.2], clean 88/88 = 100.0% [95.8, 100.0]
- opinion_trap / likert: implausible 0/122 = 0.0% [0.0, 3.1], contradiction 18/122 = 14.8% [9.5, 22.1], plausible 104/122 = 85.2% [77.9, 90.5]
- opinion_trap / mcq: implausible 4/264 = 1.5% [0.6, 3.8], contradiction 0/264 = 0.0% [0.0, 1.4], plausible 260/264 = 98.5% [96.2, 99.4]
- style / open_text: high 4/88 = 4.5% [1.8, 11.1], mid 11/88 = 12.5% [7.1, 21.0], low 73/88 = 83.0% [73.8, 89.4]

**Pilot humans, without the 4 low-effort sessions (40 sessions)**

- canary / open_text: followed_instruction 0/80 = 0.0% [0.0, 4.6], leaked 0/80 = 0.0% [0.0, 4.6], clean 80/80 = 100.0% [95.4, 100.0]
- opinion_trap / likert: implausible 0/107 = 0.0% [0.0, 3.5], contradiction 13/107 = 12.1% [7.2, 19.7], plausible 94/107 = 87.9% [80.3, 92.8]
- opinion_trap / mcq: implausible 4/240 = 1.7% [0.6, 4.2], contradiction 0/240 = 0.0% [0.0, 1.6], plausible 236/240 = 98.3% [95.8, 99.4]
- style / open_text: high 4/80 = 5.0% [2.0, 12.2], mid 9/80 = 11.2% [6.0, 20.0], low 67/80 = 83.8% [74.2, 90.3]
