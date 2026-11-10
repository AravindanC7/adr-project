# Human pilot (#30 materials → #35 pilot → #39 cleaning)

Owner: Harshini. Purpose: measure how often the hidden checks (decoy, canary) and the
behaviour check fire on **honest people**, the human false-positive rates in the failure-mode
map (AGENTS.md sections 2.2 and 6.4, step #40). Rows go in with `data_origin = "pilot"`,
`respondent_kind = "human"`.

| File | What |
|---|---|
| [consent.md](consent.md) | Consent screen, shown before question 1 (served by the server) |
| [debrief.md](debrief.md) | Debrief screen, shown after the last question (served only once finished) |
| [recruitment.md](recruitment.md) | Group message, direct message, link message, reminder |
| This file | Design decisions and the session script (checklist) |

## Design (decided Nov 3)

- **Who:** classmates aged 18 or older. Approved by Prof. Gupta in class (#24); on Nov 10 he
  approved any number of participants, so aim for 30-50 (more people narrows the human
  false-flag rates, especially for the canary and behaviour checks).
- **Format:** remote. Each person opens a link on their own laptop, any time Nov 9–13.
  Tab switches and pastes then look like real life.
- **Disclosure:** partial disclosure, then a full debrief with the option to remove answers
  (see consent.md for why).
- **Consent:** in the site, not a separate form. The server refuses to start a pilot session
  without it, and stores only `consented = true` with a random session id. No names, emails,
  IP addresses or ASU IDs.
- **Questions:** one pilot assessment of about 20–25 real items (MMLU/ARC quiz items and
  survey items from the loaders), about 15 minutes (a teammate dry run took 7). Every eligible item carries its check
  (`ADR_PICK=all`). Until #3 merges, the stub bank stands in.
- **Withdrawal:** the Stop button (any time) and "Remove my answers" (debrief) delete the
  session on the server. Closing the tab mid-way leaves a partial session; #39 drops those.

## Running it

Hosting: **an ngrok tunnel from Harshini's laptop** (decided Nov 5), so the answers never leave
the laptop except through the tunnel, and the free static domain keeps the link the same after
any restart. One command starts the server and the tunnel:

```powershell
# from the repo folder, in a PowerShell window of your own (not one Claude Code started)
Set-ExecutionPolicy -Scope Process Bypass   # this window only; changes no system setting
.\scripts\run_pilot.ps1 -Domain delicious-huddling-generic.ngrok-free.dev
# a teammate dry run instead: add  -DataDir dry_run_data
```

It sets `ADR_PILOT=1 ADR_DATA_DIR=pilot_data ADR_ITEMS=server/pilot_items.json`, listens on
127.0.0.1 only (nothing but ngrok can reach it), turns off uvicorn's access log (behind a
tunnel it would print every visitor's real IP address) and starts ngrok with
`--inspect=false` (otherwise its inspector at 127.0.0.1:4040 keeps every request in full,
answers and IPs included). It refuses to start if the consent or debrief text still has a
`<PLACEHOLDER>`, or if the data folder is inside OneDrive, Dropbox or Google Drive: those keep
deleted files in their recycle bin and history, so "Remove my answers" wouldn't really remove
them. The repo is on `D:\Code`, which isn't synced.

The laptop must stay **on, plugged in, awake and online** Nov 9–13: set Windows to never sleep
when plugged in, and turn off automatic restarts for updates for that week. If it does restart,
run the command again; the link is the same and saved answers are kept. Anyone mid-session at
that moment has to start again (their partial file is dropped in #39).

Free ngrok shows visitors a one-time "You are about to visit" page; the link message tells
them to click **Visit Site**.

`ADR_PILOT=1` turns on the consent and debrief screens and the Stop button. Never combine it
with `ADR_DEBUG_STATE=1` or `ADR_STUB_MECHANISMS=1`.

`ADR_DATA_DIR` is required in pilot mode: each session is saved to
`pilot_data/<random id>.jsonl` as it happens (`server/store.py`), so a restart loses nothing
already answered. Stop and "Remove my answers" delete the file. A file with no `finish` line
is a partial session; #39 drops it. **Never commit `pilot_data/`**, and delete it in December
2026 as the consent screen promises. The daily check, in a second PowerShell window (counts
and timings only, never answers):

```powershell
.venv\Scripts\python.exe -m server.store status
```

It prints finished sessions, ones in progress, abandoned ones (unfinished and idle over an
hour), the median time taken, and how many started each day. Without Python, the finished count
alone: `(Get-ChildItem pilot_data -Filter *.jsonl | Select-String -List '"finish"').Count`

No clock times are saved: the start record has the date only and later records have seconds
since the session started, because with about ten classmates a time of day could identify
someone. The start record also has `consent_version`, a hash of the consent text that person
agreed to.

**Removing someone later.** The done screen shows each person their random session code and
asks them to email it if they want their answers removed. Then:

```powershell
.venv\Scripts\python.exe -m server.store remove <code>
```

## Session script (remote checklist)

**By Sun Nov 8**
- [ ] #3 merged, pilot bank built (`server/pilot_items.json`, 20–25 items), and the whole
      session timed by a teammate at ≤ 25 min
- [x] Hosting decided and tested from a phone on mobile data (Nov 5)
- [ ] Group message sent; sign-up list kept separately from any data
- [ ] Teammate dry run: Arav and Archana each do one full session; then delete those sessions

**Mon Nov 9**
- [ ] Server started with `ADR_PILOT=1`; open the link once yourself, check that consent
      shows, then Stop to remove your own test session
- [ ] Link message sent

**Each day, Nov 9–13**
- [ ] Check the server is up; count finished sessions (no names)
- [ ] Answer questions from participants by DM without explaining the hidden text before they finish

**Thu Nov 12**: reminder to everyone on the sign-up list.

**Fri Nov 13 evening**: close the pilot, stop the server and copy the data for #39.
Tell the group the pilot is over, so the hidden text is no longer a secret.

**Before #39 uses the data**: read through the open-text answers for names, emails or anything
else that identifies someone, and blank them out. Nothing leaves the laptop before that.

**#39, by Sun Nov 15**: turn the finished sessions into outcome-table rows for Archana (#40):

```powershell
.venv\Scripts\python.exe -m server.clean --run-id pilot-2026-11-13
```

It writes `results/outcomes/pilot-2026-11-13.parquet` and `results/runs/pilot-2026-11-13/config.yaml`
(sessions used and dropped, rows per check), and lists any open-text answer that looks like it
holds an email, phone number, URL or name, for you to check first. `--exclude <code>` drops a
session (e.g. someone who said they used AI). The table holds outcomes only, never answer text.
The reverse-keyed IPIP "contradiction" rule is provisional until Archana confirms it.

**December 2026, after final grades**: delete `pilot_data/`, every copy of it, and the #39
outcome rows built from it. Only the totals printed in the report are kept (consent.md says so).

**If something goes wrong**
- Someone used AI anyway, or says so: ask for their code from the done screen and remove their
  session (it isn't honest-human data). Without the code we can't tell which file is theirs.
- Someone asks during the session "is this checking for AI?": say "I'll explain everything
  on the last screen" and nothing more.
- Server down: post in the group and extend by a day if needed. Never ask anyone to redo
  a session they already finished (their second try isn't naive).

## Open items (for #35)

- **Hosting:** done, ngrok from Harshini's laptop (see Running it); tested from a phone on
  mobile data on Nov 5.
- **Consent wording:** approved by Prof. Gupta (Nov 5); no second contact needed.
- **Saving data:** done (`ADR_DATA_DIR`, see Running it).
- **ngrok sees visitors' IP addresses** as part of carrying the traffic (we never store them).
  The consent screen says *we* don't record IPs, which stays true; tell the group if asked.
- **Pilot bank:** build from the loaders once #3 merges.
