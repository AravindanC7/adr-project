# Pilot consent screen (#30)

Owner: Harshini. Shown before question 1 when the server runs with `ADR_PILOT=1`.

The server reads the **On screen** section below and sends it to the page (`GET /study`), so
this file is the one place to edit the wording. Rules for that section: plain text only (no
Markdown formatting; it is shown as typed), one paragraph per block, blank line between blocks.

**Disclosure level (decided Nov 3):** partial disclosure, then a full debrief
([debrief.md](debrief.md)). Before the session we say what we record and why in general terms,
but not that some questions contain hidden text. Telling people about the hidden text first
would change how they answer (some would go looking for it or copy the question to inspect
it), and the pilot exists to measure how often honest people trip those checks by accident.
Everything is explained on the debrief screen, which also lets them remove their answers.

Approval: Prof. Vivek Gupta approved a small classmate pilot in class (step #24).

## On screen

Thanks for helping with our CSE 576 class project. We are studying how people answer online questions: quizzes and short surveys. It takes about 20 minutes.

Please answer on your own, the way you normally would. Don't use AI tools, search engines or help from anyone else. There are no wrong answers on the survey questions.

What we record: your answers, and how you answered. That means time spent on each question and counts such as how many keys you pressed, how many times you pasted and how often you left the page. We never record which keys you pressed or what you pasted.

We don't ask for your name, email or ASU ID. Please don't write anything that identifies you in the text answers. Your answers are stored under a random code and are used only for this class project. They are deleted when the course ends in December 2026, and the report shows only totals across everyone.

Taking part is voluntary and has nothing to do with your grade in any course. You can stop at any time with the Stop button, which removes everything you've answered. At the end we explain the study in full, and you can still remove your answers then.

Questions? Contact Harshini at hbalamu1@asu.edu.
