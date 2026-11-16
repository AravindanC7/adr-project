# Paper (step #3): Overleaf ACL project

Final report: 8–10 pages, ACL format, due Fri Dec 4 (AGENTS.md Sections 4 and 14).
`main.tex` has one section per row of the Section 14 table, each with a `% TODO (owner)` note.
`custom.bib` has entries for the works named in AGENTS.md; fields marked `% CHECK` still need verifying.

## Set up Overleaf (Harshini)

1. In Overleaf, open **New Project → Templates**, search the template gallery for **ACL**
   and open the official ACL / ACL Rolling Review template ("Open as Template").
2. Upload `main.tex` and `custom.bib` from this folder (replace the template's `custom.bib`).
   Then either set `main.tex` as the main document (Menu → Main document) or paste its contents
   over the template's main `.tex` file. Keep the template's `acl.sty` and `acl_natbib.bst`.
   Click Recompile and check it builds.
3. Rename the project, e.g. `CSE576 P7 Adaptive Defense Router`.
4. **Share** → invite `achidam8@asu.edu` (Arav) and `uulagan1@asu.edu` (Archana) as **editors**.
5. Paste the Overleaf link into the team tracker and into AGENTS.md (or tell Arav to add it),
   and post it in the group chat.

## Notes

- `\usepackage[final]{acl}` with named authors, since this is a course report.
  Switch to `[review]` if the team wants an anonymous version.
- Author names come from the git commit authors; check spelling and order.
- `\nocite{*}` lists every bib entry so the skeleton compiles; remove it once sections cite sources.
- The Ethics Statement has a `[N]` placeholder for the number of pilot participants.
