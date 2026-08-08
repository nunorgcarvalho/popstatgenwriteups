# popstatgenwriteups

LaTeX writeups deriving and explaining population/statistical-genetics results. See `README.md`
for the writeup index, the build commands, and the citation-key convention. Shared LaTeX
configuration lives in `tex/`, shared references in `bib/references.bib`.

## Committing: the author gets the last word before anything is committed

**This applies to this repository only** — it is about writing prose, not about how other
projects work.

These documents are Nuno's writing. When you edit one, that edit is a *proposal* until he has
read it and made his own corrections on top. A commit should capture the **settled** state of
the document — your edits *and* his revisions to them, together — not your draft on its own.

So, by default:

- **Do not commit when you finish a round of edits.** Report what you changed and leave the
  working tree dirty. He reviews your work as uncommitted changes in VS Code's source-control
  view; that is how he sees, line by line, what you wrote and what he wrote. Committing on
  finishing destroys that view.
- **His follow-up corrections are not a separate commit.** They belong in the same commit as the
  edits they are correcting, because together they are one settled version of the document.
- **Commit at the START of the next round instead.** Before your first edit of a new round,
  check the repo: if the previous round is still uncommitted — your edits plus whatever he
  layered on them — commit all of it as one commit, then begin. Everything left uncommitted at
  the end of the new round is then exactly the new work.

This is a default, not a prohibition. **If he asks you to commit, commit** — including right
after making a change. Read the request rather than the rule.

Unaffected: commit messages here carry no thesisMgr task ID, and pushing this repo is still
his call, not yours.
