# popstatgenwriteups

LaTeX writeups deriving and explaining population/statistical-genetics results. See `README.md`
for the writeup index, the build commands, and the citation-key convention. Shared LaTeX
configuration lives in `tex/`, shared references in `bib/references.bib`.

## Committing: stage every round, commit only when told

**This applies to this repository only** — it is about writing prose, not about how other
projects work.

A commit here should mark a **finished section**, not a round of editing. Getting a section into
good shape takes many rounds of back-and-forth — you edit, Nuno revises, you adjust — and none of
those rounds is worth its own commit. Staging, not committing, is what separates one round from
the next.

**At the start of every round, before you make a single edit: `git add -A`.**

That sets the baseline. Everything from before this round becomes staged; everything you then
write shows up as *unstaged*. VS Code renders the two separately, so Nuno can see at a glance
exactly which lines you just wrote, as distinct from everything that was already there. Do this
even for a one-line change, and do it even if you are about to run a build that rewrites the PDF.

**Do not commit unless Nuno asks.** Not at the end of a round, not to checkpoint before a risky
edit, not because a section looks done to you. He decides when a section is finished and says so
explicitly. When he does ask:

- Commit everything outstanding as one commit, staged and unstaged together, unless he asks for
  it split.
- If the section already has a commit and this is more work on the same section, `git commit
  --amend` onto it rather than adding a second commit — one finished section, one commit.

Never unstage, never revert, and never amend on your own initiative; staging is a review aid, and
the history is his to shape. Commit messages here carry no thesisMgr task ID, and pushing this
repo is his call, not yours.
