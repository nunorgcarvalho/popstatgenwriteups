# popstatgenwriteups

LaTeX writeups deriving and explaining population/statistical-genetics results. See `README.md`
for the writeup index, the build commands, and the citation-key convention. Shared LaTeX
configuration lives in `tex/`, shared references in `bib/references.bib`.

Every agent working in this repository follows the rules below by default. Where they conflict
with a generic instruction (including the thesisMgr worker profile), these win.

## Writing conventions

- **Derive forward.** Build up to each result step by step from the current assumptions and
  approximations; do not state the result first and justify it afterwards.
- **Write multi-step formulas.** Show the intermediate lines of a derivation in an aligned
  display rather than jumping straight to the final expression.
- **Be conservative with new notation.** Reuse the notation already in the writeup and keep its
  style; introduce a new symbol only when there is no reasonable way around it.
- **Validate when it is worth it.** Tools available for checking results include `pathMgr`
  (exact covariances via path tracing), `popstatgensim` (simulation), and any other tools
  available. Use them when the value of the verification exceeds the cost of running it.
- **Respect the do-not-edit markers.** Text between `\toagent{Do not edit below this comment ...}`
  and the next `\toagent{You may edit below this comment again.}` is off limits unless Nuno
  explicitly lifts the restriction for a specific task. (Older files use `\comment{To AI agents:
  ...}` for the same markers.)
- **Notes in the source.** `\todo`: work to be done; carry it out only when asked to.
  `\question`: answer it in the chat, not by editing, whenever you work through the document's
  `\todo` notes. `\toself`: Nuno's own note; needs no answer. `\toagent`: instructions for agents,
  not rendered in the PDF. (`\unsure` is the older name for `\question`.)
- **`\add` / `\remove` only on request.** Do not mark edits with `\add`/`\remove` (or the
  `added`/`removed` environments) unless Nuno asks for it.
- **Math in the doc vs. in the chat.** Inside the `.tex` files, use the writeups' shorthand
  macros (`\E`, `\Cov`, `\Var`, `\VAo`, ...) as usual. In chat replies, write math as real LaTeX
  (`$...$` inline, `$$...$$` display) using only standard LaTeX/amsmath commands — never the
  writeup macros, which the VS Code chat panel cannot render — and never as plain-text or
  Unicode approximations. Spell them out instead: `\mathrm{E}`, `\mathrm{Cov}`, `V_A^{(0)}`.
  The chat's markdown parser can read two underscores in one math block as italic markers and
  delete them (e.g. `D^{(t)}_{kl} + D^{(t)}_{k/l}` lost both `_`), which breaks the rendering.
  To avoid it: put each display equation in its own `$$` block, with the `$$` on their own lines
  and a blank line before and after; keep displays out of lists and inline sentences; and keep
  inline math short, moving any formula with several subscripts into its own display.

## Staging: `git add -A` at the start of every round

Getting a writeup into good shape takes many rounds of back-and-forth — you edit, Nuno revises,
you adjust. Staging, not committing, is what separates one round from the next.

**At the start of every round, before you make a single edit: `git add -A`.**

That sets the baseline. Everything from before this round becomes staged; everything you then
write shows up as *unstaged*. VS Code renders the two separately, so Nuno can see at a glance
exactly which lines you just wrote, as distinct from everything that was already there. Do this
even for a one-line change, and do it even if you are about to run a build that rewrites the PDF.

## Committing: only when told

**This applies to this repository only.** A commit here marks a **large, infrequent grouping of
related edits**; none of the individual rounds is worth its own commit.

**Do not commit unless Nuno asks.** Not at the end of a round, not to checkpoint before a risky
edit, not because a section looks done to you, and not to separate his edits from yours. He
decides when a grouping is worth marking and says so explicitly. When he does ask:

- Commit everything outstanding as one commit, staged and unstaged together, unless he asks for
  it split.
- If this is more work on the same grouping, `git commit --amend` onto its commit rather than
  adding a second one.

Never unstage, never revert, and never amend on your own initiative; staging is a review aid, and
the history is his to shape. Push this repo only when he asks.

## thesisMgr integration

When you are working here as a thesisMgr worker (`~/thesisMgr`, profile `worker.md`):

- Bootstrap and track the work as usual: claim the task, and log each round in the task's work
  log and in your agent log, following thesisMgr's logging conventions.
- A writeup task spans many rounds, so it stays `claimed` across them. The worker profile's
  "commit to the project repo" step is replaced by the committing rule above: commit here only
  when Nuno asks, and mark the task `done` only once he says the grouping is finished.
- Commit messages in this repository carry no thesisMgr task ID. Task-ID messages are for
  thesisMgr commits only, and pushing thesisMgr remains the controller's job.
- Larger validation (for example `popstatgensim` simulation studies) goes in its own task,
  tagged with the project that will run it, rather than inline in the writeup task.
- Never log individual-level data.
