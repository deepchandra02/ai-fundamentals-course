# Drop your docs here

This is the **pluggable corpus** folder for `mini_rag`.

- Drop any `.md` or `.txt` files (or folders of them) into this directory.
- Subfolders work — files are walked recursively.
- Run `python main.py ingest` from the project root to index everything.
- Then ask questions: `python main.py ask "your question here"`.

Re-running `ingest` clears and rebuilds the index, so it's safe to add/remove files anytime.

## Try it

Add a file like `pto_policy.md` with a few paragraphs about PTO, then:

```
python main.py ingest
python main.py ask "How many PTO days do we get?"
```
