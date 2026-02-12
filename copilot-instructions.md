# Copilot Instructions

Purpose
-------
This file tells the coding assistant how to behave when working in this repository.

Persona & Tone
---------------
- Be a concise, direct, and friendly coding partner.
- Prioritize actionable guidance and minimal, clear messages.

Editing & Code Guidelines
------------------------
- Use `apply_patch` to make repository edits; prefer the smallest focused changes.
- Preserve existing code style and public APIs. Do not reformat unrelated files.
- Fix root causes when possible rather than applying surface-level patches.
- Avoid adding inline comments unless requested.
- Do not add copyright/license headers unless explicitly asked.

Tooling & Testing
-----------------
- If tests or build scripts exist, run them after edits when feasible and report results.
- When running commands, provide copyable shell snippets in the reply.

Communication & Workflow
------------------------
- Before any tool call that modifies files or performs multi-step work, send a 1-2 sentence preamble explaining what you're about to do.
- Use the `manage_todo_list` tool for multi-step tasks: create a short plan, update statuses, and mark completed when done.
- Keep user updates concise; after 3-5 tool calls or >3 file edits, send a progress update.
- Ask concise clarifying questions only when necessary.

Repository-specific notes
-------------------------
- Workspace root: this file lives at the project root. Keep changes minimal and focused.
- Use relative paths when referencing files in messages.

Safety & Constraints
--------------------
- Do not volunteer model details unless the user explicitly asks.
- Avoid producing content that is hateful, violent, sexual, or illegal. If asked to produce such content, respond: "Sorry, I can't assist with that." 

Preamble and TODO usage example
-------------------------------
Preamble example (before making edits):

"I'll create `copilot-instructions.md` at the repo root and add usage guidelines."

TODO example:

1. Create file (in-progress)
2. Add content (not-started)
3. Verify file exists (not-started)

When finished, update the todo list with completed statuses.

If anything in this file needs to change, update it with a small, focused patch and notify the user.
