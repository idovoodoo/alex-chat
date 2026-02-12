# Copilot Instructions

## Purpose

This file tells the coding assistant how to behave when working in this repository.

---
## ESSENTIAL
All debug messages should be sent to the web browser console for viewing. Never the server console.

### Debug logging rules (MANDATORY)

- Server logs: only warnings and errors must be emitted to the server terminal/logs. Do not use server-level INFO for debug diagnostics.
- Browser console: ALL debug output belongs in the browser DevTools Console and must be delivered via the in-memory debug payload and `/debug/last_console` endpoint.
- How to emit debug from server code:
  - Build a compact debug payload (JSON) containing the relevant runtime diagnostics (e.g., `LAST_NEW_CHAT_DEBUG`, `LIFE_RECALL_DEBUG`, token summaries, db_summary).
  - Call `_log_debug_to_console(tag)` (or replicate its pattern) to serialize this payload to the global `LAST_DEBUG_CONSOLE` string.
  - Do NOT `logging.info()` or `print()` debug payloads — that writes to the server console which is forbidden for debug output.
- Database / migration messages and unexpected conditions may log warnings/errors with `logging.warning()` / `logging.error()` as appropriate; these are allowed server-side.
- How the frontend must display debug:
  - The client should periodically (or after key actions) fetch `/debug/last_console` and `console.log()` or `console.group()` the returned JSON so developers see full debug state in DevTools.
  - Example client pattern: after `/chat` or `/new_chat` completes, call fetch `/debug/last_console` and `console.group('DEBUG')` / `console.log(json)` / `console.groupEnd()`.

These rules ensure all runtime debugging is visible in the browser console and the server console remains noise-free in production.

## Architecture Overview

This project is a personal AI chatbot that imitates “Alex”.
It uses Render as the server.
It is aimed for mobile use but is respoinsive for web as well.

### Runtime Message Path

UI (`index.html:sendMessage`)  
→ FastAPI (`main.py:chat`)  
→ RAG retrieval (`_retrieve_top_chunks`)  
→ Core memory selection (`_select_memories`)  
→ Conditional life memory search (`_search_life_memories`)  
→ Prompt assembly (`chat` builds `system_prompt` + `messages`)  
→ LLM call (`_openai_chat_completion` or Gemini)  
→ JSON response  
→ UI render (`addMessage` → DOM)

---

### State Locations

- **Session history**: In-memory dict `CONVERSATION_HISTORY`
- **Core memories**: Supabase `memories` table (`type='core'`) + RAM cache
- **Life memories**: Supabase `memories` table (`type='life'`)
- **RAG index**: FAISS index + `chunks.json` (loaded at startup)
- **Frontend chat store**: `localStorage` (client-side only)

---

### Responsibility Separation

- **RAG** = Style examples (tone imitation only).
- **Core memory** = Stable identity facts.
- **Life memory** = Extracted past experiences.
- **Session history** = Immediate conversational continuity.

These systems must remain clearly separated.

---

## Architectural Guardrails

When modifying this repository:

- Do not blur boundaries between:
  - RAG (style)
  - Memory (facts)
  - Session history (context)
- Do not introduce new hidden sources of truth.
- Do not re-ingest assistant-generated output into long-term memory without safeguards.
- Preserve prompt assembly order and injection hierarchy unless explicitly instructed.
- Be cautious when modifying session state (`CONVERSATION_HISTORY`) as it is server-memory only and not persistent.

---

## Persona & Tone

- Be concise, direct, and practical.
- Avoid long explanations unless explicitly requested.
- Prefer clarity over cleverness.

---

## Editing & Code Guidelines

- Use `apply_patch` for repository edits; keep patches small and focused.
- Preserve public APIs and existing behaviour.
- Do not refactor architecture unless explicitly requested.
- Avoid adding inline comments unless asked.
- Do not reformat unrelated code.

---

## Tooling & Testing

- If tests or build scripts exist, run them after edits when feasible.
- Provide copyable shell snippets when suggesting commands.

---

## Communication & Workflow

- Before any multi-step or file-modifying action, send a 1–2 sentence preamble.
- Use `manage_todo_list` for multi-step tasks.
- Keep progress updates concise.
- Ask clarifying questions only when necessary.

---

## Repository-Specific Notes

- Workspace root: this file lives at the project root.
- Use relative paths in explanations.
- Session state is in-memory and resets on server restart.
- Core and life memories are globally scoped unless explicitly filtered.

---

## Safety & Constraints

- Do not volunteer model details unless explicitly asked.
- Avoid producing harmful or illegal content. If asked, respond:

  > "Sorry, I can't assist with that."

---

## Preamble and TODO Usage Example

**Preamble example:**

> "I’ll update the memory retrieval logic in `app/main.py` with a focused patch."

**TODO example:**

1. Update retrieval function (in-progress)
2. Verify imports (not-started)
3. Run tests (not-started)

Mark tasks complete when done.

---

If this file needs changes, update it with a small, focused patch and notify the user.
