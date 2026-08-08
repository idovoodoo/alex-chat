# Memory System TODO and Recommendations

Created: 2026-08-08

## Current objective

Verify and improve the end-to-end memory pipeline:

1. Load core and life memories from Supabase.
2. Generate embeddings with MiniMax.
3. Retrieve relevant memories for each chat message.
4. Include retrieved memories in the MiniMax chat prompt.
5. Analyse completed chats and save durable new memories when appropriate.

## Todo list

### Immediate: restore and verify functionality

- [x] Wait for the MiniMax embedding rate limit to reset.
- [x] Restart the local server after the rate limit resets.
- [x] Check `/debug/db` and confirm the database connection is active.
- [x] Confirm `core_memory_embeddings_built` is `true`.
- [x] Confirm `life_memory_embeddings_built` is `true`.
- [x] Confirm the life-memory count matches the database count (`55` in the database and embedding cache).
- [ ] Test an explicit direct recall question such as `you remember max?`.
- [x] Check `/debug/last_console` and confirm relevant results appear in `results_preview` (`2` results were found).
- [ ] Confirm the selected memory is included in the prompt sent to MiniMax.
- [ ] Test an ordinary message and confirm unnecessary life-memory recall is skipped.

### Response quality: use retrieved context naturally

- [ ] Treat relevant memory as a reason to expand a yes/no answer, not just as hidden background.
- [ ] Keep replies concise, but require one concrete remembered detail and a natural follow-up when the memory supports it.
- [ ] Verify an experience question such as `have you been skiing?` produces a contextual answer rather than only `yes`.

### Completed-chat memory extraction

- [x] Start a fresh chat and discuss a new durable personal fact.
- [x] Select **New Chat** to trigger analysis of the finished conversation.
- [x] Confirm the extraction call completes successfully.
- [x] Confirm duplicate checking runs for the extracted memory.
- [x] Confirm a new memory is inserted into Supabase (`db_insert_count = 1`; verify `type = 'life'` separately if not already shown by diagnostics).
- [ ] Confirm ambiguous memories containing unclear `we` references request clarification.
- [ ] Confirm the life-memory embedding cache rebuilds after new memories are saved (the diagnostic currently shows `55`; verify it becomes `56` after the insert or restart).
- [ ] Start another chat and verify the newly saved memory can be recalled.

### Latest verification notes (2026-08-08)

- MiniMax configuration is available: `MiniMax-M3`; embedding configuration is `text-embedding-3-small`.
- Supabase is reachable through the PostgreSQL pooler; `db_conn_alive = true` and `db_last_error = null`.
- Core cache: `35` database memories and `35` embeddings.
- Life cache: `55` database memories and `55` embeddings; no embedding error was reported.
- Completed-chat extraction saved one new life-memory candidate: `Steve wants to create games in Alex's name as an ongoing tribute project.`
- Duplicate checking completed with `0` duplicates skipped.
- The latest indirect query returned `2` life-memory results, but `recall_triggered = false`; explicit-trigger behavior still needs verification.
- Retrieval issue identified: nationality questions such as `where are you from` may not be semantically close enough to stored facts such as `Alex is Norwegian.`; the model can otherwise infer nationality from the England setting.
- Added an identity-question safeguard in `app/main.py` so nationality/origin questions prioritise matching identity facts from core memory, plus prompt-injection diagnostics in `/debug/last_console`.
- Added a prompt guardrail preventing nationality/origin inference from current location, conversation setting, or language.
- Follow-up test showed the first safeguard was incomplete: `_select_memories()` returned early when embedding similarity was below the normal threshold, so the identity fact never reached the prompt. Identity matches now bypass that early return, and diagnostics expose `identity_question` and `identity_memory_selected`.
- Generalised the safeguard beyond nationality: fact-shaped questions now receive a conservative lower-threshold core-memory fallback, limited to three results, while ordinary messages retain the normal threshold. `/debug/last_console` now includes `CORE_MEMORY_DEBUG` with the question type, similarity, gate, and selected previews.
- Changed life-memory retrieval to run a semantic search for every chat message. Explicit recall or past-event questions use the broader recall threshold; ordinary messages use the stricter contextual threshold and smaller result limit. Retrieval mode is included in `/debug/db` diagnostics.

### Reliability and rate-limit handling

- [ ] Add retry/backoff handling for transient MiniMax rate-limit responses.
- [ ] Prevent repeated cache rebuild attempts during the same rate-limit window.
- [ ] Cache the last successful embedding index and continue using it when a refresh fails.
- [ ] Avoid rebuilding the entire life-memory index when only one memory is added.
- [ ] Add batching limits if the MiniMax endpoint has a maximum number of texts per request.
- [ ] Add a startup health status that distinguishes database failure, embedding failure, and empty memory tables.

## Recommendations for improvements

### 1. Keep the last good memory index

Currently, a failed rebuild can leave the memory index unavailable. A safer approach is:

- Build a new index separately.
- Replace the active index only after the entire build succeeds.
- Keep the previous index if the new build fails.

This prevents a temporary MiniMax error from disabling memory retrieval completely.

### 2. Add a retry policy with backoff

For HTTP 429 or MiniMax rate-limit responses:

- Retry only a small number of times.
- Use exponential backoff.
- Stop retrying for a cooldown period after repeated failures.
- Do not retry every chat request while the service is rate-limited.

### 3. Batch embeddings safely

The implementation now uses batched embedding requests. Add a configurable batch size in case the API limits request size, for example:

- 25–100 memories per request
- Combine the resulting vectors into one local matrix
- Record which batch failed

### 4. Persist embeddings or use a vector database

Embedding all memories during every server restart is expensive and slow. Consider storing embeddings in Supabase or using a vector extension such as `pgvector`.

Benefits:

- Faster startup
- No repeated embedding charges
- Immediate database-backed similarity search
- Better scaling as memory count grows

### 5. Improve retrieval diagnostics

Record, for each query:

- Number of memories available
- Top similarity score
- Similarity threshold
- Selected memory IDs or safe previews
- Whether retrieval was core, life-recall, or contextual
- Whether the selected memories were added to the prompt

Do not log API keys or complete private conversations in production logs.

### 6. Make retrieval less dependent on trigger words

The current life-memory search is partly controlled by words such as `remember`, `when`, and `trip`. This can miss indirect questions.

A better approach is:

- Always run a lightweight semantic search.
- Use stricter thresholds for ordinary messages.
- Use a larger result limit for explicit recall questions.

Status: implemented in `app/main.py`. Verify with an indirect question that does
not contain a recall trigger, then inspect `/debug/db` and `/debug/last_console`.

### 7. Use memory IDs and metadata

Instead of caching only text, retain:

- Database ID
- Memory type
- Creation date
- Source chat/session
- Embedding model and version
- Last embedding timestamp

This makes updates, deletion, debugging, and migrations safer.

### 8. Separate extraction from saving

For completed chats, use a staged workflow:

1. Extract candidate memories.
2. Validate and normalise them.
3. Present candidates for review where appropriate.
4. Check duplicates.
5. Save approved memories.
6. Rebuild or incrementally update the index.

This reduces the risk of storing incorrect or ambiguous facts.

### 9. Avoid exposing model reasoning as the reply

The current chat completion code falls back to `reasoning_content` if `content` is empty. Prefer returning only normal assistant content. If the model returns no content, return a controlled error or retry rather than exposing internal reasoning output.

### 10. Add automated tests

Add tests covering:

- Successful batched embedding response parsing
- MiniMax error response parsing
- Rate-limit handling
- Empty memory tables
- Similarity threshold behaviour
- Core-memory retrieval
- Life-memory retrieval
- Duplicate detection
- Completed-chat extraction and saving
- Cache preservation after a failed rebuild

## Definition of done

The memory system is considered working when all of the following are true:

- [ ] The database contains the expected core and life memory counts.
- [ ] Both embedding indexes build successfully.
- [ ] A known memory is retrieved for a matching question.
- [ ] The retrieved memory is included in the MiniMax prompt.
- [ ] The chatbot uses the memory without inventing unsupported details.
- [ ] A finished chat can create a new durable memory.
- [ ] The new memory can be retrieved in a later chat.
- [ ] Temporary MiniMax rate limits do not permanently disable retrieval.
