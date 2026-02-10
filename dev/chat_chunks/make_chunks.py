import re
import sys
from pathlib import Path

# Configuration
TARGET_SPEAKER = "Alex"               # person you want to emulate
MAX_LINES_PER_CHUNK = 6                # 3–6 recommended
KEEP_PREV_CONTEXT_LINES = 2            # prefer 1–2 messages before TARGET_SPEAKER

line_re = re.compile(r"^(\d{1,2}/\d{1,2}/\d{4}), (\d{2}:\d{2}) - ([^:]+): (.*)$")


def is_noise(msg: str) -> bool:
    m = msg.strip()
    return (m == "" or m == "<Media omitted>" or m.endswith("<Media omitted>"))


def process_file(input_path: Path, output_path: Path) -> int:
    raw = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Parse messages
    messages = []
    # optional speaker alias mapping (normalize different exports)
    SPEAKER_ALIASES = {
        "Alex Faith (Son)": "Alex",
    }

    for line in raw:
        m = line_re.match(line)
        if not m:
            # Skip non-matching lines (multi-line messages could be handled later)
            continue
        date, time, speaker, msg = m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()
        # normalize speaker names using aliases
        speaker = SPEAKER_ALIASES.get(speaker, speaker)
        if is_noise(msg):
            continue
        messages.append((date, time, speaker, msg))

    # Build chunks that end with one or more TARGET_SPEAKER messages.
    # Track used message indices to ensure no overlap or reuse.
    chunks = []
    used = set()
    n = len(messages)
    i = 0

    while i < n:
        # skip already-used messages
        if i in used:
            i += 1
            continue

        # skip non-target messages
        if messages[i][2] != TARGET_SPEAKER:
            i += 1
            continue

        # collect consecutive TARGET_SPEAKER messages
        j = i
        while j < n and messages[j][2] == TARGET_SPEAKER and j not in used:
            j += 1

        target_indices = list(range(i, j))
        target_lines = [f"[{d} {t}] {s}: {m}" for (d, t, s, m) in messages[i:j]]

        # collect up to KEEP_PREV_CONTEXT_LINES immediately preceding unused messages
        prev_lines = []
        context_indices = []
        k = i - 1
        while k >= 0 and len(prev_lines) < KEEP_PREV_CONTEXT_LINES:
            if k not in used:
                d, t, s, m = messages[k]
                prev_lines.insert(0, f"[{d} {t}] {s}: {m}")
                context_indices.insert(0, k)
            k -= 1

        # mark all used indices (context + target)
        chunk_indices = context_indices + target_indices
        used.update(chunk_indices)

        # build and add chunk
        chunk_lines = prev_lines + target_lines
        chunks.append("\n".join(chunk_lines))

        # advance to next unused message
        i = j

    # Limit chunk size if needed (optional simple trimming)
    final_chunks = []
    for ch in chunks:
        lines = ch.splitlines()
        if len(lines) > MAX_LINES_PER_CHUNK:
            lines = lines[-MAX_LINES_PER_CHUNK:]
        final_chunks.append("\n".join(lines))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(final_chunks).strip() + "\n", encoding="utf-8")
    return len(final_chunks)


def main(argv):
    here = Path(__file__).parent
    logs_dir = here.parent / "chat_logs"
    outputs_dir = here / "outputs"

    paths = []
    if len(argv) > 1:
        # explicit input files provided on command line
        for p in argv[1:]:
            paths.append(Path(p))
    else:
        # process all txt files in dev/chat_logs
        paths = sorted(logs_dir.glob("*.txt"))

    total = 0
    for p in paths:
        out = outputs_dir / f"{p.stem}_chunks.txt"
        n = process_file(p, out)
        print(f"Wrote {n} chunks from {p} -> {out}")
        total += n

    print(f"Total chunks written: {total}")


if __name__ == "__main__":
    main(sys.argv)
