import re
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
SYNTHESIS_MD = ROOT / "mds" / "papers_revised" / "07_composite_kinematic_kinetic_scoring_indices" / "04_Search_Synthesis_and_Excluded_Candidates.md"
OUT_FILE = ROOT / "logs" / "notion_export" / "04_synthesis_notion.md"

RESERVED_CHARS = "\\*~`$[]<>{}|^"

def _protect_and_escape(text: str) -> str:
    # Protect bold, italic, and links
    protected = []
    def stash(match):
        protected.append(match.group(0))
        return f"\x00{len(protected)-1}\x00"

    # Protect links: [text](url)
    text = re.sub(r"\[[^\[\]\n]+\]\([^()\n]+\)", stash, text)
    # Protect bold: **bold**
    text = re.sub(r"\*\*[^*\n]+\*\*", stash, text)
    # Protect italic: *italic*
    text = re.sub(r"\*[^*\n]+\*", stash, text)

    escaped_chars = set(RESERVED_CHARS)
    out = []
    for ch in text:
        if ch in escaped_chars:
            out.append("\\" + ch)
        else:
            out.append(ch)
    text = "".join(out)

    def restore(match):
        return protected[int(match.group(1))]

    return re.sub(r"\x00(\d+)\x00", restore, text)

def convert_md_table_to_notion(table_text: str) -> str:
    lines = [l.strip() for l in table_text.strip().split("\n") if l.strip().startswith("|")]
    if len(lines) < 2:
        return table_text
    
    def split_row(line: str) -> list[str]:
        cells = line.strip().strip("|").split("|")
        return [c.strip() for c in cells]
    
    header = split_row(lines[0])
    # Line 1 is the separator: |---|---
    data_rows = [split_row(l) for l in lines[2:]]
    
    out = ['<table fit-page-width="true" header-row="true">']
    out.append("<tr>")
    for cell in header:
        out.append(f"<td>{_protect_and_escape(cell)}</td>")
    out.append("</tr>")
    
    for row in data_rows:
        out.append("<tr>")
        for cell in row:
            out.append(f"<td>{_protect_and_escape(cell)}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "\n".join(out)

def main():
    raw = SYNTHESIS_MD.read_text(encoding="utf-8")
    
    # We want to identify table blocks and convert them.
    # Tables are sequences of lines starting with |
    blocks = []
    lines = raw.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            table_text = "\n".join(table_lines)
            blocks.append(convert_md_table_to_notion(table_text))
        else:
            # Escape normal lines
            if line.startswith("#"):
                # Header lines
                m = re.match(r"^(#+)\s+(.*)$", line)
                blocks.append(f"{m.group(1)} {m.group(2)}")
            elif line.strip().startswith("-"):
                # Bullets
                m = re.match(r"^(\s*)-\s+(.*)$", line)
                bullet_content = _protect_and_escape(m.group(2))
                indent = m.group(1)
                # Convert 2 space indent to tab for Notion nesting
                if len(indent) >= 2:
                    blocks.append(f"\t- {bullet_content}")
                else:
                    blocks.append(f"- {bullet_content}")
            else:
                blocks.append(_protect_and_escape(line))
            i += 1
            
    OUT_FILE.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Converted synthesis to {OUT_FILE}")

if __name__ == "__main__":
    main()
