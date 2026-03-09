"""Generate a synthetic REPL screenshot PNG for the README."""
from PIL import Image, ImageDraw, ImageFont
import os

BG          = (18,  18,  18)
BOX_BORDER  = (80,  80,  80)
BOX_BG      = (26,  26,  26)
TOOLBAR_BG  = (43,  43,  43)
WHITE       = (220, 220, 220)
DIM         = (110, 110, 110)
CYAN        = (97,  214, 214)
GREEN       = (87,  166, 74)
YELLOW      = (220, 180, 60)
BLUE        = (86,  156, 214)
SEP_LINE    = (55,  55,  55)

W, H   = 1080, 700
MARGIN = 28
LINE_H = 22
PAD    = 14

img  = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)


def font(size=14):
    for p in [
        "C:/Windows/Fonts/CascadiaMono.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


F    = font(14)
F_SM = font(12)
F_XS = font(11)


def t(x, y, s, color=WHITE, f=None):
    draw.text((x, y), s, fill=color, font=f or F)


def hline(y, color=BOX_BORDER, x0=None, x1=None):
    draw.line([(x0 or MARGIN, y), (x1 or W - MARGIN, y)], fill=color)


# ── 1. HEADER BOX ─────────────────────────────────────────────────────────────
HX, HY = MARGIN, MARGIN
HW     = W - 2 * MARGIN

# Pre-calculate the y positions to know the exact box height needed
# title row: PAD + LINE_H+2 = 14+24 = 38   → y after = 66
# divider+10 = 10                            → y after = 76
# LLM row: LINE_H = 22                       → y after = 98
# Embed row: LINE_H+4 = 26                   → y after = 124
# divider+10 = 10                            → y after = 134
# Search/Discuss: LINE_H = 22                → y after = 156
# Docs/Hybrid: LINE_H+4 = 26                 → y after = 182
# divider+8 = 8                              → y after = 190
# ✓ text (F_SM height ~16)                   → y ends  = 206
# bottom pad = 14                            → total   = 220
HH = 222
draw.rounded_rectangle([HX, HY, HX + HW, HY + HH], radius=6, fill=BOX_BG, outline=BOX_BORDER)

y = HY + PAD
t(HX + 16, y, "🧠  Local RAG Brain", WHITE, F)
y += LINE_H + 2
hline(y, BOX_BORDER, HX + 6, HX + HW - 6)
y += 10

t(HX + 16, y, "LLM    ·  ollama/gemma",                          DIM);  y += LINE_H
t(HX + 16, y, "Embed  ·  sentence_transformers/all-MiniLM-L6-v2", DIM);  y += LINE_H + 4
hline(y, BOX_BORDER, HX + 6, HX + HW - 6)
y += 10

col2 = HX + 340
t(HX + 16, y, "Search  ·  OFF",               DIM)
t(col2,     y, "Discuss  ·  ON",               DIM)
y += LINE_H
t(HX + 16, y, "Docs    ·  210 chunks  (5 files)", DIM)
t(col2,     y, "Hybrid   ·  ON   top-k · 10   threshold · 0.3", DIM)
y += LINE_H + 4
hline(y, BOX_BORDER, HX + 6, HX + HW - 6)
y += 8

t(HX + 16, y, "✓ Embedding ready  [cpu]   ✓ Vector store ready   ✓ BM25  ·  209 docs", GREEN, F_SM)

# ── 2. HINT + SEPARATOR ───────────────────────────────────────────────────────
HB = HY + HH   # exact bottom edge of header box
t(MARGIN, HB + 10, "  Type your question  ·  /help for commands  ·  Tab to autocomplete", DIM, F_SM)
hline(HB + 28, SEP_LINE)

# ── 3. CONVERSATION ───────────────────────────────────────────────────────────
CY = HB + 44

t(MARGIN, CY, "You:", YELLOW, F)
t(MARGIN + 50, CY, "What is the main topic of my ingested documents?", WHITE)
CY += LINE_H + 4

t(MARGIN, CY, "Brain:", BLUE, F)
CY += LINE_H
for line in [
    "Based on your documents, the main topics covered are:",
    "",
    "  1.  AI Music Generation — Suno AI prompt engineering, genre tags,",
    "      and mood descriptors for high-quality music generation.",
    "",
    "  2.  Techno & Electronic Music — Style guides, BPM ranges, synthesis",
    "      parameters and arrangement patterns for techno subgenres.",
    "",
]:
    if line == "":
        CY += 5
        continue
    t(MARGIN + 10, CY, line, WHITE, F_SM)
    CY += LINE_H

t(MARGIN + 10, CY, "Sources: suno_prompts.md  ·  techno_guide.txt  ·  genre_tags.json", DIM, F_XS)
CY += LINE_H + 10

t(MARGIN, CY, "You:", YELLOW, F)
t(MARGIN + 50, CY, "/discuss", WHITE)
CY += LINE_H + 2
t(MARGIN + 10, CY, "  💬 Discussion mode OFF.", DIM, F_SM)
CY += LINE_H + 10

t(MARGIN, CY, "You:", YELLOW, F)
t(MARGIN + 50, CY, "What is the best chocolate cake recipe?", WHITE)
CY += LINE_H + 2
t(MARGIN + 10, CY, "Brain:", BLUE, F)
t(MARGIN + 72, CY, "I don't have relevant information in my documents to answer that.", DIM, F)

# ── 4. INPUT PROMPT ───────────────────────────────────────────────────────────
PY = H - 52
hline(PY - 4, SEP_LINE)
t(MARGIN, PY + 4, "You: ", YELLOW, F)
draw.rectangle([MARGIN + 46, PY + 5, MARGIN + 57, PY + 19], fill=WHITE)

# ── 5. BOTTOM TOOLBAR ─────────────────────────────────────────────────────────
TY = H - 26
draw.rectangle([0, TY, W, H], fill=TOOLBAR_BG)

tx = MARGIN


def kv(label, val):
    global tx
    t(tx, TY + 3, label, CYAN, F_SM)
    tx += int(draw.textlength(label, font=F_SM))
    t(tx, TY + 3, val, WHITE, F_SM)
    tx += int(draw.textlength(val, font=F_SM))
    t(tx, TY + 3, "  │  ", DIM, F_SM)
    tx += int(draw.textlength("  │  ", font=F_SM))


kv("LLM  ", "ollama/gemma")
kv("Embed  ", "sentence_transformers/all-MiniLM-L6-v2")
kv("discuss:", "on")
kv("search:", "off")
kv("hybrid:", "on")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repl-demo.png")
img.save(out, "PNG")
print(f"Saved: {out}")
