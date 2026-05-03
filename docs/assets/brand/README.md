# Axon brand assets

Three SVGs in this folder cover every common use:

| File | Use case | Render size |
|---|---|---|
| `axon-icon.svg` | Hero icon, README header, app icon, favicon source | 32 px – 1024 px |
| `axon-favicon.svg` | Browser favicon (simplified — strokes thicken for legibility) | 16 px – 64 px |
| `axon-wordmark.svg` | Repo header, docs banner, anywhere a horizontal lockup is needed | 80 px – 320 px width |

## The mark

The icon is a hex-framed neural node with three branches radiating from a central cell body to terminal synapses. It reads as both a graph node and a stylised neural axon — the project's namesake — and the symmetric Y-pattern of branches stays balanced at any size.

## Color

The mark ships with the Axon teal `#00d4b4` baked in. To recolor:

- The wordmark and icon both expose `currentColor` on the inline nav variant in `index.html`. Set `color` on the parent.
- Standalone SVG files use the literal `#00d4b4` and a near-black inner accent dot (`#050a14`) for the centre. Find and replace if you need a different palette.

## Embedding examples

### HTML

```html
<!-- Favicon -->
<link rel="icon" type="image/svg+xml" href="docs/assets/brand/axon-favicon.svg">

<!-- Inline icon (recolorable via CSS color) -->
<img src="docs/assets/brand/axon-icon.svg" alt="Axon" width="48" height="48">

<!-- Wordmark in a docs header -->
<img src="docs/assets/brand/axon-wordmark.svg" alt="Axon" width="240">
```

### README.md

```markdown
<p align="center">
  <img src="docs/assets/brand/axon-wordmark.svg" alt="Axon" width="280">
</p>
```

### Docs (Markdown front matter / hero block)

```markdown
![Axon](docs/assets/brand/axon-icon.svg "Private RAG. Local first. Graph native.")
```

## Concept

Three signals from a central node mirror Axon's product story:

1. **One source** — your local knowledge base (the centre)
2. **Many destinations** — citations, retrieved chunks, downstream agents (the terminals)
3. **Bounded surface** — the hex frame is the trust boundary; nothing leaves it without your action.

Use the mark whenever you'd otherwise type the word "Axon" in a headline. Don't stretch, recolor to non-teal palettes without an explicit reason, or add drop-shadows beyond the subtle teal halo already used in `index.html` nav.
