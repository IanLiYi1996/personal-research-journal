# Personal Research Journal

> *A living notebook tracking my exploration of AI, NLP, and beyond.*

---

## Browse by Section

<div class="home-grid">
<div class="home-card">
<span class="card-icon">&#128220;</span>

#### [Papers](/papers/)
Deep dives into research papers — summaries, critiques, and key takeaways from the latest in AI/ML.
</div>

<div class="home-card">
<span class="card-icon">&#128300;</span>

#### [Research Notes](/research-notes/)
Working notes, experiment logs, and structured analyses on specific research topics.
</div>

<div class="home-card">
<span class="card-icon">&#127793;</span>

#### [Topics](/topics/)
Long-running knowledge bases organized by research area — NLP, multimodal, agents, and more.
</div>

<div class="home-card">
<span class="card-icon">&#128197;</span>

#### [Weekly](/weekly/)
Week-by-week progress logs — what I read, what I built, what I learned.
</div>

<div class="home-card">
<span class="card-icon">&#128218;</span>

#### [Resources](/resources/)
Curated datasets, tools, frameworks, and reference materials.
</div>
</div>

---

## Recent Activity

| Date | Type | Title |
|------|------|-------|
| 2026-04-03 | Research Note | [HuggingFace Daily Papers Digest](/research-notes/2026-04-03-huggingface-daily-papers-digest) |
| 2026-02-27 | Research Note | [Google CE Scenario 4 — Full Design](/research-notes/2026-02-27-google-ce-scenario4-full-design) |
| 2026-02-09 | Research Note | [LLM Intro — Comprehensive Overview](/research-notes/2026-02-09-llm-intro-comprehensive) |
| 2026-02-09 | Research Note | [3D Generation Technology Survey](/research-notes/2026-02-09-3d-generation-survey) |
| 2025 | Paper | [Agentic RL Survey](/papers/2025-agentic-rl-survey) |

---

## Naming Conventions

| Folder | Pattern | Example |
|--------|---------|---------|
| `papers/` | `YYYY-short-title.md` | `2025-agentic-rl-survey.md` |
| `research-notes/` | `YYYY-MM-DD-title.md` | `2026-04-03-huggingface-daily-papers-digest.md` |
| `topics/` | `area-name/title.md` | `topics/nlp/transformers.md` |
| `weekly/` | `YYYY-WXX.md` | `2026-W06.md` |

## Quick Start

```bash
npm install            # Install dependencies
npm run docs           # Browse at http://localhost:3000

./journal.sh note <name>    # New research note
./journal.sh paper <name>   # New paper note
./journal.sh weekly         # New weekly log
./journal.sh search <kw>    # Full-text search
./journal.sh index          # Rebuild indexes
```
