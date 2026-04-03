# Personal Research Journal

A personal research journal: markdown notes organized into folders, browsable via [Docsify](https://ianliyi1996.github.io/personal-research-journal/).

## Structure

| Folder | Purpose |
|---|---|
| [papers/](papers/) | Paper reading notes |
| [research-notes/](research-notes/) | Research summaries and reflections |
| [topics/](topics/) | Topic-based notes organized by research area |
| [weekly/](weekly/) | Weekly progress logs |
| [resources/](resources/) | Datasets, tools, and useful links |

## Naming Conventions

- Papers: `papers/YYYY-short-title.md`
- Research notes: `research-notes/YYYY-MM-DD-title.md`
- Topics: `topics/area-name/title.md`
- Weekly: `weekly/YYYY-WXX.md`

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
