# Personal Research Journal

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
# Install dependencies
npm install

# Browse notes in the browser
npm run docs          # opens at http://localhost:3000
```

## CLI Tools

```bash
./journal.sh paper <name>      # Create a paper note from template
./journal.sh note <name>       # Create a research note from template
./journal.sh weekly            # Create this week's weekly note
./journal.sh search <keyword>  # Full-text search across all notes
./journal.sh index             # Regenerate all README index files
./journal.sh serve             # Start Docsify local server
```
