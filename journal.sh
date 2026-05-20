#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage: ./journal.sh <command> [args]

Commands:
  paper  <name>      Create a paper note (papers/YYYY-name.md)
  note   <name>      Create a research note (research-notes/YYYY-MM-DD-name.md)
  weekly             Create this week's weekly note (weekly/YYYY-WXX.md)
  search <keyword>   Full-text search across all markdown files
  index              Regenerate README index files for each folder
  serve              Start Docsify local server on port 3000
EOF
}

cmd_paper() {
    local name="${1:?Usage: ./journal.sh paper <name>}"
    local year
    year="$(date +%Y)"
    local filename="papers/${year}-${name}.md"

    if [[ -f "$filename" ]]; then
        echo "Already exists: $filename"
        exit 1
    fi

    cp papers/_template.md "$filename"
    echo "Created: $filename"
}

cmd_note() {
    local name="${1:?Usage: ./journal.sh note <name>}"
    local date_prefix
    date_prefix="$(date +%Y-%m-%d)"
    local filename="research-notes/${date_prefix}-${name}.md"

    if [[ -f "$filename" ]]; then
        echo "Already exists: $filename"
        exit 1
    fi

    cp research-notes/_template.md "$filename"
    echo "Created: $filename"
}

cmd_weekly() {
    local year week filename
    year="$(date +%Y)"
    week="$(date +%V)"
    filename="weekly/${year}-W${week}.md"

    if [[ -f "$filename" ]]; then
        echo "Already exists: $filename"
        exit 1
    fi

    local monday friday
    # Get Monday and Friday of the current ISO week
    monday="$(date -d "$(date +%Y-%m-%d) -$((( $(date +%u) - 1 ))) days" +%Y-%m-%d)"
    friday="$(date -d "${monday} +4 days" +%Y-%m-%d)"

    cat > "$filename" <<EOF
# Week ${week} (${monday} - ${friday})

## Goals

-

## Progress

### Monday

### Tuesday

### Wednesday

### Thursday

### Friday

## Papers Read

-

## Key Insights

-

## Next Week

-
EOF
    echo "Created: $filename"
}

cmd_search() {
    local keyword="${1:?Usage: ./journal.sh search <keyword>}"
    grep -rn --include='*.md' --color=auto -- "$keyword" \
        papers/ research-notes/ topics/ weekly/ resources/ 2>/dev/null || \
        echo "No matches found for: $keyword"
}

cmd_index() {
    # Papers index
    {
        echo "# Paper Reading Notes"
        echo ""
        echo 'Use `_template.md` as the starting point for each new paper note.'
        echo ""
        echo 'Naming: `YYYY-short-title.md`'
        echo ""
        echo "## Index"
        echo ""
        local found=false
        for f in papers/[0-9]*.md; do
            [[ -f "$f" ]] || continue
            found=true
            local basename
            basename="$(basename "$f" .md)"
            local title
            title="$(head -1 "$f" | sed 's/^#\s*//')"
            echo "- [${title}](/papers/${basename}.md)"
        done
        if [[ "$found" == false ]]; then
            echo "<!-- No papers yet -->"
        fi
    } > papers/README.md
    echo "Updated: papers/README.md"

    # Research notes index
    {
        echo "# Research Notes"
        echo ""
        echo 'Use `_template.md` as the starting point for each new note.'
        echo ""
        echo 'Naming: `YYYY-MM-DD-title.md`'
        echo ""
        echo "## Index"
        echo ""
        local found=false
        for f in research-notes/[0-9]*.md; do
            [[ -f "$f" ]] || continue
            found=true
            local basename
            basename="$(basename "$f" .md)"
            local title
            title="$(head -1 "$f" | sed 's/^#\s*//')"
            echo "- [${title}](/research-notes/${basename}.md)"
        done
        if [[ "$found" == false ]]; then
            echo "<!-- No notes yet -->"
        fi
    } > research-notes/README.md
    echo "Updated: research-notes/README.md"

    # Weekly index
    {
        echo "# Weekly Progress"
        echo ""
        echo 'Naming: `YYYY-WXX.md` (e.g., `2026-W06.md`)'
        echo ""
        echo "## Index"
        echo ""
        local found=false
        for f in weekly/[0-9]*.md; do
            [[ -f "$f" ]] || continue
            found=true
            local basename
            basename="$(basename "$f" .md)"
            local title
            title="$(head -1 "$f" | sed 's/^#\s*//')"
            echo "- [${title}](/weekly/${basename}.md)"
        done
        if [[ "$found" == false ]]; then
            echo "<!-- No weekly entries yet -->"
        fi
    } > weekly/README.md
    echo "Updated: weekly/README.md"

    # Topics index
    {
        echo "# Topics"
        echo ""
        echo "Organize notes by research area. Create subdirectories for each topic."
        echo ""
        echo "## Index"
        echo ""
        local found=false
        for d in topics/*/; do
            [[ -d "$d" ]] || continue
            found=true
            local dirname
            dirname="$(basename "$d")"
            echo "### ${dirname}"
            echo ""
            for f in "${d}"[0-9]*.md; do
                [[ -f "$f" ]] || continue
                local basename title
                basename="$(basename "$f" .md)"
                title="$(head -1 "$f" | sed 's/^#\s*//')"
                echo "- [${title}](/topics/${dirname}/${basename}.md)"
            done
            echo ""
        done
        if [[ "$found" == false ]]; then
            echo "<!-- No topics yet -->"
        fi
    } > topics/README.md
    echo "Updated: topics/README.md"

    # Sidebar
    {
        echo "- [**🏠 Home**](/)"

        echo "- **📄 Papers**"
        for f in papers/[0-9]*.md; do
            [[ -f "$f" ]] || continue
            local basename title
            basename="$(basename "$f" .md)"
            title="$(head -1 "$f" | sed 's/^#\s*//')"
            echo "  - [${title}](/papers/${basename}.md)"
        done

        echo "- **🔬 Research Notes**"
        for f in $(ls -r research-notes/[0-9]*.md 2>/dev/null); do
            [[ -f "$f" ]] || continue
            local basename title
            basename="$(basename "$f" .md)"
            title="$(head -1 "$f" | sed 's/^#\s*//')"
            echo "  - [${title}](/research-notes/${basename}.md)"
        done

        echo "- **🌿 Topics**"
        for d in topics/*/; do
            [[ -d "$d" ]] || continue
            local dirname
            dirname="$(basename "$d")"
            echo "  - **${dirname^^}**"
            for f in "${d}"[0-9]*.md; do
                [[ -f "$f" ]] || continue
                local basename title
                basename="$(basename "$f" .md)"
                title="$(head -1 "$f" | sed 's/^#\s*//')"
                echo "    - [${title}](/topics/${dirname}/${basename}.md)"
            done
        done

        echo "- **📅 Weekly**"
        for f in $(ls -r weekly/[0-9]*.md 2>/dev/null); do
            [[ -f "$f" ]] || continue
            local basename title
            basename="$(basename "$f" .md)"
            title="$(head -1 "$f" | sed 's/^#\s*//')"
            echo "  - [${title}](/weekly/${basename}.md)"
        done

        echo "- **📚 Resources**"
        echo '- <div class="sidebar-footer"><a href="https://github.com/IanLiYi1996/personal-research-journal" target="_blank">⚙️ GitHub</a><a href="/#/?id=li-yis-research-journal">❓ About</a></div>'
    } > _sidebar.md
    echo "Updated: _sidebar.md"

    update_home_md
    echo "Updated: _home.md"

    echo "Done. All indexes regenerated."
}

# Find the most recent file under a directory (by filename sort, descending),
# matching files whose basename starts with a digit and excluding _template.md.
latest_in_dir() {
    local dir="$1"
    local f
    for f in $(ls -r "${dir}"/[0-9]*.md 2>/dev/null); do
        [[ -f "$f" ]] || continue
        echo "$f"
        return 0
    done
    return 1
}

# Latest topic: walk topics/*/ subdirs, pick the file with the lexicographically
# largest basename across all subdirs.
latest_topic() {
    local best=""
    local d f basename
    for d in topics/*/; do
        [[ -d "$d" ]] || continue
        for f in "${d}"[0-9]*.md; do
            [[ -f "$f" ]] || continue
            basename="$(basename "$f")"
            if [[ -z "$best" || "$basename" > "$(basename "$best")" ]]; then
                best="$f"
            fi
        done
    done
    [[ -n "$best" ]] && echo "$best"
}

# Extract date from filename. Handles YYYY-MM-DD-* and YYYY-* and YYYY-WXX.
# Falls back to mtime YYYY-MM-DD if no date prefix is found.
file_date() {
    local f="$1"
    local base
    base="$(basename "$f" .md)"
    if [[ "$base" =~ ^([0-9]{4}-[0-9]{2}-[0-9]{2}) ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$base" =~ ^([0-9]{4}-W[0-9]{2}) ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$base" =~ ^([0-9]{4}) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        date -r "$f" +%Y-%m-%d 2>/dev/null || echo ""
    fi
}

file_title() {
    head -1 "$1" | sed 's/^#\s*//'
}

# URL path for Docsify hash routing: strip leading / and trailing .md
docsify_path() {
    local f="$1"
    echo "/${f%.md}"
}

# Replace the content between `<!-- AUTO:<MARKER>:START -->` and
# `<!-- AUTO:<MARKER>:END -->` in $file with the lines from stdin.
# Markers are preserved.
replace_block() {
    local file="$1"
    local marker="$2"
    local payload
    payload="$(cat)"

    awk -v marker="$marker" -v payload="$payload" '
        BEGIN { in_block = 0 }
        $0 ~ ("<!-- AUTO:" marker ":START -->") {
            print
            print payload
            in_block = 1
            next
        }
        $0 ~ ("<!-- AUTO:" marker ":END -->") {
            in_block = 0
            print
            next
        }
        !in_block { print }
    ' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
}

update_home_md() {
    [[ -f _home.md ]] || return 0

    local latest_note latest_topic_file
    latest_note="$(latest_in_dir research-notes || true)"
    latest_topic_file="$(latest_topic || true)"

    # --- Hero buttons ---
    {
        if [[ -n "$latest_note" ]]; then
            local p="$(docsify_path "$latest_note")"
            echo "<a href=\"#${p}\" class=\"arc-btn-primary\">Latest Digest</a>"
        fi
        if [[ -n "$latest_topic_file" ]]; then
            local p="$(docsify_path "$latest_topic_file")"
            echo "<a href=\"#${p}\" class=\"arc-btn-secondary\">Explore Topics</a>"
        fi
    } | replace_block _home.md "HERO_BUTTONS"

    # --- Recent table: 5 most recent entries across notes / topics / papers ---
    # Build a sortable list: "<date>\t<type>\t<title>\t<path>"
    {
        echo "| Date | Type | Title |"
        echo "|------|------|-------|"
        {
            local f d t p
            for f in $(ls -r research-notes/[0-9]*.md 2>/dev/null); do
                [[ -f "$f" ]] || continue
                d="$(file_date "$f")"; t="$(file_title "$f")"; p="$(docsify_path "$f")"
                printf '%s\t%s\t%s\t%s\n' "$d" "Note" "$t" "$p"
            done
            for f in topics/*/[0-9]*.md; do
                [[ -f "$f" ]] || continue
                d="$(file_date "$f")"; t="$(file_title "$f")"; p="$(docsify_path "$f")"
                printf '%s\t%s\t%s\t%s\n' "$d" "Topic" "$t" "$p"
            done
            for f in papers/[0-9]*.md; do
                [[ -f "$f" ]] || continue
                d="$(file_date "$f")"; t="$(file_title "$f")"; p="$(docsify_path "$f")"
                printf '%s\t%s\t%s\t%s\n' "$d" "Paper" "$t" "$p"
            done
        } | sort -r -k1,1 | head -5 | while IFS=$'\t' read -r d t title p; do
            echo "| ${d} | ${t} | [${title}](${p}) |"
        done
    } | replace_block _home.md "RECENT"
}

cmd_serve() {
    if ! command -v npx &>/dev/null; then
        echo "Error: npx not found. Run 'npm install' first."
        exit 1
    fi
    echo "Starting Docsify server at http://localhost:3000"
    npx docsify-cli serve . --port 3000
}

# --- Main ---
if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

command="$1"
shift

case "$command" in
    paper)  cmd_paper "$@" ;;
    note)   cmd_note "$@" ;;
    weekly) cmd_weekly ;;
    search) cmd_search "$@" ;;
    index)  cmd_index ;;
    serve)  cmd_serve ;;
    help|-h|--help) usage ;;
    *)
        echo "Unknown command: $command"
        usage
        exit 1
        ;;
esac
