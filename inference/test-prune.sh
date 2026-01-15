#!/usr/bin/env bash
query=$1
code_file=$2
code=$(cat "$code_file")
resp=$(jq -n \
  --arg q "$query" \
  --arg c "$code" \
  '{query:$q, code:$c, threshold:0.5}' \
| http POST localhost:8000/prune)

score=$(echo "$resp" | jq -r .score)
pruned=$(echo "$resp" | jq -r .pruned_code)

printf "score: %.6f\n\n" "$score"
if command -v batcat >/dev/null 2>&1; then
    echo "$pruned" | batcat --language=python --style=numbers --color=always --paging=never
else
    if command -v pygmentize >/dev/null 2>&1; then
        echo "$pruned" | pygmentize -l python -f terminal256 -O style=monokai,linenos=1
    else
        echo "$pruned"
    fi
fi
