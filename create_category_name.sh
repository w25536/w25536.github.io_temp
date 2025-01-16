#!/bin/bash

# Check if a category name was provided and validate it
if [ $# -ne 1 ] || [ -z "${1// }" ]; then
    echo "Error: Please provide a single, non-empty category name"
    echo "Usage: $0 category-name"
    exit 1
fi

# Store the category name and sanitize it
CATEGORY=$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')

# Create the category directory if it doesn't exist
mkdir -p "categories/$CATEGORY"

# Create index.html with the required content
cat > "categories/$CATEGORY/index.html" << EOF
---
layout: home
---
EOF

echo "Created category '$CATEGORY' with index.html"