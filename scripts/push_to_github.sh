#!/bin/bash
# A script to push code to GitHub using a token from the environment.

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable is not set."
    echo "Please set it using: export GITHUB_TOKEN='your_token_here'"
    exit 1
fi

# Ensure token is set for the remote URL
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/WHW1233/TriTAC.git"

# Commit message (use the first argument, or a default message)
COMMIT_MSG=${1:-"Auto update: codebase sync"}

echo "Adding changes..."
git add .

echo "Committing changes with message: '${COMMIT_MSG}'..."
git commit -m "${COMMIT_MSG}" || echo "No changes to commit."

echo "Pushing to GitHub..."
git push origin main

echo "Done!"
