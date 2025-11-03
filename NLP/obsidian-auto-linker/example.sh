#!/bin/bash

# Example usage script for Obsidian Auto Note Linker

# Set your vault path here
VAULT_PATH="/home/username/Documents/Obsidian Vault"

echo "====================================="
echo "Obsidian Auto Note Linker - Examples"
echo "====================================="
echo ""

# Example 1: Dry run with default settings
echo "Example 1: Preview changes (dry run)"
echo "Command: python3 obsidian-auto-linker.py \"$VAULT_PATH\" --dry-run"
echo ""
read -p "Press Enter to run, or Ctrl+C to skip..."
python3 obsidian-auto-linker.py "$VAULT_PATH" --dry-run
echo ""

# Example 2: Dry run with lower threshold (more links)
echo "Example 2: Preview with lower threshold (more links)"
echo "Command: python3 obsidian-auto-linker.py \"$VAULT_PATH\" --similarity-threshold 0.10 --dry-run"
echo ""
read -p "Press Enter to run, or Ctrl+C to skip..."
python3 obsidian-auto-linker.py "$VAULT_PATH" --similarity-threshold 0.10 --dry-run
echo ""

# Example 3: Dry run with JSON output
echo "Example 3: Preview and save results to JSON"
echo "Command: python3 obsidian-auto-linker.py \"$VAULT_PATH\" --dry-run --output-json results.json"
echo ""
read -p "Press Enter to run, or Ctrl+C to skip..."
python3 obsidian-auto-linker.py "$VAULT_PATH" --dry-run --output-json results.json
echo ""
echo "Results saved to: results.json"
echo ""

# Example 4: Apply changes (commented out for safety)
echo "Example 4: Apply changes to vault (CAREFUL!)"
echo "Command: python3 obsidian-auto-linker.py \"$VAULT_PATH\" --apply"
echo ""
echo "WARNING: This will modify your vault files!"
read -p "Type 'yes' to apply changes, or anything else to skip: " confirm
if [ "$confirm" = "yes" ]; then
    echo "Applying changes..."
    python3 obsidian-auto-linker.py "$VAULT_PATH" --apply
    echo ""
    echo "Done! Check your Obsidian vault to see the new links."
else
    echo "Skipped. No changes made."
fi
echo ""

echo "====================================="
echo "Examples complete!"
echo "====================================="
