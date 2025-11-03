# Obsidian Auto Note Linker

Automatically create links between related notes in your Obsidian vault by analyzing content similarity and finding mentions of note titles.

## Features

- **Smart Link Detection**: Finds natural mentions of note titles in your content
- **Content Similarity Analysis**: Uses TF-IDF to identify semantically related notes
- **Safe Linking**: Avoids linking in code blocks, URLs, and already-linked text
- **Dry-Run Mode**: Preview changes before applying them
- **Preserves Formatting**: Maintains your existing markdown structure
- **Batch Processing**: Processes all notes in your vault automatically

## Requirements

```bash
pip install scikit-learn numpy
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure .env File

Copy the example and edit with your settings:

```bash
cp .env.example .env
nano .env  # or use your preferred editor
```

Set your vault path and folders to ignore:

```bash
OBSIDIAN_VAULT_PATH=/home/username/Documents/Obsidian Vault
IGNORE_FOLDERS=Folder1,Folder2
```

### 3. Preview Changes (Recommended First Step)

```bash
# No arguments needed - reads from .env!
python3 obsidian-auto-linker.py --dry-run
```

This will show you ALL links that would be added without modifying any files.

### 4. Apply Changes to Your Vault

```bash
python3 obsidian-auto-linker.py --apply
```

## Usage Examples

### Basic Usage (with .env configured)

```bash
# Preview changes (reads vault path from .env)
python3 obsidian-auto-linker.py --dry-run

# Apply changes to vault
python3 obsidian-auto-linker.py --apply
```

### Or Specify Vault Path Manually

```bash
# Override .env vault path
python3 obsidian-auto-linker.py "/home/user/Documents/Obsidian Vault" --dry-run

# Apply changes
python3 obsidian-auto-linker.py "/home/user/Documents/Obsidian Vault" --apply
```

### Adjust Sensitivity

```bash
# Lower threshold = more links (finds more loosely related notes)
python3 obsidian-auto-linker.py "/path/to/vault" --similarity-threshold 0.10 --dry-run

# Higher threshold = fewer links (only very related notes)
python3 obsidian-auto-linker.py "/path/to/vault" --similarity-threshold 0.20 --dry-run
```

### Ignore Folders

```bash
# Skip specific root folders (won't scan or link notes in these folders)
python3 obsidian-auto-linker.py --ignore-folders "Folder1" "Folder2" --dry-run

# Or configure in .env:
# IGNORE_FOLDERS=Folder1,Folder2
```

### Ignore Generic Words

```bash
# Add custom words to ignore (in addition to defaults)
python3 obsidian-auto-linker.py --ignore-words "chapter" "section" --dry-run

# Disable default ignore list and only use your custom words
python3 obsidian-auto-linker.py --no-default-ignores --ignore-words "foo" "bar" --dry-run

# See what's being ignored in the output
python3 obsidian-auto-linker.py --dry-run
# Output: "Ignoring 2 generic note(s): README, Welcome"
# Output: "Skipped folders: Folder1, Folder2"
```

### Limit Links Per Note

```bash
# Add maximum 15 links per note
python3 obsidian-auto-linker.py "/path/to/vault" --max-links 15 --apply
```

### Save Results to JSON

```bash
# Export detailed results for review
python3 obsidian-auto-linker.py "/path/to/vault" --dry-run --output-json results.json
```

## How It Works

### 1. Scan Phase
- Finds all `.md` files in your vault (excluding `.obsidian` folder)
- Extracts content and existing links from each note
- Cleans text by removing code blocks, URLs, and existing links

### 2. Analysis Phase
- **TF-IDF Similarity**: Computes semantic similarity between all notes
- **Title Matching**: Finds exact mentions of note titles in content
- Identifies which notes should be linked together

### 3. Linking Phase
- Creates Obsidian-style `[[Note Name]]` links
- Only links mentions that aren't already in code blocks or URLs
- Respects existing links (doesn't create duplicates)
- Preserves all existing formatting

## Example Output

```
Processing vault in DRY RUN mode
============================================================

Processing: PyTorch
  Path: remote-blog/AI Learning/Deep Learning/PyTorch.md
  Related notes found: 10
  Mentions found: 0
  Links to add: 10
  - No links to add

Processing: Resume Prompting Template
  Path: Job/Resume Prompting Template.md
  Related notes found: 4
  Mentions found: 1
  Links to add: 5
  [OK] Added 1 links
    - 'PyTorch' -> [[PyTorch]]

============================================================
SUMMARY
============================================================
Total notes processed: 58
Notes modified: 9
Total mentions found: 13
Total links added: 13

[DRY RUN] No files were actually modified
Run with --apply to make changes
```


## Configuration

### .env File (Recommended)

Create a `.env` file to configure default settings:

```bash
# Required: Path to your Obsidian vault
OBSIDIAN_VAULT_PATH=/home/username/Documents/Obsidian Vault

# Folders to skip during scanning (comma-separated)
IGNORE_FOLDERS=Folder1,Folder2

# Optional: Adjust similarity threshold (0.0-1.0, default: 0.15)
SIMILARITY_THRESHOLD=0.15

# Optional: Max links per note (default: 20)
MAX_LINKS_PER_NOTE=20

# Optional: Minimum word length (default: 4)
MIN_WORD_LENGTH=4
```

**Benefits of using .env:**
- No need to type vault path every time
- Consistent configuration across runs
- Folders like "Folder1", "Folder2" are automatically skipped
- Easy to version control (add `.env` to `.gitignore`)
- Share `.env.example` with your team

## Command-Line Options

| Option | Description | Default | Can Override .env? |
|--------|-------------|---------|-------------------|
| `vault_path` | Path to vault (optional if in .env) | From .env | ✅ Yes |
| `--dry-run` | Preview without modifying files | `True` | N/A |
| `--apply` | Apply changes to the vault | `False` | N/A |
| `--similarity-threshold` | Minimum similarity score | `.env` or `0.15` | ✅ Yes |
| `--max-links` | Max links per note | `.env` or `20` | ✅ Yes |
| `--ignore-words` | Additional words to ignore | - | ✅ Adds to defaults |
| `--ignore-folders` | Root folders to skip | `.env` | ✅ Adds to .env |
| `--no-default-ignores` | Disable default ignore list | `False` | N/A |
| `--output-json` | Save results to JSON | - | N/A |

## Best Practices

### First Time Use

1. **Always start with `--dry-run`** to preview changes
2. **Review the output** to see what links will be created
3. **Adjust threshold** if you see too many or too few links
4. **Backup your vault** before running with `--apply`
5. **Test on a small subset** first if you have a large vault

### Optimal Settings

- **General Knowledge Vault**: `--similarity-threshold 0.12`
- **Technical/Specific Topics**: `--similarity-threshold 0.15` (default)
- **Highly Specialized Content**: `--similarity-threshold 0.20`

### When to Re-run

- After adding many new notes to your vault
- When you've reorganized or renamed notes
- Periodically to maintain up-to-date connections

## What Gets Linked

### ✅ Will Create Links For

- Exact mentions of note titles (case-insensitive)
- Notes with high content similarity
- Notes in regular text paragraphs

### ❌ Will NOT Link

- Text inside code blocks (`` ` `` or ` ``` `)
- Text that's already part of a link
- URLs or web addresses
- Notes already linked in the current note
- Generic/common words (see ignore list below)

## Default Ignore List

By default, the following generic words are NOT auto-linked (case-insensitive):

```
welcome, readme, index, home, about, intro, introduction,
overview, summary, table of contents, toc, notes, todo,
archive, drafts, template, templates, example, examples,
test, tests, misc, miscellaneous, other, others,
general, resources, links, references
```

You can:
- **Add more**: Use `--ignore-words "word1" "word2"`
- **Disable defaults**: Use `--no-default-ignores`
- **See what's ignored**: Check the output after scanning the vault

Example output:
```
Ignoring 2 generic note(s): README, Welcome
```

## Troubleshooting

### "scikit-learn not available"

Install the required dependency:
```bash
pip install scikit-learn numpy
```

### Too Many Links Created

Increase the similarity threshold:
```bash
python3 obsidian-auto-linker.py "/path/to/vault" --similarity-threshold 0.20 --dry-run
```

### Too Few Links Created

Decrease the similarity threshold:
```bash
python3 obsidian-auto-linker.py "/path/to/vault" --similarity-threshold 0.10 --dry-run
```

### Links Not Detected

- Make sure note filenames match the text in your content
- Check if text is inside code blocks or already linked
- Try lowering the similarity threshold

## Example Workflow

```bash
# Step 1: Preview with default settings
python3 obsidian-auto-linker.py "/home/user/Documents/Obsidian Vault" --dry-run

# Step 2: Adjust threshold if needed
python3 obsidian-auto-linker.py "/home/user/Documents/Obsidian Vault" --similarity-threshold 0.12 --dry-run

# Step 3: Export results for review
python3 obsidian-auto-linker.py "/home/user/Documents/Obsidian Vault" --dry-run --output-json links.json

# Step 4: Apply changes
python3 obsidian-auto-linker.py "/home/user/Documents/Obsidian Vault" --apply

# Step 5: Check your vault in Obsidian to see the new links!
```

## Technical Details

### Algorithm

1. **TF-IDF Vectorization**: Converts note content into numerical vectors
2. **Cosine Similarity**: Measures semantic similarity between notes
3. **Pattern Matching**: Uses regex to find exact note title mentions
4. **Smart Insertion**: Adds links while preserving existing structure

### Link Format

Creates standard Obsidian links:
```markdown
[[Note Name]]
```

For notes with aliases, you can manually create:
```markdown
[[Full Note Name|Display Text]]
```

## Contributing

Found a bug or have a feature request? Feel free to modify the code or create an issue.

## License

Free to use and modify for personal use.

## For Your Vault

### Test Run Results

After running the enhanced version on your vault:
- **Total notes**: 58
- **Notes processed**: 58
- **Notes modified**: 34 (59% of vault)
- **Total links added**: 126 (11x improvement!)
- **Notes ignored**: 2 (README, Welcome)

### Key Connections Created:

**Core Concepts Linked:**
- `TF-IDF` → [[TF-IDF Definition|TF-IDF]] (found multiple mentions)
- `hyperparameters` → [[Common Hyperparameters|hyperparameters]]
- `overfitting` → [[12. Tuning CNNs for Overfitting|overfitting]]
- `dropout` → [[4. CNN Dropout|dropout]]
- `dimensionality` → [[Curse of Dimensionality|dimensionality]]
- `PyTorch` → [[PyTorch]]
- `Cosine Similarity` → [[Cosine Similarity]]

**Smart Features:**
- Extracts core concepts from note titles (e.g., "TF-IDF Definition" → searches for "TF-IDF")
- Uses Obsidian alias format: `[[Full Note Name|Display Text]]`
- Automatically ignores overly generic terms (e.g., "learning", "model", "function")

## Tips for Maximum Effectiveness

1. **Use descriptive note titles** that match how you naturally reference concepts
2. **Write naturally** - the tool finds organic mentions of topics
3. **Run periodically** as your vault grows
4. **Check the graph view** in Obsidian after linking to visualize connections
5. **Combine with manual linking** for best results

## What Makes This Tool Unique

- **Context-aware**: Understands what should and shouldn't be linked
- **Smart similarity**: Uses ML to find semantically related content
- **Obsidian-native**: Works with standard Obsidian link syntax
- **Non-destructive**: Preserves all your existing formatting and links
- **Transparent**: Shows exactly what it's doing before and after
