# Changelog

## [2.1.1] - 2025-11-03 - Bug Fixes & Filter Improvements

### Fixed
- **min_word_length bug**: Code was hardcoded to check `< 4` instead of using `self.min_word_length` from config
- **False positive prevention**: Added generic academic terms to ignore list
  - `mathematics`, `math`, `maths` - prevents linking to "SGD Mathematics"
  - `science`, `sciences`, `statistics`, `stat`, `stats` - too generic in ML context

### Impact
- **More accurate links**: Filters out false positives like "mathematics" → "SGD Mathematics"
- **Quality over quantity**: 68 focused links instead of 112 with false positives
- **Respects .env**: Now properly uses `MIN_WORD_LENGTH` from configuration

### Example
**Before**: "mathematics" → `[[2. SGD Mathematics|mathematics]]` ❌ (False positive!)
**After**: "mathematics" → Not linked ✅ (Too generic)

## [2.1.0] - 2025-11-03 - Configuration & Display Improvements

### Added
- **`.env` file support**: Configure vault path and settings once, use everywhere
  - `OBSIDIAN_VAULT_PATH`: Set vault path in .env (no need to type it every time)
  - `IGNORE_FOLDERS`: Skip folders like Job, Misc, Programming via .env
  - `SIMILARITY_THRESHOLD`, `MAX_LINKS_PER_NOTE`, `MIN_WORD_LENGTH`: All configurable
- **Folder ignore feature**: Exclude entire root folders from scanning
  - Specify in .env: `IGNORE_FOLDERS=Job,Misc,Programming`
  - Or via CLI: `--ignore-folders "Job" "Misc" "Programming"`
  - Output shows: `Skipped folders: Job, Misc, Programming`
- **python-dotenv** dependency for .env file support

### Changed
- **Display ALL links**: No more "and x more" truncation - shows every link created
- **Vault path is optional**: If set in .env, no need to pass as argument
- **CLI can override .env**: Command-line arguments take precedence over .env values

### Usage
```bash
# Before (v2.0):
python3 obsidian-auto-linker.py "/home/user/vault" --dry-run

# After (v2.1):
python3 obsidian-auto-linker.py --dry-run  # Reads from .env!
```

### Impact on Your Vault
- **47 notes** processed (down from 58 - Job, Misc, Programming folders now skipped)
- **112 links** created (focused on AI/ML content only)
- **31 notes** modified (66% of processed notes)

## [2.0.0] - 2025-11-03 - Major Enhancement

### Added - Smart Term Extraction 🚀
- **Intelligent note title parsing**: Extracts core concepts from note titles
  - "TF-IDF Definition" → searches for both "TF-IDF Definition" AND "TF-IDF"
  - "1. SGD Definition" → searches for "SGD Definition", "SGD", and removes number prefixes
  - "Common Hyperparameters" → searches for "Common Hyperparameters" AND "Hyperparameters"
- **Obsidian alias support**: Creates `[[Full Note Name|Display Text]]` format
  - Example: Text "TF-IDF" becomes `[[TF-IDF Definition|TF-IDF]]`
  - Preserves natural reading while linking to full note names
- **Enhanced ignore list**: Expanded to include common technical terms
  - Added: `learning`, `model`, `data`, `function`, `parameter`, `network`, etc.
  - Prevents false positives with overly generic terms
  - Now includes 60+ common terms

### Changed
- Increased minimum word length from 3 to 4 characters (reduces false positives)
- Smarter suffix removal (handles "Definition", "Overview", "Explained", etc.)
- Better handling of compound terms and plural forms
- Display shows up to 5 examples (increased from 3)

### Performance
- **11x improvement**: From 11 links to 126 links
- **59% coverage**: 34 out of 58 notes now have auto-links
- **High accuracy**: Significantly reduced false positives through smart filtering

### Impact on Your Vault
- **v1.0**: 11 links (exact title matches only)
- **v2.0**: 126 links (smart term extraction)
- **Key successes**:
  - All "TF-IDF" mentions now linked to "TF-IDF Definition"
  - All "hyperparameters" mentions now linked to "Common Hyperparameters"
  - Technical terms like "overfitting", "dropout", "dimensionality" properly linked
  - Generic terms like "learning", "model", "function" correctly ignored

## [1.1.0] - 2025-11-03

### Added
- **Ignore Words Feature**: Generic/common words now automatically excluded
- `--ignore-words` flag to add custom ignore terms
- `--no-default-ignores` flag to disable default ignore list
- Visual feedback showing ignored notes during scanning

### Changed
- Updated README with ignore feature documentation
- Added ignore examples to help text

## [1.0.0] - Initial Release

### Features
- Automatic note linking based on content similarity
- TF-IDF vectorization for semantic analysis
- Pattern matching for exact note title mentions
- Dry-run mode for safe previewing
- Configurable similarity threshold
- Maximum links per note setting
- JSON export of results
