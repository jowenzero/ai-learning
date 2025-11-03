#!/usr/bin/env python3
"""
Obsidian Auto Note Linker

This script automatically creates links between related notes in an Obsidian vault
by analyzing content similarity and finding mentions of note titles.

Features:
- Scans all markdown files in the vault
- Finds related notes using TF-IDF similarity
- Detects mentions of note titles in content
- Creates Obsidian-style [[Note Name]] links
- Preserves existing links and formatting
- Provides dry-run mode to preview changes
- Avoids linking in code blocks, URLs, and already-linked text

Usage:
    python obsidian-auto-linker.py /path/to/vault --dry-run
    python obsidian-auto-linker.py /path/to/vault --apply
    python obsidian-auto-linker.py /path/to/vault --similarity-threshold 0.15
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import json

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")
    print("Falling back to simple keyword matching only.")


class ObsidianNote:
    """Represents a single Obsidian note."""

    def __init__(self, filepath: Path, vault_root: Path):
        self.filepath = filepath
        self.vault_root = vault_root
        self.relative_path = filepath.relative_to(vault_root)

        # Note name is the filename without .md extension
        self.note_name = filepath.stem

        # Extract searchable terms (core concept + full name)
        self.searchable_terms = self._extract_searchable_terms()

        # Read content
        with open(filepath, 'r', encoding='utf-8') as f:
            self.content = f.read()

        # Extract text without code blocks and existing links
        self.clean_text = self._extract_clean_text()

        # Track existing links
        self.existing_links = self._extract_existing_links()

    def _extract_searchable_terms(self) -> List[str]:
        """
        Extract searchable terms from the note name.

        For example:
        - "TF-IDF Definition" -> ["TF-IDF Definition", "TF-IDF"]
        - "1. SGD Definition" -> ["SGD Definition", "SGD", "1. SGD Definition"]
        - "Common Hyperparameters" -> ["Common Hyperparameters", "Hyperparameters"]
        """
        terms = []
        note_name = self.note_name

        # Always include the full note name
        terms.append(note_name)

        # Remove common prefixes (numbers like "1.", "2.", etc.)
        clean_name = re.sub(r'^\d+\.\s*', '', note_name)
        if clean_name != note_name:
            terms.append(clean_name)

        # Remove common suffixes that are descriptive
        suffixes_to_remove = [
            'Definition', 'Overview', 'Basics', 'Introduction', 'Intro',
            'Explained', 'Guide', 'Tutorial', 'Examples', 'Example',
            'Techniques', 'Methods', 'Concepts', 'Concept'
        ]

        for suffix in suffixes_to_remove:
            # Try to remove suffix (with optional 's' pluralization)
            pattern = r'\s+' + re.escape(suffix) + r's?$'
            core_concept = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
            if core_concept != clean_name and len(core_concept) > 2:
                terms.append(core_concept)
                # Also try removing from original note_name
                core_from_original = re.sub(pattern, '', note_name, flags=re.IGNORECASE)
                if core_from_original != note_name and core_from_original not in terms:
                    terms.append(core_from_original)

        # For compound terms, also extract key parts only if they're not too generic
        # e.g., "Common Hyperparameters" -> add "Hyperparameter" (singular)
        # But don't extract generic words like "Parameters", "Function", etc.
        words = clean_name.split()
        if len(words) > 1:
            # Add the last significant word (often the main concept)
            last_word = words[-1]
            # Only add if it's not a common generic term and is long enough
            generic_single_words = {
                'parameters', 'parameter', 'functions', 'function', 'methods', 'method',
                'values', 'value', 'results', 'result', 'features', 'feature',
                'concepts', 'concept', 'techniques', 'technique', 'approaches', 'approach',
                'models', 'model', 'systems', 'system', 'tools', 'tool',
                'frameworks', 'framework', 'libraries', 'library', 'components', 'component',
                'elements', 'element', 'examples', 'example', 'explained', 'definition',
                'definitions', 'overview', 'basics', 'guide', 'tutorial', 'introduction',
                'solutions', 'solution', 'problems', 'problem', 'questions', 'question',
                'answers', 'answer', 'issues', 'issue', 'applications', 'application'
            }
            if len(last_word) > 4 and last_word.lower() not in generic_single_words and last_word not in terms:
                terms.append(last_word)
                # Also add singular form if it ends with 's'
                if last_word.endswith('s') and len(last_word) > 5:
                    singular = last_word[:-1]
                    if singular.lower() not in generic_single_words and singular not in terms:
                        terms.append(singular)

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)

        return unique_terms

    def _extract_clean_text(self) -> str:
        """Extract text without code blocks, URLs, and existing links."""
        text = self.content

        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)

        # Remove URLs
        text = re.sub(r'https?://[^\s\)]+', '', text)

        # Remove existing Obsidian links but keep the display text
        text = re.sub(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', r'\2' if r'\2' else r'\1', text)

        return text

    def _extract_existing_links(self) -> Set[str]:
        """Extract all existing Obsidian links from the note."""
        links = set()
        # Match [[Note Name]] or [[Note Name|Alias]]
        pattern = r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]'
        for match in re.finditer(pattern, self.content):
            links.add(match.group(1))
        return links

    def __repr__(self):
        return f"ObsidianNote({self.note_name})"


class ObsidianAutoLinker:
    """Main class for auto-linking Obsidian notes."""

    # Default list of generic words to ignore when auto-linking
    DEFAULT_IGNORE_WORDS = {
        'welcome', 'readme', 'index', 'home', 'about', 'intro', 'introduction',
        'overview', 'summary', 'table of contents', 'toc', 'notes', 'todo',
        'archive', 'drafts', 'template', 'templates', 'example', 'examples',
        'test', 'tests', 'misc', 'miscellaneous', 'other', 'others',
        'general', 'resources', 'links', 'references',
        # Generic technical terms
        'definition', 'definitions', 'explained', 'guide', 'tutorial', 'basics',
        'introduction', 'methods', 'method', 'techniques', 'technique',
        'concepts', 'concept', 'examples', 'example',
        # Common words that are too generic to link
        'application', 'applications', 'solution', 'solutions', 'problem', 'problems',
        'question', 'questions', 'answer', 'answers', 'issue', 'issues',
        'data', 'model', 'models', 'system', 'systems', 'approach', 'approaches',
        'technique', 'techniques', 'method', 'methods', 'parameter', 'parameters',
        'function', 'functions', 'value', 'values', 'result', 'results',
        'feature', 'features', 'component', 'components', 'element', 'elements',
        'tool', 'tools', 'framework', 'frameworks', 'library', 'libraries',
        'learning', 'training', 'testing', 'dataset', 'location', 'locations',
        'command', 'commands', 'network', 'networks'
    }

    def __init__(self, vault_path: str, similarity_threshold: float = 0.15,
                 max_links_per_note: int = 20, min_word_length: int = 4,
                 ignore_words: Set[str] = None, ignore_folders: Set[str] = None):
        self.vault_path = Path(vault_path)
        self.similarity_threshold = similarity_threshold
        self.max_links_per_note = max_links_per_note
        self.min_word_length = min_word_length

        # Set ignore words (use default if not provided)
        if ignore_words is None:
            self.ignore_words = self.DEFAULT_IGNORE_WORDS.copy()
        else:
            self.ignore_words = {word.lower() for word in ignore_words}

        # Set ignore folders (folders to skip during scanning)
        if ignore_folders is None:
            self.ignore_folders = set()
        else:
            self.ignore_folders = {folder.strip() for folder in ignore_folders}

        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")

        # Storage for notes
        self.notes: List[ObsidianNote] = []
        self.note_by_name: Dict[str, ObsidianNote] = {}

        # Storage for similarity matrix
        self.similarity_matrix = None
        self.vectorizer = None

    def scan_vault(self):
        """Scan the vault and load all markdown notes."""
        print(f"Scanning vault: {self.vault_path}")

        # Find all .md files, excluding .obsidian folder and ignored folders
        md_files = []
        skipped_folders = []

        for root, dirs, files in os.walk(self.vault_path):
            root_path = Path(root)

            # Skip .obsidian directory
            if '.obsidian' in root_path.parts:
                continue

            # Check if this path is in an ignored folder
            # Get the immediate subfolder of vault_path
            try:
                relative_path = root_path.relative_to(self.vault_path)
                # Get the root folder name (first part of relative path)
                if relative_path.parts:
                    root_folder = relative_path.parts[0]
                    if root_folder in self.ignore_folders:
                        if root_folder not in skipped_folders:
                            skipped_folders.append(root_folder)
                        continue
            except ValueError:
                # Not a subdirectory, skip
                pass

            for file in files:
                if file.endswith('.md'):
                    md_files.append(Path(root) / file)

        print(f"Found {len(md_files)} markdown files")
        if skipped_folders:
            print(f"Skipped folders: {', '.join(sorted(skipped_folders))}")

        # Load notes
        for filepath in md_files:
            note = ObsidianNote(filepath, self.vault_path)
            self.notes.append(note)
            self.note_by_name[note.note_name] = note

        print(f"Loaded {len(self.notes)} notes")

        # Show which words are being ignored
        if self.ignore_words:
            ignored_note_count = sum(1 for note in self.notes if note.note_name.lower() in self.ignore_words)
            if ignored_note_count > 0:
                print(f"Ignoring {ignored_note_count} generic note(s): {', '.join(sorted(n.note_name for n in self.notes if n.note_name.lower() in self.ignore_words))}")

    def compute_similarity(self):
        """Compute TF-IDF similarity between all notes."""
        if not SKLEARN_AVAILABLE:
            print("Skipping similarity computation (scikit-learn not available)")
            return

        print("Computing content similarity using TF-IDF...")

        # Prepare documents
        documents = [note.clean_text for note in self.notes]

        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.8,
            stop_words='english',
            ngram_range=(1, 2)
        )

        try:
            tfidf_matrix = self.vectorizer.fit_transform(documents)

            # Compute cosine similarity
            self.similarity_matrix = cosine_similarity(tfidf_matrix)

            print(f"Similarity matrix computed: {self.similarity_matrix.shape}")
        except Exception as e:
            print(f"Error computing similarity: {e}")
            self.similarity_matrix = None

    def find_related_notes(self, note: ObsidianNote, top_k: int = 10) -> List[Tuple[ObsidianNote, float]]:
        """Find notes most related to the given note."""
        related = []

        if self.similarity_matrix is not None:
            # Find index of this note
            note_idx = self.notes.index(note)

            # Get similarity scores
            similarities = self.similarity_matrix[note_idx]

            # Get top-k most similar (excluding itself)
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]

            for idx in top_indices:
                score = similarities[idx]
                if score >= self.similarity_threshold:
                    related.append((self.notes[idx], score))

        return related

    def find_linkable_mentions(self, note: ObsidianNote) -> Dict[str, List[Tuple[int, str, str]]]:
        """
        Find mentions of other note titles in this note's content.

        Returns:
            Dict mapping note_name to list of (position, matched_text, search_term) tuples
        """
        mentions = defaultdict(list)

        # Get note content
        content = note.content

        # Find positions we should NOT link (code blocks, existing links, URLs)
        excluded_ranges = self._get_excluded_ranges(content)

        # Track positions that have been matched to avoid overlapping links
        matched_positions = set()

        # Check each other note
        for other_note in self.notes:
            if other_note.note_name == note.note_name:
                continue  # Skip self

            if other_note.note_name in note.existing_links:
                continue  # Already linked

            # Skip generic/common words that shouldn't be auto-linked
            if other_note.note_name.lower() in self.ignore_words:
                continue

            # Search for ALL searchable terms associated with this note
            # Sort by length (longest first) to prefer more specific matches
            sorted_terms = sorted(other_note.searchable_terms, key=len, reverse=True)

            for search_term in sorted_terms:
                # Skip very short terms (less than 4 chars) to avoid false positives
                if len(search_term) < 4:
                    continue

                # Skip if this term is in the ignore list
                if search_term.lower() in self.ignore_words:
                    continue

                # Search for case-insensitive mentions of the search term
                # Match whole words only
                pattern = r'\b' + re.escape(search_term) + r'\b'

                for match in re.finditer(pattern, content, re.IGNORECASE):
                    start, end = match.span()

                    # Check if this position is excluded
                    if self._is_in_excluded_range(start, excluded_ranges):
                        continue

                    # Check if this position overlaps with an existing match
                    position_range = range(start, end)
                    if any(pos in matched_positions for pos in position_range):
                        continue

                    # Add this match
                    mentions[other_note.note_name].append((start, match.group(0), search_term))

                    # Mark these positions as matched
                    matched_positions.update(position_range)

        return mentions

    def _get_excluded_ranges(self, content: str) -> List[Tuple[int, int]]:
        """Get ranges that should not be linked (code blocks, links, URLs)."""
        ranges = []

        # Code blocks (```)
        for match in re.finditer(r'```[\s\S]*?```', content):
            ranges.append(match.span())

        # Inline code (`)
        for match in re.finditer(r'`[^`]+`', content):
            ranges.append(match.span())

        # Existing Obsidian links
        for match in re.finditer(r'\[\[[^\]]+\]\]', content):
            ranges.append(match.span())

        # URLs
        for match in re.finditer(r'https?://[^\s\)]+', content):
            ranges.append(match.span())

        return ranges

    def _is_in_excluded_range(self, position: int, ranges: List[Tuple[int, int]]) -> bool:
        """Check if a position falls within any excluded range."""
        for start, end in ranges:
            if start <= position < end:
                return True
        return False

    def generate_links(self, note: ObsidianNote) -> Dict[str, int]:
        """
        Generate link suggestions for a note.

        Returns:
            Dict with stats about links to be added
        """
        link_stats = defaultdict(int)

        # Find related notes by similarity
        related_notes = self.find_related_notes(note)
        link_stats['related_notes_found'] = len(related_notes)

        # Find explicit mentions
        mentions = self.find_linkable_mentions(note)
        link_stats['mentions_found'] = sum(len(m) for m in mentions.values())

        # Prioritize: mentions first, then related notes
        links_to_add = set()

        # Add all mentions
        for note_name in mentions.keys():
            links_to_add.add(note_name)

        # Add related notes (up to max limit)
        for related_note, score in related_notes:
            if len(links_to_add) >= self.max_links_per_note:
                break
            if related_note.note_name not in links_to_add:
                links_to_add.add(related_note.note_name)

        link_stats['links_to_add'] = len(links_to_add)

        return link_stats, mentions, links_to_add

    def apply_links(self, note: ObsidianNote, mentions: Dict[str, List[Tuple[int, str, str]]],
                   dry_run: bool = True) -> str:
        """
        Apply links to the note content.

        Returns:
            Modified content with links added
        """
        content = note.content

        # Sort all mentions by position (reverse order to maintain positions)
        all_mentions = []
        for note_name, mention_list in mentions.items():
            for position, matched_text, search_term in mention_list:
                all_mentions.append((position, matched_text, note_name, search_term))

        all_mentions.sort(key=lambda x: x[0], reverse=True)

        # Apply links from end to start (to maintain positions)
        modifications = []
        for position, matched_text, note_name, search_term in all_mentions:
            # Create the link - if the matched text differs from note name, use an alias
            if matched_text.lower() == note_name.lower():
                link = f"[[{note_name}]]"
            else:
                # Use alias format: [[Note Name|Displayed Text]]
                link = f"[[{note_name}|{matched_text}]]"

            # Replace the text
            end_position = position + len(matched_text)
            content = content[:position] + link + content[end_position:]

            modifications.append({
                'position': position,
                'original': matched_text,
                'link': link,
                'target': note_name,
                'search_term': search_term
            })

        if not dry_run:
            # Write the modified content back to the file
            with open(note.filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        return content, modifications

    def process_vault(self, dry_run: bool = True):
        """Process all notes in the vault and add links."""
        print(f"\n{'='*60}")
        print(f"Processing vault in {'DRY RUN' if dry_run else 'APPLY'} mode")
        print(f"{'='*60}\n")

        total_links_added = 0
        total_mentions_found = 0
        notes_modified = 0

        results = []

        for note in self.notes:
            print(f"\nProcessing: {note.note_name}")
            print(f"  Path: {note.relative_path}")

            # Generate link suggestions
            link_stats, mentions, links_to_add = self.generate_links(note)

            print(f"  Related notes found: {link_stats['related_notes_found']}")
            print(f"  Mentions found: {link_stats['mentions_found']}")
            print(f"  Links to add: {link_stats['links_to_add']}")

            if mentions:
                # Apply links
                modified_content, modifications = self.apply_links(note, mentions, dry_run)

                total_links_added += len(modifications)
                total_mentions_found += link_stats['mentions_found']
                notes_modified += 1

                print(f"   Added {len(modifications)} links")

                # Show ALL modifications
                for i, mod in enumerate(modifications):
                    if mod['original'].lower() == mod['target'].lower():
                        print(f"    - '{mod['original']}' -> [[{mod['target']}]]")
                    else:
                        print(f"    - '{mod['original']}' -> [[{mod['target']}|{mod['original']}]]")

                results.append({
                    'note': note.note_name,
                    'path': str(note.relative_path),
                    'links_added': len(modifications),
                    'modifications': modifications
                })
            else:
                print(f"  - No links to add")

        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total notes processed: {len(self.notes)}")
        print(f"Notes modified: {notes_modified}")
        print(f"Total mentions found: {total_mentions_found}")
        print(f"Total links added: {total_links_added}")

        if dry_run:
            print(f"\n   DRY RUN: No files were actually modified")
            print(f"Run with --apply to make changes")
        else:
            print(f"\n Changes applied to vault")

        return results


def main():
    # Load .env file if it exists
    if DOTENV_AVAILABLE:
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded configuration from {env_path}")

    parser = argparse.ArgumentParser(
        description='Automatically link related notes in an Obsidian vault',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use vault path from .env file (no arguments needed)
  python obsidian-auto-linker.py --dry-run

  # Or specify vault path explicitly
  python obsidian-auto-linker.py "/path/to/vault" --dry-run

  # Apply changes to vault
  python obsidian-auto-linker.py --apply

  # Adjust similarity threshold
  python obsidian-auto-linker.py --similarity-threshold 0.2 --dry-run

  # Limit max links per note
  python obsidian-auto-linker.py --max-links 15 --apply

  # Add custom words to ignore
  python obsidian-auto-linker.py --ignore-words "chapter" "section" --dry-run

  # Ignore specific folders
  python obsidian-auto-linker.py --ignore-folders "Job" "Misc" --dry-run

  # Disable default ignores
  python obsidian-auto-linker.py --no-default-ignores --apply
        """
    )

    parser.add_argument('vault_path', nargs='?', help='Path to the Obsidian vault (optional if set in .env)')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Preview changes without modifying files (default)')
    parser.add_argument('--apply', action='store_true',
                       help='Apply changes to the vault')
    parser.add_argument('--similarity-threshold', type=float, default=0.15,
                       help='Minimum similarity score for relating notes (default: 0.15)')
    parser.add_argument('--max-links', type=int, default=20,
                       help='Maximum links to add per note (default: 20)')
    parser.add_argument('--ignore-words', type=str, nargs='*',
                       help='Additional words to ignore when auto-linking (case-insensitive)')
    parser.add_argument('--ignore-folders', type=str, nargs='*',
                       help='Root folders to skip (e.g., Job, Misc, Programming)')
    parser.add_argument('--no-default-ignores', action='store_true',
                       help='Disable default ignore list (welcome, readme, etc.)')
    parser.add_argument('--output-json', type=str,
                       help='Save results to JSON file')

    args = parser.parse_args()

    # Get vault path from CLI or .env
    vault_path = args.vault_path
    if not vault_path:
        vault_path = os.getenv('OBSIDIAN_VAULT_PATH')
        if not vault_path:
            parser.error("vault_path is required (either as argument or in .env file)")

    print(f"Using vault: {vault_path}")

    # Get configuration from .env or use defaults
    similarity_threshold = args.similarity_threshold
    if similarity_threshold == 0.15:  # Default value, check .env
        env_threshold = os.getenv('SIMILARITY_THRESHOLD')
        if env_threshold:
            similarity_threshold = float(env_threshold)

    max_links = args.max_links
    if max_links == 20:  # Default value, check .env
        env_max_links = os.getenv('MAX_LINKS_PER_NOTE')
        if env_max_links:
            max_links = int(env_max_links)

    min_word_length = 4  # Default
    env_min_word = os.getenv('MIN_WORD_LENGTH')
    if env_min_word:
        min_word_length = int(env_min_word)

    # If --apply is specified, turn off dry-run
    dry_run = not args.apply

    # Prepare ignore words list
    ignore_words = None
    if args.no_default_ignores:
        # Start with empty set if user wants no defaults
        ignore_words = set()
    else:
        # Start with default ignore words
        ignore_words = ObsidianAutoLinker.DEFAULT_IGNORE_WORDS.copy()

    # Add any additional ignore words from command line
    if args.ignore_words:
        if ignore_words is None:
            ignore_words = set()
        ignore_words.update(word.lower() for word in args.ignore_words)

    # Prepare ignore folders list
    ignore_folders = set()

    # Get from .env first
    env_ignore_folders = os.getenv('IGNORE_FOLDERS')
    if env_ignore_folders:
        ignore_folders.update(folder.strip() for folder in env_ignore_folders.split(','))

    # Add any additional ignore folders from command line (overrides .env)
    if args.ignore_folders:
        ignore_folders.update(args.ignore_folders)

    try:
        # Initialize linker
        linker = ObsidianAutoLinker(
            vault_path=vault_path,
            similarity_threshold=similarity_threshold,
            max_links_per_note=max_links,
            min_word_length=min_word_length,
            ignore_words=ignore_words,
            ignore_folders=ignore_folders
        )

        # Scan vault
        linker.scan_vault()

        # Compute similarity
        linker.compute_similarity()

        # Process vault
        results = linker.process_vault(dry_run=dry_run)

        # Save results if requested
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\n Results saved to: {args.output_json}")

    except Exception as e:
        print(f"\nL Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
