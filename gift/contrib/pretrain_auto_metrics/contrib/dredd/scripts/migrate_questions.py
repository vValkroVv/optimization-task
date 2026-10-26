#!/usr/bin/env python3
"""
Script for migrating question.jsonl files into a flat structure.

Moves all question.jsonl files from nested folders into data/arenas/ using arena names.
For example:
  data/arenas/arena-hard-auto/arena-hard-v0.1/question.jsonl
  → data/arenas/arena-hard-v0.1.jsonl
"""
import os
import shutil
from pathlib import Path
import json

def get_arena_name_from_path(file_path: str) -> str:
    """
    Extracts arena name from a question.jsonl path.
    
    Examples:
      arena-hard-auto/arena-hard-v0.1/question.jsonl → arena-hard-v0.1
      fastchat/mt_bench/question.jsonl → mt_bench
      ru-llm-arena/arena-general-v0.1/question.jsonl → arena-general-v0.1
    """
    parts = Path(file_path).parts
    # Remove 'arenas' if present at the beginning
    if 'arenas' in parts:
        parts = tuple(p for p in parts if p != 'arenas')
    
    # Remove trailing 'question.jsonl'
    parts = tuple(p for p in parts if p != 'question.jsonl')
    
    # Take the last part (arena name)
    if parts:
        return parts[-1]
    return "unknown"

def migrate_questions(arenas_dir: str = "data/arenas", dry_run: bool = False):
    """
    Migrates all question.jsonl files to a flat structure.
    
    Args:
        arenas_dir: Directory containing arenas
        dry_run: If True, only prints planned actions without moving files
    """
    arenas_path = Path(arenas_dir)
    if not arenas_path.exists():
        print(f"Error: directory {arenas_dir} not found")
        return
    
    # Find all question.jsonl files
    question_files = list(arenas_path.rglob("question.jsonl"))
    
    if not question_files:
        print(f"No question.jsonl files found in {arenas_dir}")
        return
    
    print(f"Found {len(question_files)} question.jsonl files\n")
    
    migrations = []
    for old_path in question_files:
        arena_name = get_arena_name_from_path(str(old_path.relative_to(arenas_path)))
        new_path = arenas_path / f"{arena_name}.jsonl"
        
        migrations.append({
            'old_path': old_path,
            'new_path': new_path,
            'arena_name': arena_name
        })
    
    # Check conflicts
    conflicts = []
    for mig in migrations:
        if mig['new_path'].exists() and mig['old_path'] != mig['new_path']:
            conflicts.append(mig)
    
    if conflicts:
        print("⚠️  WARNING: Conflicts found (files already exist):")
        for mig in conflicts:
            print(f"  {mig['new_path']} already exists")
        print()
        if not dry_run:
            response = input("Overwrite existing files? (yes/no): ")
            if response.lower() != 'yes':
                print("Migration cancelled")
                return
    
    # Run migration
    print("=" * 80)
    if dry_run:
        print("DRY-RUN MODE: files will not be moved")
    else:
        print("STARTING MIGRATION")
    print("=" * 80)
    print()
    
    for mig in migrations:
        old_path = mig['old_path']
        new_path = mig['new_path']
        arena_name = mig['arena_name']
        
        if old_path == new_path:
            print(f"✅ {arena_name}: already in the correct location")
            continue
        
        print(f"📦 {arena_name}:")
        print(f"   {old_path.relative_to(data_path)}")
        print(f"   → {new_path.relative_to(data_path)}")
        
        if not dry_run:
            try:
                # Create destination directory if needed
                new_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file
                shutil.move(str(old_path), str(new_path))
                print("   ✅ Moved")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        print()
    
    if not dry_run:
        print("=" * 80)
        print("MIGRATION COMPLETED")
        print("=" * 80)
        print()
        print("All question.jsonl files are now in data/arenas/ using arena names.")
        print("Example: data/arenas/arena-hard-v0.1.jsonl")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate question.jsonl files to a flat structure"
    )
    parser.add_argument(
        "--arenas-dir",
        default="data/arenas",
        help="Arena directory (default: data/arenas)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: show planned actions without moving files"
    )
    
    args = parser.parse_args()
    
    migrate_questions(arenas_dir=args.arenas_dir, dry_run=args.dry_run)
