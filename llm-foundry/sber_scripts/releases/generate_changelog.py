#!/usr/bin/env python3

import subprocess
import re
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class ChangelogGenerator:
    COMMIT_TYPES = {
        'feat': 'Features',
        'fix': 'Bug Fixes',
        'refactor': 'Refactoring',
        'test': 'Tests',
        'docs': 'Documentation',
        'ci': 'CI/CD',
        'chore': 'Maintenance',
    }

    def __init__(self):
        self.commits = []
        self.categorized_commits = defaultdict(list)

    def run_git_command(self, command: List[str]) -> str:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {' '.join(command)}")
            print(f"Error: {e.stderr}")
            return ""

    def get_last_tag(self) -> Optional[str]:
        tags = self.run_git_command(['git', 'tag', '--sort=-version:refname'])
        if tags:
            return tags.split('\n')[0]
        return None

    def get_commits_since(self, since_tag: Optional[str] = None) -> List[Dict]:
        if since_tag:
            commit_range = f"{since_tag}..HEAD"
        else:
            commit_range = "HEAD"

        log_format = "%H|%s|%an|%ae|%ad"
        commits_raw = self.run_git_command([
            'git', 'log',
            commit_range,
            '--pretty=format:' + log_format,
            '--date=short',
            '--no-merges'
        ])

        commits = []
        for line in commits_raw.split('\n'):
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 5:
                commits.append({
                    'hash': parts[0][:7],
                    'full_hash': parts[0],
                    'subject': parts[1],
                    'author': parts[2],
                    'email': parts[3],
                    'date': parts[4]
                })

        return commits

    def parse_commit_message(self, subject: str) -> Tuple[str, str, str]:
        pattern = r'^(?:\[?(\w+)\]?)(?:\(([^)]+)\))?: (.+)$'
        match = re.match(pattern, subject)

        if match:
            commit_type = match.group(1)
            scope = match.group(2) or ''
            description = match.group(3)
        else:
            commit_type = 'chore'
            scope = ''
            description = subject

        return commit_type, scope, description

    def categorize_commits(self, commits: List[Dict]):
        for commit in commits:
            commit_type, scope, description = self.parse_commit_message(commit['subject'])

            if scope:
                entry = f"**{scope}**: {description}"
            else:
                entry = description

            entry += f" ([{commit['hash']}]({commit['full_hash']}))"

            if commit_type in self.COMMIT_TYPES:
                self.categorized_commits[commit_type].append({
                    'entry': entry,
                    'author': commit['author']
                })
            else:
                self.categorized_commits['chore'].append({
                    'entry': entry,
                    'author': commit['author']
                })

    def generate_version(self) -> str:
        last_tag = self.get_last_tag()  # v0.0.4-16-04-2025

        if last_tag:
            match = re.match(r'v(\d+)\.(\d+)\.(\d+)', last_tag)
            if match:
                major, minor, patch = map(int, match.groups())
                patch += 1
            else:
                major, minor, patch = 0, 0, 1
        else:
            major, minor, patch = 0, 0, 1

        date_part = datetime.now().strftime("%d-%m-%Y")
        return f"{major}.{minor}.{patch}-{date_part}"

    def get_statistics(self, commits: List[Dict]) -> Dict:
        if not commits:
            return {
                'total_commits': 0,
                'contributors': 0,
                'files_changed': 0
            }

        contributors = set(commit['author'] for commit in commits)

        last_tag = self.get_last_tag()
        if last_tag:
            diff_stat = self.run_git_command([
                'git', 'diff', '--shortstat', f"{last_tag}..HEAD"
            ])
            files_match = re.search(r'(\d+) files? changed', diff_stat)
            files_changed = int(files_match.group(1)) if files_match else 0
        else:
            files_changed = len(self.run_git_command(['git', 'ls-files']).split('\n'))

        return {
            'total_commits': len(commits),
            'contributors': len(contributors),
            'files_changed': files_changed
        }

    def generate_changelog(self) -> str:
        last_tag = self.get_last_tag()

        print(f"Last tag: {last_tag if last_tag else 'No tags found'}")

        commits = self.get_commits_since(last_tag)

        if not commits:
            return "# Changelog\n\nNo changes since last release.\n"

        print(f"Found {len(commits)} commits since last release")

        self.categorize_commits(commits)

        version = self.generate_version()
        current_date = datetime.now().strftime("%Y-%m-%d")

        changelog = []
        changelog.append("# Changelog\n")
        changelog.append(f"## [{version}] - {current_date}\n")

        for commit_type, category_name in self.COMMIT_TYPES.items():
            if commit_type in self.categorized_commits:
                changelog.append(f"\n### {category_name}\n")
                for item in self.categorized_commits[commit_type]:
                    changelog.append(f"- {item['entry']}\n")

        stats = self.get_statistics(commits)
        changelog.append("\n### Statistics\n")
        changelog.append(f"- Total commits: {stats['total_commits']}\n")
        changelog.append(f"- Contributors: {stats['contributors']}\n")
        changelog.append(f"- Files changed: {stats['files_changed']}\n")

        if commits:
            contributors = sorted(set(commit['author'] for commit in commits))
            changelog.append("\n### 👥 Contributors\n")
            for contributor in contributors:
                changelog.append(f"- {contributor}\n")

        return ''.join(changelog)

    def save_files(self):
        changelog_content = self.generate_changelog()

        with open('CHANGELOG.md', 'w', encoding='utf-8') as f:
            f.write(changelog_content)
        print("Generated CHANGELOG.md")

        lines = changelog_content.split('\n')
        release_notes = '\n'.join(lines[:100])
        with open('release_notes.md', 'w', encoding='utf-8') as f:
            f.write(release_notes)
        print("Generated release_notes.md")

        version = self.generate_version()
        with open('release.env', 'w', encoding='utf-8') as f:
            f.write(f'RELEASE_VERSION={version}\n')
        print(f"Generated release version: {version}")

        return version


def main():
    try:
        generator = ChangelogGenerator()
        version = generator.save_files()

        print("\n" + "=" * 50)
        print("Changelog generation completed successfully!")
        print(f"Version: {version}")
        print("=" * 50)

        return 0
    except Exception as e:
        print(f"\nError generating changelog: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
