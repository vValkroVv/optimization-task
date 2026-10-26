import logging
import os
import re


def extract_stderr_traceback(stderr_file_path):
    try:
        if not os.path.exists(stderr_file_path):
            return None

        with open(stderr_file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        traceback_lines = []
        in_traceback = False
        current_traceback = []

        for line in content.split('\n'):
            if re.match(r'\[rank\d+\]:\s*Traceback \(most recent call last\):', line):
                if current_traceback:
                    traceback_lines.extend(current_traceback)
                    traceback_lines.append("")
                current_traceback = [line]
                in_traceback = True
            elif in_traceback:
                if line.strip():
                    current_traceback.append(line)
                else:
                    if current_traceback and any(
                            pattern in current_traceback[-1]
                            for pattern in ['Error:', 'Exception:', 'OSError:', 'ValueError:', 'TypeError:']
                    ):
                        traceback_lines.extend(current_traceback)
                        traceback_lines.append("")
                        current_traceback = []
                        in_traceback = False
                    else:
                        current_traceback.append(line)

        if current_traceback:
            traceback_lines.extend(current_traceback)

        return '\n'.join(traceback_lines).strip() if traceback_lines else None

    except Exception as e:
        logging.warning(f"Failed to read stderr file {stderr_file_path}: {e}")
        return None
