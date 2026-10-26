import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)


def load_prompt(value: Optional[str], *search_roots: Path, default: str = "") -> str:
    """
    Resolves a system prompt from either a plain text string or a file path.

    Resolution order:
    1. If value is empty/None — return default.
    2. Try value as an absolute path.
    3. Try value relative to each search_root in order.
    4. If no file found — treat value as literal text and return it as-is.
    """
    if not value:
        return default

    candidates = [Path(value)]
    for root in search_roots:
        candidates.append(root / value)

    for path in candidates:
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8").strip()
                logger.info(f"Loaded prompt from file: {path}")
                return text
            except Exception as e:
                logger.warning(f"Failed to read prompt file {path}: {e}")

    return value

def load_jsonl(path: str) -> List[Dict]:
    """Loads a JSONL file into a list of dicts."""
    data = []
    path = Path(path)
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return data
        
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return data

class JsonlWriter:
    """
    Writes JSONL files safely.
    """
    def __init__(self, path: str, overwrite: bool = False):
        self.path = Path(path)
        self.overwrite = overwrite
        self.mode = 'w' if overwrite else 'a'
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data: Dict):
        with open(self.path, 'a', encoding='utf-8') as f: # Always append, user logic handles clearing if overwrite needed
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def write_batch(self, data_list: List[Dict]):
        with open(self.path, 'a', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_existing_ids(path: str) -> Set[str]:
    """Returns a set of question identifiers present in a JSONL file.
    Uses 'question_id' or 'uid'"""
    ids = set()
    for item in load_jsonl(path):
        qid = item.get("question_id") or item.get("uid")
        if qid is not None:
            ids.add(str(qid))
    return ids

def normalize_question(question: Dict) -> Dict:
    """
    Normalizes a question to the universal Arena Hard format.
    
    Supported input formats:
    - arena-hard-v0.1: {'question_id': str, 'turns': [{'content': str}]}
    - arena-hard-v2.0: {'uid': str, 'prompt': str}
    - mt_bench: {'question_id': int, 'turns': [str, str, ...]}
    - ru-llm-arena: {'question_id': int, 'turns': [{'content': str}]}
    - arena-hard-logs-v3: {'question_id': str, 'turns': [{'content': str}]}
    
    Returns universal format:
    {
        'question_id': str,
        'turns': [{'content': str, 'role': str}],
        'category': str (optional),
        'cluster': str (optional),
        'subcategory': str (optional)
    }
    
    Args:
        question: Question dict in any supported format
        
    Returns:
        Normalized question dict
        
    Raises:
        ValueError: If question format is not recognized
    """
    normalized = {}
    
    # 1. Normalize question_id
    # Supported fields: question_id, uid, id
    question_id = (
        question.get('question_id') or 
        question.get('uid') or 
        question.get('id')
    )
    
    if question_id is None:
        raise ValueError("Question must contain 'question_id', 'uid', or 'id'")
    
    # Always cast to string
    normalized['question_id'] = str(question_id)
    
    # 2. Normalize turns
    if 'prompt' in question:
        # arena-hard-v2.0 format: has 'prompt', no 'turns'
        normalized['turns'] = [{
            'content': question['prompt'],
            'role': 'user'
        }]
    elif 'turns' in question:
        # Format with 'turns'
        normalized_turns = []
        
        for turn in question['turns']:
            if isinstance(turn, str):
                # mt_bench format: turns is an array of strings
                normalized_turns.append({
                    'content': turn,
                    'role': 'user'
                })
            elif isinstance(turn, dict):
                # arena-hard / ru-llm-arena format: turns is an array of objects
                normalized_turns.append({
                    'content': turn.get('content', ''),
                    'role': turn.get('role', 'user')  # Default to 'user' if role is missing
                })
            else:
                # Unknown turn type
                raise ValueError(f"Unknown turn type: {type(turn)}")
        
        if not normalized_turns:
            raise ValueError("turns cannot be empty")
        
        normalized['turns'] = normalized_turns
    else:
        # Neither 'turns' nor 'prompt' exists
        raise ValueError("Question must contain 'turns' or 'prompt'")
    
    # 3. Preserve optional fields
    optional_fields = ['category', 'cluster', 'subcategory']
    for field in optional_fields:
        if field in question:
            normalized[field] = question[field]
    
    return normalized
