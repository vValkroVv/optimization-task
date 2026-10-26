from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json
import logging
import os
import base64
import requests
import time
from pathlib import Path

from src.utils.io import load_prompt

logger = logging.getLogger(__name__)

GIGACHAT_IFT_CLIENT_ID_ENV = "GIGACHAT_IFT_CLIENT_ID"
GIGACHAT_IFT_CLIENT_SECRET_ENV = "GIGACHAT_IFT_CLIENT_SECRET"
IFT_TOKEN_URL = "https://gigachat.sberdevices.ru/v1/token"
IFT_CHAT_URL = "https://gigachat.ift.sberdevices.ru/v1/chat/completions"


def _get_ift_token() -> str:
    client_id = os.environ.get(GIGACHAT_IFT_CLIENT_ID_ENV, "").strip()
    client_secret = os.environ.get(GIGACHAT_IFT_CLIENT_SECRET_ENV, "").strip()
    if not client_id or not client_secret:
        raise ValueError(
            f"IFT API requires env vars: {GIGACHAT_IFT_CLIENT_ID_ENV} and {GIGACHAT_IFT_CLIENT_SECRET_ENV}"
        )
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8").replace("\n", "")
    headers = {
        "Authorization": f"Basic {auth}",
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Gigaclient",
    }
    resp = requests.post(IFT_TOKEN_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("access_token") or data.get("tok", "")

class AbstractModel(ABC):
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config

    @abstractmethod
    def generate_answer(self, question: Dict) -> Dict:
        """
        Generates an answer for a single question.
        Returns a dictionary representing the answer object (standard Arena format).
        """
        pass

    @abstractmethod
    def batch_generate(self, questions: List[Dict]) -> List[Dict]:
        """
        Generates answers for a batch of questions.
        """
        pass

class APIModel(AbstractModel):
    """
    API-backed model. api_type: openai | trt | ift.
    - openai: base_url/chat/completions, Bearer api_key
    - trt: base_url/chat (same contract)
    - ift: GigaChat IFT; token from env GIGACHAT_IFT_CLIENT_ID / GIGACHAT_IFT_CLIENT_SECRET

    Any config keys not in _INTERNAL_KEYS are passed through to the API payload,
    allowing provider-specific parameters (e.g. reasoning, verbosity, top_p).
    """

    _INTERNAL_KEYS = frozenset({
        "name", "path", "artifacts_dir",
        "api_base", "api_key", "api_type",
        "num_retries", "parallel_api_calls", "n_parallel",
        "system_prompt", "path_to_sys",
        "request_timeout",
    })

    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.api_type = config.get("api_type", "openai")
        self.api_base = config.get("api_base", "")
        self.api_key = config.get("api_key", "EMPTY")
        self.request_timeout = int(config.get("request_timeout", 120))
        raw_num_retries = config.get("num_retries", 3)
        try:
            self.num_retries = int(raw_num_retries)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid num_retries=%r for model %s; fallback to 3",
                raw_num_retries,
                name,
            )
            self.num_retries = 3
        if self.num_retries < 1:
            logger.warning(
                "num_retries must be >= 1 for model %s; got %s, fallback to 1",
                name,
                self.num_retries,
            )
            self.num_retries = 1
        self._base_headers = {"Content-Type": "application/json", "User-Agent": "Gigaclient"}

    @staticmethod
    def _to_primitive(value: object) -> object:
        """Convert OmegaConf containers to plain Python types for JSON serialization."""
        try:
            from omegaconf import DictConfig, ListConfig
            if isinstance(value, DictConfig):
                return {k: APIModel._to_primitive(v) for k, v in value.items()}
            if isinstance(value, ListConfig):
                return [APIModel._to_primitive(v) for v in value]
        except ImportError:
            pass
        return value

    def _headers_for_request(self) -> Dict[str, str]:
        if self.api_type == "ift":
            token = _get_ift_token()
            return {**self._base_headers, "Authorization": f"Bearer {token}"}
        return {**self._base_headers, "Authorization": f"Bearer {self.api_key}"}

    def _request_url(self) -> str:
        if self.api_type == "trt":
            return f"{self.api_base.rstrip('/')}/chat"
        if self.api_type == "ift":
            return IFT_CHAT_URL
        return f"{self.api_base.rstrip('/')}/chat/completions"

    def generate_answer(self, question: Dict) -> Dict:
        # Construct messages
        messages = []
        
        # Add system prompt if specified
        if self.config.get("path_to_sys"):
            logger.warning(
                "Config key 'path_to_sys' is deprecated and will be removed in a future version. "
                "Use 'system_prompt' instead (accepts both text and file path)."
            )
        sys_prompt_raw = self.config.get("system_prompt") or self.config.get("path_to_sys")
        if sys_prompt_raw:
            sys_prompt = load_prompt(sys_prompt_raw)
            messages.append({"role": "system", "content": sys_prompt})

        # Support both LMSYS (turns) and Official Arena Hard (prompt) formats
        turns = question.get("turns", [])
        if turns:
            for turn in turns:
                role = turn.get("role", "user")
                content = turn.get("content")
                messages.append({"role": role, "content": content})
        elif "prompt" in question:
            messages.append({"role": "user", "content": question["prompt"]})
        
        if not messages:
            logger.error(f"No content found in question: {question}")
            return None

        payload = {
            "model": self.name,
            "messages": messages,
        }
        # Pass through all non-internal config keys to the API payload.
        # This allows provider-specific parameters (e.g. reasoning, verbosity,
        # top_p, temperature, max_tokens) to be set directly in the config.
        for k, v in self.config.items():
            if k not in self._INTERNAL_KEYS and k not in payload:
                payload[k] = self._to_primitive(v)
        if self.api_type == "ift":
            payload["profanity_check"] = False

        url = self._request_url()
        headers = self._headers_for_request()
        for i in range(self.num_retries):
            try:
                logger.debug(
                    f"Sending request to {self.api_base} for QID: {question.get('question_id') or question.get('uid')}"
                )
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                logger.debug(f"Received response for QID: {question.get('question_id') or question.get('uid')}")
                response.raise_for_status()
                result = response.json()
                
                # Extract content
                answer_content = result["choices"][0]["message"]["content"]
                
                # Support both ID types in the output
                qid = question.get("question_id") or question.get("uid")
                id_key = "question_id" if "question_id" in question else "uid"

                return {
                    id_key: qid,
                    "model_id": self.name,
                    "choices": [
                        {
                            "index": 0,
                            "turns": [{"content": answer_content, "role": "assistant"}]
                        }
                    ],
                    "tstamp": time.time()
                }

            except Exception as e:
                error_msg = str(e)
                if hasattr(e, "response") and getattr(e.response, "text", None):
                    error_msg += f" | Response: {e.response.text}"
                logger.warning(f"API request failed (attempt {i + 1}/{self.num_retries}): {error_msg}")
                if i < self.num_retries - 1:
                    time.sleep(2 ** i)
        
        logger.error(
            f"Failed to generate answer for question {question.get('question_id')} "
            f"after {self.num_retries} retries."
        )
        raise RuntimeError(f"API failed after {self.num_retries} retries for model {self.name}")

    def batch_generate(self, questions: List[Dict]) -> List[Dict]:
        # For simplicity, sequential processing. 
        # In production, we'd use ThreadPoolExecutor here.
        results = []
        for q in questions:
            res = self.generate_answer(q)
            if res:
                results.append(res)
        return results

def create_model(name: str, config: Dict, api_type: Optional[str] = None) -> AbstractModel:
    """
    Creates a model from configuration.
    
    Logic:
    - If api_base exists or api_type=ift with env GIGACHAT_IFT_*: API model
    - Otherwise error (models without API should not reach this point;
      they are handled in setup_model_generation via existing generations)
    """
    cfg = {**config}
    if api_type is not None:
        cfg["api_type"] = api_type
    effective_type = cfg.get("api_type", "openai")
    has_api_base = bool(cfg.get("api_base"))
    is_ift = effective_type == "ift"
    has_ift_env = bool(os.environ.get(GIGACHAT_IFT_CLIENT_ID_ENV)) and bool(
        os.environ.get(GIGACHAT_IFT_CLIENT_SECRET_ENV)
    )
    if is_ift and not has_ift_env:
        raise ValueError(
            f"api_type=ift requires env: {GIGACHAT_IFT_CLIENT_ID_ENV}, {GIGACHAT_IFT_CLIENT_SECRET_ENV}"
        )
    if not has_api_base and not is_ift:
        raise ValueError(f"Cannot create model '{name}': api_base required for api_type openai/trt")
    return APIModel(name, cfg)
