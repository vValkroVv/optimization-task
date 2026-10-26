import concurrent.futures
import hashlib
import json
import logging
import re
import sys
import tempfile
from pathlib import Path
from tqdm import tqdm
from typing import Any
from src.judges.judge import Judge
from src.utils.io import load_jsonl, JsonlWriter

logger = logging.getLogger(__name__)
PAM_PARALLEL_VERDICT_ATTEMPTS = 5
_STRICT_VERDICT_SUFFIX = (
    "IMPORTANT: End your answer with exactly one final verdict label in the form "
    "[[A>>B]], [[A>B]], [[A=B]], [[B>A]], or [[B>>A]]."
)


class MissingVerdictError(RuntimeError):
    """A battle produced no parseable verdict label after all retries.

    Carries the full judgment_record (with each game's prompt + raw judge
    output) so the orchestrator can persist it for inspection. It is raised
    (not returned) so the battle stays excluded from judgment.jsonl and the
    metric — a null verdict written there would be miscounted as a tie.
    """

    def __init__(self, judgment_record: dict[str, Any], missing: list[int], uid: str | None) -> None:
        super().__init__(
            f"Missing verdict labels after retries; uid={uid}, games={missing}"
        )
        self.judgment_record = judgment_record
        self.missing = missing
        self.uid = uid


class JudgeRunner:
    def __init__(
        self,
        judge: Judge,
        questions_path: str,
        candidate_path: str,
        baseline_path: str,
        output_dir: str,
        candidate_model_name: str,
        baseline_model_name: str,
        overwrite: bool = False,
        output_file: str | None = None,
        parallel_api_calls: int = 1,
        verbose_errors: bool = False,
        fail_on_error: bool = False,
        category_namespace: str = "",
    ) -> None:
        self.judge = judge
        self.questions_path = Path(questions_path).resolve()
        self.candidate_path = Path(candidate_path).resolve()
        self.baseline_path = Path(baseline_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.candidate_model_name = candidate_model_name
        self.baseline_model_name = baseline_model_name
        self.parallel_api_calls = parallel_api_calls
        self.verbose_errors = verbose_errors
        self.fail_on_error = fail_on_error
        # Non-empty when several runners share one judging pool (one runner per
        # dataset): prefixes this runner's JUDGE_SETTINGS keys so runners with
        # different system prompts never collide on a shared category name.
        self.category_namespace = category_namespace
        self.fail_on_missing_verdict = bool(
            self.judge.config.get("fail_on_missing_verdict", True)
        )
        self.missing_verdict_retries = int(
            self.judge.config.get("missing_verdict_retries", 1)
        )
        self.pam_mode = bool(self.judge.config.get("pam_mode", False))
        if self.missing_verdict_retries > 1 and self.judge.config.get("temperature", 0.0) == 0.0:
            logger.warning(
                "WARNING: missing_verdict_retries=%d with temperature=0. "
                "At zero temperature the model is deterministic, so retries beyond 1 "
                "will produce identical output and waste API calls.",
                self.missing_verdict_retries,
            )

        if output_file:
            self.output_file = Path(output_file).resolve()
        else:
            self.output_file = self.output_dir / "judgment.jsonl"
        # Battles whose verdict never parsed (after retries) are dumped here for
        # inspection. Kept SEPARATE from judgment.jsonl on purpose: this file is
        # never read by the metric (a null verdict there would count as a tie)
        # and never feeds the resume skip-set, so such battles are re-judged on
        # restart rather than treated as already done.
        self.failed_output_file = self.output_file.parent / "failed_judgments.jsonl"
        self.overwrite = overwrite

        if self.judge.config.get("input_format", "prompt_template") == "structured_sbs_json":
            from src.judges.structured_sbs import StructuredSbsJudge

            self.judgment_func = StructuredSbsJudge(self.judge.model).judgment
            self.strict_retry_category = self._ns_key("__strict_retry__")
            self.judge_settings = {}
            return
        
        # Resolve submodule path from repository root instead of current working directory.
        project_root = Path(__file__).resolve().parents[2]
        self.external_repo_path = (project_root / "contribs/arena-hard-auto").resolve()
        if not self.external_repo_path.is_dir():
            raise FileNotFoundError(
                f"arena-hard-auto repo not found: {self.external_repo_path}\n"
                "Initialize the submodule: git submodule update --init --recursive contribs/arena-hard-auto"
            )
        if str(self.external_repo_path) not in sys.path:
            sys.path.insert(0, str(self.external_repo_path))
        
        # Ensure utils is a package (fix for Pupy101/Official structure missing __init__.py)
        utils_init = self.external_repo_path / "utils" / "__init__.py"
        if not utils_init.exists():
            utils_init.parent.mkdir(parents=True, exist_ok=True)
            utils_init.touch()
            logger.info(f"Created missing __init__.py at {utils_init}")

        # Hack for naming difference: gigachat vs giga
        try:
            import gigachat
            sys.modules["giga"] = gigachat
            logger.info("Mapped 'gigachat' to 'giga' in sys.modules")
        except ImportError:
            logger.warning("Package 'gigachat' not found, 'giga' import might fail.")

        try:
            from gen_judgment import judgment
            from utils.judge_utils import JUDGE_SETTINGS

            # JUDGE_SETTINGS is a module-level dict shared by every runner in the
            # process. We only ADD runner-scoped keys (see _ns_key) and never
            # mutate existing entries, so runners that later share one judging
            # pool read it concurrently without synchronization. Baseline answers
            # are passed to gen_judgment explicitly; the per-category "baseline"
            # field is only read by arena-hard-auto's own CLI entrypoint.
            # Dedicated strict category used for retries when verdict label is missing.
            self.strict_retry_category = self._ns_key("__strict_retry__")
            JUDGE_SETTINGS[self.strict_retry_category] = {
                "baseline": baseline_model_name,
                "system_prompt": f"{self.judge.system_prompt}\n\n{_STRICT_VERDICT_SUFFIX}",
            }
            
            self.judgment_func = judgment
            self.judge_settings = JUDGE_SETTINGS
            logger.info(f"Successfully imported 'judgment' from {self.external_repo_path}")
        except ImportError as e:
            logger.error(f"Failed to import gen_judgment from {self.external_repo_path}: {e}")
            raise e

    def _ns_key(self, category: str) -> str:
        """This runner's JUDGE_SETTINGS key for a category.

        gen_judgment resolves the judge system prompt through the shared
        JUDGE_SETTINGS dict at judgment time, so when several runners share one
        judging pool each runner needs its own keys: two datasets may use the
        same category name with different system prompts.
        """
        if self.category_namespace:
            return f"{self.category_namespace}::{category}"
        return category

    def _index_answers(self, path: str | Path) -> dict[str, dict[str, Any]]:
        data = load_jsonl(path)
        # Handle both LMSYS (question_id) and Official (uid) formats
        indexed = {}
        for item in data:
            key = item.get("question_id") or item.get("uid")
            if key:
                indexed[str(key)] = item
        return indexed

    def _normalize_question_for_judgment(self, question: dict[str, Any]) -> dict[str, Any]:
        """gen_judgment expects question['uid'] and question['prompt']; arena-general/mt_bench provide question_id and turns."""
        out = dict(question)
        out["uid"] = question.get("uid")
        if out["uid"] is None:
            out["uid"] = str(question.get("question_id"))
        if out.get("prompt") is None:
            turns = question.get("turns", [])
            if turns:
                parts = [
                    t if isinstance(t, str) else (t.get("content") or "")
                    for t in turns
                ]
                out["prompt"] = "\n\n---\n\n".join(parts)
            else:
                out["prompt"] = ""
        return out

    def _normalize_answer_for_judgment(self, item: dict[str, Any]) -> dict[str, Any]:
        """Normalize answer/baseline item to format expected by gen_judgment: 'model' and 'messages' with content['answer']."""
        out = dict(item)
        out["model"] = item.get("model") or item.get("model_id") or "unknown"
        if "messages" in item:
            messages = list(item["messages"])
            if messages:
                last = dict(messages[-1])
                content = last.get("content")
                if isinstance(content, str):
                    last["content"] = {"answer": content}
                elif isinstance(content, dict) and "answer" not in content:
                    last["content"] = {**content, "answer": content.get("text", "")}
                messages[-1] = last
            out["messages"] = messages
        elif "choices" in item and item["choices"]:
            turns = item["choices"][0].get("turns", [])
            text = turns[-1].get("content", "") if turns else ""
            out["messages"] = [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": {"answer": text}},
            ]
        return out

    @staticmethod
    def _extract_answer_text(item: dict[str, Any]) -> str:
        messages = item.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                content = last.get("content")
                if isinstance(content, dict):
                    return str(content.get("answer") or content.get("text") or "")
                if isinstance(content, str):
                    return content
        choices = item.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                turns = first.get("turns")
                if isinstance(turns, list) and turns:
                    content = turns[-1].get("content")
                    if isinstance(content, dict):
                        return str(content.get("answer") or content.get("text") or "")
                    if isinstance(content, str):
                        return content
        return ""

    @staticmethod
    def _normalize_answer_text_for_compare(text: str) -> str:
        # Collapse whitespace and compare semantic content, not formatting noise.
        return " ".join(str(text).split()).strip()

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _is_identical_answer(
        self,
        q_norm: dict[str, Any],
        cand_norm: dict[str, Any],
        base_norm: dict[str, Any],
    ) -> bool:
        qid = str(q_norm.get("uid") or q_norm.get("question_id"))
        category = q_norm.get("category")
        cand_model = str(cand_norm.get("model"))
        base_model = str(base_norm.get("model"))
        if cand_model == base_model:
            msg = (
                "Candidate and baseline model names are identical; refusing judgment "
                f"uid={qid} category={category} model={cand_model}"
            )
            logger.error(msg)
            raise AssertionError(msg)

        cand_text = self._normalize_answer_text_for_compare(self._extract_answer_text(cand_norm))
        base_text = self._normalize_answer_text_for_compare(self._extract_answer_text(base_norm))
        if cand_text == base_text:
            logger.warning(
                f"!!! Identical answers detected for UID {qid} (category: {category}). "
                f"Candidate and baseline both produced the same text (hash: {self._text_hash(cand_text)}). "
                "Recording automated TIE to save time and tokens. !!!"
            )
            return True
        return False

    def _annotate_game_roles_and_hashes(
        self,
        judgment_record: dict[str, Any],
        cand_norm: dict[str, Any],
        base_norm: dict[str, Any],
    ) -> None:
        games = judgment_record.get("games")
        if not isinstance(games, list):
            return

        cand_hash = self._text_hash(self._normalize_answer_text_for_compare(self._extract_answer_text(cand_norm)))
        base_hash = self._text_hash(self._normalize_answer_text_for_compare(self._extract_answer_text(base_norm)))
        role_pairs = (
            ("baseline", "candidate"),
            ("candidate", "baseline"),
        )
        for idx, game in enumerate(games):
            if not isinstance(game, dict):
                continue
            if idx < len(role_pairs):
                role_a, role_b = role_pairs[idx]
            else:
                role_a, role_b = role_pairs[idx % 2]
            game["role_a"] = role_a
            game["role_b"] = role_b
            game["model_a"] = self.baseline_model_name if role_a == "baseline" else self.candidate_model_name
            game["model_b"] = self.baseline_model_name if role_b == "baseline" else self.candidate_model_name
            game["answer_a_hash"] = base_hash if role_a == "baseline" else cand_hash
            game["answer_b_hash"] = base_hash if role_b == "baseline" else cand_hash

    def _ensure_categories_in_judge_settings(self, questions: list[dict[str, Any]]) -> None:
        """
        arena-hard-auto expects every question['category'] to exist in JUDGE_SETTINGS.
        Register this runner's key (see _ns_key) for every question category.

        The system prompt inherits an existing upstream entry for the raw
        category name (arena-hard-auto ships per-category judge protocols,
        e.g. 'hard_prompt' vs 'creative_writing'); for custom categories the
        judge's own system prompt is used.

        Only additive writes to the shared JUDGE_SETTINGS dict, performed
        before the judging pool starts.
        """
        registered = set()
        for q in questions:
            category = q.get("category")
            if not category or category in registered:
                continue
            registered.add(category)
            upstream = self.judge_settings.get(category)
            if isinstance(upstream, dict) and upstream.get("system_prompt"):
                system_prompt = upstream["system_prompt"]
            else:
                system_prompt = self.judge.system_prompt
            self.judge_settings[self._ns_key(category)] = {
                "baseline": self.baseline_model_name,
                "system_prompt": system_prompt,
            }

        if registered:
            logger.info(
                "Registered JUDGE_SETTINGS for categories: %s",
                ", ".join(sorted(str(c) for c in registered)),
            )

    @staticmethod
    def _extract_verdict_from_game(
        game: dict[str, Any],
        pattern_compiled: re.Pattern[str],
    ) -> str | None:
        if not isinstance(game, dict):
            return None
        score = game.get("score")
        if isinstance(score, str) and score.strip() and score != "N/A":
            return score.strip()
        judgment = game.get("judgment")
        text = ""
        if isinstance(judgment, dict):
            text = judgment.get("answer", "") or ""
        elif isinstance(judgment, str):
            text = judgment
        if not text:
            return None
        matches = pattern_compiled.findall(text.upper())
        matches = [m.strip() for m in matches if isinstance(m, str) and m.strip()]
        return matches[-1] if matches else None

    def _missing_verdict_indices(
        self,
        judgment_record: dict[str, Any],
        pattern_compiled: re.Pattern[str],
    ) -> list[int]:
        games = judgment_record.get("games", [])
        missing: list[int] = []
        for idx, game in enumerate(games):
            verdict = self._extract_verdict_from_game(game, pattern_compiled)
            if verdict is None:
                missing.append(idx)
        return missing

    def _run_one_judgment_to_file(self, args_dict: dict[str, Any]) -> dict[str, Any]:
        output_file = Path(args_dict["output_file"])
        if output_file.exists():
            output_file.unlink()
        self.judgment_func(args_dict)
        rows = load_jsonl(output_file)
        if not rows:
            raise RuntimeError(f"Judgment function produced no rows in {output_file}")
        return rows[-1]

    def _build_judgment_args(
        self,
        question: dict[str, Any],
        cand: dict[str, Any],
        base: dict[str, Any],
        configs: dict[str, Any],
        endpoint_dict: dict[str, Any],
        pattern_compiled: re.Pattern[str],
    ) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl",
            prefix="dredd_judgment_",
            dir=self.output_file.parent,
            delete=False,
        ) as tmpf:
            tmp_output_path = tmpf.name
        return {
            "question": question,
            "answer": cand,
            "baseline": base,
            "reference": None,
            "configs": configs,
            "output_file": tmp_output_path,
            "settings": endpoint_dict,
            "regex_pattern": pattern_compiled,
        }

    def _judge_with_retries(
        self,
        q_norm: dict[str, Any],
        cand_norm: dict[str, Any],
        base_norm: dict[str, Any],
        configs: dict[str, Any],
        endpoint_dict: dict[str, Any],
        pattern_compiled: re.Pattern[str],
        original_category: str | None = None,
    ) -> dict[str, Any]:
        # q_norm may carry a runner-scoped category (see _ns_key); when
        # original_category is given, the persisted record gets it back so
        # judgment.jsonl never leaks internal registry keys.
        def _attempt(question: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
            args_dict = self._build_judgment_args(
                question, cand_norm, base_norm, configs, endpoint_dict, pattern_compiled
            )
            try:
                record = self._run_one_judgment_to_file(args_dict)
            finally:
                tmp_path = Path(args_dict["output_file"])
                if tmp_path.exists():
                    tmp_path.unlink()
            if original_category is not None:
                record["category"] = original_category
            self._annotate_game_roles_and_hashes(record, cand_norm, base_norm)
            return record, self._missing_verdict_indices(record, pattern_compiled)

        uid = q_norm.get("uid") or q_norm.get("question_id")

        # First attempt with original category.
        judgment_record, missing = _attempt(q_norm)
        if not missing:
            return judgment_record

        # Retry with strict verdict instruction.
        q_retry = dict(q_norm)
        q_retry["category"] = self.strict_retry_category

        if self.pam_mode:
            logger.warning(
                "Missing verdict labels for uid=%s (games=%s); firing %s parallel strict retries",
                uid,
                missing,
                PAM_PARALLEL_VERDICT_ATTEMPTS,
            )
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=PAM_PARALLEL_VERDICT_ATTEMPTS
            )
            try:
                futures = [
                    executor.submit(_attempt, q_retry)
                    for _ in range(PAM_PARALLEL_VERDICT_ATTEMPTS)
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        record, record_missing = future.result()
                    except Exception:
                        logger.warning(
                            "Parallel strict retry failed for uid=%s", uid, exc_info=True
                        )
                        continue
                    if not record_missing:
                        return record
                    judgment_record, missing = record, record_missing
            finally:
                # In-flight HTTP calls cannot be cancelled; don't block on them.
                executor.shutdown(wait=False, cancel_futures=True)
        else:
            for retry_idx in range(1, self.missing_verdict_retries + 1):
                logger.warning(
                    "Missing verdict labels on attempt %s for uid=%s (games=%s); running strict retry",
                    retry_idx,
                    uid,
                    missing,
                )
                judgment_record, missing = _attempt(q_retry)
                if not missing:
                    return judgment_record

        if self.fail_on_missing_verdict:
            raise MissingVerdictError(judgment_record, missing, uid)
        logger.warning(
            "Proceeding with missing verdict labels (configured); uid=%s games=%s",
            uid,
            missing,
        )
        return judgment_record

    def prepare(self) -> list[dict[str, Any]]:
        """Validate the judgment cache and build remaining battle tasks.

        Returns a list of {"kwargs": ..., "meta": ...} dicts for
        _judge_with_retries. Must be called from the orchestrating thread
        BEFORE the judging pool starts: this is the only place that writes to
        the shared JUDGE_SETTINGS registry.
        """
        logger.info("Starting JudgeRunner (Direct Import Mode)...")

        self._skipped = False
        self._ok_count = 0
        self._fail_count = 0
        self._precheck_fail_count = 0
        self.writer: JsonlWriter | None = None
        self.failed_writer: JsonlWriter | None = None

        # 1. Non-destructive phase: parse config and load data first, so a
        # failure here (bad regex, malformed inputs) never deletes existing
        # artifacts — of this runner or of siblings prepared earlier in a group.

        # Regex Pattern: Pupy101 uses regex from config
        regex_pattern = self.judge.config.get("regex_pattern", r"\[\[([AB<>=]+)\]\]")
        pattern_compiled = re.compile(regex_pattern)

        questions = load_jsonl(self.questions_path)
        candidate_answers = self._index_answers(self.candidate_path)
        baseline_answers = self._index_answers(self.baseline_path)
        self._ensure_categories_in_judge_settings(questions)

        # 2. Check Output
        if self.output_file.exists() and self.overwrite:
            logger.info(f"Overwrite enabled. Removing existing output file: {self.output_file}")
            self.output_file.unlink()
        if self.failed_output_file.exists() and self.overwrite:
            logger.info(f"Overwrite enabled. Removing existing failed-judgments file: {self.failed_output_file}")
            self.failed_output_file.unlink()

        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # 3. Prepare Configs for the external function
        # The 'judgment' function expects a 'configs' dict and 'endpoint_dict'

        # Construct 'configs' dict mimicking what make_config returns
        configs = {
            "judge_model": self.judge.model.name,
            "system_prompt": self.judge.system_prompt,
            "prompt_template": self.judge.prompt_template, # Pass as string
            "pairwise": True,
            "temperature": self.judge.config.get("temperature", 0.0),
            "max_tokens": self.judge.config.get("max_tokens", 4096),
            "number_of_judgment_attempts": self.judge.config.get("num_retries", 2),
            "baseline": True,
            "reference": False,
            "regex_patterns": [pattern_compiled] # Changed to plural and list
        }

        # Construct 'endpoint_dict'
        # This is passed to get_answer -> chat_completion_...
        endpoint_dict = {
            "model": self.judge.model.name, # Changed from model_name to model
            "api_type": self.judge.model.config.get("api_type", "openai"),
            "request_timeout": self.judge.model.request_timeout,
            "endpoints": [{
                "api_base": self.judge.model.api_base,
                "api_key": self.judge.model.api_key,
            }],
        }

        # 4. Processing Loop
        identity = {**configs, "input_format": self.judge.config.get("input_format", "prompt_template")}
        identity.pop("number_of_judgment_attempts")
        identity["regex_patterns"] = [regex_pattern]
        identity["api_type"] = endpoint_dict["api_type"]
        identity["category_prompts"] = {
            str(q["category"]): self.judge_settings[self._ns_key(q["category"])]["system_prompt"]
            for q in questions if q.get("category")
        }

        tasks = []
        for q in questions:
            qid = str(q.get("question_id") or q.get("uid"))
            if qid not in candidate_answers:
                logger.warning(f"Missing Candidate answer for QID: {qid}")
                continue
            if qid not in baseline_answers:
                logger.warning(f"Missing Baseline answer for QID: {qid}")
                continue

            tasks.append({
                "question": q,
                "cand": candidate_answers[qid],
                "base": baseline_answers[qid]
            })

        if not tasks:
            logger.info("Nothing to judge.")
            self._skipped = True
            return []

        logger.info(f"Judging {len(tasks)} battles...")
        self.writer = JsonlWriter(str(self.output_file), overwrite=False)
        # Diagnostic sink for battles whose verdict never parsed (see __init__).
        # JsonlWriter creates the file lazily, so it stays absent unless a
        # battle actually fails.
        self.failed_writer = JsonlWriter(str(self.failed_output_file), overwrite=False)

        completed = []
        exec_tasks: list[dict[str, Any]] = []
        for task in tasks:
            q_norm = self._normalize_question_for_judgment(task["question"])
            cand_norm = self._normalize_answer_for_judgment(task["cand"])
            base_norm = self._normalize_answer_for_judgment(task["base"])
            # Keep model labels stable even when generation files were reused
            # from older runs with stale `model` fields.
            cand_norm["model"] = self.candidate_model_name
            base_norm["model"] = self.baseline_model_name

            try:
                if self._is_identical_answer(q_norm, cand_norm, base_norm):
                    # Automated Tie
                    automated_verdict = {
                        "uid": str(q_norm.get("uid") or q_norm.get("question_id")),
                        "category": q_norm.get("category"),
                        "games": [{"score": "[[A=B]]", "judgment": "Automated Tie due to identical response texts."}]
                    }
                    self._annotate_game_roles_and_hashes(automated_verdict, cand_norm, base_norm)
                    completed.append(automated_verdict)
                    continue
            except AssertionError as e:
                self._precheck_fail_count += 1
                logger.error(f"Precheck CRITICAL failure for QID={q_norm.get('uid')}: {e}")
                continue

            # Route the judgment through this runner's JUDGE_SETTINGS key; the
            # original category is restored in the persisted record.
            original_category = q_norm.get("category")
            q_judge = q_norm
            if original_category:
                q_judge = dict(q_norm)
                q_judge["category"] = self._ns_key(original_category)

            exec_tasks.append({
                "kwargs": {
                    "q_norm": q_judge,
                    "cand_norm": cand_norm,
                    "base_norm": base_norm,
                    "configs": configs,
                    "endpoint_dict": endpoint_dict,
                    "pattern_compiled": pattern_compiled,
                    "original_category": original_category,
                },
                "meta": {
                    "qid": str(q_norm.get("uid") or q_norm.get("question_id")),
                    "category": original_category,
                    "candidate_model": cand_norm.get("model"),
                    "baseline_model": base_norm.get("model"),
                },
            })
        probe_uid = None
        self.mean_var_hashes = None
        if identity["input_format"] == "structured_sbs_json" and exec_tasks:
            task = exec_tasks.pop(0)
            try:
                record = self._judge_with_retries(**task["kwargs"])
            except Exception as error:
                self._record_failure(task["meta"], error)
                raise RuntimeError("Judging completed with 1 failed task(s); cannot verify current judge identity") from error
            hashes = record["games"][0]["mean_var"]
            self.mean_var_hashes = {key: hashes[key] for key in ("protocol_sha256", "export_manifest_sha256")}
            identity["mean_var"] = self.mean_var_hashes
            probe_uid = str(record["uid"])
            completed.append(record)
        self.judge_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if self.output_file.exists():
            with self.output_file.open("rb+") as stream:
                data = stream.read()
                tail = data.rsplit(b"\n", 1)[-1]
                if tail:
                    try:
                        json.loads(tail)
                    except (ValueError, UnicodeDecodeError):
                        logger.warning("Discarding interrupted judgment tail in %s", self.output_file)
                        stream.truncate(len(data) - len(tail))
                    else:
                        stream.write(b"\n")
        cached = load_jsonl(self.output_file) if self.output_file.exists() else []
        if self.output_file.exists() and {row.get("judge_identity") for row in cached} != {self.judge_identity}:
            archive = Path(tempfile.mkdtemp(prefix="previous-judgments-", dir=self.output_file.parent))
            logger.warning("Judge identity changed or is missing; preserving old judgments in %s and rejudging", archive)
            for path in (
                self.output_file, self.failed_output_file,
                self.output_file.parent / "judgment_summary.jsonl",
                self.output_file.parent / "ci_quality.json",
            ):
                if path.exists():
                    path.rename(archive / path.name)
            cached = []
        existing_ids = {str(row.get("question_id") or row.get("uid")) for row in cached}
        for record in completed:
            uid = str(record["uid"])
            if uid not in existing_ids or uid == probe_uid:
                record["judge_identity"] = self.judge_identity
                self.writer.write(record)
                self._ok_count += 1
                existing_ids.add(uid)
        exec_tasks = [task for task in exec_tasks if task["meta"]["qid"] not in existing_ids]
        self._skipped = not exec_tasks and not self._ok_count and not self._precheck_fail_count
        return exec_tasks

    def _record_failure(self, meta: dict[str, Any], e: Exception) -> None:
        """Count and log one failed battle. Called from the orchestrating thread."""
        self._fail_count += 1
        base_msg = (
            f"Judgment failed for QID={meta['qid']}, category={meta['category']}, "
            f"candidate={meta['candidate_model']}, baseline={meta['baseline_model']}: "
            f"{type(e).__name__}: {e}"
        )
        if isinstance(e, MissingVerdictError):
            # Persist the full prompt + raw judge output for inspection.
            # The battle still counts as failed and stays out of
            # judgment.jsonl, so it never affects the metric.
            self.failed_writer.write(e.judgment_record)
            base_msg += f" | Raw judge output saved to {self.failed_output_file}"
        if isinstance(e, KeyError):
            missing_key = e.args[0] if e.args else "<unknown>"
            known_categories = ", ".join(sorted(self.judge_settings.keys()))
            base_msg += (
                f" | Likely missing category in arena-hard-auto JUDGE_SETTINGS: '{missing_key}'. "
                f"Known categories: [{known_categories}]"
            )

        if self.verbose_errors or logger.isEnabledFor(logging.DEBUG):
            logger.exception(base_msg)
        else:
            logger.error(base_msg)

    def _finalize(self) -> dict[str, int | bool]:
        """Log final per-runner stats. Called after the judging pool drains."""
        if self._skipped:
            return {"succeeded": 0, "failed": 0, "skipped": True}
        total_fail_count = self._fail_count + self._precheck_fail_count
        logger.info(
            "Judging completed. Succeeded: %s, failed: %s (runtime=%s, precheck=%s)",
            self._ok_count,
            total_fail_count,
            self._fail_count,
            self._precheck_fail_count,
        )
        return {"succeeded": self._ok_count, "failed": total_fail_count, "skipped": False}

    def run(self) -> dict[str, int | bool]:
        stats = run_judge_runners([self])[0]
        if self.fail_on_error and int(stats["failed"]) > 0:
            raise RuntimeError(f"Judging completed with {stats['failed']} failed task(s)")
        return stats


def run_judge_runners(runners: list["JudgeRunner"]) -> list[dict[str, int | bool]]:
    """Judge one or more datasets through a single shared thread pool.

    All runners must target the same judge endpoint. Battles from every runner
    are drained by one ThreadPoolExecutor sized by the largest
    parallel_api_calls in the group, so slow retries of one dataset overlap
    with regular battles of the others instead of idling the pool.

    Ordinary judgment failures return per-runner stats so
    the caller can persist summaries for every dataset before honoring
    fail_on_error (see JudgeRunner.run and dredd.run).
    Unverifiable or mixed judge identities are fatal.

    Thread-safety by construction:
    - all prepare() calls (the only JUDGE_SETTINGS writers) finish before the
      pool starts; worker threads only read it, through runner-scoped keys;
    - every result/failure is written from this thread, never from workers.
    """
    if len(runners) > 1:
        namespaces = [r.category_namespace for r in runners]
        if len(set(namespaces)) != len(namespaces) or "" in namespaces:
            raise ValueError(
                "Pooled runners must have unique non-empty category_namespace values; "
                f"got {namespaces!r}. Without them, runners sharing a category name "
                "would collide on JUDGE_SETTINGS keys."
            )

    prepared: list[tuple["JudgeRunner", dict[str, Any]]] = []
    for runner in runners:
        for task in runner.prepare():
            prepared.append((runner, task))

    if prepared:
        pool_size = max(r.parallel_api_calls for r in runners)
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
            future_to_task = {
                executor.submit(runner._judge_with_retries, **task["kwargs"]): (runner, task["meta"])
                for runner, task in prepared
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(future_to_task), desc="Judging"):
                runner, meta = future_to_task[future]
                if runner.mean_var_hashes is not None and future.exception() is None and any(
                    game["mean_var"][key] != value
                    for game in future.result()["games"] for key, value in runner.mean_var_hashes.items()
                ):
                    raise RuntimeError("Judge identity changed during scoring")
                try:
                    record = future.result()
                    record["judge_identity"] = runner.judge_identity
                    runner.writer.write(record)
                    runner._ok_count += 1
                except Exception as e:
                    runner._record_failure(meta, e)

    return [runner._finalize() for runner in runners]
