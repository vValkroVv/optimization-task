import logging
import os
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from src.models import AbstractModel
from src.utils.io import load_jsonl, get_existing_ids, JsonlWriter

logger = logging.getLogger(__name__)

class GenerationRunner:
    def __init__(self, model: AbstractModel, questions_path: str, output_dir: str, overwrite: bool = False, output_file: str | None = None, parallel_api_calls: int = 1) -> None:
        self.model = model
        self.questions_path = questions_path
        self.output_dir = Path(output_dir)
        if output_file:
            self.output_file = Path(output_file)
        else:
            self.output_file = self.output_dir / f"{self.model.name}.jsonl"
        self.overwrite = overwrite
        self.parallel_api_calls = parallel_api_calls

    def run(self) -> None:
        logger.info(f"Starting GenerationRunner for model: {self.model.name} (Parallelism: {self.parallel_api_calls})")
        
        # 1. Load Questions
        questions = load_jsonl(self.questions_path)
        if not questions:
            logger.error(f"No questions found at {self.questions_path}")
            return

        # 2. Check Existing Generations
        existing_ids = set()
        if self.output_file.exists() and not self.overwrite:
            existing_ids = get_existing_ids(self.output_file)
            logger.info(f"Found {len(existing_ids)} existing generations.")
        elif self.overwrite:
            logger.info("Overwrite mode enabled. Clearing existing file.")
            if self.output_file.exists():
                os.remove(self.output_file)

        # 3. Filter missing questions
        questions_to_process = [q for q in questions if str(q.get("question_id") or q.get("uid")) not in existing_ids]
        
        if not questions_to_process:
            logger.info("All questions already answered. Skipping generation.")
            return

        logger.info(f"Generating answers for {len(questions_to_process)} questions...")
        
        # 4. Processing Loop
        # Initialize Writer (in append mode)
        writer = JsonlWriter(self.output_file, overwrite=False)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_api_calls) as executor:
            futures = {executor.submit(self.model.generate_answer, q): q for q in questions_to_process}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Generating {self.model.name}"):
                q = futures[future]
                try:
                    answer = future.result()
                    writer.write(answer)
                except Exception as e:
                    logger.error(f"Exception during generation for QID {q.get('question_id') or q.get('uid')}: {e}")

        logger.info(f"Generation for {self.model.name} completed.")
