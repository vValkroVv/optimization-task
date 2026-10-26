import re
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from src.models import AbstractModel
from src.utils.io import load_prompt

logger = logging.getLogger(__name__)


class Judge:
    def __init__(self, model: AbstractModel, config: Dict[str, Any], project_root: Path) -> None:
        self.model = model
        self.config = config

        prompt_template_value = config.get("prompt_template", "{question}\nA:{answer_a}\nB:{answer_b}")
        self.prompt_template: str = load_prompt(prompt_template_value, project_root)

        system_prompt_value = config.get("system_prompt", "You are a judge.")
        self.system_prompt: str = load_prompt(system_prompt_value, project_root, default="You are a judge.")

        self.regex_pattern: re.Pattern[str] = re.compile(config.get("regex_pattern", r"\[\[([AB<>=]+)\]\]"))
        self.number_of_judgment_attempts: int = config.get("num_retries", 2)

    def _get_score(self, judgment: str, pairwise: bool = True) -> tuple[str | int | None, bool]:
        matches = self.regex_pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) == 0:
            return None, True # Not found, try again
        elif len(set(matches)) == 1:
            if pairwise:
                return matches[0].strip("\n"), False
            return int(matches[0]), False
        else:
            return None, False # Ambiguous

    def _get_model_answer_content(self, ans_obj: Dict) -> str:
        """Safe extraction of content from standard Arena answer format."""
        try:
            return ans_obj["choices"][0]["turns"][0]["content"]
        except (KeyError, IndexError, TypeError):
            logger.warning(f"Malformed answer object: {ans_obj}")
            return ""

    def evaluate_pairwise(self, question: Dict, answer_a: Dict, answer_b: Dict, baseline_model_name: str) -> Dict:
        """
        Runs the full pairwise evaluation logic (Arena Hard Auto style).
        1. Game 1: Candidate vs Baseline
        2. Game 2: Baseline vs Candidate
        
        Returns the full judgment object.
        """
        
        # Output structure
        output = {
            "question_id": question["question_id"],
            "model": answer_a["model_id"], # The candidate model
            "judge": self.model.name,
            "games": []
        }

        # Arena Hard Auto logic: 2 games to check position bias
        # Game 1: A = Candidate, B = Baseline
        # Game 2: A = Baseline, B = Candidate
        
        models_in_game = [
            (answer_a, answer_b), # Game 1
            (answer_b, answer_a)  # Game 2
        ]

        for game_idx, (model_1, model_2) in enumerate(models_in_game):
            # Construct the conversation history for the judge
            conv = [{"role": "system", "content": self.system_prompt}]
            
            # Prepare arguments for template
            prompt_args = {
                "question": question["turns"][0]["content"], # Simple fallback
                "question_1": question["turns"][0]["content"],
                "answer_a": self._get_model_answer_content(model_1),
                "answer_b": self._get_model_answer_content(model_2),
                "answer_1": self._get_model_answer_content(model_1),
                "answer_2": self._get_model_answer_content(model_2)
            }
            
            # Format user prompt
            # Supports both simple "{question}..." and complex "{question_1}..." templates
            try:
                user_prompt = self.prompt_template.format(**prompt_args)
            except KeyError as e:
                # Fallback if template keys don't match
                logger.warning(f"Template formatting failed: {e}. Falling back to simple concat.")
                user_prompt = f"Question: {prompt_args['question_1']}\n\nAnswer A:\n{prompt_args['answer_1']}\n\nAnswer B:\n{prompt_args['answer_2']}"

            conv.append({"role": "user", "content": user_prompt})

            # Judgment Loop (with retries)
            judgment_text = ""
            score = None
            
            for attempt in range(self.number_of_judgment_attempts):
                # Call Judge Model
                # We need to pass the conversation. 
                # Our APIModel.generate_answer expects a specific structure.
                judge_input = {
                    "question_id": f"{question['question_id']}_judge_{game_idx}_{attempt}",
                    "turns": conv
                }
                
                response = self.model.generate_answer(judge_input)
                
                if not response:
                    logger.error(f"Judge model failed to generate response.")
                    break
                
                new_judgment = self._get_model_answer_content(response)
                judgment_text += ("\n" + new_judgment)
                
                # Check for verdict
                score, try_again = self._get_score(judgment_text)
                
                conv.append({"role": "assistant", "content": new_judgment})
                
                if not try_again:
                    break
                
                # If verdict not found, ask to continue
                conv.append({"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})
            
            # Save Game Result
            result = {
                "user_prompt": user_prompt,
                "judgment": judgment_text,
                "score": score if score else "N/A"
            }
            output["games"].append(result)

        return output
