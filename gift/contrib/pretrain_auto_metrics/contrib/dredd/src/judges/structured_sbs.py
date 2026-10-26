import json

from src.models import APIModel
from src.utils.io import JsonlWriter


class StructuredSbsJudge:
    def __init__(self, model: APIModel):
        self.model = APIModel(model.name, {
            "api_type": model.api_type, "api_base": model.api_base,
            "api_key": model.api_key, "request_timeout": model.request_timeout,
            "num_retries": model.num_retries,
        })

    def judgment(self, args: dict) -> None:
        question = args["question"]
        candidate, baseline = args["answer"], args["baseline"]
        pair = {
            "schema": "gemma-mean-var-v1",
            "question": question["prompt"],
            "answer_a": baseline["messages"][-1]["content"]["answer"],
            "answer_b": candidate["messages"][-1]["content"]["answer"],
            "domain": "" if question.get("domain") is None else question["domain"],
        }
        message = {"role": "user", "content": json.dumps(pair, ensure_ascii=False)}
        response = self.model.generate_answer({"uid": question["uid"], "turns": [message]})
        result = json.loads(response["choices"][0]["turns"][0]["content"])
        hashes = {key: result[key] for key in ("protocol_sha256", "export_manifest_sha256")}
        forward = {key: value for key, value in result.items() if key not in {*hashes, "reverse"}}
        record = {
            "uid": question["uid"], "category": question.get("category"),
            "judge": self.model.name, "model": candidate["model"],
            "baseline": baseline["model"], "games": [],
        }
        for scores in (forward, result["reverse"]):
            if scores["verdict"] not in ("A>B", "B>A", "A=B"):
                raise ValueError("Structured SBS returned an invalid verdict")
            record["games"].append({
                "score": scores["verdict"], "judgment": {"answer": f"[[{scores['verdict']}]]"},
                "prompt": [message], "mean_var": {**scores, **hashes},
            })
            pair["answer_a"], pair["answer_b"] = pair["answer_b"], pair["answer_a"]
            message = {"role": "user", "content": json.dumps(pair, ensure_ascii=False)}
        JsonlWriter(args["output_file"]).write(record)
