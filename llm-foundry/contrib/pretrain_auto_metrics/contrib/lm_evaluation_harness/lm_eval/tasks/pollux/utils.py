from typing import Dict


def doc_to_text(doc):
    prompt = f"""Описание задачи:
Тебе буден дан запрос для оценки(в него могут входить инструкции, данные и т.п.), ответ на запрос для оценки, эталонный ответ (при наличии) и критерий оценки с детальной шкалой оценивания.
1. Оцени ответ на запрос согласно предоставленному критерию и описанию шкалы оценивания.
2. Твоя оценка должна пренадлежать точке на шкале оценивания и базироваться на соответсвующем описании точек шкалы.
3. Ответ должен содержать только число. Не пиши никаких дополнительных введений, объяснений или выводов.
Запрос для оценки:
{doc["instruction"]}\n"""
    if len(doc["reference_answer"]):
        prompt += f"\nЭталонный ответ:\n{doc['reference_answer']}\n"
    prompt += f"\nОтвет для оценки:\n{doc['answer']}\n\nНазвание критерия оценки:\n{doc['criteria_name']}\n\nШкала оценивания:\n{doc['rubrics']}\n\nТвоя оценка:"
    return prompt


def doc_to_target(doc):
    return round(doc["criteria_score"])


def process_results(doc, results) -> Dict[str, float]:
    target_score = doc["criteria_score"]
    pred = results[0].strip()
    pred_score = 10
    for chunk in pred.split():
        try:
            pred_score = float(chunk)
        except ValueError:
            continue

    mae = abs(pred_score - target_score)
    return {"mae": mae}
