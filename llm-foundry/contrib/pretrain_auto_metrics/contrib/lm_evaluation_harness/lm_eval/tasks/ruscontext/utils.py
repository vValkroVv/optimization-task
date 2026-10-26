# from sklearn.metrics import f1_score
import json
import re
import string
from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


def coref_to_text(doc):
    paragraph = doc["paragraph"]["text"]
    span = doc["anaphoric span"]
    prompt = f"Ответь на вопрос по этому фрагменту текста:\n{paragraph}\nТебе нужно понять, к какой сущности относится это упоминание: {span}.\nОтвет:"
    return prompt


def coref_to_target(doc):
    return int(doc["gold answer"]) - 1


def np_coref_to_target(doc):
    return "Да" if doc["gold"] else "Нет"


def disrpt_to_text(doc):
    choices = doc["choices"]
    sent_1 = doc["sent_1"]
    sent_2 = doc["sent_2"]
    options = ""
    for choice in choices:
        options += choice + ", "
    options = options[:-2]
    prompt = f"Определите связь между двумя предложениями. Возможные следующие варианты ответа: {options}.\nПредложение 1: {sent_1}\nПредложение 2: {sent_2}\nДайте только один ответ из предложенных.\nОтвет:"
    return prompt


def rudabank_to_text(doc):
    choices = doc["choices"]
    sent_1 = doc["initial_utterance"]
    sent_2 = doc["tagged_utterance"]
    options = ""
    for choice in choices:
        options += choice + ", "
    options = options[:-2]
    prompt = f"Дано начальное высказывание и ответное высказывание, определите тип ответа из следующих вариантов: {options}.\n\nНачальное высказывание: {sent_1}\nОтветное высказывание: {sent_2}\n\nДайте только один ответ из предложенных.\nОтвет:"
    return prompt


def ellipsis_to_text(doc):
    sentence = doc["sentence"]
    sentence = re.sub(r" _", "", sentence)
    sentence = re.sub(r"_", "", sentence)
    sentence = re.sub(r"\n", " ", sentence)
    prompt = f"Дано предложение:\\n{sentence}\\nОно содержит эллипсис, в нем пропущена часть информации. Постарайся восполнить как можно больше информации, не придумывай и не добавляй того, чего нет в контексте. Определи, в каком месте пропущена информация, восполни информацию и перепиши предложение с восполненной информацией.\\nОтвет:"
    # Оригинальный промпт (хуже подходит для претрейна), непонятна мотивация json, кроме помощи модели, так как используется только одно поле эллипсис в дальнейшейм
    # prompt = f"""Дано предложение {sentence}. Оно содержит эллипсис, в нем пропущена часть информации.
    # Постарайся восполнить как можно больше информации, не придумывай и не добавляй того, чего нет в контексте.
    # Определи, 1) в каком месте пропущена информация, обозначь это место нижним подчеркиванием. 2) Восполни информацию и
    # 3) напиши новое предложение с восполненой информацией.
    # Ответ дай в формате: изначальное - ответ на 1, эллипсис - ответ на 2, полное - ответ на 3. Ответ должен быть в формате json. В ответе должен быть только JSON в markdown нотации (начинаться с ```json и заканчиваться ```) без дополнительных комментариев."""
    return prompt


def ellipsis_to_target(doc):
    sentence = doc["sentence"]
    to_replace = doc["suggested ellipsis resolution"].split(", ")
    for word in to_replace:
        sentence = sentence.replace("__", word, 1)
    return sentence


def _normalize_string(s):
    return re.sub(
        r"ё",
        r"е",
        s.translate(str.maketrans("", "", string.punctuation)).lower().strip(),
    )


def _get_for_metrics(preds, golds):
    preds = [_normalize_string(pred) for pred in preds]
    golds = [_normalize_string(gold) for gold in golds]
    for_metrics = [1 if preds[i] == golds[i] else 0 for i in range(len(preds))]
    return for_metrics


# def _parse_json_ans(ans):
#     ellipsis = []
#     for text in ans:
#         try:
#             parsed_ans = json.loads(text.split("json")[1].strip("```"))
#             ellipsis.append(parsed_ans["эллипсис"])
#         except:
#             ellipsis.append("")
#     return ellipsis


def aggregate_ellipsis_acc(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    for_metrics = _get_for_metrics(preds, golds)
    return accuracy_score([1] * len(for_metrics), for_metrics)


def aggregate_ellipsis_f1(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    for_metrics = _get_for_metrics(preds, golds)
    return f1_score([1] * len(for_metrics), for_metrics)


def macro_f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="macro")
    return fscore
