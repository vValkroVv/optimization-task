from functools import partial


SUBJECTS = {
    "abstract_algebra": "абстрактной алгебре",
    "anatomy": "анатомии",
    "astronomy": "астрономии",
    "business_ethics": "деловой этике",
    "clinical_knowledge": "медицинским знаниям",
    "college_biology": "биологии в вузе",
    "college_chemistry": "химии в вузе",
    "college_computer_science": "компьютерным наукам в вузе",
    "college_mathematics": "математике в вузе",
    "college_medicine": "медицине в вузе",
    "college_physics": "физике в вузе",
    "computer_security": "компьютерной безопасности",
    "conceptual_physics": "теоретической физике",
    "econometrics": "эконометрике",
    "electrical_engineering": "электротехнике",
    "elementary_mathematics": "элементарной математике",
    "formal_logic": "формальной логике",
    "global_facts": "фактам о мире",
    "high_school_biology": "биологии в старшей школе",
    "high_school_chemistry": "химии в старшей школе",
    "high_school_computer_science": "информатике в старшей школе",
    "high_school_european_history": "истории Европы в старшей школе",
    "high_school_geography": "географии в старшей школе",
    "high_school_government_and_politics": "государству и политике в старшей школе",
    "high_school_macroeconomics": "макроэкономике в старшей школе",
    "high_school_mathematics": "математике в старшей школе",
    "high_school_microeconomics": "микроэкономике в старшей школе",
    "high_school_physics": "физике в старшей школе",
    "high_school_psychology": "психологии в старшей школе",
    "high_school_statistics": "статистике в старшей школе",
    "high_school_us_history": "истории США в старшей школе",
    "high_school_world_history": "всемирной истории в старшей школе",
    "human_aging": "старению человека",
    "human_sexuality": "человеческой сексуальности",
    "international_law": "международному праву",
    "jurisprudence": "юриспруденции",
    "logical_fallacies": "логическим ошибкам",
    "machine_learning": "машинному обучению",
    "management": "менеджменту",
    "marketing": "маркетингу",
    "medical_genetics": "медицинской генетике",
    "miscellaneous": "разным темам",
    "moral_disputes": "нравственным спорам",
    "moral_scenarios": "нравственным сценариям",
    "nutrition": "правильному питанию",
    "philosophy": "философии",
    "prehistory": "доисторической эпохе",
    "professional_accounting": "профессиональному бухгалтерскому учету",
    "professional_law": "профессиональному праву",
    "professional_medicine": "профессиональной медицине",
    "professional_psychology": "профессиональной психологии",
    "public_relations": "связям с общественностью",
    "security_studies": "исследованиям в области безопасности",
    "sociology": "социологии",
    "us_foreign_policy": "внешней политике США",
    "virology": "вирусологии",
    "world_religions": "мировым религиям",
}


def _filter_subject(dataset, subject):
    return dataset.filter(lambda doc: doc["subject"] == subject)


def _register_subject_filters():
    for name, subject in SUBJECTS.items():
        globals()[f"process_{name}"] = partial(_filter_subject, subject=subject)


_register_subject_filters()
