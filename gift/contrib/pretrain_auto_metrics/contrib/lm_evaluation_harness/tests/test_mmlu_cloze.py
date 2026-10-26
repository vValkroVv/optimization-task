"""Таск-семейство mmlu_cloze: prompt, состав запросов и девять метрик.

Все три проверки работают без модели и без torch: собирается только сам таск,
`process_results` вызывается на синтетическом списке loglikelihood.
"""

import glob
import itertools
import math
from pathlib import Path

import pytest
import yaml

import lm_eval.tasks as tasks
from lm_eval.evaluator_utils import get_task_list


MMLU_DEFAULT_GLOB = "lm_eval/tasks/mmlu/default/mmlu_*.yaml"
MMLU_CLOZE_DIR = Path("lm_eval/tasks/mmlu_cloze")

# Эталон prompt для mmlu_cloze_abstract_algebra, num_fewshot=5, нулевой документ
# test-сплита. Сверен с локальным parquet-кешем cais/mmlu. При несовпадении
# править надо реализацию, а не эталон.
EXPECTED_CONTEXT = """The following are questions (with answers) about abstract algebra.

Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
Answer: 1

Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.
Answer: False, False

Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.
Answer: True, False

Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.
Answer: True, True

Find the characteristic of the ring 2Z.
Answer: 0

Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
Answer:"""

EXPECTED_CONTEXT_LEN = 765

EXPECTED_METRICS = {
    "acc",
    "acc_norm",
    "acc_bytes",
    "acc_mutual_info",
    "nll",
    "nll_cond",
    "nll_norm",
    "nll_bytes",
    "nll_pmi",
}


@pytest.fixture(scope="module")
def abstract_algebra_task():
    task_dict = tasks.get_task_dict(["mmlu_cloze_abstract_algebra"])
    (task,) = [x.task for x in get_task_list(task_dict)]
    return task


def test_prompt_and_requests(abstract_algebra_task):
    """К2: prompt совпадает с эталоном посимвольно, запросов ровно восемь."""
    task = abstract_algebra_task
    doc = list(task.test_docs())[0]

    ctx = task.fewshot_context(doc, 5)
    assert len(EXPECTED_CONTEXT) == EXPECTED_CONTEXT_LEN
    assert len(ctx) == EXPECTED_CONTEXT_LEN
    assert ctx == EXPECTED_CONTEXT

    requests = task.construct_requests(doc, ctx)
    arguments = [r.arguments for r in requests]
    assert arguments == [
        (ctx, " 0"),
        (ctx, " 4"),
        (ctx, " 2"),
        (ctx, " 6"),
        ("", " 0"),
        ("", " 4"),
        ("", " 2"),
        ("", " 6"),
    ]


def _ratio(ll, length):
    """Деление loglikelihood на длину варианта в семантике numpy.

    Вариант нулевой длины numpy отправляет в -inf (loglikelihood всегда
    отрицателен), а не роняет вычисление, — так же ведёт себя ранжирование
    внутри process_results.
    """
    if length == 0.0:
        return math.copysign(math.inf, ll)
    return ll / length


def _expected_metrics(choices, gold, lls, unc):
    """Девять метрик по формулам К3, посчитанные независимо от кода харнесса."""
    chars = [float(len(c)) for c in choices]
    byts = [float(len(c.encode("utf-8"))) for c in choices]
    denom = sum(math.exp(ll) for ll in lls)

    def argmax(values):
        return max(range(len(values)), key=values.__getitem__)

    return {
        "acc": 1.0 if argmax(lls) == gold else 0.0,
        "acc_norm": (
            1.0 if argmax([_ratio(ll, c) for ll, c in zip(lls, chars)]) == gold else 0.0
        ),
        "acc_bytes": (
            1.0 if argmax([_ratio(ll, b) for ll, b in zip(lls, byts)]) == gold else 0.0
        ),
        "acc_mutual_info": (
            1.0 if argmax([ll - u for ll, u in zip(lls, unc)]) == gold else 0.0
        ),
        "nll": -lls[gold],
        "nll_cond": -math.log(math.exp(lls[gold]) / denom),
        "nll_norm": -lls[gold] / max(chars[gold], 1.0),
        "nll_bytes": -lls[gold] / max(byts[gold], 1.0),
        "nll_pmi": -(lls[gold] - unc[gold]),
    }


def test_metric_formulas(abstract_algebra_task):
    """К3: девять метрик считаются по формулам плана.

    Документ синтетический: на реальном документе abstract_algebra все четыре
    варианта — один символ и один байт, там acc_norm/acc_bytes и
    nll_norm/nll_bytes тождественно равны acc и nll, и тест не различал бы
    формулы. Варианты подобраны так, чтобы нормировки по символам и по байтам
    давали разные ответы.

    Случаев три, потому что одного не хватает: при любом одном наборе входов
    какие-то две метрики совпадают по значению случайно, и подмена формулы
    одной из них прошла бы незамеченной. Первые два случая с разным gold
    разводят внутри каждого семейства все пары — это проверяет замыкающий
    цикл; третий проходит по ветке ограничителя знаменателя.
    """
    task = abstract_algebra_task
    cases = [
        # варианты, gold, условные loglikelihood, безусловные
        (
            ["a", "éé", "日本", "dddd"],
            1,
            [-0.6, -1.0, -1.4, -3.0],
            [-0.1, -0.9, -0.3, -0.5],
        ),
        (
            ["a", "éé", "日本", "dddd"],
            0,
            [-0.6, -1.0, -1.4, -3.0],
            [-3.0, -0.9, -0.3, -0.5],
        ),
        # Третий случай — вариант нулевой длины: только он проходит по ветке
        # ограничителя max(..., 1.0) в nll_norm и nll_bytes. Без него снятие
        # ограничителя тест не заметил бы.
        (
            ["", "a", "éé", "日本"],
            0,
            [-0.6, -1.0, -1.4, -3.0],
            [-0.1, -0.9, -0.3, -0.5],
        ),
    ]

    per_case = []
    for choices, gold, lls, unc in cases:
        doc = {"question": "q", "choices": choices, "answer": gold}
        got = task.process_results(doc, [(ll, False) for ll in lls + unc])
        assert set(got.keys()) == EXPECTED_METRICS, sorted(got.keys())

        expected = _expected_metrics(choices, gold, lls, unc)
        for name, value in expected.items():
            assert got[name] == pytest.approx(value, abs=1e-9), (gold, name)
        per_case.append(expected)

    # Контроль силы теста: внутри каждого семейства любая пара метрик должна
    # разойтись хотя бы в одном из случаев, иначе подмена одной формулы на
    # другую прошла бы мимо ассертов выше.
    for family in (
        ("acc", "acc_norm", "acc_bytes", "acc_mutual_info"),
        ("nll", "nll_cond", "nll_norm", "nll_bytes", "nll_pmi"),
    ):
        for left, right in itertools.combinations(family, 2):
            assert any(
                round(case[left], 9) != round(case[right], 9) for case in per_case
            ), (left, right)


def test_nll_metrics_absent_when_gold_invalid(abstract_algebra_task):
    """К3: при nll_defined == False метрики nll* в словарь не попадают.

    Это существующее поведение харнесса, которое план предписывает
    воспроизвести для всех новых метрик, а не подменять нулями или NaN.
    Без этого случая снятие условия `and nll_defined` с любой из четырёх новых
    формул прошло бы незамеченным, а на документе с некорректным gold дало бы
    IndexError вместо тихого пропуска метрики.
    """
    task = abstract_algebra_task

    # answer вне диапазона вариантов: харнесс переводит gold в -100
    doc = {"question": "q", "choices": ["a", "bb", "ccc", "dddd"], "answer": 9}
    lls = [-0.6, -1.0, -1.4, -3.0]
    unc = [-0.1, -0.9, -0.3, -0.5]

    got = task.process_results(doc, [(ll, False) for ll in lls + unc])
    assert set(got.keys()) == {"acc", "acc_norm", "acc_bytes", "acc_mutual_info"}


def test_all_subject_configs_match_mcq_originals():
    """К2: все 57 пар посабджектных yaml согласованы с mmlu/default."""
    sources = sorted(Path(p) for p in glob.glob(MMLU_DEFAULT_GLOB))
    assert len(sources) == 57, len(sources)

    for source in sources:
        mcq = yaml.safe_load(source.read_text(encoding="utf-8"))
        subject = mcq["dataset_name"]
        target = MMLU_CLOZE_DIR / f"mmlu_cloze_{subject}.yaml"
        assert target.is_file(), target
        cloze = yaml.safe_load(target.read_text(encoding="utf-8"))

        assert cloze["dataset_name"] == subject, target
        assert cloze["task"] == f"mmlu_cloze_{subject}", target
        assert cloze["task_alias"] == mcq["task_alias"], target
        assert cloze["include"] == "_cloze_template_yaml", target

        category = mcq["tag"].removeprefix("mmlu_").removesuffix("_tasks")
        assert cloze["tag"] == f"mmlu_cloze_{category}_tasks", target

        assert "multiple choice " in mcq["description"], source
        assert cloze["description"] == mcq["description"].replace(
            "multiple choice ", ""
        ), target
