import pytest
import json
import os
import time
import shutil
from pathlib import Path
from unittest.mock import patch
import dredd # Импортируем сам модуль

@pytest.mark.integration
def test_dredd_new_identity_on_param_change(tmp_path):
    """
    Проверка того, что при изменении параметров модели (например, температуры)
    Dredd создает НОВУЮ директорию, а не продолжает старую.
    """
    project_dir = tmp_path / "dredd_project_identity"
    project_dir.mkdir()
    
    data_dir = project_dir / "arenas"
    data_dir.mkdir(parents=True)
    arena_file = data_dir / "test-arena.jsonl"
    questions = [{"uid": f"q{i}", "prompt": f"Q {i}", "category": "test"} for i in range(3)]
    with open(arena_file, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    model_path = project_dir / "my-model"
    model_path.mkdir()

    from omegaconf import OmegaConf
    
    # Конфиг 1: Температура 0.7
    conf_v1 = OmegaConf.create({
        "name": "mock-model",
        "path": str(model_path),
        "api_type": "openai",
        "api_base": "http://localhost:9999/v1",
        "temperature": 0.7
    })

    def mock_post(*args, **kwargs):
        return type('obj', (object,), {
            "status_code": 200,
            "json": lambda: {"choices": [{"message": {"content": "Response"}}]},
            "raise_for_status": lambda: None
        })

    with patch("dredd.DATA_ROOT", project_dir):
        # 1. Запускаем V1, прерываем на 1-м вопросе
        call_count = 0
        def mock_post_interrupted(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1: raise RuntimeError("Stop")
            return mock_post()

        try:
            with patch("requests.post", side_effect=mock_post_interrupted):
                dredd._generate(conf_v1, "test-arena", arena_file, overwrite=False)
        except RuntimeError: pass

        gen_base = model_path / "artifacts" / "arenas" / "test-arena" / "generations"
        run_dirs_v1 = sorted([d for d in gen_base.iterdir() if d.is_dir()])
        assert len(run_dirs_v1) == 1
        
        # 2. Меняем параметр: температура теперь 0.1
        conf_v2 = conf_v1.copy()
        conf_v2.temperature = 0.1

        with patch("requests.post", side_effect=mock_post):
            dredd._generate(conf_v2, "test-arena", arena_file, overwrite=False)

        # ПРОВЕРКА: Должна появиться ВТОРАЯ директория
        run_dirs_v2 = sorted([d for d in gen_base.iterdir() if d.is_dir()])
        
        # Если Dredd правильно работает, он увидит, что параметры изменились
        # и создаст новую папку.
        assert len(run_dirs_v2) == 2, f"Dredd should have created a NEW directory for different parameters. Found: {run_dirs_v2}"
        
        # Проверяем, что в новой папке все 3 ответа (полный перезапуск для новых параметров)
        new_gen_file = run_dirs_v2[-1] / "generation.jsonl"
        with open(new_gen_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 3

@pytest.mark.integration
def test_dredd_resume_logic(tmp_path):
    """
    Сценарий 1: Проверка дозапуска (resume).
    1. Полный прогон -> reference.
    2. Прерванный прогон.
    3. Дозапуск -> сравнение с reference.
    """
    project_dir = tmp_path / "dredd_project"
    project_dir.mkdir()
    
    # Создаем минимальный датасет (6 вопросов)
    data_dir = project_dir / "arenas"
    data_dir.mkdir(parents=True)
    arena_file = data_dir / "test-arena.jsonl"
    questions = []
    for i in range(6):
        questions.append({"uid": f"q{i}", "prompt": f"Question {i}", "category": "test"})
    
    with open(arena_file, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    # Модель с путем внутри нашего проекта
    model_path = project_dir / "my-model"
    model_path.mkdir()

    # Создаем конфиг (через OmegaConf для удобства)
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        "project_name": "test_resume",
        "data_root": str(project_dir),
        "arenas": ["test-arena"],
        "models": [
            {
                "name": "mock-model",
                "path": str(model_path),
                "api_type": "openai",
                "api_base": "http://localhost:9999/v1"
            }
        ]
    })
    
    model_conf = config.models[0]

    # Мокаем API
    call_count = 0
    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        time.sleep(0.05)
        return type('obj', (object,), {
            "status_code": 200,
            "json": lambda: {
                "choices": [{"message": {"content": "Mock response"}}]
            },
            "raise_for_status": lambda: None
        })

    # Патчим DATA_ROOT в модуле dredd, чтобы он смотрел на нашу временную папку
    with patch("dredd.DATA_ROOT", project_dir):
        
        # 1. ПОЛНЫЙ ПРОГОН
        with patch("requests.post", side_effect=mock_post):
            # _generate(model_conf, arena_name, questions_path, overwrite)
            dredd._generate(model_conf, "test-arena", arena_file, overwrite=True)

        # Находим файл генерации (он в папке с таймстампом)
        gen_base = model_path / "artifacts" / "arenas" / "test-arena" / "generations"
        run_dirs = sorted([d for d in gen_base.iterdir() if d.is_dir()])
        assert len(run_dirs) >= 1
        gen_file = run_dirs[-1] / "generation.jsonl"
        
        with open(gen_file, "r") as f:
            reference_data = sorted([json.loads(l) for l in f], key=lambda x: x["uid"])
        
        # Очищаем для второго захода
        shutil.rmtree(model_path / "artifacts")
        call_count = 0

        # 2. ПРЕРВАННЫЙ ПРОГОН
        def mock_post_interrupted(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise RuntimeError("Simulated interruption")
            return mock_post(*args, **kwargs)

        try:
            with patch("requests.post", side_effect=mock_post_interrupted):
                dredd._generate(model_conf, "test-arena", arena_file, overwrite=True)
        except RuntimeError:
            pass 

        # Проверяем, что файл существует и в нем меньше 6 записей (прервано)
        run_dirs = sorted([d for d in gen_base.iterdir() if d.is_dir()])
        gen_file = run_dirs[-1] / "generation.jsonl"
        with open(gen_file, "r") as f:
            interrupted_lines = [json.loads(l) for l in f]
        assert 0 < len(interrupted_lines) < 6

        # 3. ДОЗАПУСК
        # overwrite=False позволит найти существующую папку
        with patch("requests.post", side_effect=mock_post):
            dredd._generate(model_conf, "test-arena", arena_file, overwrite=False)

        # СРАВНЕНИЕ
        with open(gen_file, "r") as f:
            final_data = sorted([json.loads(l) for l in f], key=lambda x: x["uid"])
        
        assert len(final_data) == 6
        for r, f in zip(reference_data, final_data):
            assert r["uid"] == f["uid"]
            assert r["choices"][0]["turns"][0]["content"] == f["choices"][0]["turns"][0]["content"]

    print("\nResume logic tests passed successfully!")
