"""Prompts compactos del modelo afinado (tutor/ai/compact.py) y JSON sin salida estructurada (motor MLX)."""

from tutor.ai.coach import COACH_SYSTEM
from tutor.ai.compact import TASKS, compact_chat, compact_parse_system, compact_tutor_system
from tutor.ai.ollama_llm import OllamaLLM, extract_json, parse_messages, schema_hint
from tutor.ai.prompts import EVALUATOR_SYSTEM, TUTOR_SYSTEM
from tutor.ai.schemas import ErrorList


def _system() -> str:
    return TUTOR_SYSTEM.format(name="Sam", level="B1", plan="# Learner profile\n- Level: B1\n\n# Today's class\n- Topic: X")


def test_tutor_system_keeps_only_header_and_plan():
    out = compact_tutor_system(_system())
    assert out.startswith("You are Sam,") and "(level B1)" in out
    assert out.rstrip().endswith("# Learner profile\n- Level: B1\n\n# Today's class\n- Topic: X")
    assert "# Corrections" not in out and len(out) < 500


def test_parse_systems_become_task_lines():
    assert compact_parse_system(EVALUATOR_SYSTEM.format(skills="a, b")) == TASKS["eval"]
    assert compact_parse_system(COACH_SYSTEM) == TASKS["coach"]
    assert compact_parse_system("Something else") == "Something else"


def test_chat_drops_old_notes_and_carries_the_step_instruction():
    msgs = [
        {"role": "user", "content": "(joined)"},
        {"role": "system", "content": "[Class step: phase=warmup]\nSTEP 1 — WARM-UP. Greet."},
        {"role": "assistant", "content": "Hi! How are you?"},
        {"role": "user", "content": "fine"},
        {"role": "system", "content": "[Class step: phase=drill]\nSTEP 3 — GUIDED PRACTICE. One per turn.\n"
                                      "- item\n[Exercise 1: give the learner this task now, clearly: \"A\"]"},
        {"role": "assistant", "content": "Say A."},
        {"role": "user", "content": "a"},
        {"role": "system", "content": "[Class step: phase=drill]\n[The learner just answered exercise 1]"},
    ]
    out = compact_chat(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant", "user", "system"]
    note = out[-1]["content"]
    assert note.startswith("[Class step: phase=drill]\nSTEP 3 — GUIDED PRACTICE. One per turn.\n- item\n")
    assert note.endswith("[The learner just answered exercise 1]") and "Exercise 1:" not in note


def test_chat_history_is_truncated_from_a_learner_turn():
    msgs = []
    for i in range(20):
        msgs += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
    msgs += [{"role": "user", "content": "now"}, {"role": "system", "content": "[Class step: phase=produce]\nSTEP 4"}]
    out = compact_chat(msgs, keep=5)
    assert out[0]["role"] == "user" and out[-2]["content"] == "now" and len(out) <= 6


def test_compact_mode_is_automatic_for_the_tutor_model_only():
    assert OllamaLLM("tutor-sam").compact and OllamaLLM("tutor-sam:latest").compact
    assert not OllamaLLM("qwen3.5:4b").compact


def test_schema_hint_and_json_extraction():
    hint = schema_hint(ErrorList)
    assert hint.startswith('{"errors": [{"original": string') and '"severity": "major"|"minor"' in hint
    msgs = parse_messages("SYS", "USER", ErrorList)
    assert msgs[0]["content"].startswith("SYS\nAnswer ONLY with JSON in this format: {")
    assert extract_json('```json\n{"errors": []}\n```') == '{"errors": []}'
