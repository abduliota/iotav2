import pytest

from memory import (
    _detect_preferences_from_exchange,
    _has_arabic_script,
    insert_memory_item,
    maybe_write_episodic_from_exchange,
)


def test_has_arabic_script_detects_arabic():
    assert _has_arabic_script("مرحبا") is True
    assert _has_arabic_script("Hello") is False
    assert _has_arabic_script("") is False


def test_insert_memory_item_ignores_short_text(monkeypatch):
    calls = []

    class DummyClient:
        def table(self, name: str):
            assert name == "memory_item"
            return self

        def insert(self, payload):
            calls.append(payload)
            return self

        def execute(self):
            return type("R", (), {"data": []})

    from supabase_client import get_client as real_get_client

    monkeypatch.setattr("memory.get_client", lambda: DummyClient())

    mid = insert_memory_item(
        user_id="user-1",
        session_id=None,
        type_="preference",
        text="too short",
        metadata=None,
        source_message_id=None,
    )
    assert mid is None
    # No DB calls for short text
    assert calls == []


def test_detect_preferences_from_exchange_language():
    pref, delta, topics = _detect_preferences_from_exchange("Please answer in Arabic from now on.")
    assert pref == "ar"
    pref, delta, topics = _detect_preferences_from_exchange("I want English only.")
    assert pref == "en"
    pref, delta, topics = _detect_preferences_from_exchange("What is NORA?")
    assert pref is None and not topics


def test_detect_preferences_from_exchange_topics():
    _, _, topics = _detect_preferences_from_exchange("I need info on licensing and capital.")
    assert "licensing" in topics
    assert "capital" in topics


def test_detect_preferences_from_exchange_strictness():
    _, delta, _ = _detect_preferences_from_exchange("Be strict please.")
    assert delta == 1
    _, delta, _ = _detect_preferences_from_exchange("Be flexible.")
    assert delta == -1


def test_maybe_write_episodic_from_exchange_calls_insert_and_embed(monkeypatch):
    insert_calls = []
    embed_calls = []

    def fake_insert(*args, **kwargs):
        insert_calls.append((args, kwargs))
        return "mock-memory-id"

    def fake_embed(memory_item_id: str, text: str):
        embed_calls.append((memory_item_id, text))

    monkeypatch.setattr("memory.insert_memory_item", fake_insert)
    monkeypatch.setattr("memory.embed_memory_item", fake_embed)
    monkeypatch.setattr("memory.ENABLE_MEMORY_SYSTEM", True)
    monkeypatch.setattr("memory.ENABLE_EPISODIC_MEMORY_WRITES", True)

    maybe_write_episodic_from_exchange(
        user_id="u1",
        session_id="s1",
        user_message="Answer in Arabic from now on.",
        assistant_message="OK.",
        source_message_id="msg-1",
    )

    assert len(insert_calls) == 1
    (args, kwargs) = insert_calls[0]
    assert kwargs.get("user_id") == "u1"
    assert kwargs.get("session_id") == "s1"
    assert kwargs.get("type_") == "preference"
    assert "Arabic" in (kwargs.get("text") or "")
    assert kwargs.get("source_message_id") == "msg-1"

    assert len(embed_calls) == 1
    assert embed_calls[0][0] == "mock-memory-id"
    assert "Arabic" in embed_calls[0][1]


def test_maybe_write_episodic_from_exchange_no_write_when_no_preference(monkeypatch):
    insert_calls = []

    def fake_insert(*args, **kwargs):
        insert_calls.append(1)

    monkeypatch.setattr("memory.insert_memory_item", fake_insert)
    monkeypatch.setattr("memory.ENABLE_MEMORY_SYSTEM", True)
    monkeypatch.setattr("memory.ENABLE_EPISODIC_MEMORY_WRITES", True)

    maybe_write_episodic_from_exchange(
        user_id="u1",
        session_id="s1",
        user_message="What is SAMA?",
        assistant_message="Here is the answer.",
        source_message_id=None,
    )

    assert len(insert_calls) == 0

