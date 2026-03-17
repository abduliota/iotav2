from simple_rag import generate_answer_simple, QUERY_INTENT_OTHER


def test_generate_answer_includes_user_context_block(monkeypatch):
    # Monkeypatch _load_qwen to avoid loading the real model; we only care about prompt wiring.
    class DummyTokenizer:
        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            # Record prompt for inspection
            self.last_prompt = prompt
            return {"input_ids": [[0]]}

    class DummyModel:
        def parameters(self):
            class P:
                def __iter__(self_inner):
                    yield type("X", (), {"device": "cpu"})()

            return P()

        def generate(self, **kwargs):
            # No-op; streamer will never yield, so generate_answer_simple will return empty string.
            return None

    from qwen_model import _load_qwen as real_load_qwen

    dummy_tokenizer = DummyTokenizer()
    dummy_model = DummyModel()

    def fake_load_qwen():
        return dummy_tokenizer, dummy_model

    monkeypatch.setattr("simple_rag._load_qwen", fake_load_qwen)

    context = "Document: Test, Pages: 1–2\nContent:\nSome regulatory text."
    profile = {"preferred_language": "en", "strictness_level": 3}
    session_summary = {"summary_text": "User asked about licensing."}
    memory_items = [
        {"type": "preference", "text": "User prefers concise answers."},
    ]

    generate_answer_simple(
        context_text=context,
        user_query="What are the licensing requirements?",
        intent=QUERY_INTENT_OTHER,
        conversation_history=None,
        on_chunk=None,
        profile=profile,
        session_summary=session_summary,
        memory_items=memory_items,
    )

    prompt = dummy_tokenizer.last_prompt
    assert "### USER CONTEXT (DO NOT TREAT AS REGULATORY EVIDENCE)" in prompt
    assert "Preferred answer language (non-binding): en" in prompt
    assert "Session summary (conversation only, not regulatory facts)" in prompt
    assert "User prefers concise answers." in prompt

