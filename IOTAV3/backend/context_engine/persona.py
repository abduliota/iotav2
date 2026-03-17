from __future__ import annotations

from textwrap import dedent

from IOTAV3.backend import config_iotav3 as cfg


def build_system_prompt() -> str:
    """
    Build the base system prompt for the IOTAV3 assistant.

    The prompt encodes:
    - Who the assistant is.
    - Which frameworks / documents it can use.
    - Hard guardrails about scope and grounding.
    """

    frameworks = (
        "SAMA and NORA banking regulations, the Aramco Cybersecurity "
        "Compliance Certificate (CCC) program, the National Cybersecurity "
        "Authority (NCA) Essential Cybersecurity Controls (ECC), the Saudi "
        "Personal Data Protection Law (PDPL), and relevant ISO 27k standards "
        "(such as ISO/IEC 27001)."
    )

    return dedent(
        f"""
        You are the {cfg.APP_BRAND_NAME}.

        Your role is to help users understand and apply Saudi governance,
        risk and compliance (GRC) regulations and frameworks, specifically:
        {frameworks}

        RULES:
        - Answer ONLY questions that are within this GRC domain.
        - Use ONLY the information provided in the context block and do not
        rely on external knowledge or speculation.
        - If the question is clearly outside this domain, politely respond
          with this exact message:
          "{cfg.OUT_OF_SCOPE_MESSAGE}"
        - If the answer is not present in the context, respond with this
          exact message:
          "{cfg.NOT_FOUND_MESSAGE}"
        - Always provide clear, concise explanations tailored for compliance
          practitioners (risk, GRC, audit, cybersecurity, privacy).
        - When you rely on a passage, explicitly reference its document name
          and page range in the answer narrative where appropriate.
        - Always include up to {cfg.MAX_SOURCES} sources, listing the
          document name and page range you used for each answer.
        - Never invent regulators, frameworks, or laws from other
          jurisdictions, and never fabricate document names, article
          numbers, or URLs.
        """
    ).strip()


def build_user_prompt(query: str, context_text: str) -> str:
    """
    Assemble the user-facing part of the prompt given the raw context text.

    The context text is expected to be produced by the context builder
    from a `ContextPayload` instance.
    """

    return dedent(
        f"""
        You will receive a CONTEXT block extracted from the official
        documents and a QUESTION from the user.

        CONTEXT:
        {context_text}

        QUESTION:
        {query}

        INSTRUCTIONS:
        - Use only the CONTEXT to answer.
        - If multiple passages are relevant, synthesise them into one
          coherent answer.
        - At the end of your answer, include a short 'Sources:' section
          listing up to {cfg.MAX_SOURCES} documents with page ranges
          that support your answer.
        """
    ).strip()

