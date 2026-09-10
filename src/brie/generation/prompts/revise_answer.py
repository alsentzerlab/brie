"""Prompts for evidence-grounded answer revision."""

ANSWER_SYS = """Revise the answer so every clinical claim is supported by the
provided facts. Address the reviewer comment, preserve uncertainty and timing,
remove repetition, and omit identifiers. Return JSON with `answer`."""

ANSWER_USER = """<as_of>{TIMESTAMP}</as_of>
<question>{QUESTION}</question>
<draft>{ANSWER}</draft>
<review_comment>{COMMENT}</review_comment>
<facts>{FACTS}</facts>"""
