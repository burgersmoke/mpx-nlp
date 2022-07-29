from src.util import build_nlp
import pytest

nlp = build_nlp()

class TestSectionRules:
    def test_hpi(self):
        texts = ["PAST MEDICAL HISTORY:\nThis guy was sick"]
        docs = list(nlp.pipe(texts))
        for doc in docs:
            assert len(doc._.sections) == 1
            assert doc._.sections[0].category == "past_medical_history"
