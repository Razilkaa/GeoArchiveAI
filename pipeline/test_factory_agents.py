from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from factory_agents import domain_qc, local_keyword_chunks, parse_json_object, source_refs, validate_evidence


class FactoryAgentsTest(unittest.TestCase):
    def test_evidence_must_come_from_retrieved_source_markers(self):
        chunks = [{"content": "[SOURCE_PAGE: page:00012; path=T/12.jpg] факт"}]
        allowed = source_refs(chunks)
        result = {"items": [{"name": "A", "evidence": ["page:00012"]}]}
        self.assertEqual(validate_evidence(result, allowed)["status"], "pass")
        result["items"][0]["evidence"] = ["page:99999"]
        self.assertEqual(validate_evidence(result, allowed)["status"], "review")

    def test_fenced_json_is_parsed(self):
        self.assertEqual(parse_json_object('```json\n{"items": []}\n```')["items"], [])

    def test_suspicious_well_id_requires_review(self):
        result = {"items": [{"id": "Р-4о", "evidence": ["page_1.jpg"]}]}
        qc = domain_qc("wells", result, {"page_1.jpg"})
        self.assertEqual(qc["status"], "review")
        self.assertEqual(qc["suspicious_ids"], ["Р-4о"])

    def test_local_keyword_chunks_rank_matching_pages(self):
        with TemporaryDirectory() as temp:
            markdown = Path(temp) / "report.md"
            markdown.write_text(
                "## page:00001\n[SOURCE_PAGE: page:00001; path=1.jpg] карта\n"
                "## page:00002\n[SOURCE_PAGE: page:00002; path=2.jpg] скв. Р-410, скважина испытана\n",
                encoding="utf-8",
            )
            chunks = local_keyword_chunks(markdown, ("скв.", "испытан"))
            self.assertEqual(len(chunks), 1)
            self.assertIn("Р-410", chunks[0]["content"])


if __name__ == "__main__":
    unittest.main()
