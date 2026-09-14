import json
import math
from pathlib import Path
import tempfile
import unittest
from search import Document, chunks, load_documents, retrieval_metrics, validate_vector


class CoreTests(unittest.TestCase):
    def test_chunk_coverage_and_citations(self):
        doc = Document("source", "Title", "api", "0123456789" * 19)
        parts = chunks(doc, 50, 10)
        covered = set()
        for p in parts:
            self.assertEqual(p["text"], doc.text[p["start"]:p["end"]])
            self.assertEqual(p["citation"], f"corpus.json#source:{p['start']}-{p['end']}")
            covered.update(range(p["start"], p["end"]))
        self.assertEqual(covered, set(range(len(doc.text))))

    def test_chunk_ids_are_stable_and_content_sensitive(self):
        a = Document("a", "T", "api", "abc")
        self.assertEqual(chunks(a), chunks(a))
        self.assertNotEqual(chunks(a)[0]["id"], chunks(Document("a", "T", "api", "abd"))[0]["id"])

    def test_invalid_chunk_configuration(self):
        for size, overlap in [(0, 0), (10, 10), (10, -1)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                chunks(Document("a", "T", "api", "text"), size, overlap)

    def test_reject_invalid_vectors(self):
        for v in [[1], [0, 0], [1, math.nan], [math.inf, 1]]:
            with self.subTest(v=v), self.assertRaises(ValueError): validate_vector(v, 2)
        self.assertEqual(validate_vector([1, 0], 2), [1., 0.])

    def test_metrics_do_not_count_duplicate_hits(self):
        m = retrieval_metrics([["a", "a", "b"], ["x"]], [{"a", "b"}, {"z"}])
        self.assertEqual(m["recall_at_k"], .5)
        self.assertEqual(m["hit_at_1"], .5)
        self.assertEqual(m["mrr_at_k"], .5)

    def test_metrics_rank_and_misses(self):
        m = retrieval_metrics([["wrong", "right"], []], [{"right"}, {"missing"}])
        self.assertEqual(m["mrr_at_k"], .25)
        self.assertEqual(m["recall_at_k"], .5)
        self.assertEqual(m["hit_at_1"], 0)

    def test_reject_bad_evaluation(self):
        for a,b in [([], []), ([["a"]], []), ([["a"]], [set()])]:
            with self.assertRaises(ValueError): retrieval_metrics(a,b)

    def test_corpus_validation_and_relevance(self):
        root=Path(__file__).parent
        docs=load_documents(root/"corpus.json")
        ids={d.id for d in docs}
        for q in json.loads((root/"queries.json").read_text()):self.assertTrue(set(q["relevant"])<=ids)

    def test_reject_duplicate_document_ids(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"corpus.json"
            row=dict(id="a", title="T", category="api", text="body")
            p.write_text(json.dumps([row,row]))
            with self.assertRaises(ValueError):load_documents(p)


if __name__ == "__main__":unittest.main()
