"""Local reference tests, not a database server or a production authorization layer."""

import json
import unittest
from pathlib import Path

from owlrl import DeductiveClosure, OWLRL_Semantics
from pyshacl import validate
from rdflib import Dataset, Graph, Literal, Namespace, RDF, URIRef, XSD

ROOT = Path(__file__).resolve().parent
EX = Namespace("https://example.org/onto-tech/")
PUBLIC = URIRef("urn:onto-demo:asserted:public:v1")
PRIVATE = URIRef("urn:onto-demo:asserted:restricted:v1")
AT = "2026-10-23T10:00:00+08:00"


def clone(graph):
    result = Graph()
    for triple in graph:
        result.add(triple)
    return result


def load():
    schema = Graph().parse(ROOT / "ontology.ttl", format="turtle")
    shapes = Graph().parse(ROOT / "shapes.ttl", format="turtle")
    dataset = Dataset(default_union=False)
    dataset.parse(ROOT / "data.trig", format="trig")
    # The allowlist is a fixture, not a complete auth implementation.
    # Never infer over restricted data and filter only after generation.
    view = clone(schema)
    for graph_id in (PUBLIC,):
        for triple in dataset.graph(graph_id):
            view.add(triple)
    return schema, shapes, dataset, view


def conforms(graph, shapes):
    ok, _, _ = validate(
        data_graph=graph,
        shacl_graph=shapes,
        inference="none",
        meta_shacl=True,
        inplace=False,
    )
    return bool(ok)


def closure(graph):
    result = clone(graph)
    DeductiveClosure(
        OWLRL_Semantics,
        axiomatic_triples=False,
        datatype_axioms=False,
    ).expand(result)
    return result


def select(graph, environment="linux-selfhosted", at=AT, class_path=False):
    query = (ROOT / "query.rq").read_text(encoding="utf-8")
    if class_path:
        query = (
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" + query
        ).replace(
            "?release a ex:Release ;",
            "?release rdf:type/rdfs:subClassOf* ex:Release ;",
        )
    bindings = {
        "capability": EX.BatchInference,
        "environment": Literal(environment),
        "at": Literal(at, datatype=XSD.dateTime),
    }
    return [
        {str(key): str(value) for key, value in row.asdict().items()}
        for row in graph.query(query, initBindings=bindings)
    ]


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema, cls.shapes, cls.dataset, cls.base = load()

    def mutated(self):
        return clone(self.base)

    def test_01_valid_fixture(self):
        self.assertTrue(conforms(self.base, self.shapes))

    def test_02_missing_evidence_rejected(self):
        graph = self.mutated()
        graph.remove((EX.Claim21, EX.hasEvidence, None))
        self.assertFalse(conforms(graph, self.shapes))

    def test_03_missing_version_rejected(self):
        graph = self.mutated()
        graph.remove((EX.Release21, EX.version, None))
        self.assertFalse(conforms(graph, self.shapes))

    def test_04_integer_version_rejected(self):
        graph = self.mutated()
        graph.set((EX.Release21, EX.version, Literal(21)))
        self.assertFalse(conforms(graph, self.shapes))

    def test_05_invalid_interval_rejected(self):
        graph = self.mutated()
        graph.add(
            (EX.Claim21, EX.validTo, Literal(
                "2026-10-19T00:00:00+08:00", datatype=XSD.dateTime
            ))
        )
        self.assertFalse(conforms(graph, self.shapes))

    def test_06_untyped_claim_not_silently_skipped(self):
        graph = self.mutated()
        graph.remove((EX.Claim21, RDF.type, EX.SupportClaim))
        self.assertFalse(conforms(graph, self.shapes))

    def test_07_class_inference(self):
        self.assertNotIn((EX.Release21, RDF.type, EX.Release), self.base)
        self.assertIn((EX.Release21, RDF.type, EX.Release), closure(self.base))

    def test_08_inverse_inference(self):
        self.assertNotIn((EX.ProductA, EX.hasRelease, EX.Release21), self.base)
        self.assertIn(
            (EX.ProductA, EX.hasRelease, EX.Release21), closure(self.base)
        )

    def test_09_query_requires_inference_or_class_path(self):
        self.assertEqual(select(self.base), [])
        self.assertEqual(len(select(self.base, class_path=True)), 1)
        self.assertEqual(len(select(closure(self.base))), 1)

    def test_10_time_environment_and_status_filters(self):
        graph = closure(self.base)
        rows = select(graph)
        self.assertEqual({r["claim"] for r in rows}, {str(EX.Claim21)})
        self.assertEqual(select(graph, environment="windows-selfhosted"), [])
        earlier = select(graph, at="2026-10-19T10:00:00+08:00")
        self.assertEqual({r["claim"] for r in earlier}, {str(EX.Claim20)})
        at_boundary = select(graph, at="2026-10-20T00:00:00+08:00")
        self.assertEqual({r["claim"] for r in at_boundary}, {str(EX.Claim21)})

    def test_11_revoked_claim_not_returned(self):
        graph = self.mutated()
        graph.set((EX.Claim21, EX.status, Literal("revoked")))
        self.assertEqual(select(closure(graph)), [])

    def test_12_restricted_graph_not_in_public_view(self):
        self.assertGreater(len(self.dataset.graph(PRIVATE)), 0)
        self.assertFalse(list(self.base.triples((EX.PrivateClaim, None, None))))
        self.assertEqual(len(select(closure(self.base))), 1)
        self.assertNotIn("PRIVATE-DEMO", json.dumps(select(closure(self.base))))

    def test_13_dataset_retains_named_graphs(self):
        graph_ids = {graph.identifier for graph in self.dataset.graphs()}
        self.assertIn(PUBLIC, graph_ids)
        self.assertIn(PRIVATE, graph_ids)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    _, shapes, _, base = load()
    print(json.dumps({
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "asserted_view_triples": len(base),
        "expanded_view_triples": len(closure(base)),
        "shacl_conforms": conforms(base, shapes),
        "rows_after_inference": select(closure(base)),
        "scope": "local RDF/SHACL/OWL-RL reference tests; no database server",
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
