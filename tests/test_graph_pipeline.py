"""Integration tests for the metric graph pipeline."""
import asyncio

import pytest

from app.semantics.graph.contract import GraphDocument
from app.semantics.graph.metric.skeleton import GraphSkeletonEngine
from app.semantics.graph.metric.validator import GraphValidator
from app.semantics.graph.loader import KuzuLoader
from app.semantics.graph.metric.pipeline import build_metric_graph
from app.semantics.models import OSISpecification


@pytest.fixture
def tpcds_model():
    spec = OSISpecification.load_from_yaml("config/semantics/tpcds_model_sqlite.yaml")
    return spec.semantic_model[0]


class TestSkeletonEngine:
    def test_generates_all_node_labels(self, tpcds_model):
        doc = GraphSkeletonEngine(tpcds_model).build()
        labels = {n.label for n in doc.nodes}
        assert "Metric" in labels
        assert "Dimension" in labels
        assert "LogicalDataset" in labels
        assert "PhysicalField" in labels

    def test_generates_all_edge_labels(self, tpcds_model):
        doc = GraphSkeletonEngine(tpcds_model).build()
        labels = {e.label for e in doc.edges}
        assert "AGGREGATES_FROM" in labels
        assert "DERIVED_FROM" in labels
        assert "SLICES_BY" in labels
        assert "MAPS_TO" in labels
        assert "JOINS_TO" in labels

    def test_every_metric_has_aggregates_from(self, tpcds_model):
        doc = GraphSkeletonEngine(tpcds_model).build()
        metric_ids = {n.id for n in doc.nodes if n.label == "Metric"}
        agg_metrics = {e.from_ for e in doc.edges if e.label == "AGGREGATES_FROM"}
        assert metric_ids == agg_metrics, f"Missing AGGREGATES_FROM for: {metric_ids - agg_metrics}"

    def test_graph_document_roundtrip_json(self, tpcds_model):
        doc = GraphSkeletonEngine(tpcds_model).build()
        json_str = doc.model_dump_json(indent=2)
        doc2 = GraphDocument.model_validate_json(json_str)
        assert len(doc.nodes) == len(doc2.nodes)
        assert len(doc.edges) == len(doc2.edges)


class TestValidator:
    def test_valid_document_passes(self, tpcds_model):
        doc = GraphSkeletonEngine(tpcds_model).build()
        errors = GraphValidator(doc, tpcds_model).validate()
        assert len(errors) == 0, errors

    def test_broken_edge_ref_fails(self, tpcds_model):
        doc = GraphSkeletonEngine(tpcds_model).build()
        doc.edges[0].from_ = "dataset:nonexistent"
        errors = GraphValidator(doc, tpcds_model).validate()
        assert any("unknown source" in e for e in errors)

    def test_orphan_node_fails(self, tpcds_model):
        doc = GraphSkeletonEngine(tpcds_model).build()
        doc.add_node("orphan:test", "Orphan", name="test")
        errors = GraphValidator(doc, tpcds_model).validate()
        assert any("orphan" in e.lower() or "Orphan" in e for e in errors)


class TestKuzuLoader:
    def test_load_and_query(self, tpcds_model):
        doc = GraphSkeletonEngine(tpcds_model).build()
        loader = KuzuLoader(path=":memory:")
        loader.load(doc)

        r = loader.execute("MATCH (m:Metric) RETURN COUNT(m) AS cnt")
        assert r.get_next()[0] == len([n for n in doc.nodes if n.label == "Metric"])
        r = loader.execute("MATCH ()-[e:AGGREGATES_FROM]->() RETURN COUNT(e) AS cnt")
        assert r.get_next()[0] == len([e for e in doc.edges if e.label == "AGGREGATES_FROM"])
        r = loader.execute("MATCH (m:Metric {name: 'total_sales'})-[:AGGREGATES_FROM]->(f:PhysicalField) RETURN f.name")
        fields = []
        while r.has_next():
            fields.append(r.get_next()[0])
        assert len(fields) > 0


class TestPipeline:
    def test_build_metric_graph(self, tpcds_model):
        doc = asyncio.run(build_metric_graph(tpcds_model, loader=KuzuLoader(path=":memory:")))
        assert len(doc.nodes) > 0
        assert len(doc.edges) > 0
