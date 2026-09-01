"""The Source protocol: stores satisfy it structurally, and it is read-only."""

from __future__ import annotations

import inspect

from tsdive import Source
from tsdive.store.tagstore import SingleFileStore, TagStore


def test_stores_are_sources(tmp_path):
    assert isinstance(TagStore(tmp_path), Source)
    assert isinstance(SingleFileStore(tmp_path / "x.parquet"), Source)


def test_source_protocol_has_no_write_path():
    forbidden = ("write_", "save_", "update_", "delete_", "ingest_", "mutate_", "set_", "append_")
    names = [
        name
        for name, _ in inspect.getmembers(Source, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    assert names == ["list_tags", "read_window"]
    assert not any(n.lower().startswith(forbidden) for n in names)


def test_read_window_signature_matches_store():
    proto = inspect.signature(Source.read_window)
    impl = inspect.signature(TagStore.read_window)
    assert list(proto.parameters) == list(impl.parameters)
    for name, p in proto.parameters.items():
        assert p.kind == impl.parameters[name].kind
