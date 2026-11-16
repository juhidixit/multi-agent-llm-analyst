def test_entity_query_ranks_own_documents_first(retriever):
    hits = retriever.search("Lumora Systems fiscal 2024 revenue", k=3)
    assert hits and all("lumora-systems" in h.doc_id or "Lumora" in h.text for h in hits)


def test_no_match_returns_empty(retriever):
    assert retriever.search("zzzz qqqq") == []
