from core.interfaces import QueryRewriter


def test_query_rewriter_protocol_shape():
    assert hasattr(QueryRewriter, "rewrite")

    class _Ok:
        def rewrite(self, query, acl):
            return query

    assert isinstance(_Ok(), QueryRewriter)
