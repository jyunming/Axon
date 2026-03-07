"""
tests/test_tools.py

Unit tests for the agent tool schema definitions in rag_brain.tools.
"""
from rag_brain.tools import get_rag_tool_definition

EXPECTED_TOOL_NAMES = {
    "query_knowledge_base",
    "search_documents",
    "add_knowledge",
    "delete_documents",
    "ingest_directory",
    "stream_query",
}


def test_get_rag_tool_definition_returns_nonempty_list():
    tools = get_rag_tool_definition()
    assert isinstance(tools, list)
    assert len(tools) > 0


def test_all_expected_tool_names_present():
    tools = get_rag_tool_definition()
    names = {t["function"]["name"] for t in tools}
    assert names == EXPECTED_TOOL_NAMES


def test_each_tool_has_required_schema_fields():
    tools = get_rag_tool_definition()
    for tool in tools:
        assert tool.get("type") == "function", f"Tool missing type=function: {tool}"
        fn = tool.get("function", {})
        assert "name" in fn, f"Tool missing 'name': {fn}"
        assert "description" in fn, f"Tool missing 'description': {fn}"
        assert "parameters" in fn, f"Tool missing 'parameters': {fn}"
        params = fn["parameters"]
        assert params.get("type") == "object", f"Parameters type not 'object': {fn['name']}"
        assert "properties" in params, f"Parameters missing 'properties': {fn['name']}"


def test_each_tool_has_at_least_one_required_parameter():
    tools = get_rag_tool_definition()
    for tool in tools:
        fn = tool["function"]
        required = fn["parameters"].get("required", [])
        assert len(required) >= 1, f"Tool '{fn['name']}' has no required parameters"


def test_default_api_base_url_does_not_raise():
    # Just verify default call is safe (no URL used in schema but arg accepted)
    tools = get_rag_tool_definition()
    assert tools is not None


def test_custom_api_base_url_accepted():
    # Function accepts custom URL without raising
    tools = get_rag_tool_definition(api_base_url="http://myhost:9000")
    assert len(tools) == len(EXPECTED_TOOL_NAMES)


def test_query_knowledge_base_requires_query():
    tools = get_rag_tool_definition()
    tool = next(t for t in tools if t["function"]["name"] == "query_knowledge_base")
    assert "query" in tool["function"]["parameters"]["required"]


def test_search_documents_requires_query():
    tools = get_rag_tool_definition()
    tool = next(t for t in tools if t["function"]["name"] == "search_documents")
    assert "query" in tool["function"]["parameters"]["required"]


def test_add_knowledge_requires_text():
    tools = get_rag_tool_definition()
    tool = next(t for t in tools if t["function"]["name"] == "add_knowledge")
    assert "text" in tool["function"]["parameters"]["required"]


def test_delete_documents_requires_doc_ids():
    tools = get_rag_tool_definition()
    tool = next(t for t in tools if t["function"]["name"] == "delete_documents")
    assert "doc_ids" in tool["function"]["parameters"]["required"]


def test_ingest_directory_requires_path():
    tools = get_rag_tool_definition()
    tool = next(t for t in tools if t["function"]["name"] == "ingest_directory")
    assert "path" in tool["function"]["parameters"]["required"]


def test_stream_query_requires_query():
    tools = get_rag_tool_definition()
    tool = next(t for t in tools if t["function"]["name"] == "stream_query")
    assert "query" in tool["function"]["parameters"]["required"]
