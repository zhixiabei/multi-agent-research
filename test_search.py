import pytest
from tools.search import web_search, fetch_webpage

@pytest.mark.asyncio
async def test_web_search_normal():
    result = await web_search.ainvoke({"query": "人工智能发展现状", "top_k": 2})
    assert result
    assert "[搜索失败]" not in result

@pytest.mark.asyncio
async def test_fetch_webpage():
    test_url = "https://www.baidu.com"
    result = await fetch_webpage.ainvoke({"url": test_url})
    assert "网页抓取失败" not in result