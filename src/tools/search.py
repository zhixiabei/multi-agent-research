"""联网搜索工具 —— 调用 Tavily Search API"""
import os

from dotenv import load_dotenv
from langchain.tools import tool
import httpx

load_dotenv()

# Tavily API 配置
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL=os.getenv("TAVILY_SEARCH_URL")

HTTP_TIMEOUT = 20


@tool
async def web_search(query: str, top_k: int = 5) -> str:
    """
    联网搜索工具，调用 Tavily 搜索引擎获取实时信息。
    用于查询陌生知识、时效性内容、行业动态等。

    Args:
        query: 搜索关键词
        top_k: 返回结果数量，默认 5 条
    """
    body = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": top_k,
    }

    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(TAVILY_SEARCH_URL, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                return f"[搜索结果为空] 关键词: {query}"

            parts = [f"【搜索主题】{query}"]
            for i, item in enumerate(results, 1):
                title = item.get("title", "无标题")
                content = item.get("content", "")
                url = item.get("url", "")
                parts.append(f"{i}. {title}\n   {content}\n   链接: {url}")

            return "\n\n".join(parts)

    except httpx.HTTPStatusError as e:
        return f"[搜索失败] HTTP {e.response.status_code}: {e.response.text[:300]}"
    except httpx.RequestError as e:
        return f"[搜索失败] 网络请求异常: {e}"
    except Exception as e:
        return f"[搜索失败] 未知错误: {e}"


@tool
async def fetch_webpage(url: str) -> str:
    """
    网页内容抓取工具，用于读取网页完整正文。
    当搜索摘要信息不足，需要精读详情页面时使用该工具。

    Args:
        url: 待抓取的网页链接
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except Exception as e:
        return f"网页抓取失败，错误信息：{str(e)}"


# 统一导出工具集合
SEARCH_TOOLS = [web_search, fetch_webpage]
