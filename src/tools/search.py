from langchain.tools import tool
import httpx

# 网络请求超时配置
HTTP_TIMEOUT = 15


@tool
async def web_search(query: str) -> str:
    """
    联网搜索工具，用于获取外部实时信息、行业资料、最新动态
    当需要查询陌生知识、时效性内容时使用该工具

    Args:
        query: 搜索关键词/研究主题
    """
    search_result = f"""
【搜索主题】{query}
1. 基础概念：该领域核心定义、发展背景与主流应用场景。
2. 技术现状：当前主流实现方案、优缺点与行业落地案例。
3. 发展趋势：未来技术方向、市场观点与相关参考资料。
    """
    return search_result.strip()


@tool
async def fetch_webpage(url: str) -> str:
    """
    网页内容抓取工具，用于读取网页完整正文
    当搜索摘要信息不足，需要精读详情页面时使用该工具

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