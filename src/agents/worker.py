"""Worker: 接收一个子任务，联网搜索 + 查本地知识库，写出研究报告"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.models.schemas import SubTask, ResearchResult
from src.tools.search import SEARCH_TOOLS
from src.tools.rag import RAG_TOOLS

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("DEEPSEEK_BASE_URL")

llm = ChatOpenAI(model="deepseek-chat", temperature=0.3)

# 把工具放到字典里，方便按名字调用
TOOL_MAP = {t.name: t for t in SEARCH_TOOLS + RAG_TOOLS}


async def research_task(task: SubTask) -> ResearchResult:
    """
    研究一个子任务：
    1. 联网搜索
    2. 查本地知识库
    3. 用 LLM 综合两份资料写出答案
    """
    # ── 第一步：并行搜索 ──
    search_query = f"{task.question} {' '.join(task.keywords)}"

    web_tool = TOOL_MAP["web_search"]
    rag_tool = TOOL_MAP["query_knowledge"]

    web_res, rag_res = await asyncio.gather(
        web_tool.ainvoke(search_query),
        rag_tool.ainvoke(search_query)
    )

    # ── 第二步：收集信息来源 ──
    sources = []
    if "搜索主题" in str(web_res):
        sources.append("联网搜索")

    # ── 第三步：让 LLM 综合资料写答案 ──
    prompt = f"""
你是一个专业研究员。请根据以下资料回答研究问题。

研究问题：{task.question}

【联网搜索资料】
{web_res}

【本地知识库资料】
{rag_res}

要求：
1. 内容详实，结构清晰
2. 优先使用本地知识库内容
3. 如果两份资料都不够，可以补充你的知识，但要标注出来
4. 字数控制在800字以内
"""

    response = await llm.ainvoke(prompt)

    return ResearchResult(
        task_id=task.id,
        question=task.question,
        content=response.content,
        sources=sources
    )


# ── 测试入口 ──
if __name__ == "__main__":
    async def main():
        # 模拟一个 Orchestrator 拆出来的子任务
        task = SubTask(
            id=1,
            title="LangChain核心架构",
            question="LangChain的核心架构和设计理念是什么？",
            keywords=["LangChain", "架构", "设计理念"]
        )
        print(f"研究问题: {task.question}")
        print("研究中...\n")
        result = await research_task(task)
        print("=" * 50)
        print(result.content)
        print(f"\n来源: {result.sources}")

    asyncio.run(main())
