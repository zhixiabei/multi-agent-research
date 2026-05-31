import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 只导入工具，不导入复杂Agent工厂类
from src.tools.search import SEARCH_TOOLS
from src.tools.reader import FILE_TOOLS
from src.tools.rag import RAG_TOOLS

load_dotenv()

# llm
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("DEEPSEEK_BASE_URL")

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3
)
# 合并所有工具
all_tools = SEARCH_TOOLS + FILE_TOOLS + RAG_TOOLS

# 2. 手动实现简易工具调用逻辑（避开臃肿依赖）
async def run_research(topic: str) -> str:
    # 第一轮：先调用RAG检索
    rag_tool = [t for t in all_tools if t.name == "query_knowledge"][0]
    rag_res = await rag_tool.ainvoke(topic)

    # 拼接上下文 + 指令
    prompt = f"""
你是专业研究助手，请根据资料撰写完整研究报告。
规则：优先使用本地知识库内容，内容不足再结合联网信息。
研究主题：{topic}
本地知识库内容：{rag_res}
"""
    # 调用大模型生成初稿
    first_resp = await llm.ainvoke(prompt)
    content = first_resp.content

    # 简单判断：知识库无内容，则调用搜索
    if "未查询到相关内容" in rag_res:
        search_tool = [t for t in all_tools if t.name == "web_search"][0]
        search_res = await search_tool.ainvoke(topic)
        final_prompt = f"""
研究主题：{topic}
联网搜索资料：{search_res}
结合资料撰写正式研究报告。
"""
        final_resp = await llm.ainvoke(final_prompt)
        return final_resp.content

    return content

# 测试入口
if __name__ == "__main__":
    test_topic = "介绍LangChain与LangGraph的区别和应用场景"
    print("===== 单Agent 开始执行 =====")
    report = asyncio.run(run_research(test_topic))
    print("\n===== 最终报告 =====")
    print(report)