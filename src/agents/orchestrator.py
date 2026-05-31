"""Orchestrator: 把大题目拆成可执行的子任务"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.models.schemas import SubTask

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("DEEPSEEK_BASE_URL")

llm = ChatOpenAI(model="deepseek-chat", temperature=0.2)


async def decompose_topic(topic: str, num_tasks: int = 4) -> list[SubTask]:
    """
    输入一个研究主题，输出拆解后的子任务列表

    例如：
      topic = "AI对医疗行业的影响"
      → ["AI在诊断方面的应用现状", "AI药物研发进展", "医疗AI的监管政策", "AI对医生工作流程的改变"]
    """
    prompt = f"""
你是一个资深研究规划师。请把以下研究主题拆成 {num_tasks} 个独立的子问题。

规则：
1. 每个子问题要具体、可独立研究
2. 子问题合在一起能覆盖原主题的方方面面
3. 每个子问题附2-3个关键词
4. 只输出 JSON 数组，不要任何其他文字

研究主题：{topic}

输出格式（严格遵守）：
[
  {{
    "id": 1,
    "title": "子问题简短标题",
    "question": "具体的、完整的研究问题",
    "keywords": ["关键词1", "关键词2", "关键词3"]
  }}
]
"""

    response = await llm.ainvoke(prompt)
    raw = response.content.strip()

    # 清理 LLM 可能多输出的 ```json 标记
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        data = json.loads(raw)
        return [SubTask(**item) for item in data]
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Orchestrator 解析 JSON 失败，原始输出:\n{raw}") from e


# ── 测试入口 ──
if __name__ == "__main__":
    import asyncio

    async def main():
        topic = input("请输入研究主题: ")
        print(f"原始题目：{topic}\n")
        tasks = await decompose_topic(topic, num_tasks=4)
        print("拆解结果：")
        for t in tasks:
            print(f"  [{t.id}] {t.title}")
            print(f"      问题: {t.question}")
            print(f"      关键词: {', '.join(t.keywords)}")
            print()

    asyncio.run(main())
