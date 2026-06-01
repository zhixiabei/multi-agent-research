"""LangGraph 工作流：串联 Orchestrator → Worker → Critic → Synthesizer"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import operator
from typing import Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from src.agents.orchestrator import decompose_topic
from src.agents.worker import research_task
from src.agents.critic import critique_result
from src.agents.synthesizer import synthesize
from src.models.schemas import SubTask, ResearchResult, CritiqueResult, FinalReport

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("DEEPSEEK_BASE_URL")

MAX_RETRIES = 2


# ── 状态定义 ──
class ResearchState(dict):
    topic: str
    subtasks: list[SubTask]
    results: Annotated[list[ResearchResult], operator.add]   # 累加，支持增量追加
    critiques: Annotated[list[CritiqueResult], operator.add]
    failed_task_ids: list[int]
    retry_round: int
    final_report: FinalReport | None


# ── 节点函数 ──

async def decompose_node(state: ResearchState) -> dict:
    """Orchestrator 拆解主题 → N 个子任务"""
    print(f"\n{'='*50}\n[Orchestrator] 拆解主题: {state['topic']}\n{'='*50}")
    subtasks = await decompose_topic(state["topic"], num_tasks=4)
    for t in subtasks:
        print(f"  [{t.id}] {t.title} → {t.question}")
    return {
        "subtasks": subtasks,
        "results": [],
        "critiques": [],
        "retry_round": 0,
    }


async def research_node(state: ResearchState) -> dict:
    """Worker × N 并行研究"""
    # 确定哪些子任务需要研究
    failed_ids = set(state.get("failed_task_ids", []) or [])
    if failed_ids:
        targets = [t for t in state["subtasks"] if t.id in failed_ids]
        print(f"\n[Worker] 重写 {len(targets)} 个不通过的子任务: {failed_ids}")
    else:
        targets = state["subtasks"]
        print(f"\n[Worker] 并行研究 {len(targets)} 个子任务...")

    async def do(task: SubTask) -> ResearchResult:
        print(f"  → 研究: {task.question[:40]}...")
        return await research_task(task)

    new_results = await asyncio.gather(*[do(t) for t in targets])
    return {"results": list(new_results)}


async def critique_node(state: ResearchState) -> dict:
    """Critic 审核所有未审核的结果"""
    # 已有 critique 的 task_id
    reviewed_ids = {c.task_id for c in (state.get("critiques", []) or [])}

    to_review = [r for r in state["results"] if r.task_id not in reviewed_ids]
    if not to_review:
        return {}

    print(f"\n[Critic] 审核 {len(to_review)} 份研究报告...")

    async def do(r: ResearchResult) -> CritiqueResult:
        print(f"  → 审核: {r.question[:40]}...", end=" ")
        c = await critique_result(r)
        print("✅ 通过" if c.passed else "❌ 不通过")
        return c

    new_critiques = await asyncio.gather(*[do(r) for r in to_review])
    return {"critiques": list(new_critiques)}


async def synthesize_node(state: ResearchState) -> dict:
    """Synthesizer 合并所有通过审核的结果 → 最终报告"""
    passed_ids = {c.task_id for c in state["critiques"] if c.passed}
    passed_results = [r for r in state["results"] if r.task_id in passed_ids]

    print(f"\n[Synthesizer] 合并 {len(passed_results)}/{len(state['results'])} 份通过审核的结果...")

    report = await synthesize(state["topic"], passed_results)

    print(f"\n{'='*50}")
    print(f"📄 最终报告: {report.topic}")
    print(f"📌 摘要: {report.summary[:100]}...")
    print(f"📝 章节数: {len(report.sections)}")
    print(f"{'='*50}")

    return {"final_report": report}


# ── 路由函数 ──

def should_retry(state: ResearchState) -> str:
    """判断是否需要退回重写"""
    critiques = state.get("critiques", [])
    if not critiques:
        return "research"

    failed = [c for c in critiques if not c.passed]
    retry_round = state.get("retry_round", 0)

    if failed and retry_round < MAX_RETRIES:
        failed_ids = [c.task_id for c in failed]
        print(f"\n[Router] {len(failed)} 个不通过，第 {retry_round + 1}/{MAX_RETRIES} 轮重试 → {failed_ids}")
        return "research"

    return "synthesize"


# ── 构建图 ──

def build_workflow() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("decompose", decompose_node)
    graph.add_node("research", research_node)
    graph.add_node("critique", critique_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "research")
    graph.add_edge("research", "critique")

    graph.add_conditional_edges(
        "critique",
        should_retry,
        {
            "research": "research",
            "synthesize": "synthesize",
        }
    )

    graph.add_edge("synthesize", END)

    return graph.compile()


# ── 便捷入口 ──

async def run_research(topic: str) -> FinalReport:
    """运行完整研究流程，返回最终报告"""
    workflow = build_workflow()
    state = await workflow.ainvoke({"topic": topic})
    return state["final_report"]


# ── 测试入口 ──
if __name__ == "__main__":
    async def main():
        topic = input("请输入研究主题: ").strip()
        if not topic:
            topic = "LangChain和LlamaIndex的对比分析"
            print(f"使用默认主题: {topic}")

        report = await run_research(topic)

        print("\n" + "=" * 60)
        print(f"📄 最终报告：{report.topic}")
        print("=" * 60)
        print(f"\n📌 摘要\n{report.summary}")

        for i, sec in enumerate(report.sections, 1):
            print(f"\n{'─' * 50}")
            print(f"第{i}章 {sec.heading}")
            print(f"{'─' * 50}")
            print(sec.content)
            if sec.sources:
                print(f"\n  来源: {', '.join(sec.sources)}")

        print(f"\n{'─' * 50}")
        print(f"📝 结论\n{report.conclusion}")

        if report.unresolved:
            print(f"\n⚠️ 未解决的问题:")
            for item in report.unresolved:
                print(f"  • {item}")

    asyncio.run(main())
