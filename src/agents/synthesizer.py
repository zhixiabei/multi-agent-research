"""Synthesizer: 把通过审核的研究结果合并成最终报告"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.models.schemas import ResearchResult, FinalReport, ReportSection

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("DEEPSEEK_BASE_URL")

# 合成需要一定的语言组织能力，温度比 Critic 略高
llm = ChatOpenAI(model="deepseek-chat", temperature=0.3)


async def synthesize(topic: str, results: list[ResearchResult]) -> FinalReport:
    """
    把多份通过了审核的研究结果合并成一份结构化的最终报告。

    职责：
    1. 合并 —— 把各个子问题的研究结果有机整合，不是机械拼接
    2. 去重 —— 不同 Worker 可能涉及了相同的点，合并同类项
    3. 发现矛盾 —— 不同来源如果结论冲突，标注出来而不是强行统一
    4. 结构化 —— 生成摘要 + 分章节正文 + 结论

    重要：严禁编造输入中不存在的内容！
    """
    # 拼接所有研究结果，带上编号方便 LLM 引用
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[研究结果 #{i}]\n问题: {r.question}\n内容:\n{r.content}\n来源: {', '.join(r.sources)}")
    all_results = "\n\n---\n\n".join(parts)

    prompt = f"""
你是一位资深报告撰写人。请根据以下多份研究结果，撰写一份结构清晰的最终报告。

## 原始研究主题
{topic}

## 各子问题的研究结果
{all_results}

---

## 你的任务

1. **合并同类项**：不同研究结果中如果有重复的观点，合并在一起
2. **发现矛盾**：如果不同来源对同一问题的结论有冲突，在 unresolved 中标注出来
3. **重新组织**：按逻辑顺序重新组织结构，不要简单按编号罗列
4. **忠于原文**：只能基于上面给出的研究结果来写，不要编造任何新的事实、数据、人名

---

## 输出格式（严格遵守 JSON，不要额外文字）

```json
{{
  "topic": "原始主题",
  "summary": "执行摘要，200字以内，概括核心发现",
  "sections": [
    {{
      "heading": "章节标题",
      "content": "章节正文，详细但不冗余",
      "sources": ["本章引用的来源1", "来源2"]
    }}
  ],
  "conclusion": "总结，提炼关键结论和启示",
  "unresolved": ["矛盾点或未解决的问题1", "矛盾点2"]
}}
```

- sections: 2-4 个章节，每个章节聚焦一个方面
- sources: 每个章节列出它引用了哪些信息源
- unresolved: 如果所有来源结论一致，则为空列表
"""

    response = await llm.ainvoke(prompt)
    raw = response.content.strip()

    # 清理 markdown 代码块
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]  # 去掉 ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        data = json.loads(raw)
        sections = [ReportSection(**s) for s in data["sections"]]
        return FinalReport(
            topic=data["topic"],
            summary=data["summary"],
            sections=sections,
            conclusion=data["conclusion"],
            unresolved=data.get("unresolved", [])
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"Synthesizer 解析 JSON 失败，原始输出:\n{raw[:500]}") from e


# ── 测试入口 ──
if __name__ == "__main__":
    import asyncio

    async def main():
        # 用之前 Critic 测试的同一个题目，模拟 3 份通过审核的研究结果
        results = [
            ResearchResult(
                task_id=1,
                question="Python 的 GIL 是什么？",
                content="""
GIL（Global Interpreter Lock）是 CPython 中的全局解释器锁，确保同一时刻只有一个线程执行 Python 字节码。
存在原因是 CPython 的内存管理（引用计数）不是线程安全的。

GIL 对 CPU 密集型多线程程序影响最大——多线程无法利用多核。但 I/O 密集型任务中线程等待时会释放 GIL，多线程仍然有效。
                """.strip(),
                sources=["Python 官方文档"]
            ),
            ResearchResult(
                task_id=2,
                question="GIL 如何影响多线程性能？",
                content="""
在 CPU 密集型场景下，GIL 导致多线程性能甚至不如单线程（因为多了上下文切换开销）。
CPU 密集任务用 multiprocessing 可以绕过 GIL，每个进程有独立的解释器和 GIL。

I/O 密集型场景不受明显影响，因为线程在等 I/O 时会释放 GIL，其他线程可以继续执行。
                """.strip(),
                sources=["Python 官方文档", "Real Python 教程"]
            ),
            ResearchResult(
                task_id=3,
                question="有哪些绕过 GIL 的方案？",
                content="""
绕过 GIL 的主要方案：
1. multiprocessing 模块：用多进程代替多线程，每个进程有独立的 GIL
2. C 扩展：在 C 代码中手动释放 GIL（如 NumPy 的做法）
3. 替代解释器：Jython、IronPython 没有 GIL，但各自有其他限制

Python 3.13 引入了 PEP 703 的实验性无 GIL 模式，但目前还是实验性的。
                """.strip(),
                sources=["PEP 703", "Python Wiki"]
            ),
        ]

        print("=" * 60)
        print(f"合成报告：{results[0].question.split('？')[0]}等")
        print(f"输入 {len(results)} 份研究结果")
        print("=" * 60)

        report = await synthesize("Python 的 GIL 是什么？如何影响多线程性能？", results)

        print(f"\n📄 {report.topic}")
        print(f"\n📌 摘要：{report.summary}")

        for i, sec in enumerate(report.sections, 1):
            print(f"\n{'─' * 40}")
            print(f"第{i}章：{sec.heading}")
            print(f"{'─' * 40}")
            print(sec.content)
            if sec.sources:
                print(f"  来源：{', '.join(sec.sources)}")

        print(f"\n{'─' * 40}")
        print(f"📝 结论：{report.conclusion}")

        if report.unresolved:
            print(f"\n⚠️ 未解决的问题：")
            for item in report.unresolved:
                print(f"  • {item}")

    asyncio.run(main())
