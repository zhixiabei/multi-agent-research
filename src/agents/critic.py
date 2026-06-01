"""Critic: 审核 Worker 的研究结果，挑错、打分、给修改意见"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.models.schemas import ResearchResult, CritiqueResult

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("DEEPSEEK_BASE_URL")

# 审查用极低温度，保证标准一致、严格
llm = ChatOpenAI(model="deepseek-chat", temperature=0.0)


async def critique_result(result: ResearchResult) -> CritiqueResult:
    """
    审核一份研究结果，从四个维度检查：

    1. 切题性  — 是否真正回答了原始问题？有没有跑偏？
    2. 准确性  — 事实是否正确？有没有明显的胡编乱造？
    3. 完整性  — 是否涵盖了问题的关键方面？有没有重要遗漏？
    4. 可信度  — 来源是否靠谱？有没有给出引用？

    返回 CritiqueResult：
      - passed=True   → 质量合格，可以进入合成阶段
      - passed=False  → 需要退回 Worker 修改
    """
    prompt = f"""
你是一位严厉的学术审稿人。请审核以下研究报告，从四个维度评估。

## 原始研究问题
{result.question}

## 研究报告正文
{result.content}

## 声称的信息来源
{result.sources if result.sources else "（未提供来源）"}

---

## 审核标准

### 1. 切题性
- 报告是否直接回答了原始问题？
- 有没有大段跑题、答非所问？

### 2. 事实准确性
- 有没有明显的事实错误或逻辑矛盾？
- 有没有看起来像编造的数据、人名、论文标题？（重点检查）

### 3. 完整性
- 是否覆盖了问题的核心要点？
- 有没有偷懒、只写表面、回避难点？

### 4. 信息来源
- 声称的来源是否可信？
- 有没有该引用但没有引用的地方？

---

## 通过标准
- 四个维度都基本合格 → passed=true
- 存在**严重问题**（大段跑题 / 明显编造 / 核心内容缺失）→ passed=false

注意：轻微瑕疵（如个别句子不通顺、小遗漏）仍可判通过，但要记录在 issues 里。

---

## 输出格式（严格遵守 JSON，不要额外文字）

```json
{{
  "passed": true,
  "issues": ["问题描述1", "问题描述2"],
  "suggestions": ["改进建议1", "改进建议2"]
}}
```

- passed: true=通过, false=不通过
- issues: 发现的具体问题（空列表表示无问题）
  -**passed=true时**，返回一些轻微的问题或者瑕疵，比如有遗漏或者个别句子不通顺，无问题则返回空列表
  -**passed=false时**，返回你认为导致不通过的问题，让worker重写
- suggestions: 具体的改进建议，供 Worker 修改时参考（空列表表示无需修改）
  -**passed=true时，为空列表，'[]'**(报告合格)
  -**passed=false时**，返回具体的，可以操作的列表，让worker重写
"""

    response = await llm.ainvoke(prompt)
    raw = response.content.strip()

    # 清理可能的 markdown 代码块标记
    if raw.startswith("```"):
        lines = raw.split("\n")
        # 去掉第一行（```json 或 ```）
        lines = lines[1:]
        # 去掉最后一行（```）如果存在
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        data = json.loads(raw)
        return CritiqueResult(
            task_id=result.task_id,
            passed=data["passed"],
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", [])
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # fail safe：退回 Worker 重写
        print(f"[Critic 错误] JSON 解析失败，退回重审。原始输出:\n{raw[:500]}")
        return CritiqueResult(
            task_id=result.task_id,
            passed=False,
            issues=[f"Critic 自身解析异常: {str(e)}"],
            suggestions=["[系统] Critic 无法解析审查结果，请 Worker 重新撰写并确保内容格式规范"]
        )


# ── 测试入口 ──
if __name__ == "__main__":
    import asyncio

    async def main():
        # 模拟 Worker 产出的研究结果
        result = ResearchResult(
            task_id=1,
            question="Python 的 GIL 是什么？它如何影响多线程性能？",
            content="""
GIL（Global Interpreter Lock，全局解释器锁）是 CPython 解释器中的一个互斥锁，
它确保同一时刻只有一个线程在执行 Python 字节码。

GIL 存在的主要原因是 CPython 的内存管理不是线程安全的。
它导致 CPU 密集型多线程程序无法利用多核优势——即使开了多个线程，
同一时刻也只有一个在真正执行。但对于 I/O 密集型任务，
线程在等待 I/O 时会释放 GIL，所以多线程仍然有效。

绕过 GIL 的方案包括：
1. 使用 multiprocessing 模块（多进程）
2. 使用 C 扩展，在扩展中手动释放 GIL
3. 使用其他 Python 实现，如 Jython 或 IronPython（它们没有 GIL）

Python 社区多年来一直在讨论移除 GIL，Python 3.13 引入了
PEP 703 的实验性无 GIL 模式。
            """.strip(),
            sources=["Python 官方文档", "PEP 703"]
        )

        print("=" * 60)
        print(f"审核问题: {result.question}")
        print(f"报告长度: {len(result.content)} 字符")
        print("=" * 60)

        verdict = await critique_result(result)
        print(f"\n审核结果: {'✅ 通过' if verdict.passed else '❌ 不通过'}")
        print(f"\n发现的问题 ({len(verdict.issues)}):")
        for i, issue in enumerate(verdict.issues, 1):
            print(f"  {i}. {issue}")
        print(f"\n改进建议 ({len(verdict.suggestions)}):")
        for i, sug in enumerate(verdict.suggestions, 1):
            print(f"  {i}. {sug}")

    asyncio.run(main())
