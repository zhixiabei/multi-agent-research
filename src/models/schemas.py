"""统一的数据结构，所有 Agent 都用这一套"""
from pydantic import BaseModel


class SubTask(BaseModel):
    """Orchestrator 拆分出来的子任务"""
    id: int
    title: str              # 子任务标题
    question: str            # 具体要研究的问题
    keywords: list[str]      # 搜索关键词


class ResearchResult(BaseModel):
    """Worker 研究完返回的结果"""
    task_id: int
    question: str
    content: str             # 研究结果正文
    sources: list[str]       # 信息来源


class CritiqueResult(BaseModel):
    """Critic 审核完返回的意见"""
    task_id: int
    passed: bool             # 是否通过
    issues: list[str]        # 发现的问题
    suggestions: list[str]   # 改进建议


class ReportSection(BaseModel):
    """最终报告中的一个章节"""
    heading: str             # 章节标题
    content: str             # 章节正文
    sources: list[str]       # 本章引用的来源


class FinalReport(BaseModel):
    """Synthesizer 最终产出的完整报告"""
    topic: str               # 原始研究主题
    summary: str             # 执行摘要（200字以内）
    sections: list[ReportSection]  # 分章节
    conclusion: str          # 总结
    unresolved: list[str]    # 不同来源之间的矛盾 / 未解决的问题
