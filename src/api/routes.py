"""API 路由：研究 + 文件上传"""
import os, sys, uuid, tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.graph.workflow import run_research
from src.tools.reader import read_pdf, read_image
from src.tools.rag import add_knowledge

router = APIRouter()

# 内存存储（临时，后续接数据库）
_reports: dict[str, dict] = {}


# ── 请求/响应模型 ──

class ResearchRequest(BaseModel):
    topic: str


class ResearchResponse(BaseModel):
    report_id: str
    topic: str
    summary: str
    sections: list[dict]
    conclusion: str
    unresolved: list[str]
    created_at: str


class UploadResponse(BaseModel):
    file_name: str
    result: str


# ── 研究接口 ──

@router.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    """提交研究主题，运行完整 Agent 流程，返回最终报告"""
    try:
        report = await run_research(req.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"研究流程异常: {str(e)}")

    report_id = str(uuid.uuid4())[:8]

    response = ResearchResponse(
        report_id=report_id,
        topic=report.topic,
        summary=report.summary,
        sections=[s.model_dump() for s in report.sections],
        conclusion=report.conclusion,
        unresolved=report.unresolved,
        created_at=datetime.now().isoformat(),
    )

    _reports[report_id] = response.model_dump()
    return response


# ── 文件上传接口 ──

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """上传 PDF 或图片，自动提取文字并存入向量知识库"""
    ext = os.path.splitext(file.filename or "")[1].lower()

    if ext not in (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持 PDF / PNG / JPG / GIF / WebP",
        )

    # 写入临时文件
    suffix = ext if ext else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            result = await read_pdf.ainvoke({"file_path": tmp_path})
        else:
            result = await read_image.ainvoke({"image_path": tmp_path})

        return UploadResponse(file_name=file.filename or "unknown", result=result)

    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── 报告查询 ──

@router.get("/report/{report_id}")
async def get_report(report_id: str):
    """按 ID 查询历史报告"""
    report = _reports.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")
    return report


@router.get("/reports")
async def list_reports():
    """列出所有报告"""
    return list(_reports.values())
