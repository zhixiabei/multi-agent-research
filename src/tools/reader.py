import os
import base64
import mimetypes
from dotenv import load_dotenv
from langchain.tools import tool
from PyPDF2 import PdfReader
import httpx

from src.tools.rag import add_knowledge

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
HTTP_TIMEOUT = 15


@tool
async def read_pdf(file_path: str) -> str:
    """
    读取本地PDF文档，提取文字后自动语义分块存入向量知识库。
    当需要分析PDF格式资料、论文、文档时使用该工具。

    Args:
        file_path: 本地PDF文件的绝对/相对路径
    """
    try:
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        if not full_text:
            return "PDF解析完成，但未提取到有效文本内容"

        # 提取后自动语义分块入库
        file_name = os.path.basename(file_path)
        result = await add_knowledge.ainvoke({
            "text": full_text,
            "doc_source": file_name
        })
        return f"PDF「{file_name}」已处理 → {result}"

    except FileNotFoundError:
        return f"错误：文件 {file_path} 不存在"
    except Exception as e:
        return f"PDF解析失败：{str(e)}"


@tool
async def read_image(image_path: str) -> str:
    """
    解析本地图片内容，识别图片中的文字、图表、关键信息，自动存入向量知识库。
    当需要分析截图、照片、流程图、图文资料时使用该工具。

    Args:
        image_path: 本地图片文件路径
    """
    try:
        # 根据扩展名推断 MIME 类型，默认 fallback 到 image/png
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        b64_data = base64.b64encode(image_bytes).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-vl",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "详细描述这张图片里的所有内容、文字和信息"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
                        }
                    ]
                }
            ],
            "stream": False
        }

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                url=f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            resp.raise_for_status()
            res_data = resp.json()
            content = res_data["choices"][0]["message"]["content"]

        # 识别结果自动语义分块入库
        file_name = os.path.basename(image_path)
        result = await add_knowledge.ainvoke({
            "text": content,
            "doc_source": file_name
        })
        return f"图片「{file_name}」已解析 → {result}"

    except FileNotFoundError:
        return f"错误：图片 {image_path} 不存在"
    except Exception as e:
        return f"图片解析失败：{str(e)}"

FILE_TOOLS = [read_pdf, read_image]