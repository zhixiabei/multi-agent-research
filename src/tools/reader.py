import os
from dotenv import load_dotenv
from langchain.tools import tool
from PyPDF2 import PdfReader
import httpx

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
HTTP_TIMEOUT = 15


@tool
async def read_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        if not full_text:
            return "PDF解析完成，但未提取到有效文本内容"
        return full_text
    except FileNotFoundError:
        return f"错误：文件 {file_path} 不存在"
    except Exception as e:
        return f"PDF解析失败：{str(e)}"


@tool
async def read_image(image_path: str) -> str:
    """
    解析本地图片内容，识别图片中的文字、图表、关键信息
    当需要分析截图、照片、流程图、图文资料时使用该工具

    Args:
        image_path: 本地图片文件路径
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

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
                            "image_url": {"url": f"data:image/jpeg;base64,{image_bytes.encode('base64').decode()}"}
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
            return res_data["choices"][0]["message"]["content"]
    except FileNotFoundError:
        return f"错误：图片 {image_path} 不存在"
    except Exception as e:
        return f"图片解析失败：{str(e)}"


# 对外暴露工具集合
FILE_TOOLS = [read_pdf, read_image]