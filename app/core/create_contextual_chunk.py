from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import settings

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=SecretStr(settings.OPENAI_API_KEY)
)

# with open('app/static/output/kenhthongtingiuanhatruongvasinhvien/kenhthongtingiuanhatruongvasinhvien.md', 'r', encoding='utf-8') as f:
#     markdown_text = f.read()


def create_contextual_chunk(document: str, chunk: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
        Bạn là một trợ lý AI chuyên về phân tích tài liệu. Nhiệm vụ của bạn là cung cấp bối cảnh ngắn gọn, phù hợp cho một đoạn văn bản từ tài liệu được cung cấp.

        Đây là tài liệu:
        <document>
        {document}
        </document>

        Đây là đoạn văn bản cần được đặt vào bối cảnh toàn bộ tài liệu:
        <chunk>
        {chunk}
        </chunk>

        Hãy cung cấp bối cảnh ngắn gọn cho đoạn văn bản này, tuân theo các hướng dẫn sau:
        1. Xác định chủ đề chính hoặc khái niệm được thảo luận trong đoạn văn.
        2. Đề cập đến bất kỳ thông tin liên quan hoặc so sánh nào từ ngữ cảnh rộng hơn của tài liệu.
        3. Nếu có thể, nêu rõ mối liên hệ giữa thông tin này và chủ đề hoặc mục đích tổng thể của tài liệu.
        4. Bao gồm các số liệu chính, ngày tháng, liên kết web, URL hoặc phần trăm quan trọng nếu có.
        5. Không sử dụng các cụm từ như "Đoạn văn này đề cập đến", "Bối cảnh", "Ngữ cảnh", "Đoạn văn này trình bày" hoặc bất kỳ cụm từ nào tương tự.
        6. Trả lời bằng tiếng Việt và định dạng tệp Markdown.
        7. Trực tiếp cung cấp bối cảnh mà không thêm bất kỳ thông tin hay giải thích nào khác.
        8. Chỉ trả lời dựa trên tài liệu đã cung cấp, không thêm thông tin từ nguồn bên ngoài.
        Bối cảnh:
        """)

    messages = prompt.format_messages(document=document, chunk=chunk)
    response = llm.invoke(messages)

    if isinstance(response.content, str):
        return response.content
    elif isinstance(response.content, list):
        return "\n".join(str(item) for item in response.content)
    else:
        raise ValueError("Unexpected response content type")
