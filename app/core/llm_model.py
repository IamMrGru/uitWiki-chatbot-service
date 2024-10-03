# LLM Model
import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
# from langchain_anthropic import AnthropicLLM
# from langchain_huggingface import HuggingFaceEndpoint

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from app.core.config import settings
from pydantic import SecretStr

api_key = SecretStr(settings.GOOGLE_API_KEY)


def get_conversational_chain():

    prompt_template = """
    Vai trò của UITBot:
    - Bạn là một trợ lý ảo giải đáp của sinh viên tại trường đại học UIT.
    - Nhiệm vụ của bạn là trả lời các câu hỏi và thắc mắc của sinh viên một cách chi tiết và chính xác nhất.
    - Metadata cũng có thể là chứa keyword mà trong câu hỏi có. Context có thể chính là nơi chứa đáp án của câu hỏi.
    - Hãy trả lời câu hỏi dựa trên các thông tin được cung cấp trong context và metadata.
    - Trong đó, context chính là nơi chứa đáp án của câu hỏi và metadata cũng có thể trả lời một số thông tin quan trọng đi kèm
    -------------------------

    Dưới đây là thông tin cần thiết:
    *METADATA* là: ({metadata})
    ---
    *CONTEXT* là: ({context})
    ---
    *QUESTION* là: ({question})
    -------------------------

    Yêu cầu về câu trả lời:
    - Hãy đảm bảo cung cấp đầy đủ chi tiết theo METADATA,CONTEXT.
    - Hãy sắp xếp câu trả lời thành một cấu trúc đẹp. Bạn không cần phải trả lời dựa vào đâu (Dựa vào METADATA được cung cấp...., dựa vào CONTEXT ta thấy,...)
    - Đưa ra một câu trả lời tự nhiên và dễ hiểu nhất có thể.
    - Không tự trả lời mà không có trong METADATA,CONTEXT. Nếu không có hãy trả lời (Vui lòng cung cấp thêm thông tin chi tiết)
    - Bạn có thể cung cấp nội dung này được trích dẫn trong tài liệu nào.(Chỉ trích dẫn điều nào, tên tài liệu nào chứ không cần chỉ rõ trang)
    """

    # Chọn model để tạo chain
    # model=ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0) # Có thể lựa chọn model LLM khác
    model = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0,
                               api_key=api_key)
    # model=HuggingFaceEndpoint(repo_id="NlpHUST/gpt2-vietnamese",huggingfacehub_api_token='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',max_new_tokens=200)
    # model=AnthropicLLM(model="claude-2.1",api_key=os.getenv('ANTHROPIC_API_KEY'),temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "context", "question", 'metadata'])
    chain = load_qa_chain(model, chain_type="stuff",
                          prompt=prompt, verbose=True)
    return chain
