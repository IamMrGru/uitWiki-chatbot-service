# LLM Model
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
# from langchain_anthropic import AnthropicLLM
# from langchain_huggingface import HuggingFaceEndpoint

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from app.core.config import settings
from pydantic import SecretStr

api_key = SecretStr(settings.GOOGLE_API_KEY)


def filter_by_metadata(question, new_db):
    metadata_field_info = [
        AttributeInfo(
            name="title",
            description="Tên của tài liệu chứa thông tin cần truy xuất",
            type="string",
        ),
        AttributeInfo(
            name="author",
            description="Phòng ban quản lý tài liệu",
            type="string",
        ),
        AttributeInfo(
            name="publicdate",
            description="Ngày công bố tài liệu",
            type="string",
        ),
        AttributeInfo(
            name="version",
            description="Phiên bản của tài liệu (Quyết định số...)",
            type="string",
        ),
        AttributeInfo(
            name="description",
            description="Sơ lược nội dung của tài liệu",
            type="string",
        ),
        AttributeInfo(
            name="category",
            description="Loại tài liệu: Quy định, Quy chế, quy trình, hướng dẫn,...",
            type="string",
        ),
        AttributeInfo(
            name="tags",
            description="Các từ khóa liên quan đến đoạn văn cần truy xuất",
            type="string"
        ),
        AttributeInfo(
            name="target",
            description="Đối tượng cần sử dụng, áp dụng",
            type="string"
        ),
        AttributeInfo(
            name="url",
            description="Đường dẫn đến tài liệu",
            type="string"
        ),
    ]
    document_content_description = "Tóm tắt nội dung của tài liệu này để phục vụ cho việc tư vấn sinh viên "
    llm = GoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0, api_key=api_key)

    retriever = SelfQueryRetriever.from_llm(
        llm,
        new_db,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={'k': 50},
        search_type='similarity',
    )
    docs = retriever.invoke(question)
    return docs


def get_conversational_chain():

    prompt_template = """
    Tên của bạn là UITWikiBot.Bạn là một trợ lý ảo của Trường Đại học Công nghệ Thông tin UIT.
    Được phát triển bởi nhóm sinh viên UIT: Hiển Đoàn và Hải Đào dưới sự hướng dẫn của thầy Tín.
    Vai trò của bạn là:
    - Giải đáp của sinh viên tại trường Đại học Công nghệ Thông tin UIT. (LƯU Ý: Tên của trường phải luôn luôn là Trường Đại học Công nghệ Thông tin - Đại học Quốc gia Thành phố Hồ Chí Minh (UIT) . Mọi tên khác đều không chính xác)
    - Thái độ câu trả lời của bạn phải chuyên nghiệp, lịch sự và thân thiện như là một Ambassador của trường UIT. Luôn coi người hỏi là một người bạn.
        - Bạn có thể ghi nhận sắc thái của người hỏi để trả lời lại một cách phù hợp.
    - Nhiệm vụ của bạn là trả lời các câu hỏi và thắc mắc của sinh viên một cách chi tiết và chính xác nhất.
    - Tôi sẽ đưa cho bạn 3 thành phần: METADATA (Các thẻ thông tin đính kèm dữ liệu liên quan), CONTEXT (Nội dung của tài liệu được trích ra), QUESTION (Câu hỏi tôi cần trả lời).
    - Metadata cũng có thể là chứa keyword mà trong câu hỏi có. Context có thể chính là nơi chứa đáp án của câu hỏi.
    - Hãy trả lời câu hỏi dựa trên các thông tin được cung cấp trong context và metadata.
    - Trong đó, context chính là nơi chứa đáp án của câu hỏi và metadata cũng có thể trả lời một số thông tin quan trọng đi kèm
    - Context là những mẫu bối cảnh rời rạc. Do đó, bạn hãy chắt lọc, ghép nối các context để trả lời câu hỏi một cách hợp lý .
    -------------------------   

    Dưới đây là thông tin tôi sẽ cung cấp cho bạn làm nền tảng  :
    *METADATA* là: ({metadata})
    ---
    *CONTEXT nền tảng* là: ({context})
    ---
    *QUESTION của người dùng* là: ({question}?)
    -------------------------

    Yêu cầu về câu trả lời:
    - Bạn không được tự đưa ra câu trả lời mà phải dựa vào CONTEXT
    - Hãy đảm bảo cung cấp đầy đủ chi tiết theo METADATA,CONTEXT.
    - Cố gắng liên kết thông tin giữa METADATA,CONTEXT để tạo ra câu trả lời chính xác nhất.
    - Hãy sắp xếp câu trả lời thành một cấu trúc đẹp dưới dạng Markdown. Ở những câu trả lời về quy định, các bước thực hiện, hãy sắp xếp câu trả lời theo thứ tự
    - Bạn không cần phải trả lời dựa vào đâu (Dựa vào METADATA được cung cấp...., dựa vào CONTEXT ta thấy,...).Tức là không cần phải trả lời dựa vào thẻ Metadata
    - Đưa ra một câu trả lời tự nhiên và dễ hiểu nhất có thể.
    - Không tự trả lời mà không có trong METADATA,CONTEXT. Nếu không có bạn hãy trả lời (Mình không nắm được thông tin câu hỏi này) đồng thời hãy gợi ý cho người hỏi dựa theo câu hỏi họ đã đưa, tạo các câu hỏi mới để người dùng chọn 
    - Bạn phải chỉ rõ là phần bạn trích dẫn tên tài liệu chứa thông tin hoặc nội dung đó nằm ở phần nào của tài liệu đó Ở PHẦN CUỐI CỦA CÂU TRẢ LỜI CỦA BẠN. 
        (Cần chỉ rõ nằm ở tài liệu nào.Embed thêm URL thì càng tốt. Nói là Trích từ [Nội dung](https://www.uit.edu.vn).)
    - Tránh trường hợp là (sử dụng context là)
    - Với dạng câu hỏi liên quan đến chuẩn đầu ra, bạn hãy đối chiếu với từng context trong dấu [] để trả lời tương ứng với số năm và loại chương trình
        - Ví dụ như bạn phải xét từng record kiểu list có trong context [2014,QĐ...,Ko có,TOEIC 900] thì ở đây '2014' là KHÓA, 'QĐ..' là CĂN CỨ QUYẾT ĐỊNH, 'Ko có' là CHUẨN QUÁ TRÌNH, 'TOEIC 900' là CHUẨN ĐẦU RA
    - Với những dạng câu hỏi Có/Không, đầu tiên phải phản hồi Có hoặc Không, phải trả lời lại nếu như người hỏi nói sai và phải giải thích vì sao trả lời như vậy, dựa trên trích dẫn thông tin đó lấy từ tài liệu nào 
    - Những câu trả lời có đường dẫn đến link URL hay đường dẫn để download, bạn hãy embed link đó vào câu trả lời của mình.
    - Hãy embed đường dẫn tải các mẫu đơn vào tên mẫu đơn đó.
       - Ví dụ như : [Đường dẫn tải mẫu đơn](https://www.uit.edu.vn)
    - Nếu như câu hỏi của người hỏi không rõ ràng hoặc khó hiểu, bạn hãy giúp họ, gợi ý cho họ câu hỏi để bạn có thể giải quyết một cách dễ dàng (Có phải ý của bạn là....)
    """

    model = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0,
                               api_key=api_key)

    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "context", "question", 'metadata'])
    chain = load_qa_chain(model, chain_type="stuff",
                          prompt=prompt, verbose=True)
    return chain
