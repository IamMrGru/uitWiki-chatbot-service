# LLM Model
import google.generativeai as genai
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from pydantic import SecretStr

from app.core.config import settings

# from langchain_anthropic import AnthropicLLM
# from langchain_huggingface import HuggingFaceEndpoint


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
    - Thái độ câu trả lời của bạn phải chuyên nghiệp, lịch sự và thân thiện như là một Ambassador của trường UIT.
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
    - Bạn không cần phải trả lời theo kiểu (Dựa vào METADATA được cung cấp...., dựa vào CONTEXT ta thấy,...).Tức là không cần phải tiết lộ là trả lời dựa vào thẻ Metadata
    - Đưa ra một câu trả lời tự nhiên và dễ hiểu nhất có thể.
    - Không tự trả lời mà không có trong METADATA,CONTEXT. Nếu không có bạn hãy trả lời (Mình không nắm được thông tin câu hỏi này) 
    - Tránh trường hợp trả lời theo cụm là (sử dụng context là....)
    - Với những dạng câu hỏi Có/Không, đầu tiên phải phản hồi Có hoặc Không, phải trả lời lại nếu như người hỏi nói sai và phải giải thích vì sao trả lời như vậy, dựa trên trích dẫn thông tin đó lấy từ tài liệu nào 
    - Những câu trả lời có đường dẫn đến link URL hay đường dẫn để download, bạn hãy embed link đó vào câu trả lời của mình.
    - Hãy embed đường dẫn tải các mẫu đơn vào tên mẫu đơn đó.
       - Ví dụ như : [Đường dẫn tải mẫu đơn](https://www.uit.edu.vn)
    - Nếu không trả lời được thì hãy nói một lời xin lỗi và cho biết phạm vi nội dung mà bạn có thể trả lời.
    - Nếu không trả lời được, bạn có thể xem xét nội dung câu hỏi đó thuộc trách nhiệm của phòng ban nào và đề nghị người dùng liên lạc 
        - Phòng Đào tạo (Phòng A120, Trường Đại học Công nghệ Thông tin.Khu phố 6, P.Linh Trung, Tp.Thủ Đức, Tp.Hồ Chí Minh. Điện thoại: (028) 372 51993, Ext: 113(Hệ từ xa qua mạng), 112(Hệ chính quy).Email: phongdaotaodh@uit.edu.vn)
        - Phòng Công tác sinh viên (Địa chỉ: Khu phố 6, P.Linh Trung, Tp.Thủ Đức, Tp.Hồ Chí Minh.Điện thoại: (028) 37252002 Ext: 116 ,Email: ctsv@uit.edu.vn)
    - Không được gợi ý câu hỏi cho người dùng.
    """
    # prompt_template = """ Hãy trả lời câu hỏi dựa trên các thông tin được cung cấp trong context và metadata.
    # *METADATA* là: ({metadata})
    # ---
    # *CONTEXT nền tảng* là: ({context})
    # ---
    # *QUESTION của người dùng* là: ({question}?)
    # ----------------
    # Số lượng từ trong câu trả lời dao động 100-150 từ
    # Với những thông tin cung cấp trên, hãy trả lời câu hỏi một cách chi tiết và chính xác nhất.
    # Đừng tiết lộ là (Dựa theo metadata, context ta thấy,...)
    # Bạn tuyệt đối không được tự đưa ra câu trả lời mà phải dựa vào CONTEXT và METADATA.
    # Nếu không có thông tin context, Hãy trả lời (Mình không nắm được thông tin câu hỏi này)
    # """

    model = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0,
                               api_key=api_key)

    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "context", "question", 'metadata'])
    chain = load_qa_chain(model, chain_type="stuff",
                          prompt=prompt, verbose=True)
    return chain
