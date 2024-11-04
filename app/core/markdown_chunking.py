from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document


def markdown_chunking(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    header_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=header_to_split_on,
        strip_headers=False
    )

    chunks = splitter.split_text(markdown_text)

    for chunk in chunks:
        chunk.metadata = {
            'author': 'Phòng đào tạo',
            'title': 'Chương trình đào tạo ngành Thương mại điện tử 2021',
            'description': 'Chương trình đào tạo ngành Thương mại điện tử 2021',
            'url': 'https://student.uit.edu.vn/content/cu-nhan-nganh-thuong-mai-dien-tu-ap-dung-tu-khoa-15-2020'
        }

    return chunks
