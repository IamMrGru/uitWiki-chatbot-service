from langchain.text_splitter import MarkdownHeaderTextSplitter

with open('app/static/output/ctđt_tmđt_2021/ctđt_tmđt_2021.md', 'r', encoding='utf-8') as f:
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

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)
    print("\n")
