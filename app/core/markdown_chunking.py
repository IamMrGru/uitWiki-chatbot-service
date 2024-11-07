import os
import re
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from app.services.s3_service import S3Services


def download_and_read_md(s3_key: str) -> str:
    s3_client = S3Services()

    local_md_path = os.path.join("/tmp", os.path.basename(s3_key))

    s3_client.download_file(s3_key, local_md_path)

    with open(local_md_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    os.remove(local_md_path)

    return markdown_text


def preprocess_ordered_list_to_header(content, header_prefix="List Section"):
    """
    Preprocess Markdown content by replacing each ordered list with a Markdown header
    that includes a section identifier, followed by a new header.

    Parameters:
    - content: str - The original Markdown content.
    - s3_key: str - Key to identify the specific file (used for unique section names).
    - header_prefix: str - The prefix text for each header to replace the list.

    Returns:
    - str - The modified Markdown content.
    """
    # Regular expression to find ordered list items (e.g., "1. Item" or "2) Item")
    ordered_list_pattern = re.compile(r"^\s*\d+[.)]\s")

    # Split the content by lines
    lines = content.splitlines()
    new_content = []
    section_count = 1  # To number each list section

    for i, line in enumerate(lines):
        # Check if the line is an ordered list item
        if ordered_list_pattern.match(line):
            # If it's the first item in a list, add a header before it
            if i == 0 or not ordered_list_pattern.match(lines[i-1]):
                header = f"### {header_prefix} {section_count}"
                new_content.append(header)
                section_count += 1  # Increment section counter

        # Add the original line
        new_content.append(line)

    # Join the list back into a single string
    return "\n".join(new_content)


def markdown_chunking(s3_key: str, metadata) -> list[Document]:
    markdown_text = download_and_read_md(s3_key)
    processed_content = preprocess_ordered_list_to_header(markdown_text)

    header_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    metadata_processed = {}

    for meta in metadata:
        metadata_processed[meta.name] = meta.value

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=header_to_split_on,
        strip_headers=False
    )

    chunks = splitter.split_text(processed_content)

    for chunk in chunks:
        chunk.metadata = metadata_processed

    return chunks
