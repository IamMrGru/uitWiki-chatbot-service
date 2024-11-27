import os
import re
import tempfile

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from app.core.create_contextual_chunk import create_contextual_chunk
from app.services.pinecone_service import PineconeService
from app.services.s3_service import S3Services

pinecone_service = PineconeService()


def download_and_read_md(s3_key: str) -> str:
    s3_client = S3Services()

    local_md_path = os.path.join(
        tempfile.gettempdir(), os.path.basename(s3_key))

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
                header = f"### {section_count}"
                new_content.append(header)
                section_count += 1  # Increment section counter

        # Add the original line
        new_content.append(line)

    # Join the list back into a single string
    return "\n".join(new_content)


def preprocess_tables_to_header(content, header_prefix="Table Section"):
    """
    Preprocess Markdown content by adding headers before each table.

    Parameters:
    - content: str - The original Markdown content.
    - header_prefix: str - The prefix text for each header to replace the table.

    Returns:
    - str - The modified Markdown content with headers added before each table.
    """
    # Regular expression to find the start of a Markdown table (| headers or |---| separator)
    table_pattern = re.compile(r"^\s*\|.*\|\s*$")
    separator_pattern = re.compile(r"^\s*\|[ -]+\|\s*$")

    # Split the content by lines
    lines = content.splitlines()
    new_content = []
    table_count = 1  # To number each table section
    in_table = False  # Track if we're inside a table

    for i, line in enumerate(lines):
        # Detect table start (headers) or separator line
        if table_pattern.match(line) or separator_pattern.match(line):
            if not in_table:
                # Add a new header at the start of a table
                header = f"### {header_prefix} {table_count}"
                new_content.append(header)
                table_count += 1
                in_table = True  # Now inside a table
        else:
            in_table = False  # Reset when table ends

        # Add the original line
        new_content.append(line)

    # Join the list back into a single string
    return "\n".join(new_content)


async def markdown_chunking(s3_key: str, metadata) -> list[Document]:
    markdown_text = download_and_read_md(s3_key)
    processed_order_list = preprocess_ordered_list_to_header(markdown_text)
    processed_content = preprocess_tables_to_header(processed_order_list)

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

    chunks = splitter.split_text(processed_order_list)

    for index, chunk in enumerate(chunks):
        chunk.metadata = metadata_processed
        # chunk.page_content = create_contextual_chunk(
        #     processed_content, chunk.page_content
        # )
        await pinecone_service.upsert_chunk(chunk)
        print(f'Successfully upserted chunk {index + 1} of {len(chunks)}')

    return chunks
