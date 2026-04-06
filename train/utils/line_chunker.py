from pydantic import BaseModel
from typing import List


class LineChunk(BaseModel):
    start_byte: int
    end_byte: int
    text: str


def split_code_into_lines(code: str) -> List[LineChunk]:
    """Split code into line chunks with correct byte offsets.

    Avoids using str.index which breaks on repeated lines; instead, track a
    running offset. Offsets are computed assuming \n as line separator.
    """
    lines = code.split("\n")
    chunks: List[LineChunk] = []
    offset = 0
    for i, line in enumerate(lines):
        start_byte = offset
        end_byte = start_byte + len(line)
        # Next line starts after this line and one newline (except possibly last)
        offset = end_byte + 1
        chunks.append(LineChunk(start_byte=start_byte, end_byte=end_byte, text=line))
    return chunks
