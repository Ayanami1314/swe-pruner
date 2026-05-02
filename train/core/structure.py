from pydantic import BaseModel
from typing import List


class CodeGroupItem(BaseModel):
    """A contiguous code fragment (e.g. a line) with byte offsets."""
    start_byte: int
    end_byte: int
    text: str
