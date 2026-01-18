from astchunk import ASTChunkBuilder
from pathlib import Path

# Your source code
file = Path(__file__).parent / "code_stmt_splitter.py"
code = file.read_text()

# Initialize the chunk builder
configs = {
    "max_chunk_size": 512,  # Maximum non-whitespace characters per chunk
    "language": "python",  # Supported: python, java, csharp, typescript
    "metadata_template": "default",  # Metadata format for output
    "chunk_expansion": True,
}
chunk_builder = ASTChunkBuilder(**configs)

# Create chunks
chunks = chunk_builder.chunkify(code)

# Each chunk contains content and metadata
for i, chunk in enumerate(chunks):
    print(f"[Chunk {i + 1}]")
    print(f"{chunk['content']}")
    print(f"Metadata: {chunk['metadata']}")
    print("-" * 50)
