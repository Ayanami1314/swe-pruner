from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModel,
)
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from torch import Tensor
from FlagEmbedding import BGEM3FlagModel


class EmbedderAdapter(ABC):
    def embed(self, texts: list[str], batch_size: int) -> Tensor:
        pass


class QwenEmbedder(EmbedderAdapter):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_length: int = 8192,
        instruct: str = "Given a web search query, retrieve relevant passages that answer the query",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).cuda()
        self.max_length = max_length
        self.instruct = instruct

    def last_token_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"

    def embed(self, texts: list[str], batch_size: int = 32) -> Tensor:
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)

            batch_embeddings = self.last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)


class BertBasedEmbedder(EmbedderAdapter):
    def __init__(self, model_name: str, max_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.max_length = max_length

    def embed(self, texts: list[str], batch_size: int = 32) -> Tensor:
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize input texts
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Move to model device if available
            batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**batch_dict)

            # Use mean pooling over token embeddings
            attention_mask = batch_dict["attention_mask"]
            token_embeddings = outputs.last_hidden_state

            # Mask out padding tokens
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )

            # Sum embeddings and divide by number of tokens
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask

            # Normalize embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)


class BGEM3Embedder(EmbedderAdapter):
    def __init__(
        self, model_name: str = "BAAI/bge-m3", max_length: int = 8192, use_fp16=False
    ):
        self.model = BGEM3FlagModel(
            model_name, use_fp16=use_fp16
        )  # Setting use_fp16 to True speeds up computation with a slight performance degradation
        self.max_length = max_length

    def embed(self, texts: list[str], batch_size: int = 32) -> Tensor:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,  # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
        )["dense_vecs"]
        return torch.tensor(embeddings, dtype=torch.float32)


class BGECodeEmbedder(EmbedderAdapter):
    """
    Embedder for code using BGE-code model via FlagEmbedding.

    Example:
    --------
    from FlagEmbedding import FlagLLMModel
    queries = [
        "Delete the record with ID 4 from the 'Staff' table.",
        'Delete all records in the "Livestock" table where age is greater than 5'
    ]
    documents = [
        "DELETE FROM Staff WHERE StaffID = 4;",
        "DELETE FROM Livestock WHERE age > 5;"
    ]
    model = FlagLLMModel('BAAI/bge-code-v1',
                        query_instruction_format="<instruct>{}\n<query>{}",
                        query_instruction_for_retrieval="Given a question in text, retrieve SQL queries that are appropriate responses to the question.",
                        trust_remote_code=True,
                        use_fp16=True)
    embeddings_1 = model.encode_queries(queries)
    embeddings_2 = model.encode_corpus(documents)
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-code-v1",
        max_length: int = 8192,
        use_fp16: bool = False,
        query_instruction_format: str = "<instruct>{}\n<query>{}",
        query_instruction_for_retrieval: str = "Given a question in text, retrieve code snippets that are appropriate responses to the question.",
    ):
        from FlagEmbedding import FlagLLMModel

        self.model = FlagLLMModel(
            model_name,
            query_instruction_format=query_instruction_format,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            trust_remote_code=True,
            use_fp16=use_fp16,
        )
        self.max_length = max_length
        self.model_name = model_name

    def embed(
        self, texts: list[str], batch_size: int = 32, is_query: bool = False
    ) -> Tensor:
        """
        Embed texts using BGE code model.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            is_query: If True, encode as queries; if False, encode as corpus

        Returns:
            Tensor of embeddings
        """
        if is_query:
            embeddings = self.model.encode_queries(texts, batch_size=batch_size)
        else:
            embeddings = self.model.encode_corpus(texts, batch_size=batch_size)

        return torch.tensor(embeddings, dtype=torch.float32)


if __name__ == "__main__":
    # Check correctness of embedders
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ]

    print("=" * 80)
    print("Testing BertBasedEmbedder")
    print("=" * 80)
    try:
        bert_embedder = BertBasedEmbedder(
            model_name="bert-base-uncased", max_length=512
        )
        embeddings = bert_embedder.embed(texts, batch_size=2)
        print(f"✓ BertBasedEmbedder: embeddings shape = {embeddings.shape}")
        print(f"  Sample embedding norm: {torch.norm(embeddings[0]):.4f}")
    except Exception as e:
        print(f"✗ BertBasedEmbedder failed: {e}")

    print("\n" + "=" * 80)
    print("Testing BGEM3Embedder")
    print("=" * 80)
    try:
        bgem3_embedder = BGEM3Embedder(
            model_name="BAAI/bge-m3", max_length=8192, use_fp16=False
        )
        embeddings = bgem3_embedder.embed(texts, batch_size=2)
        print(f"✓ BGEM3Embedder: embeddings shape = {embeddings.shape}")
        print(f"  Embeddings type: {type(embeddings)}")
    except Exception as e:
        print(f"✗ BGEM3Embedder failed: {e}")

    print("\n" + "=" * 80)
    print("Testing QwenEmbedder")
    print("=" * 80)
    try:
        qwen_embedder = QwenEmbedder(
            model_name="Qwen/Qwen3-Embedding-0.6B", max_length=8192
        )
        embeddings = qwen_embedder.embed(texts, batch_size=2)
        print(f"✓ QwenEmbedder: embeddings shape = {embeddings.shape}")
        print(f"  Embeddings type: {type(embeddings)}")
    except Exception as e:
        print(f"✗ QwenEmbedder failed: {e}")

    print("\n" + "=" * 80)
    print("Testing BGECodeEmbedder")
    print("=" * 80)
    try:
        code_texts = [
            "def hello(): print('hello world')",
            "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        ]
        code_embedder = BGECodeEmbedder(model_name="BAAI/bge-code-v1", use_fp16=False)
        # Test corpus embedding
        embeddings = code_embedder.embed(code_texts, batch_size=2, is_query=False)
        print(f"✓ BGECodeEmbedder (corpus): embeddings shape = {embeddings.shape}")

        # Test query embedding
        query_embeddings = code_embedder.embed(
            ["get factorial of n"], batch_size=1, is_query=True
        )
        print(f"✓ BGECodeEmbedder (query): embeddings shape = {query_embeddings.shape}")

        # Compute similarity
        similarity = query_embeddings @ embeddings.T
        print(f"  Similarity scores: {similarity.numpy()}")
    except Exception as e:
        print(f"✗ BGECodeEmbedder failed: {e}")
