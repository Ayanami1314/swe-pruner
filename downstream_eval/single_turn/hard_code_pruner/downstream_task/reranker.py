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


class RerankerAdapter(ABC):
    @abstractmethod
    def rerank(
        self, query: str, documents: list[str], batch_size: int
    ) -> list[tuple[float, str]]:
        """
        Rerank documents and return scores.
        
        Returns:
            list[tuple[float, str]]: List of (score, document) tuples, sorted by score descending
        """
        pass


class BertBasedReranker(RerankerAdapter):
    def __init__(self, model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def rerank(
        self, query: str, documents: list[str], batch_size: int = 32
    ) -> dict[float, str]:
        all_scores = []

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            pairs = [[query, doc] for doc in batch_docs]

            inputs = self.tokenizer(
                pairs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                batch_scores = (
                    self.model(**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                )
            all_scores.extend(batch_scores.tolist())

        return [(score, doc) for score, doc in zip(all_scores, documents)]


class BGEV2M3Reranker(RerankerAdapter):
    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def rerank(
        self, query: str, documents: list[str], batch_size: int = 32
    ) -> dict[float, str]:
        all_scores = []

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            pairs = [[query, doc] for doc in batch_docs]

            inputs = self.tokenizer(
                pairs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                batch_scores = (
                    self.model(**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                )
            all_scores.extend(batch_scores.tolist())

        return [(score, doc) for score, doc in zip(all_scores, documents)]


class BGELLMBasedReranker(RerankerAdapter):
    def __init__(
        self,
        prompt: str,
        model_name: str,
        max_length: int = 8192,
        yes_symbol: str = "Yes",
        sep: str = "\n",
    ):
        super().__init__()
        self.prompt = prompt
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.yes_loc = self.tokenizer(yes_symbol, add_special_tokens=False)[
            "input_ids"
        ][0]
        self.sep = sep
        self.model.eval()
        self.max_length = max_length

    def get_inputs(self, pairs, tokenizer, prompt=None, max_length=1024):
        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        sep = "\n"
        prompt_inputs = tokenizer(
            prompt, return_tensors=None, add_special_tokens=False
        )["input_ids"]
        sep_inputs = tokenizer(sep, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        inputs = []
        for query, passage in pairs:
            query_inputs = tokenizer(
                f"A: {query}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length * 3 // 4,
                truncation=True,
            )
            passage_inputs = tokenizer(
                f"B: {passage}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
            item = tokenizer.prepare_for_model(
                [tokenizer.bos_token_id] + query_inputs["input_ids"],
                sep_inputs + passage_inputs["input_ids"],
                truncation="only_second",
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
            item["attention_mask"] = [1] * len(item["input_ids"])
            inputs.append(item)
        return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    def rerank(
        self, query: str, documents: list[str], batch_size: int = 16
    ) -> list[tuple[float, str]]:
        all_scores = []

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            pairs = list(zip([query] * len(batch_docs), batch_docs))

            inputs = self.get_inputs(pairs, self.tokenizer)

            with torch.no_grad():
                batch_scores = (
                    self.model(**inputs, return_dict=True)
                    .logits[:, -1, self.yes_loc]
                    .view(
                        -1,
                    )
                    .float()
                )
            all_scores.extend(batch_scores.tolist())

        return [(score, doc) for score, doc in zip(all_scores, documents)]


class QwenReranker(RerankerAdapter):
    def __init__(
        self,
        instruct: str = "Given a web search query, retrieve relevant passages that answer the query",
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        max_length: int = 8192,
        device: str = "cuda",
    ):
        # Qwen3-Reranker-0.6B is a causal language model, use AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.instruct = instruct
        self.device = device
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(
            self.prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )
        self.max_length = max_length
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

    def format_instruction(self, instruction: str, query: str, doc: str):
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = (
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
        )
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(
        self, query: str, documents: list[str], batch_size: int = 16
    ) -> list[tuple[float, str]]:
        all_scores = []

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            pairs = [
                self.format_instruction(self.instruct, query, doc) for doc in batch_docs
            ]

            inputs = self.process_inputs(pairs)

            with torch.no_grad():
                batch_scores = self.compute_logits(inputs)

            all_scores.extend(batch_scores)

        return [(score, doc) for score, doc in zip(all_scores, documents)]


class JinaReranker(RerankerAdapter):
    def __init__(self, model_name: str = "jinaai/jina-reranker-v3"):
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def rerank(
        self, query: str, documents: list[str], batch_size: int = 16
    ) -> list[tuple[float, str]]:
        all_scores = []
        all_docs = []
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            results = self.model.rerank(query, batch_docs)
            all_scores.extend(results["relevance_score"])
            all_docs.extend(results["document"])

        return [(score, doc) for score, doc in zip(all_scores, all_docs)]


class OnlineReranker(RerankerAdapter):
    """
    Reranker that uses the online pruning/reranking API service (online_serving.py).
    Extracts scores from the PruneResponse.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000",
        aggregate_method: str = "line",
        language: str = "python",
        timeout: float = 300.0,
    ):
        """
        Initialize the online reranker.

        Args:
            api_base: Base URL of the online serving API
            aggregate_method: Aggregation method ("line" or "statement")
            language: Programming language
            timeout: Request timeout in seconds
        """
        import requests
        import logging

        self.api_base = api_base.rstrip("/")
        self.aggregate_method = aggregate_method
        self.language = language
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.requests = requests

    def rerank(
        self, query: str, documents: list[str], batch_size: int = 16
    ) -> list[tuple[float, str]]:
        """
        Rerank documents using the online API.
        Returns scores from the PruneResponse.
        """
        all_scores = []
        all_docs = []
        prune_url = f"{self.api_base}/prune"

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]

            for doc in batch_docs:
                try:
                    payload = {
                        "query": query,
                        "code": doc,
                        "threshold": 0.0,  # We want scores, not pruning
                        "always_keep_first_frags": False,
                        "aggregate_method": self.aggregate_method,
                        "language": self.language,
                    }

                    response = self.requests.post(
                        prune_url, json=payload, timeout=self.timeout
                    )
                    response.raise_for_status()
                    result = response.json()

                    # Extract score from PruneResponse
                    score = result.get("score", 0.0)
                    all_scores.append(score)
                    all_docs.append(doc)
                except Exception as e:
                    self.logger.error(f"Error calling rerank API: {e}")
                    # Default to neutral score on error
                    all_scores.append(0.5)
                    all_docs.append(doc)

        return [(score, doc) for score, doc in zip(all_scores, all_docs)]


if __name__ == "__main__":
    # check correctness
    query = "What is the capital of France?"
    doc = ["Paris is the capital of France.", "Berlin is the capital of Germany."]

    rerankers = [
        QwenReranker(),
        BGEV2M3Reranker(model_name="BAAI/bge-reranker-v2-m3"),
        BertBasedReranker(model_name="bert-base-uncased"),
        JinaReranker(),
    ]

    for r in rerankers:
        d = r.rerank(query, doc, batch_size=2)
        print(d)
