from __future__ import annotations

from ast import List
from pydantic import BaseModel, Field
from typing import Any

import requests

from minisweagent.utils.log import logger


class PrunerConfig(BaseModel):
    url: str
    threshold: float
    timeout: float = 60.0
    retries: int = 1
    min_chars: int = 0
    headers: dict[str, str] = Field(default_factory=dict)
    chunk_overlap_tokens: int = 50

    class Config:
        extra = "allow"


class PrunerRequest(BaseModel):
    query: str
    code: str
    threshold: float
    always_keep_first_frags: bool = False
    chunk_overlap_tokens: int = 50


class PruneResponse(BaseModel):
    score: float
    pruned_code: str
    token_scores: list[list[str | float]]  # [[token_str, score], ...]
    kept_frags: list[int]
    origin_token_cnt: int
    left_token_cnt: int
    model_input_token_cnt: int
    error_msg: str | None = None


class PrunerClient:
    def __init__(self, config: PrunerConfig):
        self.config = config
        base_headers = {"Content-Type": "application/json"} | config.headers
        self.session = requests.Session()
        self.session.headers.update(base_headers)

    def prune(self, req: PrunerRequest) -> PruneResponse:
        if not req.query or len(req.code) <= self.config.min_chars:
            # HINT: not pruned, cnt not in statistics
            return PruneResponse(
                score=0.0,
                pruned_code=req.code,
                token_scores=[],
                kept_frags=[i + 1 for i in range(len(req.code.splitlines()))],
                origin_token_cnt=0,
                left_token_cnt=0,
                model_input_token_cnt=0,
            )
        payload = req.model_dump()
        last_error: Exception | None = None
        for _ in range(max(self.config.retries, 1)):
            try:
                response = self.session.post(self.config.url, json=payload, timeout=self.config.timeout)
                response.raise_for_status()
                return PruneResponse(**response.json())
            except Exception as exc:
                last_error = exc
        if last_error:
            logger.debug("Pruner request failed: %s", last_error)
        return PruneResponse(
            score=0.0,
            pruned_code=req.code,
            token_scores=[],
            kept_frags=[i for i in range(len(req.code.splitlines()))],
            origin_token_cnt=0,
            left_token_cnt=0,
            model_input_token_cnt=0,
            error_msg=str(last_error),
        )
