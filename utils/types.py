from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Passage:
    id: str
    text: str
    score: float


@dataclass
class TreeHopResult:
    passages: List[Passage]
    query_embedding: Optional[object] = None
    confidence: Optional[float] = None