from pydantic import BaseModel


class SearchResult(BaseModel):
    item_id: int
    final_score: float
    cosine_sim: float
    color_sim: float
    aspect_sim: float
    image_path: str


class SearchResponse(BaseModel):
    query_filename: str
    n_candidates: int
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    n_indexed: int
    index_type: str
    device: str
