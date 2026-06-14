from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    model_loaded: bool


class RecommendationResponse(BaseModel):
    client_id: str
    visit_id: str
    recommendations: list[str]
    route: str
