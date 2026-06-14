import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from server.database import Database
from server.predictor import Predictor
from server.schemas import HealthResponse, RecommendationResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n=== STARTING RECOMMENDATION SERVICE ===")

    database = Database()
    await database.connect()

    predictor = await asyncio.to_thread(Predictor)
    app.state.database = database
    app.state.predictor = predictor

    print("Recommendation service ready.\n")
    yield

    print("\nStopping recommendation service...")
    await database.disconnect()


app = FastAPI(
    title="Rental Product Recommender",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        database_connected=request.app.state.database is not None,
        model_loaded=request.app.state.predictor is not None,
    )


@app.get(
    "/recommend/{client_id}",
    response_model=RecommendationResponse,
)
async def recommend(
    client_id: str,
    request: Request,
) -> RecommendationResponse:
    print(f"Recommendation request: client_id={client_id}")

    context = await request.app.state.database.fetch_user_context(client_id)
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found.",
        )

    prediction = await asyncio.to_thread(
        request.app.state.predictor.predict,
        context,
    )

    print(
        f"Recommendation complete: client_id={client_id} "
        f"route={prediction.route}"
    )
    return RecommendationResponse(
        client_id=client_id,
        visit_id=prediction.visit_id,
        recommendations=prediction.product_ids,
        route=prediction.route,
    )
