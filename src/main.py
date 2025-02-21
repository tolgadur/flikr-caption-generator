from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up Flickr Caption Generator backend...")
    print("CORS configured for allowed origins")
    print("API router initialized")
    yield
    # Shutdown
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(title="Flickr Caption Generator", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://49.13.206.165:3000",
        "http://49.13.206.165",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


def main():
    print("Starting uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
