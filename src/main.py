from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Flickr Caption Generator")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


def main():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
