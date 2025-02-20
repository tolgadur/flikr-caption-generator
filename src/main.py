from fastapi import FastAPI
from api import router
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Flickr Caption Generator")

# Include API router
app.include_router(router)


def main():

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
