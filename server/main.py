import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from mem0 import Memory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
HISTORY_DB_TYPE = os.getenv("HISTORY_DB_TYPE", "sqlite")
HISTORY_DB_URL = os.getenv("HISTORY_DB_URL", "sqlite:///:memory:")

# Global memory instance
MEMORY_INSTANCE = None

MEM0_CONFIG = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URI,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD,
        },
    },
    "llm": {
        "provider": "openai", 
        "config": {"api_key": OPENAI_API_KEY, "temperature": 0.2, "model": "gpt-4o"}
    },
    "embedder": {
        "provider": "openai",
        "config": {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"},
    },
    "history_db": {"type": HISTORY_DB_TYPE, "url": HISTORY_DB_URL},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MEMORY_INSTANCE
    try:
        MEMORY_INSTANCE = Memory.from_config(MEM0_CONFIG)
        logging.info("Memory instance initialized")
        yield
    except Exception as e:
        logging.error(f"Failed to initialize memory: {e}")
        raise
    finally:
        if MEMORY_INSTANCE:
            try:
                MEMORY_INSTANCE.db.close()
                logging.info("Memory instance cleanup completed")
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")


app = FastAPI(
    title="Mem0 Server",
    description="A REST API server for Mem0, a memory layer for AI applications",
    version="1.0.0",
    lifespan=lifespan,
)


def get_memory_instance():
    """Dependency to get the memory instance"""
    if MEMORY_INSTANCE is None:
        raise HTTPException(status_code=500, detail="Memory instance not initialized")
    return MEMORY_INSTANCE


class Message(BaseModel):
    content: str
    role: str


class MemoryCreate(BaseModel):
    messages: List[Message]
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None


class MemorySearch(BaseModel):
    query: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    limit: int = Field(default=100, le=1000)
    filters: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = None


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception:")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


@app.post("/v1/memories/", tags=["Memories"])
def create_memory(
    memory_create: MemoryCreate, memory_instance: Memory = Depends(get_memory_instance)
):
    """
    Create a new memory.

    This endpoint allows you to add new memories to the memory store.
    At least one identifier (user_id, agent_id, or run_id) is required.
    """
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(
            status_code=400,
            detail="At least one identifier (user_id, agent_id, run_id) is required.",
        )

    params = {
        k: v for k, v in memory_create.model_dump().items()
        if v is not None and k != "messages"
    }
    try:
        response = memory_instance.add(
            messages=[m.model_dump() for m in memory_create.messages], **params
        )
        return response
    except Exception as e:
        logging.exception("Error in create_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/memories/", tags=["Memories"])
def get_all_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    memory_instance: Memory = Depends(get_memory_instance),
):
    """
    Get all memories for a given user, agent, or run.

    This endpoint retrieves all memories associated with the provided identifiers.
    At least one identifier (user_id, agent_id, or run_id) is required.
    """
    if not any([user_id, agent_id, run_id]):
        raise HTTPException(
            status_code=400,
            detail="At least one identifier (user_id, agent_id, run_id) is required.",
        )
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        return memory_instance.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories/search/", tags=["Memories"])
def search_memories(
    memory_search: MemorySearch, memory_instance: Memory = Depends(get_memory_instance)
):
    """
    Search memories based on a query.

    This endpoint allows you to search for memories using a query string.
    At least one identifier (user_id, agent_id, or run_id) is required.
    """
    if not any([memory_search.user_id, memory_search.agent_id, memory_search.run_id]):
        raise HTTPException(
            status_code=400,
            detail="At least one identifier (user_id, agent_id, run_id) is required.",
        )
    try:
        params = {
            k: v
            for k, v in memory_search.model_dump().items()
            if v is not None and k != "query"
        }
        return memory_instance.search(query=memory_search.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/memories/{memory_id}/", tags=["Memories"])
def get_memory(memory_id: str, memory_instance: Memory = Depends(get_memory_instance)):
    """
    Get a specific memory by ID.

    This endpoint retrieves a single memory using its unique identifier.
    """
    try:
        # For now, we'll use the search functionality to find the memory
        # In a production system, you might want a more direct get method
        raise HTTPException(
            status_code=501, detail="Get single memory endpoint not yet implemented"
        )
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/v1/memories/{memory_id}/", tags=["Memories"])
def update_memory(
    memory_id: str, memory_instance: Memory = Depends(get_memory_instance)
):
    """
    Update a specific memory by ID.

    This endpoint allows you to update an existing memory.
    """
    try:
        raise HTTPException(
            status_code=501, detail="Update memory endpoint not yet implemented"
        )
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/memories/{memory_id}/", tags=["Memories"])
def delete_memory(
    memory_id: str, memory_instance: Memory = Depends(get_memory_instance)
):
    """
    Delete a specific memory by ID.

    This endpoint allows you to delete a single memory using its unique identifier.
    """
    try:
        response = memory_instance.delete(memory_id)
        return response
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/memories/", tags=["Memories"])
def delete_all_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    memory_instance: Memory = Depends(get_memory_instance),
):
    """
    Delete all memories for a given user, agent, or run.

    This endpoint deletes all memories associated with the provided identifiers.
    At least one identifier (user_id, agent_id, or run_id) is required.
    """
    if not any([user_id, agent_id, run_id]):
        raise HTTPException(
            status_code=400,
            detail="At least one identifier (user_id, agent_id, run_id) is required.",
        )
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        memory_instance.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/memories/{memory_id}/history/", tags=["Memories"])
def get_memory_history(
    memory_id: str, memory_instance: Memory = Depends(get_memory_instance)
):
    """
    Get the history of changes for a specific memory.

    This endpoint retrieves the change history for a memory, showing how it has been modified over time.
    """
    try:
        history = memory_instance.history(memory_id)
        return {"history": history}
    except Exception as e:
        logging.exception("Error in get_memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")