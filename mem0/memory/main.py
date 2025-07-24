import asyncio
import concurrent
import gc
import hashlib
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import pytz
from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.enums import MemoryType
from mem0.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_update_memory_messages,
)
from mem0.memory.base import MemoryBase
from mem0.memory.setup import mem0_dir, setup_config
from mem0.memory.storage import SQLDatabaseManager
from mem0.memory.telemetry import capture_event
from mem0.memory.utils import (
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from mem0.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory


def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- add all provided session ids ----------
    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be provided.")

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


setup_config()
logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = None):
        if config is None:
            config = MemoryConfig()
        super().__init__(config)

        self.vector_store = VectorStoreFactory.create(
            config.vector_store.provider, config.vector_store.config
        )
        self.llm = LlmFactory.create(config.llm.provider, config.llm.config)
        self.embedding_model = EmbedderFactory.create(
            config.embedder.provider, config.embedder.config
        )
        self.graph = None
        if config.graph_store and config.graph_store.provider:
            self.enable_graph = True
            try:
                from mem0.utils.factory import GraphStoreFactory

                self.graph = GraphStoreFactory.create(
                    config.graph_store.provider, config.graph_store.config
                )
            except ImportError as e:
                logger.error(f"Graph store import failed: {e}")
                self.enable_graph = False
        else:
            self.enable_graph = False

        # Use SQLDatabaseManager for multi-database support
        self.db = SQLDatabaseManager(
            db_type=config.history_db.type,
            db_url=config.history_db.url,
        )
        self.collection_name = config.collection_name
        self.api_version = config.version
        logger.info("Memory initialized successfully!")

    def add(
        self,
        messages,
        user_id=None,
        agent_id=None,
        run_id=None,
        memory_type=None,
        infer=True,
        metadata=None,
        filters=None,
        prompt=None,
    ):
        """
        Add a memory to the memory store.

        Args:
            messages (str or list): The message(s) to add to the memory store.
            user_id (str, optional): The user ID associated with the memory.
            agent_id (str, optional): The agent ID associated with the memory.
            run_id (str, optional): The run ID associated with the memory.
            memory_type (str, optional): Type of memory (e.g., 'procedural'). Defaults to None.
            infer (bool, optional): Whether to infer memory from the messages. Defaults to True.
            metadata (dict, optional): Additional metadata to store with the memory. Defaults to None.
            filters (dict, optional): Additional filters to apply when querying. Defaults to None.
            prompt (str, optional): Custom prompt for memory extraction. Defaults to None.

        Returns:
            dict: A dictionary containing the result of the memory addition operation.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`
        """

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Memory type '{memory_type}' is not supported. Only 'procedural' is supported."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise ValueError("messages must be str, dict, or list[dict]")

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt
            )
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages)
        else:
            messages = parse_vision_messages(messages)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(
                self._add_to_vector_store, messages, processed_metadata, effective_filters, infer
            )
            future2 = executor.submit(self._add_to_graph, messages, effective_filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        returned_memories = vector_store_result
        if graph_result and self.enable_graph:
            returned_memories.extend(graph_result)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        return returned_memories

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") == "system"
                ):
                    continue

                message_embeddings = self.embedding_model.embed(
                    message_dict["content"], memory_action="add"
                )
                memory_id = self._create_memory(
                    message_dict["content"],
                    {message_dict["content"]: message_embeddings},
                    metadata=metadata,
                )
                returned_memories.append(
                    {
                        "id": memory_id,
                        "memory": message_dict["content"],
                        "event": "ADD",
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        parsed_messages = parse_messages(messages)
        new_retrieved_facts = get_fact_retrieval_messages(
            messages=parsed_messages,
            mem0_llm=self.llm,
            custom_prompt=self.config.custom_fact_extraction_prompt,
        )

        if not new_retrieved_facts:
            return {"results": []}

        retrieved_old_memory = []
        new_message_embeddings = {}

        for new_mem in new_retrieved_facts:
            messages_embeddings = self.embedding_model.embed(
                new_mem, memory_action="add"
            )
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                filters=filters,
                limit=5,
            )[0]

            relevant_memories = [
                mem for mem in existing_memories if float(mem.score) >= 0.7
            ]

            if relevant_memories:
                retrieved_old_memory.extend(relevant_memories[:5])

        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )

            try:
                new_memories_with_actions = self.llm.generate_response(
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )

                try:
                    new_memories_with_actions = json.loads(
                        new_memories_with_actions.get("content", "{}")
                    )
                except json.JSONDecodeError:
                    new_memories_with_actions = {}

            except Exception as e:
                logger.error(f"Error processing memory: {e}")
                return {"results": []}

        returned_memories = []

        for resp in new_memories_with_actions.get("memory", []):
            logger.info(resp)
            try:
                action_text = resp.get("text")
                if not action_text:
                    logger.info("Skipping memory entry because of empty `text` field.")
                    continue

                event_type = resp.get("event")
                if event_type in ["ADD", "UPDATE"]:
                    embeddings = new_message_embeddings.get(action_text) or self.embedding_model.embed(
                        action_text, memory_action="update"
                    )

                    if event_type == "ADD":
                        memory_id = self._create_memory(
                            action_text, {action_text: embeddings}, metadata=metadata
                        )
                        returned_memories.append(
                            {"id": memory_id, "memory": action_text, "event": "ADD"}
                        )
                    elif event_type == "UPDATE":
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        if memory_id:
                            self._update_memory(
                                memory_id, action_text, {action_text: embeddings}, metadata=metadata
                            )
                            returned_memories.append(
                                {"id": memory_id, "memory": action_text, "event": "UPDATE"}
                            )

                elif event_type == "DELETE":
                    memory_id = temp_uuid_mapping.get(resp.get("id"))
                    if memory_id:
                        self._delete_memory(memory_id)
                        returned_memories.append(
                            {"id": memory_id, "event": "DELETE"}
                        )

            except Exception as e:
                logger.error(f"Error processing memory: {e}")
                continue

        return {"results": returned_memories}

    def _add_to_graph(self, messages, filters):
        if not self.enable_graph:
            return []

        try:
            return self.graph.add(
                [m for m in messages if m.get("role") != "system"], filters=filters
            )
        except Exception as e:
            logger.error(f"Error adding to graph: {e}")
            return []

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Get all memories matching the filters.

        Args:
            user_id (str, optional): The user ID to filter memories.
            agent_id (str, optional): The agent ID to filter memories.
            run_id (str, optional): The run ID to filter memories.
            limit (int, optional): The maximum number of memories to return. Defaults to 100.
            filters (dict, optional): Additional filters to apply when querying memories.
                `filters={"actor_id": "some_user"}`.
            limit (int, optional): The maximum number of memories to return. Defaults to 100.
        """
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, effective_filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_graph_entities, future_memories] if future_graph_entities else [future_memories]
            )

            all_memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}
        else:
            return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories = self.vector_store.list(filters=filters, limit=limit)
        actual_memories = memories[0] if memories else []

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash", ""),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ):
        """
        Searches for memories based on a query

        Args:
            query (str): The query to search for.
            user_id (str, optional): The user ID to filter memories.
            agent_id (str, optional): The agent ID to filter memories.
            run_id (str, optional): The run ID to filter memories.
            limit (int, optional): The maximum number of memories to return. Defaults to 100.
            filters (dict, optional): Additional filters to apply when querying memories.
                `filters={"actor_id": "some_user"}`.
            threshold (float, optional): The minimum similarity threshold for returned memories.

        Returns:
            dict: A dictionary containing the search results.
        """
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
            },
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._search_vector_store, query, effective_filters, limit, threshold)
            future_graph_entities = (
                executor.submit(self.graph.search, query, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_graph_entities, future_memories] if future_graph_entities else [future_memories]
            )

            memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": memories_result, "relations": graph_entities_result}
        else:
            return {"results": memories_result}

    def _search_vector_store(self, query, filters, limit, threshold=None):
        embeddings = self.embedding_model.embed(query, memory_action="search")
        memories = self.vector_store.search(
            query=query,
            vectors=embeddings,
            filters=filters,
            limit=limit,
        )[0]

        # Apply threshold filtering if specified
        if threshold is not None:
            memories = [mem for mem in memories if float(mem.score) >= threshold]

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash", ""),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,  # Include similarity score
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def delete(self, memory_id):
        """Delete a memory by ID."""
        try:
            self._delete_memory(memory_id)
            capture_event(
                "mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"}
            )
            return {"message": "Memory deleted"}
        except Exception as e:
            raise ValueError(f"Memory with ID {memory_id} not found: {e}")

    def delete_all(self, user_id=None, agent_id=None, run_id=None, filters=None):
        """Delete all memories matching the filters."""
        if filters is None:
            filters = {}

        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        return {"message": f"Deleted {len(memories)} memories"}

    def history(self, memory_id):
        """Get history of a memory by ID."""
        return self.db.get_history(memory_id)

    def _create_memory(self, data, embeddings, metadata=None):
        if metadata is None:
            metadata = {}

        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(pytz.utc).isoformat()
        if "updated_at" not in metadata:
            metadata["updated_at"] = metadata["created_at"]

        memory_id = str(uuid.uuid4())
        metadata["id"] = memory_id
        metadata["data"] = data
        metadata["hash"] = str(hashlib.md5(data.encode()).hexdigest())

        vectors, ids, payloads = zip(
            *[(embedding, memory_id, metadata) for content, embedding in embeddings.items()]
        )

        self.vector_store.insert(
            vectors=vectors,
            ids=ids,
            payloads=payloads,
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        capture_event("mem0._create_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        if metadata is None:
            metadata = {}

        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(pytz.utc).isoformat()
        if "updated_at" not in metadata:
            metadata["updated_at"] = metadata["created_at"]

        messages_str = "\n".join([msg.get("content", "") for msg in messages])

        if prompt:
            system_prompt = prompt
        else:
            system_prompt = PROCEDURAL_MEMORY_SYSTEM_PROMPT

        procedural_memory = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": messages_str},
            ]
        )

        procedural_memory = remove_code_blocks(procedural_memory.get("content", ""))

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, embeddings, metadata=None):
        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}
        existing_metadata = deepcopy(existing_memory.payload)
        existing_metadata.update(new_metadata)
        existing_metadata["data"] = data
        existing_metadata["hash"] = str(hashlib.md5(data.encode()).hexdigest())
        existing_metadata["updated_at"] = datetime.now(pytz.utc).isoformat()

        vectors, ids, payloads = zip(
            *[(embedding, memory_id, existing_metadata) for content, embedding in embeddings.items()]
        )

        self.vector_store.update(
            vector_id=memory_id,
            vector=vectors[0],
            payload=existing_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=existing_metadata["created_at"],
            updated_at=existing_metadata["updated_at"],
            actor_id=existing_metadata.get("actor_id"),
            role=existing_metadata.get("role"),
        )
        capture_event("mem0._update_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

    def _delete_memory(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload["data"]
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        capture_event("mem0._delete_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id

    def reset(self):
        """
        Reset the memory store by clearing all memories and history.
        Uses the multi-database manager's reset functionality with enhanced cleanup.
        """
        logger.warning("Resetting all memories")

        gc.collect()

        # Close the vector store client if it has a close method
        if hasattr(self.vector_store, "client") and hasattr(
            self.vector_store.client, "close"
        ):
            self.vector_store.client.close()

        # Use the multi-database manager's reset method
        if hasattr(self.db, "reset"):
            self.db.reset()

        # Close the database connection
        if hasattr(self.db, "close"):
            self.db.close()

        # Reinitialize the database manager
        self.db = SQLDatabaseManager(
            type=self.config.history_db.type, url=self.config.history_db.url
        )

        # Reset vector store
        if hasattr(self.vector_store, "reset"):
            self.vector_store.reset()

        # Reset graph store if enabled
        if self.enable_graph and hasattr(self.graph, "reset"):
            self.graph.reset()

        capture_event("mem0.reset", self, {"sync_type": "sync"})

    # Async methods follow the same pattern with proper integration

    async def aadd(
        self,
        messages,
        user_id=None,
        agent_id=None,
        run_id=None,
        memory_type=None,
        infer=True,
        metadata=None,
        filters=None,
        prompt=None,
        llm=None,
    ):
        """
        Asynchronously add a memory to the memory store.

        Args:
            messages (str or list): The message(s) to add to the memory store.
            user_id (str, optional): The user ID associated with the memory.
            agent_id (str, optional): The agent ID associated with the memory.
            run_id (str, optional): The run ID associated with the memory.
            memory_type (str, optional): Type of memory (e.g., 'procedural'). Defaults to None.
            infer (bool, optional): Whether to infer memory from the messages. Defaults to True.
            metadata (dict, optional): Additional metadata to store with the memory. Defaults to None.
            filters (dict, optional): Additional filters to apply when querying. Defaults to None.
            prompt (str, optional): Custom prompt for memory extraction. Defaults to None.
            llm: Language model instance for processing. Defaults to None.

        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Memory type '{memory_type}' is not supported. Only 'procedural' is supported."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise ValueError("messages must be str, dict, or list[dict]")

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages)
        else:
            messages = parse_vision_messages(messages)

        vector_store_task = asyncio.create_task(
            self._add_to_vector_store(messages, processed_metadata, effective_filters, infer)
        )
        graph_task = asyncio.create_task(self._add_to_graph(messages, effective_filters))

        vector_store_result, graph_result = await asyncio.gather(
            vector_store_task, graph_task
        )

        returned_memories = vector_store_result
        if graph_result and self.enable_graph:
            returned_memories.extend(graph_result)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        return returned_memories

    async def _add_to_vector_store_async(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") == "system"
                ):
                    continue

                message_embeddings = await asyncio.to_thread(
                    self.embedding_model.embed, message_dict["content"], "add"
                )
                memory_id = await self._create_memory_async(
                    message_dict["content"], {message_dict["content"]: message_embeddings}, metadata=metadata
                )
                returned_memories.append(
                    {
                        "id": memory_id,
                        "memory": message_dict["content"],
                        "event": "ADD",
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        parsed_messages = parse_messages(messages)
        new_retrieved_facts = await asyncio.to_thread(
            get_fact_retrieval_messages,
            messages=parsed_messages,
            mem0_llm=self.llm,
            custom_prompt=self.config.custom_fact_extraction_prompt,
        )

        if not new_retrieved_facts:
            return {"results": []}

        retrieved_old_memory = []
        new_message_embeddings = {}

        async def process_fact_for_search(new_mem_content):
            embeddings = await asyncio.to_thread(self.embedding_model.embed, new_mem_content, "add")
            new_message_embeddings[new_mem_content] = embeddings
            existing_mems = await asyncio.to_thread(
                self.vector_store.search,
                query=new_mem_content,
                vectors=embeddings,
                filters=filters,
                limit=5,
            )
            return existing_mems[0]

        tasks = [process_fact_for_search(new_mem) for new_mem in new_retrieved_facts]
        search_results = await asyncio.gather(*tasks)

        for existing_memories in search_results:
            relevant_memories = [mem for mem in existing_memories if float(mem.score) >= 0.7]
            if relevant_memories:
                retrieved_old_memory.extend(relevant_memories[:5])

        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )

            try:
                new_memories_with_actions = await asyncio.to_thread(
                    self.llm.generate_response,
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )

                try:
                    new_memories_with_actions = json.loads(
                        new_memories_with_actions.get("content", "{}")
                    )
                except json.JSONDecodeError:
                    new_memories_with_actions = {}

            except Exception as e:
                logger.error(f"Error processing memory: {e}")
                return {"results": []}

        returned_memories = []

        for resp in new_memories_with_actions.get("memory", []):
            try:
                action_text = resp.get("text")
                if not action_text:
                    logger.info("Skipping memory entry because of empty `text` field.")
                    continue

                event_type = resp.get("event")
                if event_type in ["ADD", "UPDATE"]:
                    embeddings = new_message_embeddings.get(action_text) or await asyncio.to_thread(
                        self.embedding_model.embed, action_text, "update"
                    )

                    if event_type == "ADD":
                        memory_id = await self._create_memory_async(
                            action_text, {action_text: embeddings}, metadata=metadata
                        )
                        returned_memories.append(
                            {"id": memory_id, "memory": action_text, "event": "ADD"}
                        )
                    elif event_type == "UPDATE":
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        if memory_id:
                            await self._update_memory_async(
                                memory_id, action_text, {action_text: embeddings}, metadata=metadata
                            )
                            returned_memories.append(
                                {"id": memory_id, "memory": action_text, "event": "UPDATE"}
                            )

                elif event_type == "DELETE":
                    memory_id = temp_uuid_mapping.get(resp.get("id"))
                    if memory_id:
                        await self._delete_memory_async(memory_id)
                        returned_memories.append({"id": memory_id, "event": "DELETE"})

            except Exception as e:
                logger.error(f"Error processing memory: {e}")
                continue

        return {"results": returned_memories}

    async def _create_memory_async(self, data, embeddings, metadata=None):
        if metadata is None:
            metadata = {}

        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(pytz.utc).isoformat()
        if "updated_at" not in metadata:
            metadata["updated_at"] = metadata["created_at"]

        memory_id = str(uuid.uuid4())
        metadata["id"] = memory_id
        metadata["data"] = data
        metadata["hash"] = str(hashlib.md5(data.encode()).hexdigest())

        vectors, ids, payloads = zip(
            *[(embedding, memory_id, metadata) for content, embedding in embeddings.items()]
        )

        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=vectors,
            ids=ids,
            payloads=payloads,
        )

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )

        capture_event("mem0._create_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def _update_memory_async(self, memory_id, data, embeddings, metadata=None):
        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}
        existing_metadata = deepcopy(existing_memory.payload)
        existing_metadata.update(new_metadata)
        existing_metadata["data"] = data
        existing_metadata["hash"] = str(hashlib.md5(data.encode()).hexdigest())
        existing_metadata["updated_at"] = datetime.now(pytz.utc).isoformat()

        vectors, ids, payloads = zip(
            *[(embedding, memory_id, existing_metadata) for content, embedding in embeddings.items()]
        )

        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=vectors[0],
            payload=existing_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=existing_metadata["created_at"],
            updated_at=existing_metadata["updated_at"],
            actor_id=existing_metadata.get("actor_id"),
            role=existing_metadata.get("role"),
        )
        capture_event("mem0._update_memory", self, {"memory_id": memory_id, "sync_type": "async"})

    async def _delete_memory_async(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        prev_value = existing_memory.payload["data"]

        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        capture_event("mem0._delete_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = None):
        if config is None:
            config = MemoryConfig()
        super().__init__(config)

        self.vector_store = VectorStoreFactory.create(
            config.vector_store.provider, config.vector_store.config
        )
        self.llm = LlmFactory.create(config.llm.provider, config.llm.config)
        self.embedding_model = EmbedderFactory.create(
            config.embedder.provider, config.embedder.config
        )
        self.graph = None
        if config.graph_store and config.graph_store.provider:
            self.enable_graph = True
            try:
                from mem0.utils.factory import GraphStoreFactory

                self.graph = GraphStoreFactory.create(
                    config.graph_store.provider, config.graph_store.config
                )
            except ImportError as e:
                logger.error(f"Graph store import failed: {e}")
                self.enable_graph = False
        else:
            self.enable_graph = False

        # Use SQLDatabaseManager for multi-database support
        self.db = SQLDatabaseManager(
            db_type=config.history_db.type,
            db_url=config.history_db.url,
        )
        self.collection_name = config.collection_name
        self.api_version = config.version
        logger.info("AsyncMemory initialized successfully!")

    @classmethod
    async def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None.
                                         Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise ValueError("messages must be str, dict, or list[dict]")

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages)
        else:
            messages = parse_vision_messages(messages)

        vector_store_task = asyncio.create_task(
            self._add_to_vector_store_async(messages, processed_metadata, effective_filters, infer)
        )
        graph_task = asyncio.create_task(self._add_to_graph_async(messages, effective_filters))

        vector_store_result, graph_result = await asyncio.gather(
            vector_store_task, graph_task
        )

        returned_memories = vector_store_result.get("results", [])
        if graph_result and self.enable_graph:
            returned_memories.extend(graph_result)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        return {"results": returned_memories}

    async def _add_to_vector_store_async(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") == "system"
                ):
                    continue

                message_embeddings = await asyncio.to_thread(
                    self.embedding_model.embed, message_dict["content"], "add"
                )
                memory_id = await self._create_memory_async(
                    message_dict["content"], {message_dict["content"]: message_embeddings}, metadata=metadata
                )
                returned_memories.append(
                    {
                        "id": memory_id,
                        "memory": message_dict["content"],
                        "event": "ADD",
                        "role": message_dict["role"],
                    }
                )
            return {"results": returned_memories}

        parsed_messages = parse_messages(messages)
        new_retrieved_facts = await asyncio.to_thread(
            get_fact_retrieval_messages,
            messages=parsed_messages,
            mem0_llm=self.llm,
            custom_prompt=self.config.custom_fact_extraction_prompt,
        )

        if not new_retrieved_facts:
            return {"results": []}

        retrieved_old_memory = []
        new_message_embeddings = {}

        async def process_fact_for_search(new_mem_content):
            embeddings = await asyncio.to_thread(self.embedding_model.embed, new_mem_content, "add")
            new_message_embeddings[new_mem_content] = embeddings
            existing_mems = await asyncio.to_thread(
                self.vector_store.search,
                query=new_mem_content,
                vectors=embeddings,
                filters=filters,
                limit=5,
            )
            return existing_mems[0]

        tasks = [process_fact_for_search(new_mem) for new_mem in new_retrieved_facts]
        search_results = await asyncio.gather(*tasks)

        for existing_memories in search_results:
            relevant_memories = [mem for mem in existing_memories if float(mem.score) >= 0.7]
            if relevant_memories:
                retrieved_old_memory.extend(relevant_memories[:5])

        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )

            try:
                new_memories_with_actions = await asyncio.to_thread(
                    self.llm.generate_response,
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )

                try:
                    new_memories_with_actions = json.loads(
                        new_memories_with_actions.get("content", "{}")
                    )
                except json.JSONDecodeError:
                    new_memories_with_actions = {}

            except Exception as e:
                logger.error(f"Error processing memory: {e}")
                return {"results": []}

        returned_memories = []

        for resp in new_memories_with_actions.get("memory", []):
            try:
                action_text = resp.get("text")
                if not action_text:
                    logger.info("Skipping memory entry because of empty `text` field.")
                    continue

                event_type = resp.get("event")
                if event_type in ["ADD", "UPDATE"]:
                    embeddings = new_message_embeddings.get(action_text) or await asyncio.to_thread(
                        self.embedding_model.embed, action_text, "update"
                    )

                    if event_type == "ADD":
                        memory_id = await self._create_memory_async(
                            action_text, {action_text: embeddings}, metadata=metadata
                        )
                        returned_memories.append(
                            {"id": memory_id, "memory": action_text, "event": "ADD"}
                        )
                    elif event_type == "UPDATE":
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        if memory_id:
                            await self._update_memory_async(
                                memory_id, action_text, {action_text: embeddings}, metadata=metadata
                            )
                            returned_memories.append(
                                {"id": memory_id, "memory": action_text, "event": "UPDATE"}
                            )

                elif event_type == "DELETE":
                    memory_id = temp_uuid_mapping.get(resp.get("id"))
                    if memory_id:
                        await self._delete_memory_async(memory_id)
                        returned_memories.append({"id": memory_id, "event": "DELETE"})

            except Exception as e:
                logger.error(f"Error processing memory: {e}")
                continue

        return {"results": returned_memories}

    async def _add_to_graph_async(self, messages, filters):
        if not self.enable_graph:
            return []

        try:
            return await asyncio.to_thread(
                self.graph.add, [m for m in messages if m.get("role") != "system"], filters=filters
            )
        except Exception as e:
            logger.error(f"Error adding to graph: {e}")
            return []

    async def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Get all memories matching the filters asynchronously.

        Args:
            user_id (str, optional): The user ID to filter memories.
            agent_id (str, optional): The agent ID to filter memories.
            run_id (str, optional): The run ID to filter memories.
            limit (int, optional): The maximum number of memories to return. Defaults to 100.
            filters (dict, optional): Additional filters to apply when querying memories.
        """
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )

        all_memories_task = asyncio.create_task(
            self._get_all_from_vector_store_async(effective_filters, limit)
        )
        graph_entities_task = asyncio.create_task(
            self._get_all_from_graph_async(effective_filters, limit)
        ) if self.enable_graph else None

        if graph_entities_task:
            all_memories_result, graph_entities_result = await asyncio.gather(
                all_memories_task, graph_entities_task
            )
        else:
            all_memories_result = await all_memories_task
            graph_entities_result = None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}
        else:
            return {"results": all_memories_result}

    async def _get_all_from_vector_store_async(self, filters, limit):
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters, limit=limit)
        actual_memories = memories[0] if memories else []

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash", ""),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    async def _get_all_from_graph_async(self, filters, limit):
        if not self.enable_graph:
            return None
        return await asyncio.to_thread(self.graph.get_all, filters, limit)

    async def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ):
        """
        Searches for memories based on a query asynchronously

        Args:
            query (str): The query to search for.
            user_id (str, optional): The user ID to filter memories.
            agent_id (str, optional): The agent ID to filter memories.
            run_id (str, optional): The run ID to filter memories.
            limit (int, optional): The maximum number of memories to return. Defaults to 100.
            filters (dict, optional): Additional filters to apply when querying memories.
            threshold (float, optional): The minimum similarity threshold for returned memories.

        Returns:
            dict: A dictionary containing the search results.
        """
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "async",
                "threshold": threshold,
            },
        )

        memories_task = asyncio.create_task(
            self._search_vector_store_async(query, effective_filters, limit, threshold)
        )
        graph_entities_task = asyncio.create_task(
            self._search_graph_async(query, effective_filters, limit)
        ) if self.enable_graph else None

        if graph_entities_task:
            memories_result, graph_entities_result = await asyncio.gather(
                memories_task, graph_entities_task
            )
        else:
            memories_result = await memories_task
            graph_entities_result = None

        if self.enable_graph:
            return {"results": memories_result, "relations": graph_entities_result}
        else:
            return {"results": memories_result}

    async def _search_vector_store_async(self, query, filters, limit, threshold=None):
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")
        memories = await asyncio.to_thread(
            self.vector_store.search,
            query=query,
            vectors=embeddings,
            filters=filters,
            limit=limit,
        )

        # Apply threshold filtering if specified
        actual_memories = memories[0] if memories else []
        if threshold is not None:
            actual_memories = [mem for mem in actual_memories if float(mem.score) >= threshold]

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash", ""),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,  # Include similarity score
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    async def _search_graph_async(self, query, filters, limit):
        if not self.enable_graph:
            return None
        return await asyncio.to_thread(self.graph.search, query, filters, limit)

    async def delete(self, memory_id):
        """Delete a memory by ID asynchronously."""
        try:
            await self._delete_memory_async(memory_id)
            capture_event(
                "mem0.delete", self, {"memory_id": memory_id, "sync_type": "async"}
            )
            return {"message": "Memory deleted"}
        except Exception as e:
            raise ValueError(f"Memory with ID {memory_id} not found: {e}")

    async def delete_all(self, user_id=None, agent_id=None, run_id=None, filters=None):
        """Delete all memories matching the filters asynchronously."""
        if filters is None:
            filters = {}

        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"})
        
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters)
        actual_memories = memories[0] if memories else []
        
        for memory in actual_memories:
            await self._delete_memory_async(memory.id)

        return {"message": f"Deleted {len(actual_memories)} memories"}

    async def history(self, memory_id):
        """Get history of a memory by ID asynchronously."""
        return await asyncio.to_thread(self.db.get_history, memory_id)

    async def _create_memory_async(self, data, embeddings, metadata=None):
        if metadata is None:
            metadata = {}

        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(pytz.utc).isoformat()
        if "updated_at" not in metadata:
            metadata["updated_at"] = metadata["created_at"]

        memory_id = str(uuid.uuid4())
        metadata["id"] = memory_id
        metadata["data"] = data
        metadata["hash"] = str(hashlib.md5(data.encode()).hexdigest())

        vectors, ids, payloads = zip(
            *[(embedding, memory_id, metadata) for content, embedding in embeddings.items()]
        )

        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=vectors,
            ids=ids,
            payloads=payloads,
        )

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )

        capture_event("mem0._create_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def _create_procedural_memory(self, messages, metadata=None, prompt=None, llm=None):
        if metadata is None:
            metadata = {}

        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(pytz.utc).isoformat()
        if "updated_at" not in metadata:
            metadata["updated_at"] = metadata["created_at"]

        messages_str = "\n".join([msg.get("content", "") for msg in messages])

        if prompt:
            system_prompt = prompt
        else:
            system_prompt = PROCEDURAL_MEMORY_SYSTEM_PROMPT

        if llm:
            procedural_memory = await asyncio.to_thread(
                llm.generate_response,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": messages_str},
                ]
            )
        else:
            procedural_memory = await asyncio.to_thread(
                self.llm.generate_response,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": messages_str},
                ]
            )

        procedural_memory = remove_code_blocks(procedural_memory.get("content", ""))

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, "add")
        memory_id = await self._create_memory_async(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    async def _update_memory_async(self, memory_id, data, embeddings, metadata=None):
        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}
        existing_metadata = deepcopy(existing_memory.payload)
        existing_metadata.update(new_metadata)
        existing_metadata["data"] = data
        existing_metadata["hash"] = str(hashlib.md5(data.encode()).hexdigest())
        existing_metadata["updated_at"] = datetime.now(pytz.utc).isoformat()

        vectors, ids, payloads = zip(
            *[(embedding, memory_id, existing_metadata) for content, embedding in embeddings.items()]
        )

        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=vectors[0],
            payload=existing_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=existing_metadata["created_at"],
            updated_at=existing_metadata["updated_at"],
            actor_id=existing_metadata.get("actor_id"),
            role=existing_metadata.get("role"),
        )
        capture_event("mem0._update_memory", self, {"memory_id": memory_id, "sync_type": "async"})

    async def _delete_memory_async(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        prev_value = existing_memory.payload["data"]

        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        capture_event("mem0._delete_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def reset(self):
        """
        Reset the memory store by clearing all memories and history asynchronously.
        Uses the multi-database manager's reset functionality with enhanced cleanup.
        """
        logger.warning("Resetting all memories")

        gc.collect()

        # Close the vector store client if it has a close method
        if hasattr(self.vector_store, "client") and hasattr(
            self.vector_store.client, "close"
        ):
            self.vector_store.client.close()

        # Use the multi-database manager's reset method
        if hasattr(self.db, "reset"):
            await asyncio.to_thread(self.db.reset)

        # Close the database connection
        if hasattr(self.db, "close"):
            await asyncio.to_thread(self.db.close)

        # Reinitialize the database manager
        self.db = SQLDatabaseManager(
            db_type=self.config.history_db.type, db_url=self.config.history_db.url
        )

        # Reset vector store
        if hasattr(self.vector_store, "reset"):
            await asyncio.to_thread(self.vector_store.reset)

        # Reset graph store if enabled
        if self.enable_graph and hasattr(self.graph, "reset"):
            await asyncio.to_thread(self.graph.reset)

        capture_event("mem0.reset", self, {"sync_type": "async"})