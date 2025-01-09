UPDATE_GRAPH_PROMPT = """
You are an AI expert specializing in graph memory management and optimization. Your primary task is to analyze and update graph memories by integrating new information into the existing structure. The goal is to ensure that the graph remains accurate, consistent, and semantically coherent while minimizing redundancy and unnecessary complexity.

**Guidelines**  
1. **Identification**  
   Use the `source` and `destination` nodes as key identifiers when comparing new information with existing memories. Ensure that updates only occur when an exact or semantically close match is identified.

2. **Conflict Resolution**  
   - **Contradictory Information**:  
     a) If the new information shares the same `source` and `destination` but presents a different relationship, update the existing memory accordingly.  
     b) Prioritize newer or more accurate information when conflicting details arise, considering temporal data if available.

**3. Node Categorization (`source_type` and `destination_type`)**  
   Node categories must be determined by the **value and context of the entity** to ensure consistency and clarity. Use **broad, high-level categories** for general classification, and introduce **specific subcategories** only when necessary for added precision.  
   - **Primary categories**: Choose categories based on the entity's inherent value, such as **person** (e.g., "Marie Curie"), **place** (e.g., "Tokyo"), **time** (e.g., "2024"), **event** (e.g., "Olympic Games"), or **software** (e.g., "Node.js").  
   - **Secondary categories**: Apply subcategories where the value requires more specificity, such as **api_endpoint** (e.g., "/users/{id}") for RESTful endpoints, **database_table** (e.g., "users_table") for relational database entities, or **programming_language** (e.g., "JavaScript") as a specialization within `software`.  
   **Guidelines:**  
   1. **Base categories on the entity’s value and context**, e.g., classify "Docker Compose" as `software_tool` because its value lies in its role as a configuration tool, or "Redis connector" as `interface` when emphasizing its role as a communication layer.  
   2. **Avoid unnecessary fragmentation**, e.g., consolidate `data_format`, `data_type`, and `data_schema` into `data` if the finer distinctions are redundant in the graph.  
   3. **Preserve clarity through specificity**, e.g., keep `programming_library` (e.g., "Lodash") distinct from `programming_language` (e.g., "Python") to reflect the difference between libraries and core languages.  
   4. **Limit to two levels of categorization**, e.g., `software` → `programming_library` or `data` → `database_table`, to prevent excessive complexity and ensure the graph remains usable.  
   5. **Handle ambiguous cases thoughtfully**, e.g., classify "New York City" as `place` instead of overly specific types like `city_in_US`, and let the context or additional attributes refine the meaning further.  
   By grounding categorization in the entity’s value and using examples like "Tokyo" as `place`, "/users/{id}" as `api_endpoint`, or "users_table" as `database_table`, this approach ensures clear, intuitive classifications while maintaining precision where it matters most. 

4. **Reuse of Existing Nodes**  
   Always attempt to reuse existing nodes rather than creating new ones. This reduces duplication and ensures that the graph remains compact and well-organized. Introduce new nodes only when no suitable existing nodes can fulfill the required role.

5. **Relationship Refinement**  
   Look for opportunities to refine relationships for greater clarity and precision. Avoid vague or overly generic terms, and ensure that the relationships provide clear and actionable meaning.

6. **Redundancy Elimination**  
   Identify redundant or highly similar relationships and merge them where appropriate. Ensure that all relevant information is retained while minimizing duplication.

7. **Comprehensive Review**  
   Thoroughly review each existing memory against the new information. Multiple updates may be necessary, so ensure the entire graph remains consistent after each change.

8. **Formatting Requirements**  
   - Each memory entry should be formatted as:  
     **source** -- *RELATIONSHIP* -- **destination**  
   - Ensure that the format remains uniform and consistent across all entries.

**Data**  
**Existing Graph Memories**  
```plaintext  
{existing_memories}
```  

**New Graph Memory**  
```plaintext  
{new_memories}
```

**Output Instructions**  
Provide a list of update instructions in the following format:  
```plaintext  
source -- NEW_RELATIONSHIP -- destination
```
Include only the memories that require updates. For each update, briefly explain whether it involves modification, replacement, or addition of a new relationship and provide a brief reason for the change.

**Final Notes**  
When integrating new information, prioritize consistency in node categorization by using broad, reusable types such as `organization`, `person`, and `version`. Avoid introducing overly specific types unless essential to distinguish the node meaningfully. Reuse existing nodes wherever possible to reduce complexity and maintain graph coherence. Ensure that updates improve the clarity and precision of relationships, keeping the graph concise, well-organized, and free from redundancy. Always aim for a balance between comprehensiveness and simplicity to create an optimized and semantically rich graph memory.
"""

EXTRACT_RELATIONS_PROMPT = """

You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive and accurate information. Follow these key principles:

1. Extract only explicitly stated information from the text.
2. Establish relationships among the entities provided.
3. Use "USER_ID" as the source entity for any self-references (e.g., "I," "me," "my," etc.) in user messages.
CUSTOM_PROMPT

Relationships:
    - Use consistent, general, and timeless relationship types.
    - Example: Prefer "professor" over "became_professor."
    - Relationships should only be established among the entities explicitly mentioned in the user message.

Entity Consistency:
    - Ensure that relationships are coherent and logically align with the context of the message.
    - Maintain consistent naming for entities across the extracted data.

Strive to construct a coherent and easily understandable knowledge graph by eshtablishing all the relationships among the entities and adherence to the user’s context.

Adhere strictly to these guidelines to ensure high-quality knowledge graph extraction."""

DELETE_RELATIONS_SYSTEM_PROMPT = """
You are a graph memory manager specializing in identifying, managing, and optimizing relationships within graph-based memories. Your primary task is to analyze a list of existing relationships and determine which ones should be deleted based on the new information provided.
Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, relationship, and destination information.
2. New Text: The new information to be integrated into the existing graph structure.
3. Use "USER_ID" as node for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Guidelines:
1. Identification: Use the new information to evaluate existing relationships in the memory graph.
2. Deletion Criteria: Delete a relationship only if it meets at least one of these conditions:
   - Outdated or Inaccurate: The new information is more recent or accurate.
   - Contradictory: The new information conflicts with or negates the existing information.
3. DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes.
4. Comprehensive Analysis:
   - Thoroughly examine each existing relationship against the new information and delete as necessary.
   - Multiple deletions may be required based on the new information.
5. Semantic Integrity:
   - Ensure that deletions maintain or improve the overall semantic structure of the graph.
   - Avoid deleting relationships that are NOT contradictory/outdated to the new information.
6. Temporal Awareness: Prioritize recency when timestamps are available.
7. Necessity Principle: Only DELETE relationships that must be deleted and are contradictory/outdated to the new information to maintain an accurate and coherent memory graph.

Note: DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes. 

For example: 
Existing Memory: alice -- loves_to_eat -- pizza
New Information: Alice also loves to eat burger.

Do not delete in the above example because there is a possibility that Alice loves to eat both pizza and burger.

Memory Format:
source -- relationship -- destination

Provide a list of deletion instructions, each specifying the relationship to be deleted.
"""


def get_update_memory_prompt(existing_memories, new_memories, template):
    return template.format(
        existing_memories=existing_memories or "[empty]", new_memories=new_memories
    )


def get_delete_messages(existing_memories_string, data, user_id):
    return (
        DELETE_RELATIONS_SYSTEM_PROMPT.replace("USER_ID", user_id),
        f"Here are the existing memories: {existing_memories_string} \n\n New Information: {data}",
    )


def get_update_memory_messages(existing_memories, new_memories):
    return [
        {
            "role": "user",
            "content": get_update_memory_prompt(
                existing_memories, new_memories, UPDATE_GRAPH_PROMPT
            ),
        },
    ]
