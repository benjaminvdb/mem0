import re

UPDATE_GRAPH_PROMPT = """
You are an AI expert specializing in graph memory management and optimization. Your task is to analyze existing graph memories alongside new information, and update the relationships in the memory list to ensure the most accurate, current, and coherent representation of knowledge.

Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, target, and relationship information.
2. New Graph Memory: Fresh information to be integrated into the existing graph structure.

Guidelines:
1. Identification: Use the source and target as primary identifiers when matching existing memories with new information.
2. Conflict Resolution:
   - If new information contradicts an existing memory:
     a) For matching source and target but differing content, update the relationship of the existing memory.
     b) If the new memory provides more recent or accurate information, update the existing memory accordingly.
3. Comprehensive Review: Thoroughly examine each existing graph memory against the new information, updating relationships as necessary. Multiple updates may be required.
4. Consistency: Maintain a uniform and clear style across all memories. Each entry should be concise yet comprehensive.
5. Semantic Coherence: Ensure that updates maintain or improve the overall semantic structure of the graph.
6. Temporal Awareness: If timestamps are available, consider the recency of information when making updates.
7. Relationship Refinement: Look for opportunities to refine relationship descriptions for greater precision or clarity.
8. Redundancy Elimination: Identify and merge any redundant or highly similar relationships that may result from the update.

Task Details:
- Existing Graph Memories:
{existing_memories}

- New Graph Memory: {memory}

Output:
Provide a list of update instructions, each specifying the source, target, and the new relationship to be set. Only include memories that require updates.
"""

EXTRACT_ENTITIES_PROMPT = """
You are an advanced algorithm designed to extract structured information from text to construct efficient and accurate knowledge graphs. Your primary goal is to **capture as much useful, explicitly stated information as possible**, while avoiding unnecessary or irrelevant details. Focus on clarity, precision, and practicality.

**Core Principles:**
1. **Extract Explicit Information**: Include only information directly stated in the text. Avoid assumptions or inferred data.
2. **Optimize Nodes and Relationships**: Identify key **entities** (nodes), assign appropriate **tags**, and define clear **relationships** between them using underscore-separated names.
3. **Prioritize Reuse**: Reuse existing nodes and relationships whenever possible to prevent duplication and redundancy.
4. **Focus on Key Targets**: Select main subjects and objects in the text as primary nodes and targets in relationships.

**Rules for Nodes:**
- **Names and Types**: Use precise, widely recognized identifiers in English when possible. Retain native-language names for unique entities (e.g., "Phở").
- **Tagging Conventions**: Assign concise, lowercase, underscore-separated tags to nodes (e.g., `person`, `place`, `concept`, `event`).
- **Consistency**: Use the most complete and descriptive form of an entity throughout the graph.

**Rules for Relationships:**
- **Clarity and Simplicity**: Use general, timeless, and descriptive relationship names (e.g., `also_known_as`, `likes`, `originates_from`).
- **Consistent Naming**: Relationships must be lowercase, concise, and use underscores (e.g., `acquainted_with`, `is_part_of`).
- **Avoid Overcomplication**: Keep relationships straightforward and directly related to the text.

**Examples:**

**Example 1**
**Input**:
"B-2 Spirit also known as Stealth Bomber."
**Extracted Relationships**:
```json
[
  {
    "source": "B-2 Spirit",
    "relationship": "also_known_as",
    "target": "Stealth Bomber"
  }
]
```
**Node Tags**:
- "B-2 Spirit" (tag: `fighter_aircraft`)
- "Stealth Bomber" (tag: `strategic_bomber`)

**Example 2**
**Input**:
"Tôi thích ăn Phở vì nó đặc trưng hương vị của đồ ăn Việt Nam."
**Extracted Relationships**:

```json
[
  {
    "source": "USER_ID",
    "relationship": "likes",
    "target": "Phở"
  },
  {
    "source": "Phở",
    "relationship": "originates_from",
    "target": "Vietnam"
  }
]
```

**Key Practices:**
- **Maximize Useful Content**: Extract all significant information that contributes to understanding the text's meaning.
- **Avoid Overuse and Off-topic Details**: Do not include irrelevant or speculative data.
- **Simplify for Reuse**: Design nodes and relationships for seamless integration with existing data.
- **Clarity and Coherence**: Ensure the knowledge graph is easy to understand and logically connected.
- **Consistency in Naming**: Use standard, widely accepted terms to enhance reusability.

Adhere strictly to these guidelines to construct knowledge graphs that are **efficient, accurate, and practically useful**, capturing the richness of the input text without unnecessary complexity.
"""


def get_update_memory_prompt(existing_memories, memory, template):
    return template.format(existing_memories=existing_memories, memory=memory)


def get_update_memory_messages(existing_memories, memory):
    return [
        {
            "role": "user",
            "content": get_update_memory_prompt(existing_memories, memory, UPDATE_GRAPH_PROMPT),
        },
    ]

def sanitize_graph_item(item):
    def transform(value):
        # return value.lower().replace(" ", "_") if value else value
        return convert_unicode_escapes(value)

    keys = ["source", "source_type", "relationship", "destination", "destination_type"]
    return {key: transform(item[key]) for key in keys if key in item}


def convert_unicode_escapes(content):
    # Pattern to match Unicode escape sequences like "\u00XX"
    pattern = r"\\u[0-9a-fA-F]{4}"

    # Function to replace each match with its corresponding UTF-8 character
    def replace_match(match):
        unicode_escape = match.group(0)
        # Convert the Unicode escape to a character
        return unicode_escape.encode().decode("unicode_escape")

    # Replace all Unicode escape sequences in the input string
    return re.sub(pattern, replace_match, content)
