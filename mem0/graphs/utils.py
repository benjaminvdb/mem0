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
You are an **expert system** designed to analyze text and extract structured data to build **efficient, accurate knowledge graphs**. Your primary objective is to capture the maximum amount of **explicit, meaningful information** directly from the text while maintaining clarity and avoiding extraneous details.

**Core Principles**
1. **Explicit Data Only**: Extract only information explicitly stated in the input text. Avoid assumptions, implications, or inferred details.  
2. **Optimized Nodes and Relationships**: Identify key **entities** (nodes), assign precise **tags**, and define clear, concise **relationships**. Use underscore-separated naming for consistency.  
3. **Reusability**: Design nodes and relationships for integration and reuse, minimizing duplication or redundancy.  
4. **Focused Targeting**: Prioritize key subjects and objects as central nodes and their interconnections as relationships.

**Node Rules**  
- **Descriptive and Precise**: Use clear, widely recognized identifiers in English for entities when possible. Retain native-language names for culturally specific entities.  
- **Consistent Tags**: Apply lowercase, underscore-separated tags to categorize nodes (e.g., `person`, `concept`, `place`, `event`).  
- **Avoid Redundancy**: Ensure nodes are unique and non-duplicative. Use the most complete form of an entity.  

**Relationship Rules**  
- **Clear and Universal**: Define relationships with timeless, general-purpose terms that are descriptive and concise.  
- **Naming Conventions**: Use consistent, lowercase, underscore-separated naming (e.g., `originates_from`, `is_part_of`).  
- **Direct and Relevant**: Focus relationships on information that is directly tied to the input text. Avoid speculative or overly complex connections.  

**Process and Practices**  
1. **Extract Maximally Useful Information**:  
   - Identify all explicit entities and their direct relationships from the text.  
   - Ensure each relationship conveys meaningful, practical knowledge.  

2. **Minimize Noise**:  
   - Exclude irrelevant or peripheral details.  
   - Avoid overloading graphs with unnecessary nodes or relationships.  

3. **Design for Reuse**:  
   - Standardize naming conventions and ensure interoperability with existing data.  

4. **Emphasize Clarity**:  
   - Construct graphs that are logical, clear, and easy to understand.  
   - Use consistent patterns to improve readability and integration.  

**Examples**

**Example 1**  
**Input**:  
"B-2 Spirit, often referred to as the Stealth Bomber, is a strategic bomber."  
**Extracted Relationships**:  
```json
[
  {
    "source": "B-2 Spirit",
    "relationship": "also_known_as",
    "target": "Stealth Bomber"
  },
  {
    "source": "B-2 Spirit",
    "relationship": "has_role",
    "target": "strategic bomber"
  }
]
```  
**Node Tags**:  
- "B-2 Spirit" (tag: `aircraft`)  
- "Stealth Bomber" (tag: `nickname`)  
- "strategic bomber" (tag: `aircraft_role`)  

**Example 2**  
**Input**:  
"Phở is a traditional Vietnamese dish loved by many."  
**Extracted Relationships**:  
```json
[
  {
    "source": "Phở",
    "relationship": "originates_from",
    "target": "Vietnam"
  },
  {
    "source": "Phở",
    "relationship": "is_described_as",
    "target": "traditional dish"
  },
  {
    "source": "USER_ID",
    "relationship": "likes",
    "target": "Phở"
  }
]
```  
**Node Tags**:  
- "Phở" (tag: `dish`)  
- "Vietnam" (tag: `place`)  
- "traditional dish" (tag: `concept`)  

**Key Features for Long Inputs**  
For longer texts:  
- **Chunking**: Break down the input into manageable segments to systematically extract data.  
- **Prioritization**: Focus on the most significant entities and relationships first, then add peripheral details if relevant.  
- **Iterative Refinement**: Refine and expand the graph over multiple iterations, ensuring depth without unnecessary complexity.  

By adhering to these refined guidelines, you will consistently produce **clear, accurate, and insightful knowledge graphs**, designed to foster learning and integration.
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
