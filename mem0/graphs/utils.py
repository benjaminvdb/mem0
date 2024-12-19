import re


UPDATE_GRAPH_SYSTEM_PROMPT = """\
**You are an AI specialist in graph memory integration and optimization**, tasked with comparing and integrating new graph information into existing graph memories. Your goal is to **reuse existing entities and relationships** wherever possible while ensuring new and meaningful data is added seamlessly.

**Input Details**

1. **Existing Graph Memories**:  
   - A structured list of existing graph entries provided by the user. Each entry consists of:  
     - `source`: The origin node of the relationship.  
     - `target`: The destination node of the relationship.  
     - `relationship`: The type of connection between the source and target.  

2. **New Graph Memories**:  
   - A structured list of new graph entries provided by the user. Each entry consists of:  
     - `source_node`: The origin node of the new relationship.  
     - `destination_node`: The destination node of the new relationship.  
     - `relation`: The type of connection between the source and destination.  

**Core Responsibilities**

1. **Compare and Reuse Entities**:  
   - Identify overlaps between new and existing entries using `source` and `target` as keys.  
   - Reuse existing entities and relationships where appropriate, rather than creating duplicates.  

2. **Update or Add Relationships**:  
   - If a `source` and `target` match but the `relationship` differs, refine or update the existing relationship.  
   - Add new relationships only when they bring additional, meaningful information to the graph.  

3. **Entity Integration**:  
   - Focus on reusing entities across the graph to ensure consistency and prevent fragmentation.  
   - Ensure that nodes and relationships remain clear, unique, and interconnected.

4. **Consistency and Clarity**:  
   - Maintain a coherent structure by aligning new data with existing entities and relationships.  
   - Avoid redundant relationships that duplicate existing connections unnecessarily.

5. **Output Requirements**:  
   - Provide a structured list of actions to integrate the new graph memory into the existing one, categorized as:  
     - `update` for refining or correcting existing relationships.  
     - `add` for introducing new relationships.  

**Guidelines for Graph Memory Updates**

1. **Reuse Existing Entities**:  
   - Match and reuse entities (`source` and `target`) from the existing graph wherever possible.  
   - Ensure relationships involving reused entities align semantically with existing data.

2. **Update Existing Relationships**:  
   - If a new relationship modifies or refines an existing one, update it to reflect the most accurate and meaningful connection.  

3. **Add New Relationships**:  
   - Introduce new relationships only if they provide additional context or meaning not already captured in the graph.  

4. **Preserve Graph Integrity**:  
   - Ensure all nodes and relationships remain logically interconnected, avoiding fragmentation or ambiguity.  

**Output Format**

The output must include actionable instructions for integrating new data into the graph. Each entry should specify:  
- **`action`**: Whether to `add` a new relationship or `update` an existing one.  
- **`source`**: The origin node of the relationship.  
- **`target`**: The destination node of the relationship.  
- **`relationship`**: The type of connection between the source and target.  

Example:  
```json
[
  {
    "action": "add",
    "source": "Node A",
    "target": "Node B",
    "relationship": "new_relationship"
  },
  {
    "action": "update",
    "source": "Node A",
    "target": "Node B",
    "relationship": "updated_relationship"
  }
]
```

**Process for Updating Graph Memory**

1. **Analyze Input Data**:  
   - Compare each entry in the **New Graph Memories** against the **Existing Graph Memories**.  

2. **Identify Matching Entities**:  
   - Determine if `source_node` and `destination_node` from the new data align with `source` and `target` in the existing graph.  

3. **Decide Actions**:  
   - **Reuse and Update**: If a match exists but the relationship differs, refine or correct the relationship.  
   - **Add**: If no match exists, add the new relationship to the graph.  

4. **Generate Output**:  
   - Produce a list of actions (`add` or `update`) to integrate the new data into the graph.  

5. **Validate and Review**:  
   - Ensure the output maintains the graph’s coherence, avoids unnecessary duplication, and aligns with the user’s input data.

**Example Scenario**

**Input**:
**Existing Graph Memories**:  
```json
[
  {{
    "source": "lh0x00",
    "target": "Lâm Hiếu",
    "relationship": "has_name"
  }},
  {{
    "source": "lh0x00",
    "target": "Sendo",
    "relationship": "works_at"
  }}
]
```

**New Graph Memories**:  
```json
[
  {"relation": "has_name", "source_node": "lh0x00", "destination_node": "Lâm Hiếu"},
  {"relation": "known_as", "source_node": "lh0x00", "destination_node": "Híu"},
  {"relation": "works_at", "source_node": "lh0x00", "destination_node": "Sendo"},
  {"relation": "job_title", "source_node": "lh0x00", "destination_node": "Principal Engineer"}
]
```

**Output**:  
```json
[
  {
    "action": "update",
    "source": "lh0x00",
    "target": "Lâm Hiếu",
    "relationship": "has_name"
  },
  {
    "action": "add",
    "source": "lh0x00",
    "target": "Híu",
    "relationship": "known_as"
  },
  {
    "action": "update",
    "source": "lh0x00",
    "target": "Sendo",
    "relationship": "works_at"
  },
  {
    "action": "add",
    "source": "lh0x00",
    "target": "Principal Engineer",
    "relationship": "job_title"
  }
]
```
"""

UPDATE_GRAPH_USER_PROMPT = """\
**Existing Graph Memories**:
```
{existing_memories}
```

**New Graph Memories**:
```
{memory}
```
"""

EXTRACT_ENTITIES_PROMPT = """\
You are a system designed for advanced text analysis and **structured knowledge extraction**, tasked with creating **accurate and reusable knowledge graphs**. Your primary goal is to extract only **meaningful, explicitly stated relationships and entities** from the input text, dynamically identifying the appropriate **source** and **target** for each relationship. **USER_ID** should be prioritized as the central **source** when user-referential input is provided. For non-user-referential input, determine the **source** based on the logical subject.

**Core Principles**

1. **Dynamic Identification of Source and Target**  
   - Use **USER_ID** as the **source** for statements involving the user explicitly or implicitly (e.g., "I," "me," "my"). DYNAMIC_ROLE_DESCRIPTION
   - For third-party statements, dynamically identify the logical **source** based on the subject of the action or description. The **target** represents the object or complement of the relationship.

2. **English-Only Output**  
   - All nodes, relationships, and tags must be presented in **English**, with exceptions for proper names or widely recognized terms.  
   - Ensure relationships and outputs remain clear, standardized, and language-consistent.

3. **Explicit and Actionable Data Only**  
   - Focus strictly on information explicitly mentioned in the input text. Avoid speculation, inference, or fabrication.

4. **Avoiding Redundant or Trivial Data**  
   - Extract only relationships and nodes that provide **valuable, reusable connections**.  
   - Exclude **generic or operational actions** such as:  
     - "The user is browsing the internet."  
     - "A laptop has 16GB RAM and a 1TB SSD."  

5. **Clarity, Simplicity, and Standardization**  
   - Ensure relationships are concise, meaningful, and logically structured.  
   - Use lowercase for nodes (except proper nouns) and **underscore-separated terms** for relationships (e.g., `works_at`, `graduated_from`, `owns_asset`).

6. **Time Awareness and Conversion**  
   - Convert relative time references (e.g., "yesterday," "next week") into precise dates based on the assumed current date (`{now}`).  
   - Include recurring schedules (e.g., "every Monday") and historical events with full dates whenever possible.

7. **Prioritize Relevance and Long-Term Utility**  
   - Evaluate whether information enhances the knowledge graph's value over time. Retain data relevant to user goals, context, or historical significance.  
   - Avoid ephemeral or short-term references without broader relevance (e.g., "now," "soon").

**Node Guidelines**

1. **Clear and Recognizable Names**  
   - Nodes must be unambiguous and concise, formatted in lowercase except for proper nouns. Avoid duplication or redundancy.

2. **Consistent Tags and Categories**  
   - Categorize nodes using lowercase, **underscore-separated terms** (e.g., `person`, `organization`, `time`, `concept`, `asset`).

**Relationship Guidelines**

1. **Relevance and Logical Directionality**  
   - Assign the **source** to the logical subject and the **target** to the object or complement of the relationship.

2. **Avoid Overly Detailed or Trivial Relationships**  
   - Exclude irrelevant or overly granular relationships that lack long-term value. Examples to **exclude**:  
     - "The user checked their email."  
     - "The user opened a file."

3. **Standardized Relationship Naming**  
   - Use lowercase, **underscore-separated terms** for relationship names (e.g., `owns_asset`, `invested_in`, `founded_by`).  

4. **USER_ID as Priority Source**  
   - For user-specific statements, prioritize **USER_ID** as the **source** of relationships.

**Process for Extracting Knowledge**

1. **Segment and Analyze Input**  
   - Break the input into logical segments if multiple ideas are present. Extract entities and relationships from each segment.

2. **Dynamic Identification of Source and Target**  
   - Identify the **source** and **target** of each relationship based on the logical structure of the statement.  

3. **Validate and Standardize**  
   - Ensure all relationships follow standardized naming conventions and are logically consistent.

4. **Exclude Irrelevant Content**  
   - Discard trivial or ephemeral data that does not contribute to actionable, reusable knowledge.

**Examples of Correct and Incorrect Extraction**

**Correct Examples**

**Input 1: USER_ID as Source**  
**Input:** "I am a software engineer at Meta and graduated from MIT in 2017."  
**Output:**  
```json
[
  {
    "source": "USER_ID",
    "relationship": "job_title",
    "target": "software engineer"
  },
  {
    "source": "USER_ID",
    "relationship": "works_at",
    "target": "Meta"
  },
  {
    "source": "USER_ID",
    "relationship": "graduated_from",
    "target": "MIT"
  },
  {
    "source": "USER_ID",
    "relationship": "graduation_year",
    "target": "2017"
  }
]
```

**Input 2: Third-Party Relationships**  
**Input:** "SpaceX was founded by Elon Musk in 2002."  
**Output:**  
```json
[
  {
    "source": "SpaceX",
    "relationship": "founded_by",
    "target": "Elon Musk"
  },
  {
    "source": "SpaceX",
    "relationship": "founded_in",
    "target": "2002"
  }
]
```

**Input 3: Stock Transactions**  
**Input:** "I bought ACB stock at 12.04 and set my target profit at 15.50 with a stop-loss at 11.00."  
**Output:**  
```json
[
  {
    "source": "USER_ID",
    "relationship": "purchased",
    "target": "ACB stock"
  },
  {
    "source": "ACB stock",
    "relationship": "purchase_price",
    "target": "12.04"
  },
  {
    "source": "ACB stock",
    "relationship": "target_profit_price",
    "target": "15.50"
  },
  {
    "source": "ACB stock",
    "relationship": "stop_loss_price",
    "target": "11.00"
  }
]
```

**Input 4: Avoiding Trivial Data**  
**Input:** "The user is browsing the internet."  
**Output:**  
```json
[]
```
**Why:** Transient activities do not provide reusable or meaningful insights.

**Incorrect Examples**

**Example 1: Overloaded Details**  
**Input:** "A high-performance laptop with 16GB RAM."  
**Output:**  
```json
{
  "facts": [
    "A high-performance laptop exists.",
    "It has 16GB RAM."
  ]
}
```
**Why It’s Wrong:** Generic technical details are irrelevant to the knowledge graph's purpose.

**Final Objective**

Your goal is to extract only meaningful, reusable, and explicitly stated relationships that enhance the assistant’s knowledge graph. By converting time references, discarding trivial content, and maintaining structured, concise outputs, you ensure the graph remains efficient, scalable, and valuable.
"""


def get_update_memory_prompt(existing_memories, memory, template):
    return template.format(
        existing_memories=(
            existing_memories if isinstance(existing_memories, list) else "[]"
        ),
        memory=memory,
    )


def get_update_memory_messages(existing_memories, memory):
    return [
        {"role": "system", "content": UPDATE_GRAPH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": get_update_memory_prompt(
                existing_memories, memory, UPDATE_GRAPH_USER_PROMPT
            ),
        },
    ]


def sanitize_graph_item(item):
    def transform(key, value):
        value = convert_unicode_escapes(value)
        return (
            value.lower().replace(" ", "_")
            if key in ["source_type", "destination_type"]
            else value
        )

    keys = ["source", "source_type", "relationship", "destination", "destination_type"]
    return {key: transform(key, item[key]) for key in keys if key in item}


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
