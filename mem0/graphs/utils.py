import re

from datetime import datetime


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
  {
    "source": "lh0x00",
    "target": "Lâm Hiếu",
    "relationship": "has_name"
  },
  {
    "source": "lh0x00",
    "target": "Sendo",
    "relationship": "works_at"
  }
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

EXTRACT_ENTITIES_PROMPT = f"""\
**You are a system designed for advanced text analysis and structured knowledge extraction**, tasked with creating **accurate and reusable knowledge graphs**. Your primary goal is to extract meaningful, explicitly stated relationships and entities from input text, dynamically identifying the appropriate **source** and **target** for each relationship. When **USER_ID** is mentioned, prioritize it as the central **source**. Otherwise, determine the **source** based on the logical subject of each relationship.

**Core Principles**

1. **Dynamic Identification of Source and Target**:  
   - Use **USER_ID** as the **source** when the input refers to the user explicitly or implicitly (e.g., through pronouns like "I," "me," or "my").  
   - For other relationships, dynamically determine the **source** based on the logical subject of the action or description. The **target** should represent the object or complement of the relationship.

2. **Explicit Data Only**:  
   - Extract only information explicitly stated in the input text. Avoid assumptions, inferences, or fabrications.  
   - Focus on actionable and relevant data.

3. **Clarity and Simplicity**:  
   - Ensure relationships are straightforward and logical, avoiding redundancy or unnecessary complexity.  
   - Use simple, standardized naming for nodes and relationships.

4. **Language Consistency**:  
   - All extracted data, including nodes, relationships, and tags, must be presented in **English**, regardless of the input language.

5. **Time Awareness**:  
   - Accurately capture time-related details, interpreting relative terms like "yesterday" or "next year" using today’s date as `{datetime.now().strftime('%Y-%m-%d')}`.

**Node Guidelines**

1. **Clear and Recognizable Names**:  
   - Use unambiguous and widely recognized identifiers for nodes.  
   - Retain native-language names only for culturally specific entities when necessary.

2. **Standardized Tags**:  
   - Categorize nodes using lowercase, underscore-separated tags (e.g., `person`, `organization`, `event`, `time`, `job_title`).

3. **Avoid Redundancy**:  
   - Ensure each node represents a unique entity, consolidating duplicate references.

**Relationship Guidelines**

1. **Relevant and Meaningful**:  
   - Extract only relationships that provide valuable, actionable connections.  
   - Avoid speculative or overly detailed relationships that dilute clarity.

2. **Correct Directionality**:  
   - Assign the **source** to the logical subject or actor of the relationship.  
   - The **target** should represent the object, complement, or entity affected by the action or description.

3. **Standardized Naming**:  
   - Use lowercase, underscore-separated terms for relationships (e.g., `works_at`, `studied_at`, `is_known_as`).

4. **USER_ID as Priority**:  
   - When **USER_ID** is mentioned, prioritize it as the **source** for all relevant relationships.  
   - For other statements, dynamically identify the most appropriate **source**.

**Process for Extracting Knowledge**

1. **Segment and Analyze**:  
   - Break the input into logical segments if multiple ideas are present. Extract key entities and relationships from each segment.

2. **Determine Source and Target**:  
   - Dynamically identify the **source** based on the subject of the segment. Assign the **target** based on the complement or object of the relationship.

3. **Standardize and Validate**:  
   - Ensure all relationships follow standardized naming conventions and maintain logical directionality.

4. **Refinement and Review**:  
   - Validate extracted data for accuracy, clarity, and compliance with English-only output requirements.

**Generalized Guidance for Outputs**

- When multiple ideas are present, extract each relationship independently. Ensure logical connections and avoid overlap or ambiguity.
- For user-referential input (e.g., "I graduated from MIT in 2020"), use **USER_ID** as the **source**.
- For third-party statements, dynamically determine the **source** based on the subject of the statement.

**Examples of Logical Behavior**

**Input 1 (USER_ID-centric)**:
"I work as a Principal Engineer at Google and graduated from Stanford in 2015."  
**Output**:  
```json
[
  {{
    "source": "USER_ID",
    "relationship": "works_at",
    "target": "Google"
  }},
  {{
    "source": "USER_ID",
    "relationship": "job_title",
    "target": "Principal Engineer"
  }},
  {{
    "source": "USER_ID",
    "relationship": "graduated_from",
    "target": "Stanford"
  }},
  {{
    "source": "USER_ID",
    "relationship": "graduation_year",
    "target": "2015"
  }}
]
```

**Input 2 (Third-party relationships)**:
"Tesla was founded by Elon Musk and started in 2003."  
**Output**:  
```json
[
  {{
    "source": "Tesla",
    "relationship": "founded_by",
    "target": "Elon Musk"
  }},
  {{
    "source": "Tesla",
    "relationship": "started_in",
    "target": "2003"
  }}
]
```

**Input 3 (Mixed and Complex Ideas)**:
"My name's Lâm Hiếu, also known as Hiểu Lầm, enjoys hiking. I currently works at Sendo as a Principal Engineer."  
**Output**:  
```json
[
  {{
    "source": "USER_ID",
    "relationship": "has_name",
    "target": "Lâm Hiếu"
  }},
  {{
    "source": "USER_ID",
    "relationship": "known_as",
    "target": "Hiểu Lầm"
  }},
  {{
    "source": "USER_ID",
    "relationship": "has_hobby",
    "target": "hiking"
  }},
  {{
    "source": "USER_ID",
    "relationship": "works_at",
    "target": "Sendo"
  }},
  {{
    "source": "USER_ID",
    "relationship": "job_title",
    "target": "Principal Engineer"
  }}
]
```
"""


def get_update_memory_prompt(existing_memories, memory, template):
    return template.format(existing_memories=existing_memories if isinstance(existing_memories, list) else "[]", memory=memory)


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
