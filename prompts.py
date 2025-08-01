from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-AWQ")

with open("variable_definitions/default_definitions.json", "r") as file:
    default = json.load(file)

with open("variable_definitions/ontological_definitions.json", "r") as file:
    ontology = json.load(file)


def prompt_er(entities, debug=False):
    system = (
        "\n--- ROLE ---\n"
        "You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged."
    )
    user = (
        "The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.\n"
        "Here are the rules for identifying duplicates:\n"
        "1. Entities with minor typographical differences should be considered duplicates.\n"
        "2. Entities with different formats but the same content should be considered duplicates.\n"
        "3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.\n"
        # add more here to ignore codes like the ukb, study codes, etc. If there are differences in numbers, dates, codes, identifiers, etc, do not merge results
        "4. If the entities refers to different numbers, dates, codes, medical ids, or products, do not merge entities.\n"
        "5. If the entities contain any numerical differences, do not merge entities.\n\n"
        "Here is the list of entities to process:\n"
        f"{entities}\n\n"
        "Please identify duplicates, merge them, and provide the merged list."
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )
    if debug:
        print(prompt)
    return prompt

def default_prompt(text, debug=False):
    system_prompt = (
        "# Knowledge Graph Instructions\n"
        "## 1. Overview\n"
        "You are a top-tier algorithm designed for extracting information in structured "
        "formats to build a knowledge graph. You will focus on extracting information from biomedical literature.\n"
        "Try to capture as much information from the text as possible without "
        "sacrificing accuracy. Do not add any information that is not explicitly "
        "mentioned in the text.\n"
        "- **Nodes** represent entities and concepts.\n"
        "## 2. Labeling Nodes\n"
        "- **Consistency**: Ensure you use available types for node labels.\n"
        "Ensure you use basic or elementary types for node labels.\n"
        "- For example, when you identify an entity representing a person, "
        "always label it as **'person'**. Avoid using more specific terms "
        "like 'mathematician' or 'scientist'."
        "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
        "names or human-readable identifiers found in the text.\n"
        "- **Relationships** represent connections between entities or concepts.\n"
        "Ensure consistency and generality in relationship types when constructing "
        "knowledge graphs. Instead of using specific and momentary types "
        "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
        "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
        "Specifically, focus on extracting relationship types pertaining to causality, such as A CAUSES B, or A IS_ASSOCIATED_WITH B.\n"
        "## 3. Coreference Resolution\n"
        "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
        "ensure consistency.\n"
        'If an entity, such as "John Doe", is mentioned multiple times in the text '
        'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
        "always use the most complete identifier for that entity throughout the "
        'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
        "Remember, the knowledge graph should be coherent and easily understandable, "
        "so maintaining consistency in entity references is crucial.\n"
        "## 4. Strict Compliance\n"
        "Adhere to the rules strictly. Non-compliance will result in termination."
    )
    
    user = (
        "Tip: Make sure to answer in the correct format and do not include any explanations. "
        "Use the given format to extract information from the following input:\n\n"
        "-- INPUT --\n"
        f"{ text }"
    )

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )
    if debug:
        print(prompt)
    return prompt

def ontology_prompt(text, debug=False):
    return var_def_prompt(text, definitions="ontology", debug=False)

def var_def_prompt(text, definitions="default" ,debug=False):
    def_map = default
    if definitions == "ontology":
        def_map = ontology
    
    system_prompt = (
        "# Knowledge Graph Instructions\n"
        "## 1. Overview\n"
        "You are a top-tier algorithm designed for extracting information in structured "
        "formats to build a knowledge graph. You will focus on extracting information from biomedical literature.\n"
        "Try to capture as much information from the text as possible without "
        "sacrificing accuracy. Do not add any information that is not explicitly "
        "mentioned in the text.\n"
        "- **Nodes** represent entities and concepts.\n"
        "- Focus on capturing all entities and concepts in the text. Specifically, pay special attention to the following terms:\n"
        f"- {def_map.get('Sex')}\n"
        f"- {def_map.get('PEG')}\n"
        f"- {def_map.get('Sleep')}\n"
        f"- {def_map.get('Depression')}\n"
        f"- {def_map.get('Anxiety')}\n"
        f"- {def_map.get('Obesity')}\n"
        f"- {def_map.get('Alcohol')}\n"
        f"- {def_map.get('Fear_avoidance')}\n"
        f"- {def_map.get('Catastrophizing')}\n"
        f"- {def_map.get('CCI')}\n"
        f"- {def_map.get('Education')}\n"
        f"- {def_map.get('Financial_level')}\n"
        f"- {def_map.get('Age')}\n"
        f"- {def_map.get('Smoking')}\n"
        "You **MUST** extract all entities related to the above terms.\n"
        "## 2. Labeling Nodes\n"
        "- **Consistency**: Ensure you use available types for node labels.\n"
        "Ensure you use basic or elementary types for node labels.\n"
        "- For example, when you identify an entity representing a person, "
        "always label it as **'person'**. Avoid using more specific terms "
        "like 'mathematician' or 'scientist'."
        "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
        "names or human-readable identifiers found in the text.\n"
        "- **Relationships** represent connections between entities or concepts.\n"
        "Ensure consistency and generality in relationship types when constructing "
        "knowledge graphs. Instead of using specific and momentary types "
        "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
        "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
        "Specifically, focus on extracting relationship types pertaining to causality, such as A CAUSES B, or A IS_ASSOCIATED_WITH B.\n"
        "## 3. Coreference Resolution\n"
        "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
        "ensure consistency.\n"
        'If an entity, such as "John Doe", is mentioned multiple times in the text '
        'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
        "always use the most complete identifier for that entity throughout the "
        'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
        "Remember, the knowledge graph should be coherent and easily understandable, "
        "so maintaining consistency in entity references is crucial.\n"
        "## 4. Strict Compliance\n"
        "Adhere to the rules strictly. Non-compliance will result in termination."
    )
    
    user = (
        "Tip: Make sure to answer in the correct format and do not include any explanations. "
        "Use the given format to extract information from the following input:\n\n"
        "-- INPUT --\n"
        f"{ text }"
    )

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )
    if debug:
        print(prompt)
    return prompt

def summarize_community(community, debug=False):
    system = (
        "\n--- ROLE ---\n"
        "You are an expert in health care, medicine, chronic pain and its associated health conditions.\n"
    )

    user = (
        "--- INSTRUCTIONS ---\n"
        "Based on the provided nodes and relationships that belong to the same graph community, generate a 750 word natural language summary of the provided information."
        "Use the explicit nodes mentioned instead of coreferences, and focus your summary on the relationships provided.  Output only the summary without any commentary, citations, or extraneous information.\n"

        "--- GRAPH COMMUNITY ---\n"
        f"{ community }\n"
        "--- SUMMARY ---\n"
        "Summary: "
    )

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )
    if debug:
        print(prompt)
    return prompt

def reduce(question, var1, var2, report, definitions, debug=False):
    system = (
        "\n--- ROLE ---\n"
        "You are an expert in the field of chonic pain, health care, and medicine. Given the question, output True or False. You will be reading a report from the leading experts in chronic lower back pain. "
        "Use the relevant information in the report, as well as the definitions given, as context to help you answer the following True/False question.\n"
        "Begin by carefully breaking down the relevant information given in the report, noting any observations or conclusions that you can draw from the given information. "
        "Afterwards, identify any connections between the key points in the reports. Reread the report multiple times to make sure you identify all key details and all connections "
        "within the report. Then, think through the question step by step, thoroughly explaining your thought process. Use the information given in the report and your general "
        "knowledge to help you reason through the question and make your final answer. Remember, DO NOT base your conclusion solely on the report and DO NOT interpret the absence "
        "of information as evidence that the answer to the following question is false. If there is not enough information in the report, think through the question step by step "
        "again, using your general knowledge and reasoning skills to finalize your conclusion instead of relying solely on the report.\n"
        "Based on your reasoning, output True or False as your final conclusion. Output your reasoning and conclusion explicitly in valid JSON using this exact format:\n"
        "{\n"
        '  "reasoning": [\n'
        '    {"reasoning_step": "Your first reasoning step here"},\n'
        '    {"reasoning_step": "Your second reasoning step here"}\n'
        '  ],\n'
        '  "conclusion": true\n'
        "}\n\n"
        "Specifically, each reasoning step must contain your intermediate thought process, and your final answer to the question must be in the conclusion field. "
        "Your final conclusion MUST be consistent with your reasoning. "
        "Failure to follow directions will result in immediate termination."
    )
    user = (
        "--- QUESTION ---\n"
        f"{ question }\n"
        "\n--- VARIABLE DEFINITIONS ---\n"
        "Variable 1:\n"
        f"{ var1 }\n"
        "Definition of Variable 1:\n"
        f"{ definitions.get(var1, var1) }\n"
        "Variable 2:\n"
        f"{ var2 }\n"
        "Definition of Variable 2:\n"
        f"{ definitions.get(var2, var2) }\n\n"
        "--- REPORT ---\n"
        f"{ report }\n\n"
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_bos=False,
        add_generation_prompt=True,
    )
    if debug:
        print(prompt)
    return prompt


def predict(question, var1, var2, definitions, debug=False):

    system = (
        "\n--- ROLE ---\n"
        "You are an expert in the field of chonic pain, health care, and medicine. Given the question, output True or False. "
        "Think through the question step by step, thoroughly explaining your thought process. Use the definitions and your general knowledge to help you reason through the question and make your final answer.\n"
        "Based on your reasoning, output True or False as your final conclusion. Output your reasoning and conclusion explicitly in valid JSON using this exact format:\n"
        "{\n"
        '  "reasoning": [\n'
        '    {"reasoning_step": "Your first reasoning step here"},\n'
        '    {"reasoning_step": "Your second reasoning step here"}\n'
        '  ],\n'
        '  "conclusion": true\n'
        "}\n\n"
        "Specifically, each reasoning step must contain your intermediate thought process, and your final answer to the question must be in the conclusion field. "
        "Your final conclusion MUST be consistent with your reasoning. "
        "Failure to follow directions will result in immediate termination."
    )

    user = (
        "--- QUESTION ---\n"
        f"{ question }\n"
        "\n--- VARIABLE DEFINITIONS ---\n"
        "Variable 1:\n"
        f"{ var1 }\n"
        "Definition of Variable 1:\n"
        f"{ definitions.get(var1, var1) }\n"
        "Variable 2:\n"
        f"{ var2 }\n"
        "Definition of Variable 2:\n"
        f"{ definitions.get(var2, var2) }\n"
    )

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )
    if debug:
        print(prompt)
    return prompt

def analyze_inconsistencies(query, reasoning, result, debug=False):
    system = (
        "\n--- TASK ---\n"
        "I am evaluating inconsistencies between the thought process of Large Language Models and their final answers to questions. "
        "You will be given a model's attempt to reason through the given question, and you will compare the reasoning process to its final conclusion. "
        "I want you to determine whether the final conclusion is consistent with the model's reasoning. Output True if the final conclusion is consistent with the model's reasoning "
        "or False if the model's final conclusion is inconsistent with the model's reasoning."
    )

    user = (
        "\n--- QUESTION --- \n"
        f"{query}\n"
        "\n--- REASONING --- \n"
        f"{reasoning}\n"
        "\n--- FINAL CONCLUSION --- \n"
        f"{result}\n"
        "\n--- QUESTION ---\n"
        "Is the reasoning consistent with the final conclusion?"
    )

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )
    if debug:
        print(prompt)
    return prompt

def summarize(output, var1, var2, definitions, debug=False):
    system = (
        "\n--- ROLE ---\n"
        f"You are an expert in medicine and healthcare, aiming to summarize important documents to help physicians learn about { var1 }, and { var2 }, defined as {definitions.get(var1, var1)} and {definitions.get(var2, var2)}.\n"
        "You will be given output from a knowledge graph, consisting of entities, relationships, and summaries collected over the graph.\n"
        f"Focus on summarizing the information mainly within the relationships and the reports collected from the graph. Capture all details that may directly or indirectly relate {var1} to {var2}. \n"
        "Output only the summary and do not add any pre-amble, acknowledgements, or extraneous information such as citations. Failure to follow directions will result in immediate termination."
    )

    user = (
        "--- INSTRUCTIONS ---\n"
        "Given the output from the knowledge graph, summarize the output into a 750 word technical report.\n"
        f"Focus on extracting all details that pertain to { var1 }, { var2 }, and the relationships between { var1 } and { var2 },"
        "ignoring all other irrelevant pieces of information.\n"
        "\n--- KNOWLEDGE GRAPH OUTPUT ---\n"
        f"{ output }\n"
    )

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )
    if debug:
        print(prompt)
    return prompt

# ----------------------
# GraphReader‑style prompts for kg_builder.py
# ----------------------

# Prompt to extract atomic facts from a text chunk
atomic_fact_prompt = (
    "# Instruction: Extract atomic facts\n"
    "From the following text chunk, pull out every minimal, self‑contained statement (“atomic fact”).\n"
    "Do NOT add commentary—just return JSON.\n"
    "Output format (exactly):\n"
    "  {{\"atomic_facts\": [\"fact1\",\"fact2\",...]}}\n\n"
    "-- TEXT CHUNK --\n"
    "{chunk}\n\n"
    "-- JSON OUTPUT --"
)

# Prompt to extract key elements (entities/concepts) from one atomic fact
key_element_prompt = (
    "# Instruction: Extract key elements\n"
    "Given the following atomic fact, identify each entity or concept mentioned.\n"
    "Do NOT add commentary—just return JSON.\n"
    "Output format (exactly):\n"
    "  {{\"key_elements\": [\"entity1\",\"entity2\",...]}}\n\n"
    "-- ATOMIC FACT --\n"
    "{fact}\n\n"
    "-- JSON OUTPUT --"
)
