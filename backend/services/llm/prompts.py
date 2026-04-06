# ── Extraction Prompts ────────────────────────────────────
# Three variants for MSc research comparison
# Each variant is tested against the same diagrams
# to measure which produces most accurate extraction

EXTRACTION_PROMPTS = {

    # ── Zero Shot ─────────────────────────────────────────
    # No examples given — model relies purely on training
    # Baseline measurement for research
    "zero_shot": """
Analyze this software architecture diagram and extract its structure.

Return ONLY a valid JSON object with exactly this structure:
{
  "components": [
    {
      "id": "unique_id",
      "name": "Component Name",
      "type": "service|database|gateway|queue|cache|cdn|load_balancer|client|storage|other",
      "technology": "specific technology if visible e.g. Redis, PostgreSQL, NGINX",
      "position": {"x": 0, "y": 0},
      "metadata": {
        "role": "what this component does",
        "bottleneck_risk": "low|medium|high",
        "scalability": "horizontal|vertical|both",
        "security_surface": "low|medium|high",
        "responsibilities": ["responsibility 1", "responsibility 2"],
        "suggestions": ["improvement 1", "improvement 2"]
      }
    }
  ],
  "connections": [
    {
      "id": "conn_1",
      "source": "source_component_id",
      "target": "target_component_id",
      "label": "protocol or data type e.g. REST, gRPC, SQL",
      "direction": "unidirectional|bidirectional",
      "protocol": "HTTP|HTTPS|TCP|WebSocket|AMQP|etc",
      "data_type": "JSON|binary|stream|etc"
    }
  ],
  "arch_type": "microservices|monolith|serverless|event_driven|layered|other",
  "confidence_score": 0.95
}
Important: Be concise. Max 2 responsibilities and 1 suggestion per component.
Keep all text values under 80 characters.

Return ONLY the JSON. No explanation. No markdown. No code blocks.
""",

    # ── Few Shot ──────────────────────────────────────────
    # One example provided — helps model understand expected format
    # Tests if examples improve extraction accuracy
    "few_shot": """
Analyze this software architecture diagram and extract its structure.

Here is an example of the expected output format for a simple architecture:

EXAMPLE INPUT: A diagram showing a web browser connecting to a load balancer, which connects to two app servers, both connecting to a database.

EXAMPLE OUTPUT:
{
  "components": [
    {"id": "c1", "name": "Web Browser", "type": "client", "technology": null, "position": {"x": 0, "y": 0}, "metadata": {"role": "End user client", "bottleneck_risk": "low", "scalability": "horizontal", "security_surface": "low", "responsibilities": ["Send HTTP requests"], "suggestions": []}},
    {"id": "c2", "name": "Load Balancer", "type": "load_balancer", "technology": "NGINX", "position": {"x": 200, "y": 0}, "metadata": {"role": "Distributes traffic", "bottleneck_risk": "medium", "scalability": "horizontal", "security_surface": "medium", "responsibilities": ["Route requests", "Health checks"], "suggestions": ["Add redundancy"]}},
    {"id": "c3", "name": "App Server 1", "type": "service", "technology": "Node.js", "position": {"x": 400, "y": -100}, "metadata": {"role": "Application logic", "bottleneck_risk": "low", "scalability": "horizontal", "security_surface": "medium", "responsibilities": ["Process requests"], "suggestions": []}},
    {"id": "c4", "name": "Database", "type": "database", "technology": "PostgreSQL", "position": {"x": 600, "y": 0}, "metadata": {"role": "Persistent storage", "bottleneck_risk": "high", "scalability": "vertical", "security_surface": "high", "responsibilities": ["Store data"], "suggestions": ["Add read replicas"]}}
  ],
  "connections": [
    {"id": "e1", "source": "c1", "target": "c2", "label": "HTTPS", "direction": "bidirectional", "protocol": "HTTPS", "data_type": "JSON"},
    {"id": "e2", "source": "c2", "target": "c3", "label": "HTTP", "direction": "unidirectional", "protocol": "HTTP", "data_type": "JSON"},
    {"id": "e3", "source": "c3", "target": "c4", "label": "SQL", "direction": "bidirectional", "protocol": "TCP", "data_type": "SQL"}
  ],
  "arch_type": "layered",
  "confidence_score": 0.92
}

Now analyze the provided diagram and return the same JSON structure.
Return ONLY the JSON. No explanation. No markdown. No code blocks.
""",

    # ── Chain of Thought ──────────────────────────────────
    # Instructs model to reason step by step before returning JSON
    # Tests if reasoning improves accuracy and reduces hallucination
    "chain_of_thought": """
Analyze this software architecture diagram carefully.

Follow these steps in order:

STEP 1 - IDENTIFY ALL COMPONENTS:
Look at every box, cylinder, cloud shape, or labeled element.
List every component you can see, including its label and shape type.

STEP 2 - CLASSIFY EACH COMPONENT:
For each component, determine its type:
- Rectangles/boxes with service names → service
- Cylinders → database
- Diamond shapes → gateway
- Parallelograms → queue
- Rounded boxes labeled cache/redis → cache
- Cloud shapes → cdn or storage
- Hexagons → load_balancer
- Browser/mobile icons → client

STEP 3 - IDENTIFY ALL CONNECTIONS:
Look for every arrow, line, or connection between components.
Note the direction and any labels on the connection.

STEP 4 - ASSESS RISKS:
For each component consider:
- Is it a single point of failure? → bottleneck_risk: high
- Can it scale easily? → scalability type
- Does it handle sensitive data? → security_surface level

STEP 5 - BUILD THE JSON:
Now construct the final JSON using everything identified above.

Return ONLY this JSON structure, nothing else:
{
  "components": [
    {
      "id": "unique_id",
      "name": "Component Name",
      "type": "service|database|gateway|queue|cache|cdn|load_balancer|client|storage|other",
      "technology": "specific technology if visible",
      "position": {"x": 0, "y": 0},
      "metadata": {
        "role": "what this component does",
        "bottleneck_risk": "low|medium|high",
        "scalability": "horizontal|vertical|both",
        "security_surface": "low|medium|high",
        "responsibilities": ["responsibility 1"],
        "suggestions": ["improvement 1"]
      }
    }
  ],
  "connections": [
    {
      "id": "conn_id",
      "source": "source_id",
      "target": "target_id",
      "label": "protocol",
      "direction": "unidirectional|bidirectional",
      "protocol": "HTTP|HTTPS|TCP|etc",
      "data_type": "JSON|binary|etc"
    }
  ],
  "arch_type": "microservices|monolith|serverless|event_driven|layered|other",
  "confidence_score": 0.95
}

Return ONLY the JSON. No explanation. No markdown. No code blocks.
"""
}

# ── Chat Prompts ──────────────────────────────────────────
CHAT_SYSTEM_PROMPT = """
You are an expert software architect and system design consultant.
You have been given a software architecture diagram that has been 
extracted into structured JSON format.

Your role is to:
1. Answer questions about this specific architecture
2. Identify bottlenecks, scaling issues, and security concerns
3. Suggest improvements based on industry best practices
4. Compare this architecture to real world systems like Netflix, Uber, Amazon

Always refer to the specific components visible in this architecture.
Never give generic answers — always relate your response to the actual
components and connections provided in the architecture context.

Architecture context will be provided with each message.
"""

CHAT_INTERVIEW_PROMPT = """
You are an expert system design interviewer at a top tech company (FAANG level).
You have been given a software architecture diagram in JSON format.

When answering questions, structure your response exactly like a model 
interview answer:

CLARIFYING ASSUMPTIONS:
State any assumptions you are making about scale, traffic, and requirements.

HIGH LEVEL DESIGN:
Explain the overall architecture approach and key design decisions.

KEY COMPONENTS:
Walk through each major component, its role, and why it was chosen.

BOTTLENECKS & TRADEOFFS:
Identify what could fail under load and what tradeoffs were made.

IMPROVEMENTS:
Suggest 2-3 concrete improvements with justification.

Always refer to the specific components in the provided architecture.
Teach the user HOW to think about system design, not just what the answer is.
"""

# ── Component Explanation Prompt ──────────────────────────
COMPONENT_EXPLAIN_PROMPT = """
You are an expert software architect analyzing a specific component 
within a larger system architecture.

Given the full architecture context and the specific component details,
provide a detailed analysis.

Return ONLY a valid JSON object with exactly this structure:
{
  "component_id": "the component id",
  "component_name": "the component name",
  "role": "clear explanation of what this component does in THIS specific architecture",
  "responsibilities": [
    "specific responsibility 1",
    "specific responsibility 2",
    "specific responsibility 3"
  ],
  "bottleneck_risk": "low|medium|high",
  "bottleneck_explanation": "why this risk level in context of this architecture",
  "scalability": "horizontal|vertical|both",
  "scalability_explanation": "how this component scales",
  "security": "low|medium|high",
  "security_explanation": "what security considerations apply",
  "suggestions": [
    "concrete improvement 1",
    "concrete improvement 2"
  ],
  "interview_talking_points": [
    "key point to mention in system design interview 1",
    "key point to mention in system design interview 2"
  ]
}

Return ONLY the JSON. No explanation. No markdown. No code blocks.
"""
