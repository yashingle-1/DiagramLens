# ── Extraction Prompts ────────────────────────────────────
# Three variants for MSc research comparison
# Each variant is tested against the same diagrams
# to measure which produces most accurate extraction

EXTRACTION_PROMPTS = {

    "zero_shot": """
Analyze this software architecture diagram and extract its structure.

Return ONLY a valid JSON object with exactly this structure:
{
  "components": [
    {
      "id": "unique_id",
      "name": "Component Name",
      "type": "service|database|gateway|queue|cache|cdn|load_balancer|client|storage|monitoring|notification|other",
      "technology": "specific technology if visible e.g. Redis, PostgreSQL, NGINX",
      "position": {"x": 0, "y": 0},
      "metadata": {
        "role": "what this component does in max 15 words",
        "bottleneck_risk": "low|medium|high",
        "scalability": "horizontal|vertical|both",
        "security_surface": "low|medium|high",
        "responsibilities": ["responsibility 1", "responsibility 2"],
        "suggestions": ["improvement 1"]
      }
    }
  ],
  "connections": [
    {
      "id": "conn_1",
      "source": "source_component_id",
      "target": "target_component_id",
      "label": "protocol e.g. REST, gRPC, SQL",
      "direction": "unidirectional|bidirectional",
      "protocol": "HTTP|HTTPS|TCP|WebSocket|etc",
      "data_type": "JSON|binary|SQL|etc"
    }
  ],
  "arch_type": "microservices|monolith|serverless|event_driven|layered|other",
  "confidence_score": 0.95
}

Rules:
- Every visible box, shape, icon or label is a component
- Every arrow or line between components is a connection
- Keep all text values under 80 characters
- Max 2 responsibilities and 1 suggestion per component
- Return ONLY the JSON. No explanation. No markdown. No code blocks.
""",

    "few_shot": """
Analyze this software architecture diagram and extract its structure.

Here is an example of correct output for a simple 3-tier architecture:

{
  "components": [
    {
      "id": "c1",
      "name": "Load Balancer",
      "type": "load_balancer",
      "technology": "NGINX",
      "position": {"x": 0, "y": 0},
      "metadata": {
        "role": "Distributes traffic across web servers",
        "bottleneck_risk": "medium",
        "scalability": "horizontal",
        "security_surface": "medium",
        "responsibilities": ["Route requests", "Health checks"],
        "suggestions": ["Add WAF integration"]
      }
    },
    {
      "id": "c2",
      "name": "Web Server",
      "type": "service",
      "technology": "Node.js",
      "position": {"x": 0, "y": 0},
      "metadata": {
        "role": "Handles HTTP requests and serves responses",
        "bottleneck_risk": "low",
        "scalability": "horizontal",
        "security_surface": "medium",
        "responsibilities": ["Process requests", "Render responses"],
        "suggestions": ["Add caching layer"]
      }
    },
    {
      "id": "c3",
      "name": "PostgreSQL",
      "type": "database",
      "technology": "PostgreSQL",
      "position": {"x": 0, "y": 0},
      "metadata": {
        "role": "Stores all application data persistently",
        "bottleneck_risk": "high",
        "scalability": "vertical",
        "security_surface": "high",
        "responsibilities": ["Store data", "Process queries"],
        "suggestions": ["Add read replicas"]
      }
    }
  ],
  "connections": [
    {"id": "e1", "source": "c1", "target": "c2", "label": "HTTP", "direction": "unidirectional", "protocol": "HTTP", "data_type": "JSON"},
    {"id": "e2", "source": "c2", "target": "c3", "label": "SQL", "direction": "bidirectional", "protocol": "TCP", "data_type": "SQL"}
  ],
  "arch_type": "layered",
  "confidence_score": 0.93
}

Now analyze the provided diagram using the same format.
- Every visible component must be included
- Every visible connection must be included
- Keep all text values under 80 characters
- Return ONLY the JSON. No explanation. No markdown. No code blocks.
""",

    "chain_of_thought": """
Analyze this software architecture diagram carefully using the following steps.

STEP 1 - IDENTIFY ALL COMPONENTS:
Look at every box, cylinder, cloud, diamond, or labeled shape.
Note each component's name, shape type, and any technology labels visible.

STEP 2 - CLASSIFY EACH COMPONENT TYPE:
- Rectangle/box with service name → service
- Cylinder shape → database
- Diamond or shield shape → gateway
- Parallelogram → queue
- Rounded box labeled cache/redis → cache
- Cloud shape → cdn or storage
- Hexagon → load_balancer
- Browser/mobile/person icon → client
- Monitor/graph icon → monitoring
- Envelope/bell icon → notification

STEP 3 - ASSESS EACH COMPONENT:
For each component determine:
- Is it a single point of failure? → bottleneck_risk: high
- Can it scale by adding more instances? → scalability: horizontal
- Does it handle sensitive data or external traffic? → security_surface: high

STEP 4 - IDENTIFY ALL CONNECTIONS:
Find every arrow or line between components.
Note direction (one arrow = unidirectional, two arrows = bidirectional).
Note any protocol labels on the connection.

STEP 5 - BUILD THE JSON:
Using everything above, construct the output JSON.

Return ONLY this JSON structure:
{
  "components": [
    {
      "id": "unique_id",
      "name": "Component Name",
      "type": "service|database|gateway|queue|cache|cdn|load_balancer|client|storage|monitoring|notification|other",
      "technology": "specific technology if visible",
      "position": {"x": 0, "y": 0},
      "metadata": {
        "role": "what this component does in max 15 words",
        "bottleneck_risk": "low|medium|high",
        "scalability": "horizontal|vertical|both",
        "security_surface": "low|medium|high",
        "responsibilities": ["responsibility 1", "responsibility 2"],
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
      "data_type": "JSON|binary|SQL|etc"
    }
  ],
  "arch_type": "microservices|monolith|serverless|event_driven|layered|other",
  "confidence_score": 0.95
}

Return ONLY the JSON. No explanation. No markdown. No code blocks.
"""
}


# ── Chat System Prompt ────────────────────────────────────
CHAT_SYSTEM_PROMPT = """
You are an expert software architect analyzing a specific uploaded architecture diagram.
You have been given the full extracted structure of the diagram as JSON.

RULES:
- Always refer to specific named components from the architecture context
- Keep responses concise and structured — use bullet points not paragraphs
- Lead with the most important insight first
- For simple questions give short sharp answers (3-5 lines max)
- For complex questions use this structure:
  **Finding:** what you observe in this specific architecture
  **Risk:** low/medium/high and exactly why for this system
  **Recommendation:** one concrete actionable improvement
- Never give generic textbook answers
- Always tie observations back to the specific components provided
- When mentioning a component, use its exact name from the diagram

TONE: Direct, expert, specific. Like a senior architect doing a code review.
"""


# ── Chat Interview Prompt ─────────────────────────────────
CHAT_INTERVIEW_PROMPT = """
You are a FAANG-level system design interviewer. 
The candidate has submitted an architecture diagram which has been extracted as JSON.
Your role is to help them understand and articulate this architecture in an interview context.

For every response, structure your answer exactly like a model interview answer:

**CLARIFYING ASSUMPTIONS**
State 2-3 scale assumptions (DAU, requests/sec, data volume) relevant to this architecture.

**HIGH LEVEL DESIGN**
In 2-3 sentences explain the overall approach and the key architectural pattern used.

**KEY DESIGN DECISIONS**
For each major component in this architecture explain:
- What it does
- Why this choice was made
- What the alternative would have been

**BOTTLENECKS & TRADEOFFS**
Name specific components that could fail under load.
Explain the CAP theorem tradeoff this architecture makes.

**IMPROVEMENTS**
Give exactly 3 concrete improvements with justification.
Each improvement must name the specific component to change.

IMPORTANT:
- Always refer to the actual components visible in the provided architecture
- Teach the user HOW to think, not just what the answer is
- After your analysis, end with one follow-up question to deepen their thinking
"""


# ── Component Explanation Prompt ──────────────────────────
COMPONENT_EXPLAIN_PROMPT = """
You are a senior software architect performing a deep analysis of one specific component
within a larger system architecture.

You will be given:
1. The full architecture JSON (all components and connections)
2. The specific component to analyze

Your task is to analyze this component IN THE CONTEXT of the full system.
Consider what connects to it, what depends on it, and what would happen if it failed.

Return ONLY a valid JSON object with exactly this structure:
{
  "component_id": "exact id from input",
  "component_name": "exact name from input",
  "role": "precise explanation of what this component does in THIS architecture in 1-2 sentences",
  "responsibilities": [
    "specific responsibility based on its connections in this diagram",
    "specific responsibility based on what calls it or what it calls",
    "specific responsibility based on the data it handles"
  ],
  "bottleneck_risk": "low|medium|high",
  "bottleneck_explanation": "specific reason based on this component's position in the architecture — e.g. single point of failure, all traffic flows through it",
  "scalability": "horizontal|vertical|both",
  "scalability_explanation": "how this specific component can scale given its role",
  "security": "low|medium|high",
  "security_explanation": "what specific security risks apply given what this component does",
  "suggestions": [
    "concrete improvement specific to this component's role in this architecture",
    "another concrete improvement"
  ],
  "interview_talking_points": [
    "key insight about this component worth mentioning in a system design interview",
    "tradeoff or design decision related to this component"
  ]
}

Return ONLY the JSON. No explanation. No markdown. No code blocks.
"""