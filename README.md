# Data Center Weather Agent v3

A production-grade multi-agent LangGraph system that provides secure, intelligent weather forecasting for data center locations using the Model Context Protocol (MCP).

**Architecture**: LangGraph Multi-Agent Supervisor Pattern with 7 specialized agents, 3-tier memory system, and fine-tuned ML classifiers for cost-efficient inference.

**Key Features**:
- 🤖 **Fine-Tuned Classifiers**: Gemma 3 270M (security) + DistilBERT (intent) - saves LLM credits
- 🧠 **Three-Tier Memory**: Context Management + SQLite Persistence + RAG Semantic Search
- 🔒 **Multi-Layer Security**: Real-time threat detection with adaptive learning
- ⚡ **Production Ready**: Python 3.13, MPS/CUDA support, graceful error handling

## Overview

This system securely answers: *"What is the weather forecast of the data center?"*

### Core Architecture

**7 Specialized Agents**:
1. **Security Agent**: Fine-tuned Gemma 3 270M classifier (268M params) - detects threats without LLM credits
2. **Intent Agent**: Fine-tuned DistilBERT classifier (66M params) - validates topic relevance
3. **IP Agent**: Discovers data center's public IP and geographic coordinates
4. **Weather Agent**: Retrieves current weather forecast data
5. **Safety Agent**: Validates output doesn't leak sensitive system information
6. **Learning Agent**: Analyzes security violations and adapts threat detection
7. **Supervisor Agent**: Orchestrates workflow and makes routing decisions

**3-Tier Memory System**:
- **Tier 1**: Context Management - Token tracking, compression, deduplication (28K limit)
- **Tier 2**: Persistent Memory - SQLite for agent decisions, patterns, query history
- **Tier 3**: RAG Semantic Search - ChromaDB + sentence-transformers for semantic retrieval

### Machine Learning Components

**Fine-Tuned Security Classifier**:
- Model: Gemma 3 270M-it (google/gemma-3-270m-it)
- Purpose: Detect prompt injection, credential extraction, role manipulation
- Training: PyTorch 2.10.0, transformers 5.0.0, MPS/CUDA support
- Inference: Local on-device, no API costs
- Location: `models/checkpoints/best_model.pt`

**Fine-Tuned Intent Classifier**:
- Model: DistilBERT base uncased
- Purpose: Classify weather/location vs off-topic queries
- Inference: Local on-device, no API costs
- Location: `models/intent_checkpoints/best_intent_model.pt`

**Training Pipeline**:
```bash
# Train security classifier
python3 -m src.training.train

# Train intent classifier
python3 -m src.training.train_intent
```

### Why LangGraph Multi-Agent Supervisor Architecture?

This implementation uses **LangGraph's supervisor pattern** with specialized agents:

- **Security First**: Multi-layer threat detection with adaptive learning from violations
- **Separation of Concerns**: Each agent has a single, well-defined responsibility
- **Production-Ready**: Enterprise-grade security logging and violation tracking
- **Adaptive Learning**: System improves threat detection from historical attacks
- **Best Practices**: Follows LangGraph 2026 multi-agent patterns
- **Maintainable**: Clear agent boundaries with typed state management

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│              Data Center Weather Multi-Agent System                  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │                  Supervisor Agent                          │      │
│  │              (Orchestrates all agents)                     │      │
│  └─────┬─────┬──────┬──────┬──────┬──────┬────────────────────┘      │
│        │     │      │      │      │      │                           │
│   ┌────▼──┐ ┌▼────┐┌▼───┐┌▼────┐┌▼────┐┌▼────────┐                   │
│   │Secur- │ │Int- ││IP  ││Wea- ││Safe-││Learning │                   │
│   │ ity   │ │ent  ││Agt ││ther ││ ty  ││  Agent  │                   │
│   │Agent  │ │Agt  ││    ││Agt  ││Agt  ││         │                   │
│   └───┬───┘ └──┬──┘└─┬──┘└──┬──┘└──┬──┘└────┬────┘                   │
│       │        │     │      │      │        │                        │
│       │        │     │      │      │        │                        │
│       │        │     ▼      ▼      │        ▼                        │
│       │        │  ┌───────────────┐│   ┌───────────┐                 │
│       │        │  │  MCP Server   ││   │security_  │                 │
│       │        │  │ ┌──────────┐  ││   │logs/      │                 │
│       │        │  │ │ipify     │  ││   │ insights  │                 │
│       │        │  │ │ip_to_geo │  ││   │ violations│                 │
│       │        │  │ │weather   │  ││   └───────────┘                 │
│       │        │  │ └──────────┘  ││                                 │
│       │        │  └───────────────┘│                                 │
│       └────────┴────────┴──────────┴──────────────────────────────   │
│         Threat Detection & Intent Classification Pipeline            │
└──────────────────────────────────────────────────────────────────────┘
```

### Components

#### 1. MCP Server (`server/`)
- **Technology**: Python, FastMCP, uvicorn
- **Port**: 8000 (SSE transport)
- **Tools**:
  - `ipify`: Get public IP address 
  - `ip_to_geo`: Convert IP to lat/lon (ip-api.com)
  - `weather_forecast`: Fetch weather data (Open-Meteo)
- **Features**: Input validation, error handling, structured logging

#### 2. LangGraph Multi-Agent System (`agent/`)
- **Technology**: Python, LangGraph (Supervisor Pattern), LangChain, Google Gemini
- **Architecture**: LangGraph Supervisor Pattern with 7 Specialized Agents
- **Agents**:
  - **Security Agent**: Threat detection with adaptive learning
  - **Intent Agent**: Query classification and scope enforcement
  - **IP Agent**: Location discovery via MCP tools
  - **Weather Agent**: Forecast retrieval and formatting
  - **Safety Agent**: Output validation and sanitization
  - **Learning Agent**: Pattern analysis and insights generation
  - **Supervisor Agent**: Workflow orchestration and routing
- **Features**: 
  - LLM fallback (Gemini to LongCat)
  - Real-time security logging
  - Automatic insights updates
  - Comprehensive violation tracking
  - Adaptive threat detection

## State Management

All data flows through a strongly typed `SharedState`:

```python
class SharedState(TypedDict):
    # User input
    question: str
    
    # Security analysis
    is_safe_query: bool | None
    security_threat_type: str | None
    
    # Intent classification
    is_weather_question: bool | None
    
    # IP and location data
    public_ip: str | None
    latitude: float | None
    longitude: float | None
    
    # Weather data
    weather_data: str | None
    
    # Final answer
    answer: str | None
    output_safe: bool | None
    
    # Routing and control
    next_agent: str | None
    current_agent: str | None
    
    # Error tracking
    error: str | None
    
    # Messages for LLM context
    messages: list[BaseMessage]
```

## Graph Structure

```
START
  ↓
┌─────────────────┐
│ Security Agent  │ → Analyze for threats
└────────┬────────┘
         │ [returns to supervisor]
         ↓
┌─────────────────┐
│ Supervisor      │ → Route based on security result
└────────┬────────┘
         │
         ├───[THREAT]────────────────────────────┐
         │                                       ↓
         │                             ┌─────────────────┐
         │                             │ Learning Agent  │
         │                             └────────┬────────┘
         │                                      ↓
         │                                     END
         │
         └───[SAFE]────────────────────────────┐
                                                ↓
                                     ┌─────────────────┐
                                     │  Intent Agent   │
                                     └────────┬────────┘
                                              ↓ [returns to supervisor]
                                     ┌─────────────────┐
                                     │ Supervisor      │
                                     └────────┬────────┘
                                              │
                       ├───[OFF-TOPIC]────────────────────┐
                       │                                   ↓
                       │                          END
                       │
                       └───[WEATHER RELATED]──────────────┐
                                                           ↓
                                                ┌─────────────────┐
                                                │   IP Agent      │
                                                └────────┬────────┘
                                                         ↓
                                                ┌─────────────────┐
                                                │ Weather Agent   │
                                                └────────┬────────┘
                                                         ↓
                                                ┌─────────────────┐
                                                │ Supervisor      │
                                                │ (Generate LLM   │
                                                │   Response)     │
                                                └────────┬────────┘
                                                         ↓
                                                ┌─────────────────┐
                                                │  Safety Agent   │
                                                └────────┬────────┘
                                                         ↓
                                                        END
```

### Agent Responsibilities

1. **Security Agent**: Analyzes input for malicious patterns, logs violations, updates insights
2. **Intent Agent**: Classifies if query is weather/location related
3. **IP Agent**: Discovers public IP and resolves to coordinates
4. **Weather Agent**: Fetches weather forecast for location
5. **Safety Agent**: Validates output doesn't leak system information
6. **Learning Agent**: Analyzes violation patterns and generates insights
7. **Supervisor Agent**: Orchestrates all routing and generates LLM responses

### Security Features

**Threat Detection**:
- Prompt extraction attempts
- Credential extraction attempts
- Role manipulation attempts
- Security bypass attempts
- Config inspection attempts

**Adaptive Learning**:
- Automatic violation logging (JSONL format)
- Real-time insights generation (JSON)
- Pattern analysis from historical attacks
- Enhanced threat detection using learned patterns

**Security Logs**:
```
security_logs/
├── violations_YYYYMMDD.jsonl    # Daily violation log
└── insights.json                 # Aggregated threat intelligence
```

### Supervisor Routing Logic

The supervisor makes intelligent routing decisions:

```python
async def route(self, state: SharedState) -> SharedState:
    # Security threat detected - refuse and learn
    if state.get("is_safe_query") is False:
        return route_to_learning_agent()
    
    # Query passed security - check intent
    if state.get("is_weather_question") is None:
        return route_to_intent_agent()
    
    # Off-topic query - refuse
    if state.get("is_weather_question") is False:
        return generate_refusal()
    
    # Normal workflow routing
    if not state.get("public_ip"):
        return route_to_ip_agent()
    
    if not state.get("weather_data"):
        return route_to_weather_agent()
    
    if not state.get("answer"):
        return generate_answer()
    
    # Validate output
    return route_to_safety_agent()
```

## Setup

### Prerequisites

- **Python 3.13** (recommended) - onnxruntime requires ≤3.13 for RAG features
- Python 3.14 not supported (onnxruntime incompatibility)
- Google Gemini API key (free tier available)
- Optional: LongCat API key (fallback)
- HuggingFace account (for downloading gated Gemma 3 model)

**Hardware**:
- 4GB+ RAM for model inference
- MPS (Apple Silicon) or CUDA GPU recommended for training
- 2GB disk space for models and dependencies

### Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/devrodrigoreis/datacenter_weather_agent_v3.git
   cd datacenter_weather_agent_v3
   ```

2. **Create Virtual Environment** (Python 3.13):
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
   Key dependencies:
   - PyTorch 2.10.0
   - transformers 5.0.0
   - chromadb 1.4.1 (RAG)
   - onnxruntime 1.23.2 (required for ChromaDB)
   - sentence-transformers 5.2.2 (embeddings)
   - langgraph, langchain, mcp

4. **Authenticate with HuggingFace** (for Gemma 3 model):
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```
   Enter your HuggingFace token (get it from https://huggingface.co/settings/tokens)

5. **Configure API Keys**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env`:
   ```bash
   GOOGLE_API_KEY=your_gemini_api_key_here
   LONGCAT_API_KEY=your_longcat_api_key_here  # Optional
   ```
   
   Get API keys:
   - Gemini: https://ai.google.dev/
   - LongCat: https://longcat.chat/ (optional)

6. **Download Pre-trained Models** (if not training from scratch):
   ```bash
   # Models should be in:
   # models/checkpoints/best_model.pt          (security classifier)
   # models/intent_checkpoints/best_intent_model.pt (intent classifier)
   ```

## Usage

### Start the System

You need two terminal windows:

**Terminal 1 - Start MCP Server**:
```bash
python3 -m server.main
```

Expected output:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Run Agent**:
```bash
python3 -m agent.main
```

Expected output:
```
============================================================
Data Center Weather Agent (Multi-Agent System v2)
With Context Management
============================================================

✓ Connected to MCP Server
[OK] LLM configured with fallback: Gemini -> LongCat
[OK] Context manager initialized (max: 28K tokens, threshold: 20K)
[OK] Memory store initialized (SQLite database)
[Initializing RAG memory system...]
[OK] RAG memory initialized (0 documents)
  [Loading fine-tuned security classifier...]
  [OK] Fine-tuned classifier loaded - saving LLM credits!
  [Loading fine-tuned intent classifier...]
  [OK] Fine-tuned intent classifier loaded - saving LLM credits!
[OK] Multi-agent system ready

Agents:
  - Security Agent (input validation)
  - Intent Agent (topic classification)
  - IP Agent (location discovery)
  - Weather Agent (forecast retrieval)
  - Safety Agent (output validation)
  - Learning Agent (pattern analysis)
  - Supervisor (orchestration)

NEW Features (Tier 1):
  ✓ Context compression (auto)
  ✓ Token tracking
  ✓ Message deduplication

NEW Features (Tier 2):
  ✓ Persistent memory (SQLite)
  ✓ Agent decision logging
  ✓ Pattern learning
  ✓ Query history

NEW Features (Tier 3):
  ✓ RAG semantic retrieval (ChromaDB)
  ✓ Similar threat detection
  ✓ Similar query matching
  ✓ Vector embeddings (all-MiniLM-L6-v2)
------------------------------------------------------------
```

### Example Session

```
============================================================
Data Center Weather Agent (Custom StateGraph)
============================================================

Connected to MCP Server
LLM configured with fallback: Gemini -> LongCat

Graph structure built successfully

Agent Ready! (Type 'quit' to exit)
------------------------------------------------------------

Enter your question: What is the weather forecast of the data center?

============================================================
EXECUTION TRACE
============================================================

[Step 1: IP Discovery]
  Tool: ipify
  Result: 174.162.142.78

[Step 2: Location Resolution]
  Tool: ip_to_geo
  Input: 174.162.142.78
  Result: 40.3495, -111.8998

[Step 3: Weather Retrieval]
  Tool: weather_forecast
  Input: lat=40.3495, lon=-111.8998
  Result: Temperature: 0.2 C, Windspeed: 2.2 km/h

[Step 4: Answer Generation]
  Generated answer successfully

============================================================
FINAL ANSWER
============================================================

The data center, located at 40.3495°N, 111.8998°W, currently has a 
temperature of 0.2°C and a wind speed of 2.2 km/h.

Enter your question: Where is the data center located?

[... processes IP and location lookup ...]

FINAL ANSWER

The data center is located at coordinates 40.3495°N, 111.8998°W 
(approximately in Utah, United States).

Enter your question: quit
```

## File Structure

```
datacenter_weather_agent_v3/
├── server/
│   ├── main.py              # MCP server with FastMCP
│   ├── tools.py             # Tool implementations (ipify, ip_to_geo, weather)
│   └── __init__.py
├── agent/
│   ├── main.py              # Multi-agent system (1427 lines)
│   ├── client.py            # MCP client wrapper
│   └── __init__.py
├── src/
│   ├── data.py              # DataLoaderFactory for training
│   ├── inference/
│   │   ├── security_checker.py    # Gemma 3 270M inference
│   │   └── intent_checker.py      # DistilBERT inference
│   ├── memory/
│   │   ├── context_manager.py     # Tier 1: Context compression
│   │   ├── agent_memory.py        # Tier 2: SQLite persistence
│   │   └── rag_memory.py          # Tier 3: ChromaDB semantic search
│   ├── model/
│   │   └── classifier.py          # PromptSecurityClassifier
│   └── training/
│       ├── train.py               # Security classifier training
│       └── train_intent.py        # Intent classifier training
├── models/
│   ├── checkpoints/
│   │   └── best_model.pt          # Trained security classifier
│   └── intent_checkpoints/
│       └── best_intent_model.pt   # Trained intent classifier
├── data/
│   └── raw/                       # Training datasets
├── memory/
│   └── agent_memory.db            # SQLite database (Tier 2)
├── security_logs/
│   ├── insights.json              # Threat intelligence
│   └── violations_YYYYMMDD.jsonl  # Daily violations
├── config.yaml                    # Model and training config
├── README.md                      # This file
├── ARCHITECTURE.md                # Deep technical documentation
├── .env.example                   # API key template
├── .env                           # Your API keys (create this)
└── requirements.txt               # Python dependencies
```

## Key Features

### 1. Robust Input Validation

**Server-Side**:
- IP address format validation (IPv4/IPv6)
- IPv4 octet range checking (0-255)
- Coordinate range validation (lat: -90 to 90, lon: -180 to 180)
- Type checking on all parameters

**Client-Side**:
- Response structure validation
- Non-empty content verification
- Runtime error detection

### 2. Comprehensive Error Handling

**Per-Node Error Handling**:
```python
async def get_ip_node(state, tools):
    try:
        # ... tool execution ...
        return success_state
    except Exception as e:
        logger.error(f"Error in get_ip_node: {e}")
        return error_state
```

**Conditional Routing to Error Node**:
- Any node failure routes to centralized error handler
- User-friendly error messages
- Agent remains responsive (doesn't crash)

### 3. LLM Fallback Strategy ### -> Extra Feature <-

Automatic failover on rate limits:
```python
gemini_llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")
longcat_llm = ChatOpenAI(model="LongCat-Flash-Chat", ...)
llm = gemini_llm.with_fallbacks([longcat_llm])
```

### 4. Detailed Execution Trace

Every operation is logged:
- Tool calls with inputs and outputs
- State transitions
- Error conditions
- LLM reasoning

Good for debugging and monitoring.

### 5. Clean Execution Trace

The agent produces clean, focused output by:
- Suppressing verbose library logs (httpx, google_genai)
- Showing only essential step information
- Automatically handling LLM fallbacks silently

User-facing trace shows only what matters:
```
[Step 1: IP Discovery]
  Tool: ipify
  Result: 174.162.142.78
```

Backend logging still captures details for debugging.

## Tool Calling Strategy

The agent enforces a **strict sequential workflow**:

1. **IP Discovery** (ipify)
   - No prerequisites
   - Must succeed before proceeding

2. **Location Resolution** (ip_to_geo)
   - Requires: `public_ip` from step 1
   - Validates IP format before calling
   - Parses lat/lon from response

3. **Weather Retrieval** (weather_forecast)
   - Requires: `latitude` and `longitude` from step 2
   - Validates coordinate ranges
   - Returns formatted weather string

4. **Answer Generation** (LLM)
   - Requires: All collected data
   - Synthesizes natural language response
   - Maintains conversation context

**Validation**: Each conditional edge verifies prerequisites exist before routing to the next node.

## Production Enhancements

### Enable Checkpointing (Conversation Memory) 
### Its an extra intend for productions environments. 

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# In build_graph():
async with SqliteSaver.from_conn_string("./checkpoints.db") as memory:
    graph = workflow.compile(checkpointer=memory)

# In main():
config = {"configurable": {"thread_id": "user-123"}}
result = await graph.ainvoke(state, config=config)
```

### Add Human-in-the-Loop

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Compile with interrupt
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["generate_answer"]  # Pause before final answer
)

# Resume after approval
graph.invoke(None, config=config)  # Continue from interrupt
```

### Multi-Agent Extension

Create specialized sub-agents:

```python
weather_specialist = build_weather_subgraph()
location_specialist = build_location_subgraph()

workflow.add_node("weather", weather_specialist)
workflow.add_node("location", location_specialist)
workflow.add_conditional_edges("start", route_to_specialist, {...})
```

## Training Fine-Tuned Classifiers

### Why Fine-Tune?

**Cost Savings**: Fine-tuned classifiers eliminate 2 LLM API calls per query:
- Security check: ~500 tokens saved
- Intent classification: ~300 tokens saved
- **Total savings**: ~800 tokens per query = significant cost reduction at scale

**Performance**: Local inference is faster and more reliable than API calls

### Security Classifier Training

```bash
# Train Gemma 3 270M for security classification
python3 -m src.training.train
```

**Configuration** (`config.yaml`):
```yaml
model:
  name: "google/gemma-3-270m-it"
  num_labels: 2
  dropout: 0.1
  max_length: 512

training:
  num_epochs: 10
  batch_size: 16
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 100
  gradient_accumulation_steps: 2
  fp16: true  # Mixed precision (CUDA only)
```

**Training Data** (data/raw/):
- `merged_train_prompts.json` - Combined dataset (preferred)
- `hf_train_prompts.json` - HuggingFace dataset
- `exploit_prompts.json` - Security exploits
- `train_prompts.json` - Synthetic data

**Output**:
- `models/checkpoints/best_model.pt` - Best validation checkpoint
- `models/checkpoints/final_model.pt` - Final model
- `models/checkpoints/training_history.json` - Metrics

### Intent Classifier Training

```bash
# Train DistilBERT for intent classification
python3 -m src.training.train_intent
```

**Configuration**:
```yaml
intent_model:
  name: "distilbert-base-uncased"
  num_labels: 2
  dropout: 0.1
  max_length: 128
```

**Output**:
- `models/intent_checkpoints/best_intent_model.pt`

### Hardware Requirements

- **Apple Silicon (MPS)**: Supported, automatic detection
- **NVIDIA GPU (CUDA)**: Supported with mixed precision (fp16)
- **CPU**: Fallback, slower training

**Training Time** (Apple M1 Pro):
- Security classifier: ~30 minutes (10 epochs, 16 batch size)
- Intent classifier: ~15 minutes (lighter model)

## Troubleshooting

### Server Connection Failed

**Symptom**: `connection refused` or `server not found`

**Solution**:
1. Verify server is running: `curl http://localhost:8000/sse`
2. Check no other process is using port 8000
3. Review server logs for startup errors

### LLM Rate Limit Errors

**Symptom**: `429 RESOURCE_EXHAUSTED`

**Solution**:
- Fallback to LongCat triggers automatically if configured
- Wait 60 seconds and retry
- Consider upgrading to Gemini paid tier

### Import Errors

**Symptom**: `ModuleNotFoundError`

**Solution**:
```bash
# Always run from project root:
cd /path/to/datacenter_weather_agent
python3 -m agent.main 
```

### Tool Validation Errors

**Symptom**: `Invalid IP address format` or `Invalid latitude`

**Solution**:
- Check server logs for detailed error
- Verify external APIs are accessible
- Test manually: `curl https://api.ipify.org?format=json`

## Performance

**Typical Metrics**:
- Cold start: 1-2 seconds (library loading)
- Tool execution: 3-5 seconds (3× HTTP requests)
- LLM synthesis: 0.5-1 second
- Total: ~5-8 seconds per query
- Memory: ~50MB runtime

**Bottlenecks**:
- External API latency (ipify, ip-api, open-meteo)
- LLM generation time - free tier api keys have a significant slower performance, please be patient
- Network conditions

## Limitations

1. **Sequential Execution**: Tools run in sequence (could be parallelized in some cases)
2. **Public IP Only**: Cannot determine location of internal/private IPs
3. **VPN Aware**: Location reflects VPN exit node if active
4. **No Retry Logic**: Failed tool calls immediately error (can add exponential backoff)
5. **Single Query**: Processes one question at a time (can add batch support)
6. **No Persistence**: State cleared between sessions (unless checkpointing enabled)

## Extending the Agent

### Add a New Tool

1. **Implement in `server/tools.py`**:
   ```python
   async def get_timezone(latitude: float, longitude: float) -> str:
       # Implementation
       return timezone_data
   ```

2. **Register in `server/main.py`**:
   ```python
   @mcp.tool()
   async def timezone(latitude: float, longitude: float) -> str:
       return await get_timezone(latitude, longitude)
   ```

3. **Add Node in `agent/main.py`**:
   ```python
   async def get_timezone_node(state, tools):
       tool = next(t for t in tools if t.name == "timezone")
       result = await tool.ainvoke({
           "latitude": state["latitude"],
           "longitude": state["longitude"]
       })
       return {**state, "timezone": result}
   
   workflow.add_node("get_timezone", get_timezone_node)
   workflow.add_edge("fetch_weather", "get_timezone")
   workflow.add_edge("get_timezone", "generate_answer")
   ```

### Modify Routing Logic

Change conditional edges to support alternative flows:

```python
def route_after_ip(state: AgentState):
    if state.get("error"):
        return "error"
    
    # New: Check if IP is internal/private
    ip = state.get("public_ip", "")
    if ip.startswith("192.168.") or ip.startswith("10."):
        return "handle_private_ip"  # New node
    
    return "resolve_location"
```

## Testing

### Unit Tests

Test individual nodes:

```python
import pytest
from agent.main import get_ip_node

@pytest.mark.asyncio
async def test_get_ip_node():
    mock_tools = [MockIPifyTool()]
    state = {"question": "test", "messages": []}
    
    result = await get_ip_node(state, mock_tools)
    
    assert result["public_ip"] is not None
    assert result["error"] is None
```

### Integration Tests

Test full graph execution:

```python
@pytest.mark.asyncio
async def test_full_workflow(mcp_server_running):
    graph = await build_graph(client, llm)
    initial_state = {...}
    
    final_state = await graph.ainvoke(initial_state)
    
    assert final_state["answer"] is not None
    assert "temperature" in final_state["answer"].lower()
```

### LangSmith Monitoring (From docs, last access Jan 30 2026)

Enable tracing for production debugging:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT="datacenter-weather"
```

Every execution will be traced in LangSmith dashboard.

## Security Considerations

1. **API Keys**: Never commit `.env` to version control
2. **Server Exposure**: MCP server has no authentication (local use only)
3. **Rate Limiting**: External APIs may ban abusive usage
4. **Input Validation**: Server validates all tool inputs
5. **Error Messages**: Avoid leaking sensitive data in logs

## Dependencies

### Core Framework
- **langgraph**: Graph-based agent framework
- **langchain**: LLM abstraction layer
- **langchain-google-genai**: Gemini integration
- **langchain-openai**: OpenAI-compatible APIs (LongCat)
- **mcp**: Model Context Protocol SDK
- **httpx**: Async HTTP client

### Machine Learning
- **torch**: 2.10.0 - PyTorch deep learning framework
- **transformers**: 5.0.0 - HuggingFace transformers
- **accelerate**: 1.12.0 - Training optimization
- **datasets**: 4.5.0 - Dataset management

### Memory & RAG (Tier 3)
- **chromadb**: 1.4.1 - Vector database
- **sentence-transformers**: 5.2.2 - Text embeddings
- **onnxruntime**: 1.23.2 - Required by ChromaDB

### Utilities
- **python-dotenv**: Environment variable management
- **uvicorn**: ASGI server for MCP
- **pydantic**: Data validation
- **pydantic-settings**: Settings management

**Python Version Requirements**:
- Python 3.13 recommended (full RAG support)
- Python 3.14 NOT supported (onnxruntime incompatibility)
- Python 3.12 supported

See `requirements.txt` for complete list with pinned versions.

## Learn More

- [LangGraph Documentation](https://langchain.com/docs/langgraph)
- [MCP Specification](https://modelcontextprotocol.io/)
- [StateGraph Tutorial](https://langchain.com/docs/langgraph/tutorials/state_graph)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)

## License

This is a demonstration project for personal purposes. No warranty provided.

---

**Built with**: LangGraph Custom StateGraph (Production Best Practices, January 2026)
