import os
from pathlib import Path
from typing import List, Dict, TypedDict, Any, Annotated
from datetime import datetime
import json

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
import openai
import uvicorn
from fastapi.staticfiles import StaticFiles

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Get the directory containing the script
script_dir = Path(__file__).parent.absolute()

# Load environment variables from .env file in the same directory as the script
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Set up OpenAI API key
openai.api_key = api_key

class State(TypedDict):
    messages: List[Any]  # Can contain HumanMessage or AIMessage
    memory: Dict[str, Any]
    context: Dict[str, Any]
    emotional_state: str
    current_topic: str

class ChatBot:
    def __init__(self):
        self.conversations: Dict[str, State] = {}
        self.graph = self.create_graph()

    def create_graph(self) -> StateGraph:
        # Create the graph
        graph = StateGraph(State)

        # Define the memory processing node
        def process_memory_node(state: State) -> State:
            if len(state["messages"]) > 0:
                last_msg = state["messages"][-1]
                # Access content properly based on message type
                last_content = last_msg.content.lower() if isinstance(last_msg, (HumanMessage, AIMessage)) else str(last_msg).lower()
                
                topics = {
                    "stress": ["stress", "overwhelm", "pressure"],
                    "anxiety": ["anxiety", "worry", "nervous"],
                    "mood": ["feel", "feeling", "felt"],
                    "sleep": ["sleep", "tired", "rest"],
                    "relationships": ["friend", "family", "relationship"]
                }
                
                for topic, keywords in topics.items():
                    if any(keyword in last_content for keyword in keywords):
                        if topic not in state["memory"]:
                            state["memory"][topic] = []
                        state["memory"][topic].append({
                            "message": last_content,
                            "timestamp": datetime.now().isoformat()
                        })
            return state

        # Define the context analysis node
        def analyze_context_node(state: State) -> State:
            if len(state["messages"]) > 0:
                last_msg = state["messages"][-1]
                last_content = last_msg.content.lower() if isinstance(last_msg, (HumanMessage, AIMessage)) else str(last_msg).lower()
                
                # Update emotional state
                emotional_indicators = {
                    "stress": ["stressed", "overwhelmed", "pressure"],
                    "anxiety": ["anxious", "worried", "nervous"],
                    "sadness": ["sad", "down", "depressed"],
                    "positive": ["better", "good", "happy"]
                }
                
                for emotion, indicators in emotional_indicators.items():
                    if any(indicator in last_content for indicator in indicators):
                        state["emotional_state"] = emotion
                        break
                        
                # Update context with memory references
                state["context"]["references"] = []
                for topic, memories in state["memory"].items():
                    for memory in memories:
                        if any(word in memory["message"] for word in last_content.split()):
                            state["context"]["references"].append(memory["message"])
            return state

        # Define the LLM response generation node
        async def generate_response_node(state: State) -> State:
            try:
                client = openai.OpenAI()
                
                # Prepare context
                context_message = (
                    f"Current emotional state: {state['emotional_state']}\n"
                    f"Current topic: {state['current_topic']}\n"
                    f"Memory references: {json.dumps(state['context'].get('references', []))}"
                )
                
                messages = [
                    SystemMessage(content=(
                        "You are a supportive and empathetic mental health chatbot. "
                        "Use the provided context and memory to maintain conversation continuity. "
                        "Reference previous discussions when relevant to show understanding."
                    )),
                    SystemMessage(content=context_message)
                ]
                
                # Add recent conversation history
                for msg in state["messages"][-5:]:
                    if isinstance(msg, (HumanMessage, AIMessage)):
                        messages.append(msg)
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "system" if isinstance(m, SystemMessage) else 
                               "user" if isinstance(m, HumanMessage) else "assistant",
                        "content": m.content
                    } for m in messages],
                    max_tokens=150,
                    temperature=0.7
                )
                
                # Add response to state
                state["messages"].append(AIMessage(content=response.choices[0].message.content))
                
                return state
                
            except Exception as e:
                print(f"Error generating response: {e}")
                state["messages"].append(AIMessage(content="I apologize, but I'm having trouble generating a response right now."))
                return state

        # Add nodes to graph with unique names
        graph.add_node("process_memory_step", process_memory_node)
        graph.add_node("analyze_context_step", analyze_context_node)
        graph.add_node("generate_response_step", generate_response_node)

        # Define edges with new node names
        graph.set_entry_point("process_memory_step")
        graph.add_edge("process_memory_step", "analyze_context_step")
        graph.add_edge("analyze_context_step", "generate_response_step")

        return graph.compile()

    def initialize_session(self, session_id: str) -> None:
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "messages": [],
                "memory": {},
                "context": {},
                "emotional_state": "unknown",
                "current_topic": "general"
            }

    async def generate_response(self, user_input: str, session_id: str) -> str:
        self.initialize_session(session_id)
        state = self.conversations[session_id]
        
        # Add user message as HumanMessage
        state["messages"].append(HumanMessage(content=user_input))
        
        # Process through graph
        updated_state = await self.graph.ainvoke(state)
        
        # Update conversation state
        self.conversations[session_id] = updated_state
        
        # Return the last bot message content
        last_message = updated_state["messages"][-1]
        return last_message.content if isinstance(last_message, AIMessage) else str(last_message)

# Initialize FastAPI app
app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create chatbot instance
chatbot = ChatBot()

# Update the root route to serve the HTML file
@app.get("/")
async def read_root():
    return FileResponse('static/chat.html')

@app.post("/chat")
async def chat(message: str = Form(...), request: Request = None):
    session_id = f"{request.client.host}_{request.headers.get('user-agent', 'unknown')}"
    response = await chatbot.generate_response(message, session_id)
    return {"response": response}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)