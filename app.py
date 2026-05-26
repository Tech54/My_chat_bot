import os
import sqlite3
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# LangChain Agent Imports
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
app = Flask(__name__)

print("Initializing V6 Enterprise AI Agent...")

# --- 1. SQL DATABASE SETUP (Logging & Analytics) ---
def init_db():
    conn = sqlite3.connect('pratinik_logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_logs 
                 (id INTEGER PRIMARY KEY, user_msg TEXT, bot_reply TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# --- 2. VECTOR DATABASE SETUP ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./pratinik_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 3. ADVANCED AGENT ROUTING (Tools) ---
# Tool A: The Knowledge Searcher
@tool
def search_knowledge_base(query: str) -> str:
    """Use this tool to search the Pratinik Infotech database for answers about services, policies, or FAQs."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# Tool B: The Escalation Router
@tool
def transfer_to_human(reason: str) -> str:
    """Use this tool ONLY if the user explicitly asks to speak to a human, live agent, or if they are highly angry/frustrated."""
    return "SYSTEM_ESCALATION_TRIGGER: I will transfer you to a human agent immediately. Please hold."

tools = [search_knowledge_base, transfer_to_human]

# --- 4. THE AGENT BRAIN ---
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

system_prompt = """You are an advanced support agent for Pratinik Infotech. 
You have access to tools. 
If the user asks a question about the company, ALWAYS use the 'search_knowledge_base' tool first to find the answer. Do not guess.
If the user asks to speak to a human, use the 'transfer_to_human' tool.
Be professional and concise."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chat_history = []

# --- 5. FLASK ROUTES ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    user_input = request.form["msg"]
    
    try:
        # Route through the Agent
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        # --- THE BUG FIX ---
        answer = response["output"]
        # If Gemini returns a list of metadata blocks, extract just the text
        if isinstance(answer, list):
            answer = answer[0].get("text", str(answer))
        else:
            answer = str(answer)
        # -------------------
        
        chat_history.extend([HumanMessage(content=user_input), AIMessage(content=answer)])

        # Analytics: Secretly log the conversation to SQLite (Now safely a string!)
        conn = sqlite3.connect('pratinik_logs.db')
        c = conn.cursor()
        c.execute("INSERT INTO chat_logs (user_msg, bot_reply) VALUES (?, ?)", (user_input, answer))
        conn.commit()
        conn.close()

        return answer
        
    except Exception as e:
        print(f"Agent Error: {e}")
        return "I am experiencing a system glitch. Please try again."
    
@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    global chat_history
    chat_history = []
    return "Memory Cleared"

# --- 6. ADMIN DASHBOARD ROUTES ---
@app.route("/admin")
def admin_dashboard():
    # Fetch latest analytics logs
    conn = sqlite3.connect('pratinik_logs.db')
    c = conn.cursor()
    c.execute("SELECT * FROM chat_logs ORDER BY timestamp DESC LIMIT 50")
    logs = c.fetchall()
    conn.close()
    return render_template("admin.html", logs=logs)

@app.route("/admin/add", methods=["POST"])
def admin_add_faq():
    # Dynamically inject new knowledge without restarting the server!
    q = request.form["question"]
    a = request.form["answer"]
    combined_text = f"Question: {q}\nAnswer: {a}"
    
    # Add directly to the live Chroma database
    vectorstore.add_texts(texts=[combined_text])
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)
