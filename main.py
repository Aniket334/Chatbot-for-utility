from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, RootModel
from typing import Dict, Any, List
import pandas as pd 
import plotly.express as px
import json
import numpy as np # <-- NEW: Import NumPy for type checking

# Assuming these files are in your 'modules' directory
from modules import llm_router, db_manager, forecasting_engine 

# --- Custom JSON Encoder for NumPy/Pandas Fix ---

class CustomEncoder(json.JSONEncoder):
    """
    Handles serialization of NumPy/Pandas objects that the standard json.JSONEncoder misses.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert NumPy integer/float to native Python int/float
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Convert NumPy array to Python list
        return json.JSONEncoder.default(self, obj)

# --- Configuration (Inject Custom Encoder) ---
# We configure the built-in json module to use our custom encoder
# This ensures that when Plotly generates its JSON, it uses the correct serialization.
json_dumps = lambda obj: json.dumps(obj, cls=CustomEncoder)
# If using FastAPI's built-in JSONResponse, we might need to patch the app's default encoder, 
# but injecting it into the plot helper is the safest bet for the plot JSON.

# Initialize FastAPI app
app = FastAPI(title="GridOps Backend API")

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic V2 Models ---

class DataFrameRow(RootModel):
    root: Dict[str, Any]

class DataFrameData(BaseModel):
    rows: List[DataFrameRow]

class ChatRequest(BaseModel):
    message: str
    model_type: str 

class ChatResponse(BaseModel):
    role: str = "assistant"
    type: str = "data" 
    content: str
    intent: str
    data: DataFrameData = None
    sql: str = None
    insight: Dict[str, Any] = None
    plot_json: str = None 


# --- Plotly Helper ---
def generate_plotly_json(df: pd.DataFrame, insight: Dict[str, Any]) -> str | None:
    vt = insight.get("visualization_type")
    x_col = insight.get("x_column")
    y_col = insight.get("y_column")
    
    if df.empty or not x_col or not y_col:
        return None
        
    if vt == "line":
        fig = px.line(df, x=x_col, y=y_col, template="plotly_dark")
    elif vt == "bar":
        fig = px.bar(df, x=x_col, y=y_col, template="plotly_dark")
    else:
        return None 
    
    # Use the CustomEncoder when dumping the Plotly figure dict to JSON
    # This is the line that required the fix!
    return json.dumps(fig.to_dict(), cls=CustomEncoder)


# --- API ENDPOINTS ---

@app.get("/api/health")
def get_health():
    """Simple check to ensure the backend is running."""
    return {"status": "ok", "service": "GridOps LLM Router"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    user_query = request.message
    model_type = request.model_type

    try:
        # 1. Intent Classification
        intent = llm_router.classify_intent(user_query, model_type=model_type)
        
        df = pd.DataFrame()
        generated_sql = ""
        insight = {}
        
        # 2. Routing Logic
        if intent == "REVENUE_FORECAST":
            
            # Call Forecasting Engine
            df, msg = forecasting_engine.predict_revenue_for_date(user_query)
            
            if df.empty:
                return ChatResponse(
                    type="error",
                    content=f"Forecasting Failed: {msg}",
                    intent=intent
                )
            
            insight = {"summary": msg, "visualization_type": "line", "x_column": "date_time", "y_column": "forecasted_revenue"}

        else: # SQL_QUERY
            
            # 2a. Generate SQL
            generated_sql, raw_llm_response = llm_router.generate_sql(user_query, model_type=model_type)
            
            # Check for LLM errors 
            if generated_sql.startswith("-- SYSTEM ERROR"):
                 return ChatResponse(
                    type="error",
                    content=generated_sql.replace("-- SYSTEM ERROR: ", ""),
                    intent=intent
                )

            # 2b. Execute SQL
            try:
                df = db_manager.run_select_query(generated_sql)
            except Exception as e:
                return ChatResponse(
                    type="error",
                    content=f"SQL Execution Failed: {str(e)}",
                    intent=intent
                )

            # 2c. Analyze Data
            insight = llm_router.analyze_data(df, user_query, model_type=model_type)
        
        # --- Final Response Construction ---
        
        # Convert DataFrame to list of dictionaries for JSON compatibility
        # Note: df.to_dict('records') also produces NumPy types, but Pydantic/RootModel handles 
        # those conversions implicitly when possible. However, the custom encoder helps guarantee safety.
        data_rows = df.to_dict('records') if not df.empty else []
        
        # --- NEW: Generate Plotly JSON using the fixed helper ---
        plot_output_json = generate_plotly_json(df, insight)

        return ChatResponse(
            type="data",
            content=insight.get("summary", "Data retrieved successfully."),
            intent=intent,
            data=DataFrameData(rows=data_rows),
            sql=generated_sql,
            insight=insight,
            plot_json=plot_output_json
        )

    except Exception as e:
        # Catch unexpected Python errors
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# --- RUN INSTRUCTIONS ---

# To run the FastAPI server, use your terminal:
# uvicorn main:app --reload --port 8000