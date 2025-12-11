import requests
import json
import re
from datetime import datetime

# --- CONFIGURATION ---
# Replace with your actual Groq API Key
GROQ_API_KEY = "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"  # Put your 'gsk_...' key here

# Groq uses OpenAI-compatible endpoints
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Best model for SQL & Logic on Groq
MODEL = "llama-3.3-70b-versatile" 

def call_groq(prompt, timeout=30):
    """
    Sends the prompt to Groq's ultra-fast API.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1, # Low temp for deterministic SQL
        "max_tokens": 1024
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status() 
        
        data = response.json()
        return data['choices'][0]['message']['content']
        
    except requests.exceptions.HTTPError as e:
        return f"API Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"Connection Error: {e}"

# --- APP LOGIC ---

def classify_intent(query):
    """Decides if the user wants historical data (SQL) or future predictions (Revenue)."""
    current_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
    You are an intent classifier for an Energy Analytics system. Analyze the user query carefully.
    
    Context: Today is {current_date}.
    
    1. If the query explicitly asks about **REVENUE** AND a future date/time (e.g., "next week's revenue", "forecasted revenue"), return JSON: {{"intent": "REVENUE_FORECAST"}}
    
    2. For all other queries, including:
       - Queries about **LOAD** (future or historical, e.g., "predicted load next week")
       - Queries about historical usage or user data.
       - Queries that do not mention REVENUE.
       
       Return JSON: {{"intent": "SQL_QUERY"}}
    
    Output JSON ONLY.
    Query: {query}
    """
    # CALL GROQ
    resp = call_groq(prompt)
    
    if "REVENUE_FORECAST" in resp: 
        return "REVENUE_FORECAST"
    return "SQL_QUERY"

def generate_sql(query):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_day = datetime.now().strftime("%A")
    
    prompt = f"""
    Role: Expert SQL Generator.
    Database Schema:
    - users (username TEXT, meter_id INTEGER)
    - meter_loads (meter_id INTEGER, date_time TEXT, forecasted_load REAL)
    
    Context: 
    - Today is {current_time} ({current_day}).
    - Dates in DB are formatted 'YYYY-MM-DD'.
    - For relative dates like "next Thursday", calculate the exact date based on Today.
    
    Task: Convert this query to executable SQLite SQL. Return ONLY the SQL code. No markdown.
    Query: {query}
    """
    
    # 1. Call Groq
    resp = call_groq(prompt)
    
    # 2. Safety Check for API Errors
    if "API Error" in resp or "Connection Error" in resp:
        return f"-- SYSTEM ERROR: {resp}", resp

    # 3. Clean SQL (Groq models are usually clean, but safety first)
    sql_match = re.search(r"```sql\n(.*?)\n```", resp, re.DOTALL)
    if sql_match:
        sql = sql_match.group(1)
    else:
        sql = re.sub(r"```", "", resp).strip()
        
    return sql, resp

def analyze_data(df, query):
    if df.empty: 
        return {"summary": "No data found matching your query.", "visualization_type": "table"}
    
    data_sample = df.head(5).to_dict()
    prompt = f"""
    Role: Data Analyst.
    Task: Analyze this data snippet and the user's query.
    Return a JSON object with:
    1. "summary": A 1-sentence insight.
    2. "visualization_type": "line", "bar", or "table".
    3. "x_column": The best column for X-axis (or null).
    4. "y_column": The best column for Y-axis (or null).
    
    Query: {query}
    Data Sample: {data_sample}
    JSON Output Only:
    """
    # CALL GROQ
    resp = call_groq(prompt)
    
    try:
        match = re.search(r"\{.*\}", resp, re.DOTALL).group(0)
        return json.loads(match)
    except:
        return {"summary": "Data retrieved successfully.", "visualization_type": "table", "raw": resp}