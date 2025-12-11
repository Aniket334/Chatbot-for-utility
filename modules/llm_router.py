import requests
import json
import re
from datetime import datetime

# --- CONFIGURATION ---

# 1. LOCAL CONFIG (Ollama)
OLLAMA_API = "http://localhost:11434/api/generate"
# Make sure you have pulled this model in terminal: ollama pull gemma3:12b
LOCAL_MODEL = "gemma3:12b"  

# 2. CLOUD CONFIG (Groq)
# Replace with your actual Groq API Key
GROQ_API_KEY = "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" 
CLOUD_API_URL = "https://api.groq.com/openai/v1/chat/completions"
CLOUD_MODEL = "llama-3.3-70b-versatile"

# --- ROUTER FUNCTIONS ---

def call_llm(prompt, model_type="local"):
    """
    Routes the request to either Local (Ollama) or Cloud (Groq).
    """
    if model_type == "cloud":
        return call_cloud_api(prompt)
    else:
        return call_local_api(prompt)

def call_local_api(prompt):
    """
    Calls Ollama running locally.
    """
    try:
        payload = {
            "model": LOCAL_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        r = requests.post(OLLAMA_API, json=payload, timeout=300)
        return r.json().get('response', 'Error: No response from Ollama')
    except Exception as e:
        return f"Local Connection Error: {e}"

def call_cloud_api(prompt):
    """
    Calls Groq API (Cloud).
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    payload = {
        "model": CLOUD_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    try:
        r = requests.post(CLOUD_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except requests.exceptions.HTTPError as e:
        return f"Cloud API Error ({r.status_code}): {r.text}"
    except Exception as e:
        return f"Cloud Connection Error: {e}"

# --- APP LOGIC FUNCTIONS ---

def classify_intent(query, model_type="local"):
    """
    FIXED: Uses strict JSON parsing to correctly classify intent,
    preventing false positives from messy LLM output.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
    You are an intent classifier.
    Context: Today is {current_date}.
    1. If query is about REVENUE AND a future date -> JSON: {{"intent": "REVENUE_FORECAST"}}
    2. Else -> JSON: {{"intent": "SQL_QUERY"}}
    Output JSON ONLY. Query: {query}
    """
    
    resp = call_llm(prompt, model_type)
    
    try:
        # 1. Use regex to find and strip the JSON object
        json_match = re.search(r"\{.*\}", resp.strip(), re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # 2. Strictly check the intent key value
            if data.get("intent") == "REVENUE_FORECAST":
                return "REVENUE_FORECAST"
        
    except json.JSONDecodeError:
        # If parsing fails, we continue and default to SQL_QUERY
        pass

    # Default to SQL_QUERY if parsing failed, or the intent was anything else.
    return "SQL_QUERY"

def generate_sql(query, model_type="local"):
    """Generates SQL based on the user query."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_day = datetime.now().strftime("%A")
    
    prompt = f"""
    Role: Expert SQL Generator.
    Schema: users (username TEXT, meter_id INTEGER), meter_loads (meter_id INTEGER, date_time TEXT, forecasted_load REAL)
    Context: Today is {current_time} ({current_day}).
    Task: Convert to SQLite SQL. Return ONLY SQL. No markdown.
    Query: {query}
    """
    
    resp = call_llm(prompt, model_type)
    
    # Safety Check: If the API returned an error message, don't execute it as SQL
    if "Error" in resp and ("Connection" in resp or "API" in resp):
        return f"-- SYSTEM ERROR: {resp}", resp

    # Clean SQL (remove markdown code blocks)
    sql_match = re.search(r"```sql\n(.*?)\n```", resp, re.DOTALL)
    if sql_match:
        sql = sql_match.group(1)
    else:
        sql = re.sub(r"```", "", resp).strip()
        
    return sql, resp

def analyze_data(df, query, model_type="local"):
    """Generates insights from the dataframe."""
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
    
    resp = call_llm(prompt, model_type)
    
    try:
        match = re.search(r"\{.*\}", resp, re.DOTALL).group(0)
        return json.loads(match)
    except:
        return {"summary": "Analysis failed or raw text returned.", "visualization_type": "table", "raw": resp}