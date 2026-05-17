import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from duckduckgo_search import DDGS
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Prophet Hacks Superforecaster - Base Rate Engine")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "openai/gpt-4o-mini" 

class EventInput(BaseModel):
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: Optional[str] = None
    description: Optional[str] = None
    category: str
    rules: str
    close_time: str
    outcomes: List[str]

def get_live_context(query: str) -> str:
    """Advanced AI-native search using Tavily."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("Warning: TAVILY_API_KEY not found, returning no data.")
        return "No data."
        
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": tavily_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": 5
    }
    
    try:
        response = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        context_str = f"--- TAVILY AI ANSWER ---\n{data.get('answer', 'No direct answer generated.')}\n\n"
        context_str += "--- SOURCE RESULTS ---\n"
        
        for i, res in enumerate(data.get("results", [])):
            context_str += f"[{i+1}] {res.get('title')}: {res.get('content')}\n\n"
            
        return context_str
    except Exception as e:
        print(f"Tavily Search failed: {e}")
        return "No data."

def call_openrouter(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY")
        
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a dual-engine agent: a strict forensic text parser AND an expert historical odds-maker. You output valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {response.text}")
        
    return response.json()["choices"][0]["message"]["content"]

SUPERFORECASTER_PROMPT = r"""
You are an elite dual-engine forecaster optimizing for Brier Score. 

EVENT TO INVESTIGATE:
- Title: {title}
- Category: {category}
- Allowed Outcomes: {outcomes}

LIVE SEARCH CONTEXT:
---
{search_context}
---

STEP 1: TEXT VERIFICATION
Read the Search Context. Is the EXACT, official winner of the 2026 event explicitly announced in the text?
- If YES: Set "status" to "CONFIRMED", and set "identified_winner" to the exact string match.
- If NO, or if the text is from a past year/different event: Set "status" to "UNKNOWN".

STEP 2: BASE RATE GENERATION (THE ODDS MAKER)
You must generate an educated probability distribution based on your vast pre-2024 internal knowledge of these entities. 
- Do NOT use a flat uniform distribution (e.g. 50/50).
- Identify the historical heavyweights, incumbents, or #1 seeds. 
- For binary events: Assign ~0.60 to the historical favorite and ~0.40 to the underdog.
- For leagues/multi-outcome: Assign ~0.30 to the biggest historical heavyweight, ~0.20 to the runner-up, and distribute the rest among the underdogs. 

REQUIRED JSON OUTPUT:
{{
  "investigation": "Brief explanation of search findings.",
  "status": "CONFIRMED", 
  "identified_winner": "Exact String or None",
  "base_rates": [
    {{"market": "Outcome 1", "probability": 0.60}},
    {{"market": "Outcome 2", "probability": 0.40}}
  ]
}}
"""

@app.post("/predict")
async def predict(event: EventInput):
    search_query = event.title
    search_context = get_live_context(search_query)
    
    prompt = SUPERFORECASTER_PROMPT.format(
        title=event.title,
        category=event.category,
        outcomes=json.dumps(event.outcomes),
        search_context=search_context
    )
    
    try:
        llm_raw_response = call_openrouter(prompt)
        investigation = json.loads(llm_raw_response)
        
        status = investigation.get("status", "UNKNOWN")
        identified = investigation.get("identified_winner", "None")
        base_rates = investigation.get("base_rates", [])
        
        probabilities = []
        num_outcomes = len(event.outcomes)
        
        if status == "CONFIRMED" and identified in event.outcomes:
            # TEXT PARSER WINS: We have hard proof. 0.95 to the winner.
            for out in event.outcomes:
                if out == identified:
                    probabilities.append({"market": out, "probability": 0.95})
                else:
                    probabilities.append({"market": out, "probability": round(0.05 / (num_outcomes - 1), 4)})
        else:
            
            base_rate_dict = {item["market"]: item["probability"] for item in base_rates}
            
            for out in event.outcomes:
                raw_prob = base_rate_dict.get(out, 1.0 / num_outcomes)
                
                
                if raw_prob > 0.60:
                    raw_prob = 0.60
                    
                probabilities.append({"market": out, "probability": raw_prob})

        
        total = sum(p["probability"] for p in probabilities)
        for p in probabilities:
            p["probability"] = round(p["probability"] / total, 4)

        
        final_payload = {"probabilities": probabilities}
        
        for prob_obj in probabilities:
            market_name = prob_obj["market"].lower()
            if market_name == "yes":
                final_payload["p_yes"] = prob_obj["probability"]
            elif market_name == "no":
                final_payload["p_no"] = prob_obj["probability"]
                
        if "p_yes" not in final_payload:
            final_payload["p_yes"] = probabilities[0]["probability"]
        if "p_no" not in final_payload:
            final_payload["p_no"] = round(1.0 - final_payload.get("p_yes", 0.5), 4)

        return final_payload
        
    except Exception as e:
        print(f"Error: {e}")
        num_outcomes = len(event.outcomes)
        fallback_prob = round(1.0 / num_outcomes, 4)
        return {
            "probabilities": [{"market": o, "probability": fallback_prob} for o in event.outcomes],
            "p_yes": fallback_prob, "p_no": fallback_prob
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)