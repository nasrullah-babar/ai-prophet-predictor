import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Prophet Hacks Superforecaster - Ultimate Calibration Build")

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
            {"role": "system", "content": "You are a forensic data extraction agent. You break down text details into strict structured logic without missing fine print."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {response.text}")
        
    return response.json()["choices"][0]["message"]["content"]

SUPERFORECASTER_PROMPT = r"""
You are an expert quantitative forecaster optimizing for Brier Score verification.

=== MARKET DETAILS ===
Title: {title}
Description/Rules: {description}
Allowed Outcomes: {outcomes}

=== SEARCH DOCUMENTS ===
{search_context}

=== FORENSIC ALGORITHM ===
Fill out the following structural inspection fields step-by-step:

1. "temporal_date_check": Analyze the dates mentioned in the search text. Does the text explicitly refer to the correct year and timeframe of the target event? (True/False)
2. "direct_resolution_found": Identify if an official entity explicitly crowns a winner or provides an exact terminal count for the target question. (True/False)
3. "vote_breakdown_check": If this is a judicial or legislative vote, isolate the entire raw count string. Map exactly which side favored the target entity mentioned in the title/rules, and which side opposed it.
4. "final_status": Is the exact answer to the target market definitively proven by the text AND temporally verified for the correct year in Step 1? Select exactly one: "CONFIRMED" or "UNKNOWN".
5. "verified_winner": If status is CONFIRMED, provide the exact string match from the Allowed Outcomes list. If UNKNOWN, output "None".

=== PRE-2024 ODDS BASE RATES ===
Generate baseline odds for fallback based on pre-2024 prominence. Never use a flat uniform distribution. Give historical heavyweights or favorites a clear edge (~0.60 for binary favorites, ~0.35 for league heavyweights).

REQUIRED JSON OUTPUT FORMAT:
{{
  "temporal_date_check": true,
  "direct_resolution_found": true,
  "vote_breakdown_check": "Analyze majority vs minority splits...",
  "final_status": "CONFIRMED",
  "verified_winner": "Exact String Match from Allowed Outcomes list",
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
        description=event.description or event.rules,
        outcomes=json.dumps(event.outcomes),
        search_context=search_context
    )
    
    try:
        llm_raw_response = call_openrouter(prompt)
        investigation = json.loads(llm_raw_response)
        
        status = investigation.get("final_status", "UNKNOWN")
        identified = str(investigation.get("verified_winner", "None")).strip()
        base_rates = investigation.get("base_rates", [])
        
        probabilities = []
        num_outcomes = len(event.outcomes)
        
        if status == "CONFIRMED" and identified in event.outcomes:
            # Fact locked: 0.95 to the explicit match
            for out in event.outcomes:
                if out == identified:
                    probabilities.append({"market": out, "probability": 0.95})
                else:
                    probabilities.append({"market": out, "probability": round(0.05 / (num_outcomes - 1), 4)})
        else:
            # Fallback to bounded base rates
            base_rate_dict = {str(item["market"]).strip(): item["probability"] for item in base_rates}
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