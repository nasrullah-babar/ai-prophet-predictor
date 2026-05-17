import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Prophet Hacks Superforecaster - Two-Pass Production Build V2")

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
    """Two-Pass Search to capture definitive facts AND future projections safely."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "No data."
        
    headers = {"Content-Type": "application/json"}
    context_str = ""
    
    
    payload_fact = {
        "api_key": tavily_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": 3
    }
    
    
    payload_forecast = {
        "api_key": tavily_key,
        "query": f"{query} current odds polling prediction consensus",
        "search_depth": "advanced",
        "include_answer": False,
        "max_results": 3
    }
    
    try:
        
        res1 = requests.post("https://api.tavily.com/search", headers=headers, json=payload_fact)
        if res1.status_code == 200:
            data1 = res1.json()
            context_str += f"=== PASS 1: HISTORICAL & CONCLUSIVE FACTS ===\n"
            context_str += f"Summary Answer: {data1.get('answer', 'None')}\n"
            for res in data1.get("results", []):
                context_str += f"- {res.get('title')}: {res.get('content')}\n"
    except Exception as e:
        print(f"Pass 1 search failed: {e}")
        
    try:
        
        res2 = requests.post("https://api.tavily.com/search", headers=headers, json=payload_forecast)
        if res2.status_code == 200:
            data2 = res2.json()
            context_str += f"\n=== PASS 2: LIVE FORECASTING & ODDS PROJECTIONS ===\n"
            for res in data2.get("results", []):
                context_str += f"- {res.get('title')}: {res.get('content')}\n"
    except Exception as e:
        print(f"Pass 2 search failed: {e}")
        
    return context_str if context_str else "No data found."

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
            {"role": "system", "content": "You are a quantitative forecasting system. You meticulously cross-reference query rules against textual records to extract accurate probabilities. You output strict JSON."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {response.text}")
        
    return response.json()["choices"][0]["message"]["content"]

SUPERFORECASTER_PROMPT = r"""
You are an expert prediction market tool optimizing for Brier Score verification.

=== MARKET DETAILS ===
Title: {title}
Description/Rules: {description}
Allowed Outcomes: {outcomes}

=== MERGED SEARCH contexts ===
{search_context}

=== FORENSIC ALGORITHM ===
Analyze the text buffers step-by-step:

1. "temporal_date_check": Identify the exact year or timeline of the events in the search context. Do they match the target market timeframe? (True/False)
2. "vote_tally_deduction": If evaluating a vote tally (e.g. Supreme Court or Senate records), map out the explicit division. Identify which number maps to the exact condition listed in the market description/rules.
3. "final_status": 
   - If a conclusive final winner/count is officially documented and matches your date check, output "CONFIRMED".
   - If the event is uncompleted/future, but explicit live betting odds, polling percentages, or expert models are documented, output "PROJECTION".
   - If text data is ambiguous, missing, or contradictory, output "UNKNOWN".
4. "verified_outcome": If status is CONFIRMED, output the exact string match from Allowed Outcomes. Otherwise, output "None".
5. "live_odds_weights": If status is PROJECTION, list the outcomes and their derived implied probabilities based on the live text data.

REQUIRED JSON OUTPUT FORMAT:
{{
  "temporal_date_check": true,
  "vote_tally_deduction": "Detailed textual breakdown of numerical counts to avoid inversion traps...",
  "final_status": "CONFIRMED",
  "verified_outcome": "Exact String Match from Allowed Outcomes",
  "live_odds_weights": [
    {{"market": "Outcome 1", "probability": 0.65}}
  ],
  "base_rates": [
    {{"market": "Outcome 1", "probability": 0.55}}
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
        identified = str(investigation.get("verified_outcome", "None")).strip()
        live_odds = investigation.get("live_odds_weights", [])
        base_rates = investigation.get("base_rates", [])
        
        probabilities = []
        num_outcomes = len(event.outcomes)
        
        
        def get_clean_mapping(target_list, weights_list, max_clamp):
            weights_dict = {str(item["market"]).strip(): item["probability"] for item in weights_list}
            output_list = []
            for item in target_list:
                prob = weights_dict.get(str(item).strip(), 1.0 / len(target_list))
                if prob > max_clamp:
                    prob = max_clamp
                output_list.append({"market": item, "probability": prob})
            return output_list

        if status == "CONFIRMED" and identified in event.outcomes:
            
            for out in event.outcomes:
                if out == identified:
                    probabilities.append({"market": out, "probability": 0.95})
                else:
                    probabilities.append({"market": out, "probability": round(0.05 / (num_outcomes - 1), 4)})
                    
        elif status == "PROJECTION" and live_odds:
           
            probabilities = get_clean_mapping(event.outcomes, live_odds, 0.75)
                
        else:
            
            probabilities = get_clean_mapping(event.outcomes, base_rates, 0.60)

        
        total = sum(p["probability"] for p in probabilities)
        for p in probabilities:
            p["probability"] = round(p["probability"] / total, 4)

        final_payload = {"probabilities": probabilities}
        for prob_obj in probabilities:
            market_name = str(prob_obj["market"]).lower()
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
        print(f"Error encountered: {e}")
        num_outcomes = len(event.outcomes)
        fallback_prob = round(1.0 / num_outcomes, 4)
        return {
            "probabilities": [{"market": o, "probability": fallback_prob} for o in event.outcomes],
            "p_yes": fallback_prob, "p_no": fallback_prob
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)