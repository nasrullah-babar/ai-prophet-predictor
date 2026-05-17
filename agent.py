import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from duckduckgo_search import DDGS
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Prophet Hacks Superforecaster Agent FINAL")

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
    """Un-poisoned search: maximizes context gathering using raw, natural language queries."""
    context_str = ""
    with DDGS() as ddgs:
        try:
            
            news_results = [r for r in ddgs.news(query, max_results=5)]
            if news_results:
                context_str += "--- LATEST NEWS RESULTS ---\n"
                for i, res in enumerate(news_results):
                    context_str += f"[{i+1}] Date: {res.get('date', 'N/A')}\nTitle: {res.get('title')}\nSnippet: {res.get('body')}\n\n"
        except Exception:
            pass
            
        try:
            
            web_results = [r for r in ddgs.text(query, max_results=8)]
            if web_results:
                context_str += "--- STANDARD WEB RESULTS ---\n"
                for i, res in enumerate(web_results):
                    context_str += f"[{i+1}] Title: {res.get('title')}\nSnippet: {res.get('body')}\n\n"
        except Exception as e:
            print(f"Text search backup failed: {e}")
            
    return context_str if context_str else "No real-time search context available."

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
            {"role": "system", "content": "You are a forensic, mathematically disciplined prediction market analyst. You extract facts and output strict, valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {response.text}")
        
    res_json = response.json()
    return res_json["choices"][0]["message"]["content"]


SUPERFORECASTER_PROMPT = """
You are an elite, highly forensic prediction market analyst. Your objective is to extract the exact factual resolution to the provided event using the live search context.

Event Details:
- Title: {title}
- Allowed Outcomes: {outcomes}
- Rules: {rules}

Search Context:
{search_context}

Step-by-Step Execution:
1. "evidence_extraction": Meticulously read every snippet. Look for exact matches to the Allowed Outcomes. Identify if the event has concluded and a winner has been crowned, or if it is still ongoing.
2. "confidence_scoring": Rate the quality of the evidence (e.g., 'Definitive Winner Found', 'Partial Lead Found', 'No Relevant Data').
3. "probabilities": Assign probabilities based on this strict mathematical framework:
   - DEFINITIVE WINNER: If the search context explicitly confirms that an outcome has won, clinched, or officially resolved, assign that specific outcome 0.95. Distribute the remaining 0.05 equally among all other outcomes.
   - UNKNOWN / NO DATA: If the context does not explicitly name the confirmed winner, you MUST assign a perfectly equal, UNIFORM distribution to all outcomes (e.g., 0.50 each for 2, 0.05 each for 20). DO NOT GUESS a favorite if you lack proof. Guessing wrong incurs a massive Brier penalty.

You MUST respond strictly with a valid JSON object matching this structure:
{{
  "evidence_extraction": "Detailed analysis of the snippets...",
  "confidence_scoring": "Explanation of the assigned tier...",
  "probabilities": [
    {{"market": "Outcome 1", "probability": 0.50}},
    {{"market": "Outcome 2", "probability": 0.50}}
  ]
}}
Ensure the "market" labels EXACTLY match the provided {outcomes} list.
"""

@app.post("/predict")
async def predict(event: EventInput):
    
    search_query = event.title
    
    search_context = get_live_context(search_query)
    
    prompt = SUPERFORECASTER_PROMPT.format(
        title=event.title,
        rules=event.rules,
        outcomes=json.dumps(event.outcomes),
        search_context=search_context
    )
    
    try:
        llm_raw_response = call_openrouter(prompt)
        prediction_data = json.loads(llm_raw_response)
        
        if "probabilities" not in prediction_data:
            raise ValueError("Invalid structure returned by LLM")
            
        probs = prediction_data["probabilities"]
        total = sum(p["probability"] for p in probs)
        
        # Enforce exactly 1.0 sum
        if not (0.99 <= total <= 1.01):
            for p in probs:
                p["probability"] = round(p["probability"] / total, 4)
                
        # --- HACKATHON DAY 1 PATCH ---
        for prob_obj in prediction_data["probabilities"]:
            market_name = prob_obj["market"].lower()
            if market_name == "yes":
                prediction_data["p_yes"] = prob_obj["probability"]
            elif market_name == "no":
                prediction_data["p_no"] = prob_obj["probability"]
                
        if "p_yes" not in prediction_data:
            prediction_data["p_yes"] = prediction_data["probabilities"][0]["probability"]
        if "p_no" not in prediction_data:
            prediction_data["p_no"] = round(1.0 - prediction_data.get("p_yes", 0.5), 4)

        return prediction_data
        
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