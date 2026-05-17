import json
import requests
import time  


with open("resolved_events.json", "r") as f:
    events = json.load(f)

brier_scores = []
print(f"Backtesting agent on {len(events)} resolved events...\n")

for event in events:
    ticker = event["market_ticker"]
    title = event["title"]
    
    
    try:
        res = requests.post("http://localhost:8000/predict", json=event)
        prediction = res.json()
        
        
        probs = {p["market"]: p["probability"] for p in prediction["probabilities"]}
        resolved = event["resolved_outcome"]
        
        if resolved and "value" in resolved:
            actual_winners = resolved["value"] 
            
            
            event_brier = 0.0
            for outcome in event["outcomes"]:
                p_i = probs.get(outcome, 0.0)
                outcome_i = 1.0 if outcome in actual_winners else 0.0
                event_brier += (p_i - outcome_i) ** 2
            
            brier_scores.append(event_brier)
            print(f"🔹 {ticker} | Brier: {event_brier:.4f} | {title[:50]}...")
            
    except Exception as e:
        print(f"❌ Failed to get prediction for {ticker}: {e}")

    
    time.sleep(2.5) 

if brier_scores:
    avg_brier = sum(brier_scores) / len(brier_scores)
    print("\n" + "="*40)
    print(f"🎯 FINAL AGENT LOCAL BRIER SCORE: {avg_brier:.4f}")
    print("=(Perfect score is 0.0, completely wrong is 2.0)=")
    print("="*40)
else:
    print("No scores collected.")