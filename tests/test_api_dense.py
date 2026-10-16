import urllib.request
import json

url = "http://127.0.0.1:8000/api/recommendations/user_alpha?limit=4"
req = urllib.request.urlopen(url)
print("Status Code:", req.status)
data = json.loads(req.read().decode("utf-8"))
print(f"Received {len(data)} recommendations:")
for item in data:
    title = item.get("title")
    cat = item.get("category")
    pct = item.get("match_percentage")
    score = item.get("similarity_score")
    print(f" - {title} [{cat}] | Match: {pct}% (score: {score})")
