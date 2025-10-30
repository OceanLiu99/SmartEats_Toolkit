# schema placeholder
ReRankSchema = {
  "name": "TopPicks",
  "schema": {
    "type": "object",
    "properties": {
      "choices": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "restaurant_id": {"type": "string"},
            "restaurant_name": {"type": "string"},
            "estimated_cost_per_person": {"type": "number"},
            "justification": {"type": "string"},
            "match_reasons": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
          },
          "required": [
            "restaurant_id","restaurant_name","justification","match_reasons","confidence"
          ],
          "additionalProperties": False
        }
      }
    },
    "required": ["choices"],
    "additionalProperties": False
  },
  "strict": True
}