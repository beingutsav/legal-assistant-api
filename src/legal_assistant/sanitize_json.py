import json

def sanitize_to_raw_json(input_string):
    """
    Removes any markdown formatting (like ```json ... ```) and returns clean JSON in JSON format.

    Args:
        input_string (str): The input string containing JSON-like content.

    Returns:
        str: The sanitized JSON string (properly formatted with double quotes).
    """
    try:
        # Remove the markdown backticks if present
        if input_string.startswith("```") and input_string.endswith("```"):
            # Strip backticks and optional `json` keyword
            input_string = input_string.strip("```json").strip("```").strip()

        # Convert the sanitized string into a proper Python dictionary
        sanitized_dict = json.loads(input_string)

        # Convert the Python dictionary back to a proper JSON string
        sanitized_json = json.dumps(sanitized_dict, indent=2)  # Indented for better readability
        return sanitized_json
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

# Example usage
input_string = """```json
{
  "isNewResearchRequired": false,
  "responseToUser": "As mentioned previously, anticipatory bail in murder cases is possible but rare. The court will consider the evidence, your conduct, and the public interest. The *Mohammad Rafi* case highlights the importance of your conduct."
}
```"""

sanitized_json = sanitize_to_raw_json(input_string)
if sanitized_json:
    print("Sanitized JSON:\n", sanitized_json)
