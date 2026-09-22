PROMPT_VERSION = "caption_attributes_v1"


def annotation_prompt() -> str:
    return (
        "Analyze the visible person in the image and return only the requested JSON. "
        "Write one concise factual English caption describing only visible appearance, "
        "clothing and carried items. "
        "Do not infer identity, relationships, location, intent or hidden details. "
        "Use exact attribute codes from the JSON Schema and return all fields. "
        "For a single-label field choose an allowed code or null when not visible. "
        "For multi-label fields return every visible allowed code. Use [] only when "
        "the region is visible and none apply, or null when it cannot be determined. "
        "Use unknown and none only where the schema permits them."
    )
