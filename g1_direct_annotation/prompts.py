PROMPT_VERSION = "caption_attributes_v3"

CAPTION_FEW_SHOTS = """
Caption style examples only — never copy their gender, clothing, or accessories into the current image:
- A middle-aged man with short, straight black hair wears a solid blue short-sleeve shirt, beige knee-length shorts, and slippers in a casual style.
- A young adult female with an average build and long black hair in a bun wears a solid beige short-sleeve T-shirt and a long white skirt.
- An adult male with short straight black hair wears a solid black short-sleeve T-shirt, white shorts, and slippers. He holds a phone and wears eyeglasses.
""".strip()


def annotation_prompt() -> str:
    return (
        "Analyze the visible person in the image carefully and return only the requested JSON. "
        "Write a factual 20–35 word English caption in 1–3 short sentences, modeled on a "
        "person-annotation dataset. It is person-centric: do not describe the floor, background, "
        "scene, walking, standing, or other activity. Include every clearly visible, material descriptor in this "
        "order when available: age presentation, gender presentation, body build, hair length/"
        "texture/style/color, upper garment type/sleeves/color/pattern, lower garment type/color, "
        "footwear, bag/accessories/carried objects, and clothing style. Do not use a vague 'person' "
        "caption when a visible gender or age presentation can safely be stated. Omit uncertain "
        "claims rather than guessing. Keep the caption consistent with the returned attributes. "
        "Do not infer identity, relationships, location, intent or hidden details. "
        "Use exact attribute codes from the JSON Schema and return all fields. "
        "For a single-label field choose an allowed code or null when not visible. "
        "For multi-label fields return every visible allowed code. Use [] only when "
        "the region is visible and none apply, or null when it cannot be determined. "
        "Use unknown and none only where the schema permits them.\n\n"
        + CAPTION_FEW_SHOTS
    )
