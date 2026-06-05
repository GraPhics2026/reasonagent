"""Reason Agent for edit and hybrid routes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .api_client import MLLMClient, extract_json_object
from .schemas import ReasonResult, ReasoningType, VQACheck


DEFAULT_REASON_PROMPT = Path("prompts/reason_system.txt")
VALID_REASONING_TYPES = {"physical", "temporal", "causal", "story"}


def load_reason_system_prompt(path: str | Path = DEFAULT_REASON_PROMPT) -> str:
    prompt_path = Path(path)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return (
        "You are the Reason Agent for counterfactual image editing. "
        "Return strict JSON with reasoning_chain, edit_prompt or scene_prompt, "
        "target_objects, and vqa_checklist."
    )


def run_reason_agent(
    image_path: str | Path,
    instruction: str,
    mode: Literal["edit", "hybrid"] = "edit",
    dry_run: bool | None = None,
    client: MLLMClient | None = None,
) -> ReasonResult:
    """Infer counterfactual visual outcome from an image and instruction."""

    mllm = client or MLLMClient()
    use_dry_run = (not mllm.configured) if dry_run is None else dry_run
    if use_dry_run:
        return heuristic_reason_result(instruction, mode=mode)

    system_prompt = load_reason_system_prompt()
    user_prompt = f"""Mode: {mode}
Hypothetical instruction:
{instruction.strip()}

Inspect the image carefully. Extract fine-grained visual cues before inferring the edit.
For edit mode, fill edit_prompt (English, image-editor ready).
For hybrid mode, fill scene_prompt (English, text-to-image ready) instead of edit_prompt.

IMPORTANT — hybrid mode camera & spatial composition:
Since T2I generates from scratch with no image reference, the camera perspective and
spatial layout of the original MUST be preserved. Extract these as visual_cues:
- Camera viewpoint: eye-level / low angle / high angle / overhead, and the viewing
  direction relative to the room (e.g. "eye-level view facing the window wall")
- Spatial positions: label every object by its position relative to frame edges —
  LEFT, RIGHT, CENTER, FOREGROUND, BACKGROUND (e.g. "window on the RIGHT wall",
  "sofa in the CENTER foreground", "plant on the windowsill at CENTER-RIGHT")
- Frame composition: what occupies each zone of the image — left third, center,
  right third, upper half, lower half

IMPORTANT — hybrid mode scene style extraction:
Include these style cues in your visual_cues list:
- Dominant color palette (e.g. "warm golden-hour tones", "cool blue-gray winter light")
- Lighting quality and direction (e.g. "soft diffused light from left window", "harsh midday sun")
- Surface materials and textures (e.g. "rustic wood grain", "matte ceramic", "glossy glass")
- Overall atmosphere (e.g. "serene and quiet", "bright and lively", "cozy indoor warmth")

CRITICAL — hybrid mode person identity preservation:
Since T2I generates from scratch with no image reference, person appearance is ENTIRELY
determined by the text description. T2I models have strong training bias toward
Caucasian/white faces and default to older/generic ages. To preserve the original
person's identity, you MUST describe:
- Race/ethnicity: "East Asian man", "Chinese man", "Black woman" — NOT just "man"/"woman"
- Skin tone: "light skin with warm undertone", "olive complexion", "dark brown skin"
- APPROXIMATE AGE: "young man in his mid-20s", "middle-aged woman around 40",
  "elderly man with wrinkles" — THIS IS CRITICAL. Without age, T2I defaults to older.
- Facial features: "round face, almond-shaped eyes", "broad nose, full lips"
- Hair: texture, color, style (e.g. "curly black hair", "straight dark brown hair")
- Body build: "lean build", "stocky build", "tall and slender"
These details MUST appear in both visual_cues AND scene_prompt. Without them, T2I will
generate a completely different person with default Caucasian features and wrong age.

CRITICAL — hybrid mode light behavior for photorealism:
T2I models need explicit descriptions of light behavior to generate photorealistic results.
Without these details, generated images look flat and lose ray-tracing quality. In your
scene_prompt and visual_cues, ALWAYS describe:
- Shadow sharpness: are shadows CRISP/SHARP (typical of direct sunlight) or SOFT/DIFFUSE
  (typical of overcast or indirect light)? e.g. "crisp diagonal shadows of the window
  frame cast across the floor"
- Specular highlights: which surfaces have bright highlights? e.g. "bright specular
  highlight on the glossy ceramic vase surface", "glare on the glass window pane"
- Light caustics/refraction: if glass or water is present, describe any light patterns
  e.g. "light caustics from the glass create patterned highlights on the table"
Without shadow sharpness and highlight descriptions, T2I images lose their 3D depth
and look flat/artificial. THIS IS THE #1 CAUSE OF LOST PHOTOREALISM.

CRITICAL — hybrid mode room corner geometry for indoor scenes:
For INDOOR SCENES, the T2I model MUST know what occupies each side of the frame.
Ambiguous left/right boundaries cause the model to invent wide open spaces where
there should be walls. In your scene_prompt and visual_cues, ALWAYS describe:
- Where is the nearest WALL CORNER? Does the left side show a corner where two walls
  meet, or does a single wall continue flat to the left edge?
- What does the adjacent wall look like? Color, texture, any features (decorations,
  windows, doors)?
- Example: "The leftmost part of the frame shows the corner where the window wall
  meets the adjacent wall at a 90-degree angle. The adjacent wall is painted the
  same soft light blue matte finish and recedes toward the left."
- Example: "The wall continues flat to the left edge of the frame with no corner
  visible — it is a single uninterrupted wall surface."
Without corner geometry, T2I imagines open space where there should be walls.
THIS IS THE #1 CAUSE OF WRONG SPATIAL LAYOUT IN INDOOR IMAGES.

CRITICAL for hybrid mode — scene_prompt is a STANDALONE T2I prompt:
It will be sent directly to an image generator WITHOUT the original image.
Therefore scene_prompt MUST describe the ENTIRE scene from scratch, NOT just the changed parts.

THE SCENE PROMPT MUST BE A FORENSIC-LEVEL DESCRIPTION OF THE ORIGINAL IMAGE,
with ONLY the instruction's changes applied. Think of it as: "describe this exact
photograph to an artist who cannot see it, then apply changes to specific elements."

RULES (violating any of these will produce bad results):
1. DESCRIBE EVERY SURFACE AND OBJECT in the original image precisely:
   - Floor: material, color, texture, pattern (e.g. "light gray tile floor with
     subtle grout lines" NOT generic "wooden floor")
   - Walls: color, finish, any visible features (e.g. "light blue matte wall with
     no decorations" NOT vague "smooth matte walls")
   - Window: size, frame material/color, number of panes, position on wall
   - Windowsill: depth, material, color, width relative to window
   - Plant pot: shape, size, color, material, exact position on windowsill
   - Plant: leaf shape, count, size, color, growth direction
   - Curtains: fabric, color, how they hang, which side of window
   - Any other objects visible (leaf, furniture, decorations)
   - **PEOPLE: race/ethnicity, skin tone, facial features, hair, body build**
2. DO NOT INVENT OBJECTS that are not in the original image. No "bedside lamp",
   no "framed picture on wall", no "breeze" unless they actually exist in the
   original image. If an object is not in the image, do NOT describe it.
3. ONLY CHANGE what the instruction asks to change:
   - For "become night": change ONLY the sky/lighting/atmosphere. The floor,
     walls, windowsill, pot, plant, leaf, curtains — SAME materials, colors,
     shapes, positions as the original.
   - For "split into two bouquets": change ONLY the count and arrangement of
     flowers. Keep the SAME flower type, color, and size for both bouquets.
     Table, light, background — SAME as original.
   - For "add many people": ADD people but keep ALL original elements identical.
4. The first sentence of scene_prompt must establish the camera perspective
   (e.g. "An eye-level photograph of a room...")
5. The scene_prompt must be 150-300 words of concrete visual description.
   No vague adjectives without concrete referents.

- The instruction may require two types of changes:
  a) PRESERVATIVE: add elements while keeping the original atmosphere.
     For these, keep lighting and atmosphere as-is.
  b) TRANSFORMATIVE: change time-of-day, weather, or global atmosphere.
     For these, describe ONLY the FINAL state — NEVER mention the original
     state or say "as the scene transitions". The scene_prompt should read
     as if the final state is the ONLY state that ever existed.
- BAD example (DO NOT narrate transition): "A bright sunlit room. As night
  falls, the room becomes darker..." — T2I reads "bright sunlit" and generates
  a daytime image regardless of what comes after.
- BAD example (DO NOT invent objects): "Warm light from a bedside lamp off-frame
  casts amber shadows" — if there is no lamp in the original image, T2I may
  hallucinate a lamp or mismatched lighting.
- BAD example (DO NOT be vague): "A serene room with smooth walls" — too short,
  T2I fills in its own imagination. The prompt must be 150+ words.

GOOD example for "sunny indoor room → night" (note the forensic detail of
the ORIGINAL room, preserved exactly, with ONLY lighting/sky changed):

"An eye-level photograph of a bright indoor space viewed from slightly left
of center, facing a wall with a large multi-pane window on the right side. The
wall is painted a soft light blue with a matte finish. The window has a thin
dark metal frame divided into rectangular panes. A shallow white windowsill runs
the full width of the window. On the left side of the windowsill sits a small
round ceramic pot in off-white, about 15cm tall, with a single green plant
growing upward — broad oval leaves with pointed tips. Sheer white curtains hang
from a rod above the window, one panel on the right side drawn slightly open.
The floor is covered in large square tiles in a pale cream color with faint
beige veins. In the foreground, slightly left of center on the floor, lies a
single small dried brown leaf. Natural daylight enters from the window on the
right, casting soft diagonal shadows of the window frame across the floor and
lower wall. Photorealistic, sharp focus, natural colors."

Now for the night version, change ONLY the lighting/sky/atmosphere — all
physical objects, materials, and positions remain identical to the above:

"An eye-level photograph of a room viewed from slightly left of center, facing
a wall with a large multi-pane window on the right side. The wall is painted a
soft light blue with a matte finish, now appearing darker in the low light. The
window has a thin dark metal frame divided into rectangular panes; through the
glass, a dark night sky dotted with faint stars is visible. A shallow white
windowsill runs the full width of the window. On the left side of the windowsill
sits a small round off-white ceramic pot, about 15cm tall, with a single green
plant — broad oval leaves with pointed tips, now softly silhouetted against the
dark window. Sheer white curtains hang from a rod above the window, one panel on
the right side drawn slightly open. The floor is covered in large square tiles
in a pale cream color with faint beige veins, now dimly visible. In the
foreground, slightly left of center on the tile floor, lies a single small dried
brown leaf. The room is illuminated by a soft, cool moonlight filtering through
the window, creating pale blue shadows across the floor and wall.
Photorealistic, sharp focus, natural colors."

Notice: the original scene's floor (pale cream tiles), pot (off-white ceramic,
round, small), plant (broad oval leaves, pointed tips), leaf (small, dried, brown,
floor foreground left-center), windowsill (shallow, white, full window width),
curtains (sheer white, rod-mounted, right panel partially open), wall (soft light
blue matte), and window (multi-pane, thin dark metal frame) are ALL preserved
exactly. ONLY the lighting, sky, and atmosphere changed.

This is the level of detail and fidelity required for EVERY scene_prompt.
If you cannot describe an element precisely because you cannot see it clearly
in the image, describe what you CAN see -- but never guess or invent.

Similarly, the vqa_checklist MUST include verification items for ALL parts of the
instruction -- not just the changes but also any newly requested elements. Also include
checks for spatial consistency (e.g. "Is the window on the RIGHT wall?", "Is the plant
still on the windowsill?").

Think of it this way: if someone reads your scene_prompt aloud, they should be able to
visualize the ENTIRE image, not guess what it looks like.
"""
    raw = mllm.chat_vision(image_path, user_prompt, system_prompt=system_prompt, temperature=0.2)
    data = extract_json_object(raw)
    return parse_reason_result(data, mode=mode, instruction=instruction)


def parse_reason_result(
    data: dict[str, object],
    mode: Literal["edit", "hybrid"],
    instruction: str,
) -> ReasonResult:
    chain = [str(item).strip() for item in data.get("reasoning_chain", []) if str(item).strip()]
    if not chain:
        chain = [f"Instruction received: {instruction.strip()}"]

    checklist_raw = data.get("vqa_checklist", [])
    checklist: list[VQACheck] = []
    if isinstance(checklist_raw, list):
        for item in checklist_raw:
            if isinstance(item, dict):
                question = str(item.get("q", "")).strip()
                if question:
                    checklist.append(
                        VQACheck(q=question, expected=str(item.get("expected", "yes")).strip() or "yes")
                    )
            elif isinstance(item, str) and item.strip():
                checklist.append(VQACheck(q=item.strip()))

    targets = _string_list(data.get("target_objects"))
    preserve = _string_list(data.get("preserve_objects"))
    visual_cues = _string_list(data.get("visual_cues"))
    physics = _string_list(data.get("physics_implications"))
    reasoning_type = _parse_reasoning_type(data.get("reasoning_type"))
    if reasoning_type is None:
        reasoning_type = infer_reasoning_type(instruction)

    edit_prompt = str(data.get("edit_prompt", "")).strip() or None
    scene_prompt = str(data.get("scene_prompt", "")).strip() or None

    if mode == "edit" and not edit_prompt:
        edit_prompt = fallback_edit_prompt(instruction, targets, physics, preserve)
    if mode == "hybrid" and not scene_prompt:
        scene_prompt = fallback_scene_prompt(instruction, targets, physics)

    if not checklist:
        checklist = default_vqa_checklist(reasoning_type, targets, physics)

    return ReasonResult(
        mode=mode,
        reasoning_chain=chain,
        vqa_checklist=checklist,
        edit_prompt=edit_prompt,
        scene_prompt=scene_prompt,
        target_objects=targets,
        reasoning_type=reasoning_type,
        visual_cues=visual_cues,
        physics_implications=physics,
        preserve_objects=preserve,
    )


def finalize_edit_prompt(result: ReasonResult) -> str:
    """Compose an editor-ready prompt with physics and preservation hints."""

    base = (result.edit_prompt or "").strip()
    if not base:
        return ""
    suffix_parts: list[str] = []
    if result.physics_implications:
        joined = "; ".join(result.physics_implications[:3])
        if joined.lower() not in base.lower():
            suffix_parts.append(f"Expected outcome: {joined}")
    if result.preserve_objects:
        joined = ", ".join(result.preserve_objects[:5])
        if joined.lower() not in base.lower():
            suffix_parts.append(f"Keep unchanged: {joined}")
    if not suffix_parts:
        return base
    return f"{base.rstrip('.')}. {' '.join(suffix_parts)}"


def build_reason_context(result: ReasonResult) -> str:
    parts: list[str] = []
    if result.reasoning_type:
        parts.append(f"Reasoning type: {result.reasoning_type}")
    if result.visual_cues:
        parts.append("Visual cues from source image: " + "; ".join(result.visual_cues[:4]))
    if result.physics_implications:
        parts.append("Expected physical/causal outcome: " + "; ".join(result.physics_implications[:4]))
    if result.preserve_objects:
        parts.append("Must preserve: " + ", ".join(result.preserve_objects[:5]))
    return "\n".join(parts)


def heuristic_reason_result(instruction: str, mode: Literal["edit", "hybrid"]) -> ReasonResult:
    text = instruction.strip()
    lower = text.lower()
    reasoning_type = infer_reasoning_type(text)

    if any(token in lower for token in ["ice", "冰"]):
        return _build_heuristic_result(
            mode=mode,
            text=text,
            reasoning_type="physical",
            visual_cues=["sharp-edged solid ice cubes", "container or plate surface", "ambient lighting"],
            physics_implications=["ice melts into liquid water", "edges soften and volume shrinks"],
            targets=["ice cubes"],
            preserve=["plate or container", "background", "unrelated objects"],
            edit_prompt=(
                "The ice cubes have partially melted into clear water on the plate. "
                "Edges are softened and smaller puddles of water are visible. "
                "Keep the plate and background unchanged. Realistic photo."
            ),
            checklist=[
                VQACheck(q="Are the ice cubes melted or visibly softened?", expected="yes"),
                VQACheck(q="Is there visible liquid water from melting?", expected="yes"),
                VQACheck(q="Are the plate and background unchanged?", expected="yes"),
            ],
        )

    if any(token in lower for token in ["跷跷板", "seesaw"]):
        return _build_heuristic_result(
            mode=mode,
            text=text,
            reasoning_type="physical",
            visual_cues=["large elephant and small squirrel on grass", "open field with sky"],
            physics_implications=[
                "elephant sits on one end of a seesaw",
                "elephant side is lower because it is heavier",
                "squirrel end is raised high",
            ],
            targets=["elephant", "squirrel", "seesaw"],
            preserve=["grass field", "sky", "overall scene style"],
            edit_prompt=(
                "Place the elephant and squirrel on a wooden seesaw in the grassy field. "
                "The elephant sits on one end with its side low near the ground; "
                "the squirrel sits on the other end lifted high. "
                "Preserve the lawn and sky."
            ),
            checklist=[
                VQACheck(q="Are the elephant and squirrel on a seesaw?", expected="yes"),
                VQACheck(q="Is the elephant's side lower than the squirrel's side?", expected="yes"),
                VQACheck(q="Are the grass and sky largely preserved?", expected="yes"),
            ],
        )

    if any(token in lower for token in ["盘子", "plate", "陶瓷", "ceramic", "blue", "蓝"]):
        return _build_heuristic_result(
            mode=mode,
            text=text,
            reasoning_type="physical",
            visual_cues=["plate surface material and color", "food or objects on the plate"],
            physics_implications=["plate material/color changes to the requested counterfactual"],
            targets=["plate"],
            preserve=["food on plate", "table", "background"],
            edit_prompt=(
                "Change the plate to the requested material/color while keeping the food, "
                "table, and background unchanged. Realistic photo."
            ),
            checklist=[
                VQACheck(q="Does the plate show the requested material or color change?", expected="yes"),
                VQACheck(q="Are food and background largely unchanged?", expected="yes"),
            ],
        )

    edit_prompt = fallback_edit_prompt(text, [], [], [])
    checklist = default_vqa_checklist(reasoning_type, [], [])
    return _build_heuristic_result(
        mode=mode,
        text=text,
        reasoning_type=reasoning_type,
        visual_cues=[],
        physics_implications=[f"visible outcome if: {text}"],
        targets=[],
        preserve=["unchanged background and unrelated objects"],
        edit_prompt=edit_prompt,
        checklist=checklist,
    )


def _build_heuristic_result(
    mode: Literal["edit", "hybrid"],
    text: str,
    reasoning_type: ReasoningType,
    visual_cues: list[str],
    physics_implications: list[str],
    targets: list[str],
    preserve: list[str],
    edit_prompt: str,
    checklist: list[VQACheck],
) -> ReasonResult:
    chain = [
        f"Dry-run heuristic ({reasoning_type} reasoning).",
        f"Instruction: {text}",
        "Converted into a structured edit plan.",
    ]
    if mode == "hybrid":
        return ReasonResult(
            mode=mode,
            reasoning_chain=chain,
            scene_prompt=fallback_scene_prompt(text, targets, physics_implications),
            vqa_checklist=checklist,
            target_objects=targets,
            reasoning_type=reasoning_type,
            visual_cues=visual_cues,
            physics_implications=physics_implications,
            preserve_objects=preserve,
        )
    return ReasonResult(
        mode=mode,
        reasoning_chain=chain,
        edit_prompt=edit_prompt,
        vqa_checklist=checklist,
        target_objects=targets,
        reasoning_type=reasoning_type,
        visual_cues=visual_cues,
        physics_implications=physics_implications,
        preserve_objects=preserve,
    )


def infer_reasoning_type(instruction: str) -> ReasoningType:
    lower = instruction.lower()
    if any(token in lower for token in ["sunset", "night", "morning", "later", "after hours", "snow", "winter",
                                         "日落", "夜晚", "时间", "雪", "冬天", "大雪", "下雪", "季节"]):
        return "temporal"
    if any(token in lower for token in ["land", "collide", "because", "导致", "因此", "interaction", "落在", "碰撞"]):
        return "causal"
    if any(token in lower for token in ["hidden", "secret", "story", "texture", "隐藏", "故事", "纹理"]):
        return "story"
    return "physical"


def default_vqa_checklist(
    reasoning_type: ReasoningType | None,
    targets: list[str],
    physics: list[str],
) -> list[VQACheck]:
    items = [VQACheck(q="Does the image reflect the requested counterfactual change?", expected="yes")]
    if physics:
        items.append(VQACheck(q=f"Is this visible outcome satisfied: {physics[0]}?", expected="yes"))
    if reasoning_type == "physical" and len(items) < 3:
        items.append(VQACheck(q="Are unrelated background elements largely preserved?", expected="yes"))
    if targets and len(items) < 3:
        items.append(VQACheck(q=f"Are the target objects/regions ({', '.join(targets[:3])}) edited as intended?", expected="yes"))
    return items[:4]


def fallback_edit_prompt(
    instruction: str,
    targets: list[str],
    physics: list[str],
    preserve: list[str],
) -> str:
    target_text = ", ".join(targets) if targets else "the relevant region"
    physics_text = physics[0] if physics else instruction.strip()
    preserve_text = ", ".join(preserve) if preserve else "all unrelated objects, layout, and lighting"
    return (
        f"Edit the image so this counterfactual holds: {physics_text}. "
        f"Focus changes on {target_text}. Preserve {preserve_text}."
    )


def fallback_scene_prompt(instruction: str, targets: list[str], physics: list[str]) -> str:
    """Generate a complete, standalone T2I scene description — not an edit instruction.

    This is used when the MLLM fails to produce a valid scene_prompt in hybrid mode.
    The prompt must describe the entire scene from scratch since T2I has no access
    to the original image.
    """
    outcome = physics[0] if physics else instruction.strip()
    target_text = ", ".join(targets) if targets else "relevant scene elements"
    parts = [
        f"A photorealistic image depicting {target_text}.",
        f"The visual change applied: {outcome}.",
        "All original subjects, their appearance, spatial layout, background, and lighting are preserved.",
        f"Context: {instruction.strip()}.",
        "Clear composition, highly detailed, professional photography.",
    ]
    return " ".join(parts)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _parse_reasoning_type(value: object) -> ReasoningType | None:
    raw = str(value or "").strip().lower()
    return raw if raw in VALID_REASONING_TYPES else None
