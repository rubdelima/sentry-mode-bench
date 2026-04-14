from dataclasses import dataclass

base_prompt = """
You are a security analyst. Analyze the provided images and classify only the occurrences listed below.
"""

@dataclass
class SituationPrompt:
    name: str
    prompt: str


@dataclass
class GroupPrompt:
    name: str
    situations: list[SituationPrompt]

    def get_prompt(self) -> str:
        situations_text = "\n".join(
            f"{situation.name}: {situation.prompt}" for situation in self.situations
        )
        
        return (
            f"{base_prompt.strip()}\n\n"
            f"{situations_text}\n\n"
            "Return STRICT JSON ONLY (no explanation, no markdown, no extra text).\n"
            "Output format must be an object with class names as keys and boolean as values.\n"
            "You may include only detected classes with true to save tokens, for example:\n"
            "{\"person_trying_to_open_door\": true}\n"
            "If none is detected, return {}."
        )


group_1 = GroupPrompt(
    name="direct_human_interactions_and_vandalism",
    situations=[
        SituationPrompt(
            name="person_adjusting_car_wipers",
            prompt="If you observe someone adjusting or moving the vehicle's windshield wipers.",
        ),
        SituationPrompt(
            name="person_adjusting_mirror",
            prompt="If you observe someone adjusting or moving a vehicle mirror.",
        ),
        SituationPrompt(
            name="person_applying_material_on_vehicle",
            prompt="If you observe someone applying any material to the vehicle surface.",
        ),
        SituationPrompt(
            name="person_applying_stickers",
            prompt="If you observe someone putting stickers on the vehicle.",
        ),
        SituationPrompt(
            name="person_hitting_window",
            prompt="If you observe someone hitting a vehicle window with their body or hand.",
        ),
        SituationPrompt(
            name="person_kicking_car_side",
            prompt="If you observe someone kicking the side of the vehicle.",
        ),
        SituationPrompt(
            name="person_leaning_on_car",
            prompt="If you observe someone leaning their body on the vehicle.",
        ),
        SituationPrompt(
            name="person_lowering_car_antenna",
            prompt="If you observe someone lowering or pushing down the vehicle antenna.",
        ),
        SituationPrompt(
            name="person_pushing_car",
            prompt="If you observe someone pushing the vehicle with force.",
        ),
        SituationPrompt(
            name="person_raising_car_antenna",
            prompt="If you observe someone raising or pulling up the vehicle antenna.",
        ),
        SituationPrompt(
            name="person_removing_stickers",
            prompt="If you observe someone removing stickers from the vehicle.",
        ),
        SituationPrompt(
            name="person_running_fingers_on_window",
            prompt="If you observe someone sliding their fingers across a vehicle window.",
        ),
        SituationPrompt(
            name="person_scratching_car_with_keys",
            prompt="If you observe someone scratching the vehicle surface with keys or a key-like object.",
        ),
        SituationPrompt(
            name="person_sitting_on_car_hood",
            prompt="If you observe someone sitting on the vehicle hood.",
        ),
        SituationPrompt(
            name="person_throwing_object_at_vehicle",
            prompt="If you observe someone throwing any object at the vehicle.",
        ),
        SituationPrompt(
            name="person_touching_car",
            prompt="If you observe someone touching any part of the vehicle.",
        ),
    ],
)


group_2 = GroupPrompt(
    name="suspicious_behaviors",
    situations=[
        SituationPrompt(
            name="person_holding_crowbar",
            prompt="If you observe someone holding a crowbar near the vehicle.",
        ),
        SituationPrompt(
            name="person_holding_gun_near_vehicle",
            prompt="If you observe someone holding a gun near the vehicle.",
        ),
        SituationPrompt(
            name="person_holding_rock_near_vehicle",
            prompt="If you observe someone holding a rock near the vehicle.",
        ),
        SituationPrompt(
            name="person_holding_tire_or_spare_wheel",
            prompt="If you observe someone carrying or holding a tire or spare wheel near the vehicle.",
        ),
        SituationPrompt(
            name="person_holding_wheel_wrench",
            prompt="If you observe someone holding a wheel wrench near the vehicle.",
        ),
        SituationPrompt(
            name="person_looking_inside_vehicle",
            prompt="If you observe someone peering into the vehicle interior through windows.",
        ),
        SituationPrompt(
            name="person_opening_fuel_cap",
            prompt="If you observe someone opening or trying to open the vehicle fuel cap.",
        ),
        SituationPrompt(
            name="person_pointing_at_vehicle",
            prompt="If you observe someone pointing at the vehicle.",
        ),
        SituationPrompt(
            name="person_recording_or_photographing_vehicle",
            prompt="If you observe someone recording or photographing the vehicle.",
        ),
        SituationPrompt(
            name="person_staring_at_vehicle",
            prompt="If you observe someone repeatedly staring at the vehicle in a focused way.",
        ),
        SituationPrompt(
            name="person_using_flashlight_toward_vehicle",
            prompt="If you observe someone aiming a flashlight toward the vehicle.",
        ),
        SituationPrompt(
            name="person_walking_around_vehicle",
            prompt="If you observe someone walking around the vehicle.",
        ),
        SituationPrompt(
            name="person_wearing_balaclava_near_vehicle",
            prompt="If you observe someone wearing a balaclava near the vehicle.",
        ),
    ],
)


group_3 = GroupPrompt(
    name="theft_attempts_and_theft",
    situations=[
        SituationPrompt(
            name="person_breaking_window",
            prompt="If you observe someone trying to break a vehicle window.",
        ),
        SituationPrompt(
            name="person_inserting_something_through_window",
            prompt="If you observe someone trying to insert an object through a vehicle window.",
        ),
        SituationPrompt(
            name="person_removing_hubcap",
            prompt="If you observe someone removing or trying to remove a hubcap.",
        ),
        SituationPrompt(
            name="person_removing_mirror",
            prompt="If you observe someone removing or trying to remove a vehicle mirror.",
        ),
        SituationPrompt(
            name="person_removing_tire",
            prompt="If you observe someone removing or trying to remove a vehicle tire.",
        ),
        SituationPrompt(
            name="person_throwing_rock_at_the_vehicle",
            prompt="If you observe someone throwing a rock at the vehicle.",
        ),
        SituationPrompt(
            name="person_trying_to_open_door",
            prompt="If you observe someone trying to open a vehicle door without clear owner behavior.",
        ),
        SituationPrompt(
            name="person_trying_to_open_hood",
            prompt="If you observe someone trying to open the vehicle hood.",
        ),
        SituationPrompt(
            name="person_trying_to_open_trunk",
            prompt="If you observe someone trying to open the vehicle trunk.",
        ),
        SituationPrompt(
            name="person_using_scanner_on_vehicle",
            prompt="If you observe someone using a scanner-like device on or near the vehicle.",
        ),
        SituationPrompt(
            name="person_using_tool_on_vehicle_lock",
            prompt="If you observe someone using a tool on the vehicle lock area.",
        ),
    ],
)


ALL_GROUPS: list[GroupPrompt] = [group_1,group_2,group_3]