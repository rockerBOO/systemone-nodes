from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

Question = io.Custom("SYSTEMONE_QUESTION")


def parse_criteria(question_type, text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if question_type == "score":
        return lines
    criteria = {}
    for line in lines:
        key, sep, description = line.partition(":")
        if not sep:
            raise ValueError(f"Criteria line must be 'key: description', got {line!r}")
        criteria[key.strip()] = description.strip()
    return criteria


class SystemOneQuestion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SystemOneQuestion",
            display_name="SystemOne Question",
            category="SystemOne",
            inputs=[
                io.String.Input("name", default="question"),
                io.Combo.Input("type", options=["choice", "score", "noul"]),
                io.String.Input("instructions", multiline=True),
                io.String.Input(
                    "criteria",
                    multiline=True,
                    tooltip="choice: one 'key: description' per line. score: one level per line, lowest first. noul: optional 'true: ...' and 'false: ...' lines.",
                ),
            ],
            outputs=[Question.Output()],
        )

    @classmethod
    def execute(cls, name, type, instructions, criteria) -> io.NodeOutput:
        question = {"type": type, "instructions": instructions}
        parsed = parse_criteria(type, criteria)
        if parsed:
            question["criteria"] = parsed
        return io.NodeOutput((name, question))


class SystemOneExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SystemOneQuestion,
        ]


async def comfy_entrypoint() -> SystemOneExtension:
    return SystemOneExtension()
