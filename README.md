# SystemOne Nodes

ComfyUI nodes that ask a SystemOne server typed questions about some text and branch the workflow on the answers — e.g. pick a LoRA from the prompt.

<img width="2119" height="1159" alt="Image" src="https://github.com/user-attachments/assets/3fce6860-075a-4f90-a0b7-9e729db5be38" />

## Install

Clone into `ComfyUI/custom_nodes/`, then install the package into ComfyUI's environment:

    cd ComfyUI && uv pip install --no-deps -e custom_nodes/systemone-nodes

Requires `requests` (already a ComfyUI dependency).

For a hosted server, set `TYPESAFE_API_KEY` in ComfyUI's environment. It is sent as a Bearer token and never stored in workflows.

## Nodes

- **SystemOne Question** — one question: `name`, `type` (choice / score / noul), `instructions`, `criteria`.
  - choice: one `key: description` per line
  - score: one level per line, lowest first (2–10 levels)
  - noul: optional `true: ...` / `false: ...` lines
- **SystemOne** — sends `state` and all connected questions in one request to `server_url` (default `http://localhost:8765/v1/systemone`).
- **SystemOne Answer** — picks one answer by `name`. Outputs `choice` (STRING), `index` (INT), `value` (FLOAT), `confidence` (FLOAT), `probabilities` (FLOAT list). Noul has no confidence and outputs 1.0.
- **SystemOne Threshold Switch** — `value >= threshold` outputs `on_true`, otherwise `on_false`. Only the selected branch runs.
- **SystemOne Choice Switch** — outputs `option{index}` (10 inputs). Only the selected option runs.

## Examples

- LoRA by name: make choice keys LoRA filenames → Answer `choice` → Convert String To Combo → LoraLoader `lora_name`.
- LoRA by branch: Answer `index` → Choice Switch over the MODEL outputs of several LoraLoaders.
- Confidence gate: Answer `confidence` → Threshold Switch (0.7) between the chosen branch and a default.
- Strength from score: Answer `value` → Math Expression → LoraLoader `strength_model`.
