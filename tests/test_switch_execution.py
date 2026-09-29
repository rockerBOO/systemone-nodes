import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import execution  # noqa: E402
import nodes  # noqa: E402
from comfy_api.latest import io  # noqa: E402

from systemone_nodes.nodes import SystemOneChoiceSwitch, SystemOneThresholdSwitch  # noqa: E402

RAN = []


class Probe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SystemOneTestProbe",
            category="test",
            inputs=[io.String.Input("tag")],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, tag) -> io.NodeOutput:
        RAN.append(tag)
        return io.NodeOutput(tag)


class Sink(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SystemOneTestSink",
            category="test",
            is_output_node=True,
            inputs=[io.String.Input("value", force_input=True)],
            outputs=[],
        )

    @classmethod
    def execute(cls, value) -> io.NodeOutput:
        RAN.append(("sink", value))
        return io.NodeOutput()


class FakeServer:
    client_id = None
    last_node_id = None

    def send_sync(self, event, data, sid=None):
        pass


@pytest.fixture
def run_prompt(monkeypatch):
    for node in (Probe, Sink, SystemOneChoiceSwitch, SystemOneThresholdSwitch):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node.define_schema().node_id, node)
    RAN.clear()

    def run(prompt):
        executor = execution.PromptExecutor(FakeServer(), cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
        executor.execute(prompt, "systemone-test", {}, ["9"])
        assert executor.success, executor.status_messages
        return RAN

    return run


def probe(tag):
    return {"class_type": "SystemOneTestProbe", "inputs": {"tag": tag}}


def sink(from_node):
    return {"class_type": "SystemOneTestSink", "inputs": {"value": [from_node, 0]}}


def test_choice_switch_runs_only_selected_branch(run_prompt):
    ran = run_prompt({
        "1": probe("a"),
        "2": probe("b"),
        "3": probe("c"),
        "4": {
            "class_type": "SystemOneChoiceSwitch",
            "inputs": {"index": 1, "option0": ["1", 0], "option1": ["2", 0], "option2": ["3", 0]},
        },
        "9": sink("4"),
    })

    assert ran == ["b", ("sink", "b")]


def test_threshold_switch_runs_only_selected_branch(run_prompt):
    ran = run_prompt({
        "1": probe("yes"),
        "2": probe("no"),
        "4": {
            "class_type": "SystemOneThresholdSwitch",
            "inputs": {"value": 0.3, "threshold": 0.8, "on_true": ["1", 0], "on_false": ["2", 0]},
        },
        "9": sink("4"),
    })

    assert ran == ["no", ("sink", "no")]
