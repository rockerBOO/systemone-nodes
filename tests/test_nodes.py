import pytest

from systemone_nodes.nodes import SystemOneQuestion, parse_criteria


def test_choice_criteria_keep_line_order():
    text = "anime: Anime or illustrated\n\nphoto: Photorealistic: shot on a camera\n"

    assert list(parse_criteria("choice", text).items()) == [
        ("anime", "Anime or illustrated"),
        ("photo", "Photorealistic: shot on a camera"),
    ]


def test_choice_line_without_colon_raises():
    with pytest.raises(ValueError, match="'photo'"):
        parse_criteria("choice", "anime: Anime\nphoto")


def test_score_criteria_is_ordered_list():
    assert parse_criteria("score", "Calm\n\n Frustrated \nVery angry") == ["Calm", "Frustrated", "Very angry"]


def test_question_output():
    name, question = SystemOneQuestion.execute("route", "choice", "Which team?", "1: Refunds\n2: Damaged").result[0]

    assert name == "route"
    assert question == {"type": "choice", "instructions": "Which team?", "criteria": {"1": "Refunds", "2": "Damaged"}}


def test_noul_without_criteria_omits_criteria():
    _, question = SystemOneQuestion.execute("angry", "noul", "Is the customer angry?", "").result[0]

    assert question == {"type": "noul", "instructions": "Is the customer angry?"}
