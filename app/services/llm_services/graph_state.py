from typing import TypedDict


class PipelineState(TypedDict):
    file_path: str
    document: str
    test_cases_lv1: str
    test_cases_final: dict