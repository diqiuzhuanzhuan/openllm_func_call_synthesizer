from pathlib import Path

with (Path(__file__).parents[0] / Path("templates") / Path("critic_system_header.md")).open("r") as f:
    __BASE_CRITIC_SYSTEM_HEADER = f.read()

CRITIC_SYSTEM_HEADER = __BASE_CRITIC_SYSTEM_HEADER.replace("{context_prompt}", "", 1)


with (Path(__file__).parents[0] / Path("templates") / Path("critic_function_call_system_header.md")).open("r") as f:
    __BASE_CRITIC_FUNCTION_CALL_SYSTEM_HEADER = f.read()
    # noqa: E501

CRITIC_FUNCTION_CALL_SYSTEM_HEADER = __BASE_CRITIC_FUNCTION_CALL_SYSTEM_HEADER.replace("{context_prompt}", "", 1)

with (Path(__file__).parents[0] / Path("templates") / Path("query_generate_system_header.md")).open("r") as f:
    __BASE_QUERY_GENERATE_SYSTEM_HEADER = f.read()

QUERY_GENERATE_SYSTEM_HEADER = __BASE_QUERY_GENERATE_SYSTEM_HEADER
