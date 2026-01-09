#!/usr/bin/env bash

set -e

python  apps/main.py \
    synthesizer=default \
    synthesizer.mcp_servers.ugreen_mcp.transport="http://192.168.111.11:8000/mcp" \
    synthesizer.query_generation.enable=True \
    synthesizer.query_generation.function_docs="examples/function_docs.example.json" \
    synthesizer.function_call_generation.enable=True \
    synthesizer.function_call_generation.function_dataset="data/function_query" \
    synthesizer.critic=True


exit 0