#!/bin/bash

# Run all NLL computation scripts
# Note: Claude and Gemini are commented out as they are already done

echo "Starting NLL computation for all models..."

# Claude 
# echo "Running Claude NLL computation..."
# python claude_nll.py

# Gemini 
echo "Running Gemini NLL computation..."
python gemini_nll.py

# GPT4
# echo "Running GPT4 NLL computation..."
# python gpt4_nll.py

# GPT4o
# echo "Running GPT4o NLL computation..."
# python gpt4o_nll.py

# LlaMA3
echo "Running LlaMA3 NLL computation..."
python llama3_nll.py

# Qwen
echo "Running Qwen NLL computation..."
python qwen_nll.py

echo "All NLL computations completed!"
