#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================="
echo "Setting up pipeline dependencies"
echo "================================================="

# mini-swe-agent (installed as an editable package)
echo -e "\n--- Installing mini-swe-agent ---"
pip install -e "$SCRIPT_DIR/mini-swe-agent"

# kgCompass
echo -e "\n--- Installing kgCompass requirements ---"
pip install -r "$SCRIPT_DIR/kgCompass/requirements.txt"

echo -e "\n================================================="
echo "✅ Setup complete"
echo "================================================="
echo ""
echo "Next: fill in your API keys in config.env"
echo "  DEEPSEEK_API_KEY=..."
echo "  GITHUB_TOKEN=..."
