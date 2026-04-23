from dotenv import load_dotenv
import os
from github import Github

load_dotenv()

# GitHub Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Bailian
BAILIAN_API_KEY = os.getenv("GEMINI_API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jpassword")

# Knowledge Graph Configuration
MAX_CANDIDATE_METHODS = 500
MAX_SEARCH_DEPTH = 2
SEARCH_SPACE = 50

# Connection strength coefficients
CONNECTION_FACTOR = 0.5
WEAK_CONNECTION = 1.0
NORMAL_CONNECTION = WEAK_CONNECTION * CONNECTION_FACTOR
STRONG_CONNECTION = NORMAL_CONNECTION * CONNECTION_FACTOR

# Dataset Configuration
DATASET_NAME = "princeton-nlp/SWE-bench_Lite"

# Graph Configuration
DECAY_FACTOR = 0.6
VECTOR_SIMILARITY_WEIGHT = 0.3

# API Configuration
DEEPSEEK_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-2.5-flash"
REVIEW_MODEL_NAME = "gemini-2.5-flash"

# LLM Request Configuration
MAX_TOKENS = 8192
TEMPERATURE = 0.3
TOP_P = 0.95

MAX_INPUT_LENGTH = 65536 / 3
LLM_LOC_MAX = 10
CANDIDATE_LOCATIONS_MAX = 20