OPENAI_KEY = 'sk-XX' # *Default
OPENAI_URL = "XX"
CLAUDE_API_KEY = 'YOUR_KEY' # *Optional


# For search, choose one of the following:
SEMANTIC_SCHOLAR_API_KEY = 'XX' # *Default


# SELECT_MODEL = "llama"
SELECT_MODEL = "gpt-4o"
MODEL_PATH = "PATH/Llama-3.1-8B-Instruct"
EMBD_MODEL = "text-embedding-3-large"

ASPECT_K=[4]
FINAL_K=[4]
ASPECT_TYPE="dynamic" # "single" # "dynamic" "fixed"
DP_TYPE="DP"  # "Greedy" "Combine"

DATA_PATH = "../data/ground_new"
HTML_PATH = "../data/arxiv_html/"
MD_PATH = "../data/arxiv_md/"
SAVE_PATH = f"../res"