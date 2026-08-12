from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # LLM Provider Configuration
    LLM_PROVIDER: str = "groq"  # Options: "groq" or "local"

    # Groq API Configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_TEMPERATURE: float = 0.0
    GROQ_MAX_TOKENS: int = 40000
    GROQ_REASONING_EFFORT: str = "medium"

    # Local LLM Configuration
    LOCAL_LLM_API_URL: str = "http://localhost:8001/v1/chat/completions"
    LOCAL_LLM_MODEL: str = "openai/gpt-oss-120b"
    LOCAL_LLM_TEMPERATURE: float = 0.0
    LOCAL_LLM_MAX_TOKENS: int = 40000

    # Reviewer LLM Configuration (separate model for biochemistry validation)
    REVIEWER_LLM_API_URL: str = "http://localhost:8002/v1/chat/completions"
    REVIEWER_LLM_MODEL: str = "aaditya/Llama3-OpenBioLLM-70B"
    REVIEWER_LLM_TEMPERATURE: float = 0.0
    REVIEWER_LLM_MAX_TOKENS: int = 4096

    # Server Configuration
    API_PORT: int = 6000
    API_HOST: str = "0.0.0.0"

    # LLM request timeout (seconds). Must exceed a model's load/wake time behind llm-router
    # (vLLM sleep mode: a wake is ~seconds, but the first load after deploy can take minutes,
    # e.g. OpenBio ~3.5 min). The old hardcoded 120s was shorter, so reviewer calls timed out.
    LLM_REQUEST_TIMEOUT: int = 600

    # Pathway Clustering Configuration (Step 1 pathway-theme grouping)
    # Markov Clustering (van Dongen 2000) replaces the legacy connected-components
    # partitioning; it resolves the single-linkage chaining problem via flow simulation and
    # is deterministic. Defaults below mirror aPEAR's published defaults (ievaKer/aPEAR,
    # R/cluster.R + R/similarity.R): mcl(sim, addLoops=FALSE, max.iter=500, expansion=2,
    # inflation=2.5) run on the full pairwise Jaccard similarity matrix with NO threshold.
    CLUSTER_METHOD: str = "mcl"               # "mcl" | "connected_components"
    CLUSTER_INFLATION: float = 2.5            # MCL inflation/granularity — aPEAR default (2.5)
    CLUSTER_SIMILARITY_FLOOR: float = 0.0     # no threshold: aPEAR feeds the full Jaccard matrix
    CLUSTER_GENE_SOURCE: str = "de"           # similarity gene set: "de" (current) | "full"
    CLUSTER_MIN_SIZE: int = 2                 # min pathways per cluster (aPEAR minClusterSize=2)
    CLUSTER_JACCARD_THRESHOLD: float = 0.25   # legacy connected_components edge cutoff (NOT aPEAR)

    # Pipeline Configuration
    PIPELINE_OUTPUT_DIR: str = "output/pipeline_jobs"  # Directory to save pipeline job outputs
    PIPELINE_ENABLE_VALIDATION: bool = True  # Enable step output validation to catch LLM hallucinations
    PIPELINE_STRICT_MODE: bool = False  # Fail pipeline if validation errors found (default: log warnings only)

    # Baseline Comparison Configuration
    ENABLE_BASELINE_COMPARISON: bool = False  # Set to True to run baseline comparisons alongside pipeline
    OPENAI_API_KEY: str = ""
    OPENAI_BASELINE_MODEL: str = "gpt-5.4"
    OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_BASELINE_MODEL: str = "claude-sonnet-4-20250514"
    ANTHROPIC_API_URL: str = "https://api.anthropic.com/v1/messages"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # tolerate extra keys in .env / environment (don't crash on unknown vars)


settings = Settings()
