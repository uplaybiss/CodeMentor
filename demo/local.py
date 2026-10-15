"""python -m demo.local [--port 8013] -- never selects a paid provider."""

import argparse
import os
from pathlib import Path


def offline_environment(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "LLM_PROVIDER": "demo",
            "VECTOR_ENABLED": "false",
            "APP_NAME": "CodeMentor",
            "DATABASE_URL": f"sqlite:///{(data_dir / 'codementor.sqlite').as_posix()}",
            "HARNESS_CHECKPOINT_PATH": str(data_dir / "checkpoints.sqlite"),
            "ANONYMIZED_TELEMETRY": "False",
            "LANGCHAIN_TRACING_V2": "false",
            "LANGSMITH_TRACING": "false",
            "HF_HUB_OFFLINE": "1",
            "TIKTOKEN_CACHE_DIR": str(Path(__file__).parents[1] / "demo" / "token_cache"),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--data-dir", type=Path, default=Path("data/local_demo"))
    args = parser.parse_args()
    offline_environment(args.data_dir.resolve())
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
