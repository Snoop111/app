# Bedrock RAG Chat

This Streamlit application queries an AWS Bedrock Knowledge Base and displays answers with citations.

## Environment Variables

Create a `.env` file in the project root with the following keys:

```
REGION=<aws-region>
MODEL_ID=<foundation-model-id>
EMBEDDING_MODEL_ID=<embedding-model-id>
KB_ID=<knowledge-base-id>
RAG_MODEL_ID=<foundation-model-id-for-rag>
MAX_TOKENS=<token-limit>
RATE_LIMIT_PER_MINUTE=<api-calls-per-minute>
LOG_LEVEL=INFO
```

`LOG_LEVEL` controls the log verbosity (e.g. `DEBUG`, `INFO`).

## Running

Install the required dependencies and run the app with:

```
streamlit run appv2.py
```

Run tests with:

```
pytest
```

