# Deployment guide

**Streamlit Community Cloud (current public demo).** New app → repo `Shivjahagirdar/signalforge-ai-product-intelligence`,
branch `main`, main file `app.py`. Dependencies come from `requirements.txt`; theme/server limits from
`.streamlit/config.toml`. Optional secret: `ANTHROPIC_API_KEY` (enables LLM narration; never required).

**Docker.** `docker build -t signalforge . && docker run -p 8501:8501 signalforge` (non-root user, health check on
`/_stcore/health`; CI builds and health-checks the image).

**Local.** `pip install -r requirements.txt && streamlit run app.py`.

**Verify a deployment.** `SF_E2E_URL=https://<app-url> pytest -m e2e tests/e2e` runs the full browser journey and the
second-visitor isolation check against the live URL.
