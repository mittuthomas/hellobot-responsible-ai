import os
import tempfile

# Tests are hermetic: offline stub LLM + throwaway database
os.environ.setdefault("FORCE_OFFLINE", "true")
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/test.db"
