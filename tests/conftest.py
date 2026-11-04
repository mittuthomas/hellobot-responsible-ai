import os
import tempfile

# Tests are hermetic: offline stub LLM + throwaway database
os.environ.setdefault("FORCE_OFFLINE", "true")
_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["CHECKPOINT_DB"] = f"{_tmp}/checkpoints.db"
