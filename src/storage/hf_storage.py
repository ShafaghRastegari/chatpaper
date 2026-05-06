import os
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")


def get_repo_id() -> str:
    return f"{HF_USERNAME}/chatpaper-data"


def is_hf_configured() -> bool:
    """Returns True only if HF credentials are available."""
    return bool(HF_TOKEN and HF_USERNAME)


def ensure_dataset_repo():
    if not is_hf_configured():
        return
    api = HfApi(token=HF_TOKEN)
    repo_id = get_repo_id()
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"HF dataset repo exists: {repo_id}")
    except Exception:
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)
            print(f"Created HF dataset repo: {repo_id}")
        except Exception as e:
            if "already exists" in str(e) or "409" in str(e):
                print(f"HF dataset repo already exists: {repo_id}")
            else:
                print(f"Warning: Could not create HF repo: {e}")


def upload_file(local_path: str, path_in_repo: str) -> bool:
    try:
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=get_repo_id(),
            repo_type="dataset",
        )
        return True
    except Exception as e:
        print(f"HF upload error: {e}")
        return False


def download_file(path_in_repo: str, local_path: str) -> bool:
    try:
        downloaded = hf_hub_download(
            repo_id=get_repo_id(),
            filename=path_in_repo,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="/tmp/hf_downloads",
        )
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(downloaded, local_path)
        return True
    except EntryNotFoundError:
        return False
    except Exception as e:
        print(f"HF download error: {e}")
        return False


def save_json_to_hf(data, path_in_repo: str) -> bool:
    tmp_path = Path("/tmp") / ("hf_" + path_in_repo.replace("/", "_"))
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    result = upload_file(str(tmp_path), path_in_repo)
    tmp_path.unlink(missing_ok=True)
    return result


def load_json_from_hf(path_in_repo: str, default=None):
    tmp_path = f"/tmp/hf_{path_in_repo.replace('/', '_')}"
    success = download_file(path_in_repo, tmp_path)
    if not success:
        return default
    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def save_chat(chat_data: dict) -> bool:
    if not is_hf_configured():
        return False
    path = f"chats/{chat_data['session_id']}.json"
    return save_json_to_hf(chat_data, path)


def load_all_chats() -> list:
    if not is_hf_configured():
        return []
    try:
        api = HfApi(token=HF_TOKEN)
        files = list(api.list_repo_files(repo_id=get_repo_id(), repo_type="dataset"))
        chat_files = [f for f in files if f.startswith("chats/") and f.endswith(".json")]
    except Exception:
        return []
    chats = []
    for file_path in chat_files:
        chat = load_json_from_hf(file_path, default=None)
        if chat:
            chats.append(chat)
    chats.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return chats


def delete_chat(session_id: str) -> bool:
    if not is_hf_configured():
        return False
    try:
        api = HfApi(token=HF_TOKEN)
        api.delete_file(
            path_in_repo=f"chats/{session_id}.json",
            repo_id=get_repo_id(),
            repo_type="dataset",
        )
        return True
    except Exception as e:
        print(f"Delete chat error: {e}")
        return False


def save_related_papers(data: dict) -> bool:
    if not is_hf_configured():
        return False
    return save_json_to_hf(data, "related_papers.json")


def load_related_papers() -> dict:
    if not is_hf_configured():
        return {}
    return load_json_from_hf("related_papers.json", default={})