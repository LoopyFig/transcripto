from huggingface_hub import snapshot_download

snapshot_download(
  repo_id = "Systran/faster-whisper-base",
  local_dir = "./models/faster-whisper-base",
  local_dir_use_symlinks = False
)