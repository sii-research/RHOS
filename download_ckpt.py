from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="THyanNK/L1FLOW",
    repo_type="dataset",
    allow_patterns="results/pusht/run_0/**", 
    local_dir="./results/L1FLOW/pusht/run_0/", 
)