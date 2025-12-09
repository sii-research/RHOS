from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="THyanNK/L1FLOW",
    repo_type="dataset",
    allow_patterns="results/pusht/run_0/**",  # 下载 run_0 及其所有子内容
    local_dir="./results/L1FLOW/pusht/run_0/",      # 本地保存路径
    # local_dir_use_symlinks=False      # 若希望文件真实存在（而非 symlink），可加此行
)