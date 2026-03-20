from huggingface_hub import snapshot_download

repo_id = "lmms-lab/llava-onevision-qwen2-7b-ov"
local_dir = "./llava_onevision_qwen2_7b_ov"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
)
print("done")
