from huggingface_hub import snapshot_download

repo_id = "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
local_dir = "/home/betty/models/llava_onevision_qwen2_7b_ov_chat_hf"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
)
print("done")
