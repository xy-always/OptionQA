# /bin/bash
models=(
    /mnt/disk4/xy/MobileLLM/hf-models/ggml-model-Llama-3.1-8B-Instruct-q4_0.gguf
    /mnt/disk4/xy/MobileLLM/hf-models/ggml-model-Llama-3.1-70B-Instruct-q4_0.gguf
    /mnt/disk4/xy/MobileLLM/hf-models/ggml-model-Llama3-Med42-8B-q4_0.gguf
    /mnt/disk4/xy/MobileLLM/hf-models/ggml-model-Llama3-Med42-70B-q4_0.gguf
    /mnt/disk4/xy/MobileLLM/hf-models/ggml-model-Med-LLaMA3-8B-q4_0.gguf
    /mnt/disk4/xy/MobileLLM/hf-models/ggml-model-meditron-7b-q4_0.gguf
    /mnt/disk4/xy/MobileLLM/hf-models/ggml-model-Mixtral-8x7B-Instruct-v0.1-q4_0.gguf
    /mnt/disk4/xy/MobileLLM/hf-models/ggml-model-Qwen2.5-Aloe-Beta-7B-q4_0.gguf
    /home/wsy/project/MobileLLM/hf-models/ggml-model-llama-2-7b-chat-q4_0.gguf
    /home/wsy/project/MobileLLM/hf-models/ggml-model-llama-2-70b-chat-q4_0.gguf
    /home/wsy/project/MobileLLM/hf-models/ggml-model-qwen2.5-3b-q4_0.gguf
    /home/wsy/project/MobileLLM/hf-models/ggml-model-qwen2.5-72b-q4_0.gguf
)
# M42_8b
dataset_name=MMLU-anatomy
for model in "${models[@]}"; do
    python main_multi.py \
        --data_dir ./ \
        --generator_model_name_or_path $model \
        --generator_type mobile \
        --evaluator_model_name_or_path gpt-4-1106-preview \
        --dataset_name $dataset_name \
        --output_dir ./results/$dataset_name/ \
        --device 1 \
        --max_context_window 8192
done