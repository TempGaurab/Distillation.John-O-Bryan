#pip install mlx-lm
#python finetune.py

# Low memory (16GB):
#python finetune.py --lora-layers 4 --batch-size 2 --lora-rank 4

# More memory (32GB+):
#python finetune.py --lora-layers 16 --batch-size 8 --lora-rank 16

# Change the target model:
#python finetune.py --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Skip fusing (just keep the adapter):
#python finetune.py --no-fuse