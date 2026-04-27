gauurab@nku23N0091 Distillation.John-O-Bryan % mlx_lm.lora \
 --model mlx-community/Qwen2.5-7B-Instruct-4bit \
 --train \
 --data data/ \
 --iters 1000 \
 --batch-size 4 \
 --mask-prompt \
 --grad-checkpoint
Loading pretrained model
Fetching 9 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [02:13<00:00, 14.83s/it]
Download complete: : 4.30GB [02:13, 32.2MB/s] █████████████████████████████████████████████████▎ | 4/9 [02:13<03:50, 46.13s/it]
Loading datasets
Training
Trainable parameters: 0.151% (11.534M/7615.617M)
Starting training..., iters: 1000
Calculating loss...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [01:42<00:00, 6.82s/it]
Iter 1: Val loss 0.928, Val took 102.264s
Iter 10: Train loss 0.476, Learning Rate 1.000e-05, It/sec 0.066, Tokens/sec 65.688, Trained Tokens 9935, Peak mem 8.743 GB
Iter 20: Train loss 0.356, Learning Rate 1.000e-05, It/sec 0.072, Tokens/sec 63.772, Trained Tokens 18791, Peak mem 8.743 GB
Iter 30: Train loss 0.363, Learning Rate 1.000e-05, It/sec 0.070, Tokens/sec 66.032, Trained Tokens 28254, Peak mem 8.743 GB
Iter 40: Train loss 0.370, Learning Rate 1.000e-05, It/sec 0.070, Tokens/sec 67.328, Trained Tokens 37836, Peak mem 8.743 GB
Iter 50: Train loss 0.356, Learning Rate 1.000e-05, It/sec 0.063, Tokens/sec 58.651, Trained Tokens 47095, Peak mem 9.053 GB
Iter 60: Train loss 0.340, Learning Rate 1.000e-05, It/sec 0.064, Tokens/sec 61.788, Trained Tokens 56705, Peak mem 9.883 GB
Iter 70: Train loss 0.327, Learning Rate 1.000e-05, It/sec 0.065, Tokens/sec 62.117, Trained Tokens 66310, Peak mem 9.883 GB
Iter 80: Train loss 0.333, Learning Rate 1.000e-05, It/sec 0.079, Tokens/sec 63.487, Trained Tokens 74323, Peak mem 9.883 GB
Iter 90: Train loss 0.309, Learning Rate 1.000e-05, It/sec 0.066, Tokens/sec 68.022, Trained Tokens 84681, Peak mem 9.883 GB
Iter 100: Train loss 0.376, Learning Rate 1.000e-05, It/sec 0.060, Tokens/sec 58.826, Trained Tokens 94479, Peak mem 12.201 GB
Iter 100: Saved adapter weights to adapters/adapters.safetensors and adapters/0000100_adapters.safetensors.
Iter 110: Train loss 0.300, Learning Rate 1.000e-05, It/sec 0.061, Tokens/sec 65.630, Trained Tokens 105213, Peak mem 12.201 GB
Iter 120: Train loss 0.300, Learning Rate 1.000e-05, It/sec 0.066, Tokens/sec 62.519, Trained Tokens 114746, Peak mem 12.201 GB
Iter 130: Train loss 0.276, Learning Rate 1.000e-05, It/sec 0.074, Tokens/sec 65.942, Trained Tokens 123686, Peak mem 12.201 GB
Iter 140: Train loss 0.221, Learning Rate 1.000e-05, It/sec 0.065, Tokens/sec 63.570, Trained Tokens 133410, Peak mem 12.201 GB
Iter 150: Train loss 0.236, Learning Rate 1.000e-05, It/sec 0.070, Tokens/sec 69.507, Trained Tokens 143275, Peak mem 12.201 GB
Iter 160: Train loss 0.196, Learning Rate 1.000e-05, It/sec 0.068, Tokens/sec 62.304, Trained Tokens 152374, Peak mem 12.201 GB
Iter 170: Train loss 0.232, Learning Rate 1.000e-05, It/sec 0.071, Tokens/sec 68.856, Trained Tokens 162123, Peak mem 12.201 GB
Iter 180: Train loss 0.195, Learning Rate 1.000e-05, It/sec 0.080, Tokens/sec 65.184, Trained Tokens 170316, Peak mem 12.201 GB
Iter 190: Train loss 0.173, Learning Rate 1.000e-05, It/sec 0.074, Tokens/sec 63.501, Trained Tokens 178934, Peak mem 12.201 GB
Calculating loss...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [01:39<00:00, 6.63s/it]
Iter 200: Val loss 0.374, Val took 99.435s
Iter 200: Train loss 0.210, Learning Rate 1.000e-05, It/sec 0.070, Tokens/sec 66.287, Trained Tokens 188407, Peak mem 12.201 GB
Iter 200: Saved adapter weights to adapters/adapters.safetensors and adapters/0000200_adapters.safetensors.
Iter 210: Train loss 0.212, Learning Rate 1.000e-05, It/sec 0.067, Tokens/sec 67.749, Trained Tokens 198506, Peak mem 12.201 GB
Iter 220: Train loss 0.210, Learning Rate 1.000e-05, It/sec 0.068, Tokens/sec 61.284, Trained Tokens 207486, Peak mem 12.201 GB
Iter 230: Train loss 0.247, Learning Rate 1.000e-05, It/sec 0.056, Tokens/sec 61.599, Trained Tokens 218566, Peak mem 12.247 GB
Iter 240: Train loss 0.227, Learning Rate 1.000e-05, It/sec 0.070, Tokens/sec 61.416, Trained Tokens 227384, Peak mem 12.247 GB
Iter 250: Train loss 0.205, Learning Rate 1.000e-05, It/sec 0.063, Tokens/sec 64.013, Trained Tokens 237521, Peak mem 12.247 GB
Iter 260: Train loss 0.151, Learning Rate 1.000e-05, It/sec 0.066, Tokens/sec 65.551, Trained Tokens 247473, Peak mem 12.247 GB
Iter 270: Train loss 0.108, Learning Rate 1.000e-05, It/sec 0.074, Tokens/sec 62.691, Trained Tokens 255923, Peak mem 12.247 GB
Iter 280: Train loss 0.158, Learning Rate 1.000e-05, It/sec 0.057, Tokens/sec 56.039, Trained Tokens 265699, Peak mem 12.247 GB
Iter 290: Train loss 0.116, Learning Rate 1.000e-05, It/sec 0.069, Tokens/sec 64.182, Trained Tokens 274978, Peak mem 12.247 GB
Iter 300: Train loss 0.119, Learning Rate 1.000e-05, It/sec 0.066, Tokens/sec 60.997, Trained Tokens 284181, Peak mem 12.247 GB
Iter 300: Saved adapter weights to adapters/adapters.safetensors and adapters/0000300_adapters.safetensors.
Iter 310: Train loss 0.138, Learning Rate 1.000e-05, It/sec 0.070, Tokens/sec 66.886, Trained Tokens 293718, Peak mem 12.247 GB
Iter 320: Train loss 0.136, Learning Rate 1.000e-05, It/sec 0.070, Tokens/sec 70.530, Trained Tokens 303797, Peak mem 12.247 GB
Iter 330: Train loss 0.114, Learning Rate 1.000e-05, It/sec 0.071, Tokens/sec 70.190, Trained Tokens 313738, Peak mem 12.247 GB
Iter 340: Train loss 0.148, Learning Rate 1.000e-05, It/sec 0.065, Tokens/sec 69.810, Trained Tokens 324507, Peak mem 12.247 GB
Iter 350: Train loss 0.119, Learning Rate 1.000e-05, It/sec 0.071, Tokens/sec 69.351, Trained Tokens 334333, Peak mem 12.247 GB
Iter 360: Train loss 0.109, Learning Rate 1.000e-05, It/sec 0.072, Tokens/sec 63.194, Trained Tokens 343153, Peak mem 12.247 GB
Iter 370: Train loss 0.118, Learning Rate 1.000e-05, It/sec 0.067, Tokens/sec 66.650, Trained Tokens 353088, Peak mem 12.247 GB
Iter 380: Train loss 0.115, Learning Rate 1.000e-05, It/sec 0.064, Tokens/sec 58.200, Trained Tokens 362228, Peak mem 12.247 GB
Iter 390: Train loss 0.101, Learning Rate 1.000e-05, It/sec 0.053, Tokens/sec 56.514, Trained Tokens 372927, Peak mem 12.247 GB
Calculating loss...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [01:48<00:00, 7.24s/it]
Iter 400: Val loss 0.555, Val took 108.679s
Iter 400: Train loss 0.054, Learning Rate 1.000e-05, It/sec 0.067, Tokens/sec 66.501, Trained Tokens 382904, Peak mem 12.247 GB
Iter 400: Saved adapter weights to adapters/adapters.safetensors and adapters/0000400_adapters.safetensors.
Iter 410: Train loss 0.057, Learning Rate 1.000e-05, It/sec 0.064, Tokens/sec 62.414, Trained Tokens 392722, Peak mem 12.247 GB
Iter 420: Train loss 0.047, Learning Rate 1.000e-05, It/sec 0.072, Tokens/sec 67.197, Trained Tokens 402088, Peak mem 12.247 GB
Iter 430: Train loss 0.060, Learning Rate 1.000e-05, It/sec 0.066, Tokens/sec 64.740, Trained Tokens 411888, Peak mem 12.247 GB
Iter 440: Train loss 0.066, Learning Rate 1.000e-05, It/sec 0.064, Tokens/sec 68.477, Trained Tokens 422512, Peak mem 12.247 GB
Iter 450: Train loss 0.049, Learning Rate 1.000e-05, It/sec 0.073, Tokens/sec 63.751, Trained Tokens 431289, Peak mem 12.247 GB
Iter 460: Train loss 0.039, Learning Rate 1.000e-05, It/sec 0.076, Tokens/sec 63.791, Trained Tokens 439658, Peak mem 12.247 GB
Iter 470: Train loss 0.050, Learning Rate 1.000e-05, It/sec 0.067, Tokens/sec 62.105, Trained Tokens 448921, Peak mem 12.247 GB
Iter 480: Train loss 0.059, Learning Rate 1.000e-05, It/sec 0.065, Tokens/sec 62.270, Trained Tokens 458510, Peak mem 12.247 GB
Iter 490: Train loss 0.056, Learning Rate 1.000e-05, It/sec 0.070, Tokens/sec 66.394, Trained Tokens 467998, Peak mem 12.247 GB
Iter 500: Train loss 0.052, Learning Rate 1.000e-05, It/sec 0.076, Tokens/sec 66.454, Trained Tokens 476719, Peak mem 12.247 GB
Iter 500: Saved adapter weights to adapters/adapters.safetensors and adapters/0000500_adapters.safetensors.
Iter 510: Train loss 0.047, Learning Rate 1.000e-05, It/sec 0.063, Tokens/sec 61.039, Trained Tokens 486365, Peak mem 12.247 GB
Iter 520: Train loss 0.020, Learning Rate 1.000e-05, It/sec 0.063, Tokens/sec 61.129, Trained Tokens 496053, Peak mem 12.247 GB
Iter 530: Train loss 0.022, Learning Rate 1.000e-05, It/sec 0.066, Tokens/sec 61.735, Trained Tokens 505360, Peak mem 12.247 GB
Iter 540: Train loss 0.021, Learning Rate 1.000e-05, It/sec 0.069, Tokens/sec 68.975, Trained Tokens 515356, Peak mem 12.247 GB
Iter 550: Train loss 0.023, Learning Rate 1.000e-05, It/sec 0.061, Tokens/sec 64.102, Trained Tokens 525839, Peak mem 12.247 GB
Iter 560: Train loss 0.020, Learning Rate 1.000e-05, It/sec 0.065, Tokens/sec 62.371, Trained Tokens 535473, Peak mem 12.247 GB
Iter 570: Train loss 0.018, Learning Rate 1.000e-05, It/sec 0.073, Tokens/sec 66.629, Trained Tokens 544606, Peak mem 12.247 GB
Iter 580: Train loss 0.026, Learning Rate 1.000e-05, It/sec 0.067, Tokens/sec 68.574, Trained Tokens 554903, Peak mem 12.247 GB
Iter 590: Train loss 0.048, Learning Rate 1.000e-05, It/sec 0.059, Tokens/sec 57.855, Trained Tokens 564690, Peak mem 12.247 GB
Calculating loss...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [01:54<00:00, 7.62s/it]
Iter 600: Val loss 0.616, Val took 114.395s
Iter 600: Train loss 0.014, Learning Rate 1.000e-05, It/sec 0.074, Tokens/sec 65.775, Trained Tokens 573546, Peak mem 12.247 GB
Iter 600: Saved adapter weights to adapters/adapters.safetensors and adapters/0000600_adapters.safetensors.
Iter 610: Train loss 0.021, Learning Rate 1.000e-05, It/sec 0.072, Tokens/sec 66.143, Trained Tokens 582740, Peak mem 12.247 GB
Iter 620: Train loss 0.029, Learning Rate 1.000e-05, It/sec 0.062, Tokens/sec 58.595, Trained Tokens 592118, Peak mem 12.247 GB
Iter 630: Train loss 0.025, Learning Rate 1.000e-05, It/sec 0.073, Tokens/sec 63.433, Trained Tokens 600811, Peak mem 12.247 GB
Iter 640: Train loss 0.015, Learning Rate 1.000e-05, It/sec 0.065, Tokens/sec 61.920, Trained Tokens 610324, Peak mem 12.247 GB
Iter 650: Train loss 0.010, Learning Rate 1.000e-05, It/sec 0.067, Tokens/sec 65.203, Trained Tokens 620030, Peak mem 12.247 GB
Iter 660: Train loss 0.007, Learning Rate 1.000e-05, It/sec 0.067, Tokens/sec 65.720, Trained Tokens 629849, Peak mem 12.247 GB
Iter 670: Train loss 0.010, Learning Rate 1.000e-05, It/sec 0.071, Tokens/sec 65.805, Trained Tokens 639073, Peak mem 12.247 GB
Iter 680: Train loss 0.012, Learning Rate 1.000e-05, It/sec 0.063, Tokens/sec 60.282, Trained Tokens 648649, Peak mem 12.247 GB
Iter 690: Train loss 0.013, Learning Rate 1.000e-05, It/sec 0.068, Tokens/sec 64.705, Trained Tokens 658199, Peak mem 12.247 GB
Iter 700: Train loss 0.008, Learning Rate 1.000e-05, It/sec 0.074, Tokens/sec 62.715, Trained Tokens 666685, Peak mem 12.247 GB
Iter 700: Saved adapter weights to adapters/adapters.safetensors and adapters/0000700_adapters.safetensors.
Iter 710: Train loss 0.021, Learning Rate 1.000e-05, It/sec 0.056, Tokens/sec 58.232, Trained Tokens 677074, Peak mem 12.247 GB
Iter 720: Train loss 0.011, Learning Rate 1.000e-05, It/sec 0.067, Tokens/sec 62.589, Trained Tokens 686441, Peak mem 12.247 GB
Iter 730: Train loss 0.008, Learning Rate 1.000e-05, It/sec 0.067, Tokens/sec 58.796, Trained Tokens 695197, Peak mem 12.247 GB
Iter 740: Train loss 0.013, Learning Rate 1.000e-05, It/sec 0.062, Tokens/sec 65.145, Trained Tokens 705655, Peak mem 12.247 GB
Iter 750: Train loss 0.008, Learning Rate 1.000e-05, It/sec 0.065, Tokens/sec 64.971, Trained Tokens 715613, Peak mem 12.247 GB
Iter 760: Train loss 0.009, Learning Rate 1.000e-05, It/sec 0.069, Tokens/sec 60.519, Trained Tokens 724439, Peak mem 12.247 GB
Iter 770: Train loss 0.005, Learning Rate 1.000e-05, It/sec 0.071, Tokens/sec 65.053, Trained Tokens 733599, Peak mem 12.247 GB
Iter 780: Train loss 0.005, Learning Rate 1.000e-05, It/sec 0.064, Tokens/sec 64.873, Trained Tokens 743722, Peak mem 12.247 GB
Iter 790: Train loss 0.007, Learning Rate 1.000e-05, It/sec 0.058, Tokens/sec 61.026, Trained Tokens 754297, Peak mem 12.247 GB
Calculating loss...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [01:46<00:00, 7.10s/it]
Iter 800: Val loss 0.787, Val took 106.479s
Iter 800: Train loss 0.013, Learning Rate 1.000e-05, It/sec 0.058, Tokens/sec 57.421, Trained Tokens 764231, Peak mem 12.247 GB
Iter 800: Saved adapter weights to adapters/adapters.safetensors and adapters/0000800_adapters.safetensors.
Iter 810: Train loss 0.007, Learning Rate 1.000e-05, It/sec 0.058, Tokens/sec 61.308, Trained Tokens 774861, Peak mem 12.247 GB
Iter 820: Train loss 0.007, Learning Rate 1.000e-05, It/sec 0.078, Tokens/sec 64.267, Trained Tokens 783080, Peak mem 12.247 GB
Iter 830: Train loss 0.009, Learning Rate 1.000e-05, It/sec 0.074, Tokens/sec 69.998, Trained Tokens 792515, Peak mem 12.247 GB
Iter 840: Train loss 0.004, Learning Rate 1.000e-05, It/sec 0.069, Tokens/sec 60.362, Trained Tokens 801276, Peak mem 12.247 GB
Iter 850: Train loss 0.006, Learning Rate 1.000e-05, It/sec 0.063, Tokens/sec 60.387, Trained Tokens 810839, Peak mem 12.247 GB
Iter 860: Train loss 0.004, Learning Rate 1.000e-05, It/sec 0.072, Tokens/sec 65.654, Trained Tokens 819964, Peak mem 12.247 GB
Iter 870: Train loss 0.004, Learning Rate 1.000e-05, It/sec 0.069, Tokens/sec 64.012, Trained Tokens 829304, Peak mem 12.247 GB
Iter 880: Train loss 0.004, Learning Rate 1.000e-05, It/sec 0.069, Tokens/sec 62.484, Trained Tokens 838399, Peak mem 12.247 GB
Iter 890: Train loss 0.004, Learning Rate 1.000e-05, It/sec 0.065, Tokens/sec 63.863, Trained Tokens 848186, Peak mem 12.247 GB
Iter 900: Train loss 0.003, Learning Rate 1.000e-05, It/sec 0.072, Tokens/sec 64.976, Trained Tokens 857207, Peak mem 12.247 GB
Iter 900: Saved adapter weights to adapters/adapters.safetensors and adapters/0000900_adapters.safetensors.
Iter 910: Train loss 0.002, Learning Rate 1.000e-05, It/sec 0.071, Tokens/sec 69.367, Trained Tokens 867024, Peak mem 12.247 GB
Iter 920: Train loss 0.002, Learning Rate 1.000e-05, It/sec 0.071, Tokens/sec 62.881, Trained Tokens 875869, Peak mem 12.247 GB
Iter 930: Train loss 0.002, Learning Rate 1.000e-05, It/sec 0.059, Tokens/sec 55.323, Trained Tokens 885274, Peak mem 12.247 GB
Iter 940: Train loss 0.007, Learning Rate 1.000e-05, It/sec 0.060, Tokens/sec 58.893, Trained Tokens 895107, Peak mem 12.247 GB
Iter 950: Train loss 0.003, Learning Rate 1.000e-05, It/sec 0.072, Tokens/sec 68.642, Trained Tokens 904682, Peak mem 12.247 GB
Iter 960: Train loss 0.006, Learning Rate 1.000e-05, It/sec 0.058, Tokens/sec 61.587, Trained Tokens 915223, Peak mem 12.247 GB
Iter 970: Train loss 0.003, Learning Rate 1.000e-05, It/sec 0.073, Tokens/sec 65.059, Trained Tokens 924089, Peak mem 12.247 GB
Iter 980: Train loss 0.007, Learning Rate 1.000e-05, It/sec 0.066, Tokens/sec 64.961, Trained Tokens 933890, Peak mem 12.247 GB
Iter 990: Train loss 0.002, Learning Rate 1.000e-05, It/sec 0.072, Tokens/sec 66.830, Trained Tokens 943160, Peak mem 12.247 GB
Calculating loss...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [01:47<00:00, 7.15s/it]
Iter 1000: Val loss 0.826, Val took 107.386s
Iter 1000: Train loss 0.004, Learning Rate 1.000e-05, It/sec 0.063, Tokens/sec 60.155, Trained Tokens 952651, Peak mem 12.247 GB
Iter 1000: Saved adapter weights to adapters/adapters.safetensors and adapters/0001000_adapters.safetensors.
Saved final weights to adapters/adapters.safetensors.
gauurab@nku23N0091 Distillation.John-O-Bryan % mlx_lm.lora --model ... --adapter-path adapters --data data/ --test

Loading pretrained model
Traceback (most recent call last):
File "/Library/Frameworks/Python.framework/Versions/3.12/bin/mlx*lm.lora", line 6, in <module>
sys.exit(main())
^^^^^^
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/lora.py", line 362, in main
run(types.SimpleNamespace(\*\*args))
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/lora.py", line 322, in run
model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/utils.py", line 477, in load
model_path = \_download(path_or_hf_repo, revision=revision)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/utils.py", line 237, in \_download
snapshot_download(
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/huggingface_hub/utils/\_validators.py", line 85, in \_inner_fn
validate_repo_id(arg_value)
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/huggingface_hub/utils/\_validators.py", line 139, in validate_repo_id
raise HFValidationError(
huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars, '-', '*' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: '...'.
gauurab@nku23N0091 Distillation.John-O-Bryan % mlx_lm.lora \
 --model mlx-community/Qwen2.5-3B-Instruct-4bit \
 --adapter-path adapters \
 --data data/ \
 --test
Loading pretrained model
Fetching 9 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:23<00:00, 2.60s/it]
Download complete: : 1.75GB [00:23, 74.7MB/s] ██████████████████████████████████████████████████▊ | 4/9 [00:23<00:40, 8.04s/it]
Loading datasets
Testing
Calculating loss...: 0%| | 0/7 [00:00<?, ?it/s]
Traceback (most recent call last):
File "/Library/Frameworks/Python.framework/Versions/3.12/bin/mlx_lm.lora", line 6, in <module>
sys.exit(main())
^^^^^^
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/lora.py", line 362, in main
run(types.SimpleNamespace(\**args))
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/lora.py", line 340, in run
evaluate_model(args, model, test_set)
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/lora.py", line 299, in evaluate_model
test_loss = evaluate(
^^^^^^^^^
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/tuner/trainer.py", line 193, in evaluate
losses, toks = loss(model, *batch)
^^^^^^^^^^^^^^^^^^^
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/tuner/trainer.py", line 79, in default_loss
logits = model(inputs)
^^^^^^^^^^^^^
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/models/qwen2.py", line 173, in **call**
out = self.model(inputs, cache, input_embeddings)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/models/qwen2.py", line 153, in **call**
h = layer(h, mask, c)
^^^^^^^^^^^^^^^^^
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/models/qwen2.py", line 117, in **call**
r = self.self_attn(self.input_layernorm(x), mask, cache)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/models/qwen2.py", line 65, in **call**
queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
^^^^^^^^^^^^^^
File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mlx_lm/tuner/lora.py", line 97, in **call**
z = (self.dropout(x) @ self.lora_a) @ self.lora_b
~~~~~~~~~~~~~~~~^~~~~~~~~~~~~
ValueError: [matmul] Last dimension of first input with shape (4,320,2048) must match second to last dimension of second input with shape (3584,8).
gauurab@nku23N0091 Distillation.John-O-Bryan % mlx_lm.lora \
 --model mlx-community/Qwen2.5-7B-Instruct-4bit \
 --adapter-path adapters \
 --data data/ \
 --test
Loading pretrained model
Fetching 9 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 20832.64it/s]
Download complete: : 0.00B [00:01, ?B/s] | 0/9 [00:00<?, ?it/s]
Loading datasets
Testing
Calculating loss...: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:51<00:00, 7.41s/it]
Test loss 1.837, Test ppl 6.280.
gauurab@nku23N0091 Distillation.John-O-Bryan % mlx_lm.fuse \
 --model mlx-community/Qwen2.5-7B-Instruct-4bit \
 --adapter-path adapters \
 --save-path fused_model/
Loading pretrained model
Fetching 9 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 23876.49it/s]
Download complete: : 0.00B [00:00, ?B/s] | 0/9 [00:00<?, ?it/s]
README.md: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [00:00<00:00, 3.40MB/s]
gauurab@nku23N0091 Distillation.John-O-Bryan % mlx_lm.generate \
 --model fused_model/ \
 --prompt "A convex polygon has 19 sides. Find the sum of the degree measures of the interior angles." \
 --max-tokens 600
==========
To find the sum of the interior angles of a convex polygon with 19 sides, we can use the formula for the sum of the interior angles of an \(n\)-sided polygon. The formula is:

\[
\text{Sum of interior angles} = (n-2) \times 180^\circ
\]

Here, \(n\) is the number of sides of the polygon. For a polygon with 19 sides, we substitute \(n = 19\) into the formula:

\[
\text{Sum of interior angles} = (19-2) \times 180^\circ
\]

First, calculate \(19 - 2\):

\[
19 - 2 = 17
\]

Next, multiply 17 by 180:

\[
17 \times 180 = 3060
\]

Therefore, the sum of the interior angles of a convex polygon with 19 sides is:

\[
\boxed{3060^\circ}
\]
==========
Prompt: 50 tokens, 75.685 tokens-per-sec
Generation: 225 tokens, 32.028 tokens-per-sec
Peak memory: 4.410 GB
gauurab@nku23N0091 Distillation.John-O-Bryan % mlx_lm.lora \
 --model /Users/gauurab/Documents/Projects/Distillation.John-O-Bryan/fused_model/ \
 --data /Users/gauurab/Documents/Projects/Distillation.John-O-Bryan/data/ \
 --test
Loading pretrained model
Loading datasets
Testing
Calculating loss...: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:51<00:00, 7.37s/it]
Test loss 1.941, Test ppl 6.969.
