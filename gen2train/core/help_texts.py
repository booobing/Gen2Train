"""고급 탭에 보여줄 파라미터 설명.

sd-scripts 원본 argparse help(영/일 혼용, 짧고 기술적)보다 실사용 관점의 한국어 설명을
우선 노출하기 위한 매핑이다. 여기 없는 dest는 원본 help 텍스트로 자동 대체된다.
"""

HELP_TEXTS: dict[str, str] = {
    # --- Accelerate / 리소스 ---
    "mixed_precision": (
        "학습 중 VRAM 사용량과 속도를 결정한다.\n"
        "fp16: 가장 일반적, 속도 빠르고 VRAM 절약.\n"
        "bf16: 최신 GPU에서 더 안정적일 수 있음.\n"
        "no: 정밀도는 가장 높지만 느리고 VRAM을 많이 쓴다."
    ),
    "num_processes": "학습에 사용할 프로세스(보통 GPU) 개수. GPU 1개면 1.",
    "num_machines": "학습에 사용할 컴퓨터(머신) 대수. 개인 사용자는 항상 1.",
    "num_cpu_threads_per_process": "데이터 준비에 사용할 CPU 스레드 수. 기본값을 그대로 둬도 대부분 괜찮다.",
    "dynamo_backend": "PyTorch 2.0 이상의 실험적 컴파일 최적화 기능. 아직 불안정할 수 있어 잘 모르면 no(사용 안 함)를 권장한다.",
    "dynamo_mode": "dynamo_backend 사용 시 최적화 모드. 기본값 그대로 둬도 된다.",
    "multi_gpu": "GPU 2개 이상을 함께 사용할지 여부. GPU가 1개면 반드시 꺼둔다.",
    "gpu_ids": "여러 GPU 중 실제로 사용할 GPU 번호. 예: \"0,1\" -> 0번과 1번 GPU 사용.",
    "main_process_port": "여러 GPU가 서로 통신할 때 쓰는 네트워크 포트. 다른 프로그램과 겹치지만 않으면 기본값을 쓴다.",

    # --- LoRA 네트워크 ---
    "network_dim": (
        "LoRA의 랭크(rank). 학습 가능한 표현력/용량을 결정한다.\n"
        "낮을수록(8~32) 파일이 작고 과적합이 덜하며, 높을수록(64~128) 디테일을 더 담을 수 있지만 과적합 위험도 커진다.\n"
        "결함 하나를 학습한다면 16~32면 충분한 경우가 많다."
    ),
    "network_alpha": "LoRA 학습 강도 스케일. 보통 network_dim의 절반 값(예: dim 32 -> alpha 16)이 무난하다.",
    "network_dropout": "학습 중 LoRA 일부를 무작위로 꺼서 과적합을 억제하는 기법. 과적합이 심할 때 0.1~0.2를 시도해볼 수 있다. 기본은 0(사용 안 함).",
    "rank_dropout": "network_dropout과 유사하게, rank 차원 일부를 무작위로 꺼서 과적합을 억제한다.",
    "module_dropout": "학습 중 LoRA 모듈 자체를 무작위로 통째로 꺼서 과적합을 억제한다.",
    "network_weights": "이어서 학습할 기존 LoRA 파일 경로. 처음부터 새로 학습한다면 비워둔다.",
    "dim_from_weights": "network_weights로 불러온 LoRA 파일의 rank(dim) 값을 자동으로 읽어와 사용한다.",
    "scale_weight_norms": "과적합 방지용 가중치 크기 제한. 특별한 문제가 없다면 0(비활성화)으로 둔다.",
    "unet_lr": "이미지 생성을 담당하는 U-Net 부분의 학습률. 비워두면 공용 Learning rate를 사용한다.",
    "text_encoder_lr": "캡션(텍스트) 이해를 담당하는 Text Encoder의 학습률. 보통 U-Net보다 절반 정도 낮게 설정하면 좋은 결과가 나오는 경우가 많다.",
    "network_train_unet_only": "Text Encoder는 그대로 두고 U-Net 부분만 학습한다.",
    "network_train_text_encoder_only": "U-Net은 그대로 두고 Text Encoder 부분만 학습한다.",

    # --- 학습 기본 ---
    "train_batch_size": "한 번에 학습할 이미지 수. 클수록 VRAM을 많이 쓴다. VRAM이 부족하면 1로 낮추고 대신 gradient_accumulation_steps를 올려서 보완한다.",
    "gradient_accumulation_steps": (
        "여러 스텝에 걸쳐 기울기를 누적한 뒤 한 번에 반영해, VRAM은 적게 쓰면서도 실질적으로 큰 배치처럼 학습하는 기법.\n"
        "예: batch_size=1, 이 값=8이면 실질 배치=8과 비슷한 효과.\n"
        "batch_size를 키웠더니 스텝당 시간이 비정상적으로 느려졌다면, batch_size를 1~2로 낮추고 이 값을 8~16으로 올리는 걸 권장한다."
    ),
    "max_train_epochs": "전체 데이터셋을 몇 번 반복 학습할지. max_train_steps와 둘 중 하나만 사용한다(둘 다 채우면 epoch 기준이 우선).",
    "max_train_steps": "총 학습 스텝 수로 종료 시점을 지정. epoch 대신 스텝 수 기준으로 끝내고 싶을 때 사용한다.",
    "save_every_n_epochs": "몇 epoch마다 중간 체크포인트를 저장할지. 여러 시점의 결과물을 비교해보고 싶을 때 유용하다.",
    "seed": "재현성을 위한 난수 시드. 같은 시드+같은 설정이면 항상 같은 결과가 나온다. -1이면 매번 랜덤.",
    "cache_latents": "이미지를 매번 새로 변환하지 않고 미리 변환해 저장해두는 기능. 학습 속도가 크게 빨라지므로 켜는 것을 권장한다.",
    "cache_latents_to_disk": "cache_latents 결과를 디스크에 저장해 다음 학습에도 재사용한다. 데이터셋이 크면 유용하다.",
    "optimizer_type": "학습을 최적화하는 알고리즘. AdamW8bit이 VRAM 절약과 성능의 균형이 좋아 기본으로 권장된다.",
    "lr_scheduler": "학습 중 learning rate를 어떻게 변화시킬지에 대한 계획. cosine이나 cosine_with_restarts가 보편적으로 좋은 성능을 낸다.",
    "lr_scheduler_num_cycles": "cosine_with_restarts 등에서 학습률을 몇 번 반복(재시작)할지.",
    "lr_scheduler_power": "polynomial 스케줄러를 쓸 때의 감소 곡선 강도.",
    "learning_rate": "학습률. 모델이 한 번에 얼마나 크게 업데이트될지 결정하는 값.\n너무 크면 학습이 불안정해지고, 너무 작으면 학습이 느리거나 제대로 배우지 못한다.",
    "lr_warmup_steps": "학습 초반 learning rate를 0에서 목표치까지 서서히 끌어올리는 구간(스텝 수). 초기 학습 안정성을 높여준다.",
    "enable_bucket": "다양한 가로세로 비율의 이미지를 비슷한 비율끼리 묶어 불필요한 잘림 없이 학습하는 기능. 거의 항상 켜두는 것을 권장한다.",
    "min_bucket_reso": "버킷팅이 다룰 최소 해상도.",
    "max_bucket_reso": "버킷팅이 다룰 최대 해상도.",
    "bucket_reso_steps": "버킷 해상도를 몇 단위로 나눌지 (작을수록 세밀하게 나뉜다).",
    "clip_skip": "텍스트 인코더의 마지막 몇 개 레이어를 건너뛸지. 애니메이션 계열 모델은 보통 2를 쓰고, 실사 계열은 비워두는(1) 경우가 많다.",
    "xformers": "메모리 효율적인 attention 연산을 사용해 VRAM을 절약하고 속도를 높인다. 대부분의 경우 켜두는 것이 유리하다.",
    "sdpa": "PyTorch 내장 메모리 효율 attention. xformers 대신 사용할 수 있는 대안.",
    "gradient_checkpointing": "연산 중간 결과를 저장하지 않고 필요할 때 다시 계산해 VRAM을 절약하는 기법. 대신 속도는 약간 느려진다. VRAM이 부족할 때 켠다.",
    "shuffle_caption": "캡션 안의 태그 순서를 매 학습마다 무작위로 섞는다. 여러 태그가 있을 때 특정 순서에 과의존하는 것을 막아준다.",
    "keep_tokens": "shuffle_caption을 켰을 때, 앞에서부터 몇 개 태그는 순서를 섞지 않고 고정할지 (트리거 단어를 맨 앞에 고정하고 싶을 때 유용).",
    "caption_dropout_rate": "학습 중 캡션 전체를 일정 확률로 비워서(무캡션) 학습하는 기법. 과적합 억제에 도움될 수 있다.",
    "caption_dropout_every_n_epochs": "몇 epoch마다 caption_dropout_rate를 적용할지.",

    # --- 샘플 이미지 ---
    "sample_every_n_epochs": "몇 epoch마다 미리보기 이미지를 생성할지.",
    "sample_every_n_steps": "몇 스텝마다 미리보기 이미지를 생성할지.",
    "sample_sampler": "미리보기 이미지 생성에 사용할 샘플러 (예: euler_a, dpm++ 2m karras 등).",
    "sample_prompts": "미리보기 생성에 사용할 프롬프트. 학습이 잘 되고 있는지, 과적합이 오는지 실시간으로 확인하는 데 유용하다.",

    # --- 저장 / 기타 ---
    "save_model_as": "결과물 저장 형식. safetensors가 가장 안전하고 빠르며 권장된다.",
    "training_comment": "완성된 모델 파일 안에 남기는 개인 메모. 어떤 설정으로 학습했는지, 실험 결과가 어땠는지 기록해두면 나중에 비교하기 좋다.",
    "no_half_vae": "SDXL의 VAE를 fp32로 강제 실행해 NaN 오류를 방지한다. VRAM을 더 쓰지만 안정성이 높아진다 (SDXL 전용).",
    "cache_text_encoder_outputs": "SDXL 텍스트 인코더 결과를 미리 계산해 저장, VRAM을 아낀다. shuffle_caption 등 캡션을 매번 바꾸는 기능과는 함께 쓸 수 없다 (SDXL 전용).",
    "full_fp16": "가능한 모든 연산을 fp16으로 강제해 VRAM을 더 아낀다. 불안정할 수 있어 VRAM이 정말 부족할 때만 권장.",
    "full_bf16": "full_fp16과 유사하되 bf16을 사용한다.",
    "prior_loss_weight": "정규화(양품) 이미지 손실에 곱해지는 가중치. 1.0이 기본이며, 결함 학습이 정상 개념을 너무 많이 덮어쓴다 싶으면 낮춰본다.",
    "resolution": "학습 이미지 해상도(가로,세로). 결함이 작거나 디테일이 중요하면 해상도를 높이는 게 유리하지만 VRAM 사용량도 늘어난다.",

    # --- 로깅 / 콘솔 ---
    "console_log_level": "로그 출력 레벨을 설정한다. 기본값은 INFO.",
    "console_log_simple": "rich 콘솔 대신 단순한 형태로 로그를 출력한다.",
    "logging_dir": "로그를 남길 폴더. 이 폴더에 TensorBoard용 로그가 저장된다.",
    "log_with": "로그 출력에 사용할 도구 (tensorboard / wandb / all).",
    "log_prefix": "로그 폴더 이름 앞에 붙일 문자열.",
    "log_tracker_name": "로그에 사용할 tracker 이름. 기본은 스크립트별 기본값.",
    "wandb_run_name": "WandB에 표시될 세션 이름.",
    "log_tracker_config": "로그에 사용할 tracker 설정 파일 경로.",
    "wandb_api_key": "학습 시작 전 로그인에 사용할 WandB API 키.",
    "log_config": "학습 설정을 로그로 남긴다.",

    # --- 모델 ---
    "v2": "Stable Diffusion 2.0 계열 모델을 불러온다.",
    "v_parameterization": "v-parameterization 학습을 사용한다 (SD2.1-v 등 특정 모델 전용).",
    "tokenizer_cache_dir": "토크나이저를 캐싱할 폴더. 인터넷 연결 없이 학습할 때 사용한다.",
    "disable_mmap_load_safetensors": "safetensors 로드 시 mmap을 사용하지 않는다. WSL 환경 등에서 모델 로딩 속도 개선에 도움이 된다.",

    # --- 데이터셋 / 캡션 ---
    "cache_info": "메타 정보(캡션·이미지 크기)를 캐싱해 데이터셋 로딩 속도를 높인다. DreamBooth 방식에서만 사용 가능.",
    "caption_separator": "캡션 태그를 구분하는 문자. 기본은 쉼표(,).",
    "caption_extention": "캡션 파일 확장자 (하위 호환용 옵션. Gen2Train은 caption_extension을 고정 사용하므로 신경 쓸 필요 없음).",
    "keep_tokens_separator": "캡션을 고정 부분과 가변 부분으로 나누는 구분자. 이 구분자 앞의 태그는 셔플되지 않는다. 지정하지 않으면 keep_tokens 값으로 고정 개수를 정한다.",
    "secondary_separator": "캡션의 보조 구분자. 셔플/드롭아웃 이후 caption_separator로 치환된다.",
    "enable_wildcard": "캡션에 와일드카드(예: '{image|picture|rendition}')를 사용할 수 있게 한다.",
    "caption_prefix": "캡션 앞에 붙일 문자열.",
    "caption_suffix": "캡션 뒤에 붙일 문자열.",
    "color_aug": "약한 색상 증강(augmentation)을 사용한다.",
    "flip_aug": "좌우 반전 증강을 사용한다.",
    "face_crop_aug_range": "얼굴 중심 크롭 증강과 그 배율(예: 2.0,4.0)을 설정한다.",
    "random_crop": "랜덤 크롭을 사용한다 (얼굴 중심 증강과 함께 화풍 학습 시 사용).",
    "debug_dataset": "학습은 하지 않고 디버그용으로 이미지를 화면에 표시한다.",
    "bucket_no_upscale": "이미지를 확대하지 않고 버킷을 생성한다.",
    "resize_interpolation": "필요 시 사용할 리사이즈 보간법. 기본은 area (lanczos/nearest/bilinear/bicubic/area 중 선택).",
    "token_warmup_min": "N개의 태그부터 시작해 점점 늘려가며 학습한다.",
    "token_warmup_step": "N스텝(N<1이면 N×max_train_steps) 만에 태그 길이가 최대가 된다. 기본은 0(처음부터 최대).",
    "alpha_mask": "이미지의 알파 채널을 손실(loss) 계산용 마스크로 사용한다.",
    "dataset_class": "임의의 데이터셋 클래스를 사용할 때의 클래스 경로 (package.module.Class).",
    "caption_tag_dropout_rate": "쉼표로 구분된 태그를 드롭아웃(제거)할 확률(0.0~1.0).",
    "in_json": "데이터셋 메타데이터 json 파일 (finetune 방식 전용. Gen2Train은 항상 DreamBooth 방식을 쓰므로 사용하지 않음).",
    "dataset_repeats": "캡션으로 학습할 때 데이터셋을 반복할 횟수.",
    "max_token_length": "텍스트 인코더의 최대 토큰 길이 (기본 75, 150 또는 225 지정 가능).",
    "weighted_captions": "'(token:1.3)' 형태의 가중치 캡션을 사용한다. 괄호 안에 쉼표를 넣으면 셔플/드롭아웃이 깨질 수 있다.",
    "conditioning_data_dir": "ControlNet 등에서 사용하는 조건부 이미지 데이터 폴더.",

    # --- Hugging Face 업로드 ---
    "huggingface_repo_id": "업로드할 Hugging Face 저장소 이름.",
    "huggingface_repo_type": "업로드할 Hugging Face 저장소 종류.",
    "huggingface_path_in_repo": "업로드할 파일의 Hugging Face 저장소 내 경로.",
    "huggingface_token": "Hugging Face 토큰.",
    "huggingface_repo_visibility": "Hugging Face 저장소 공개 설정 ('public'=공개, 'private' 또는 비워두면 비공개).",
    "save_state_to_huggingface": "학습 상태(state)를 Hugging Face에도 저장한다.",
    "resume_from_huggingface": "Hugging Face에 저장된 상태로부터 이어서 학습한다.",
    "async_upload": "Hugging Face 업로드를 비동기로 진행한다.",

    # --- 저장 ---
    "save_precision": "저장 시 정밀도를 변경해서 저장한다.",
    "save_every_n_steps": "N 스텝마다 체크포인트를 저장한다 (save_every_n_epochs 대신 스텝 기준으로 저장하고 싶을 때).",
    "save_n_epoch_ratio": "지정한 비율로 체크포인트를 저장한다 (예: 5면 학습 전체에서 최소 5개 파일 저장).",
    "save_last_n_epochs": "N epoch 단위로 저장할 때 최근 N개만 남기고 이전 체크포인트는 삭제한다.",
    "save_last_n_epochs_state": "학습 상태(state)는 최근 N epoch분만 저장한다 (save_last_n_epochs 설정보다 우선 적용).",
    "save_last_n_steps": "N 스텝이 지날 때까지만 체크포인트를 유지한다 (지나면 이전 것부터 삭제).",
    "save_last_n_steps_state": "학습 상태(state)를 N 스텝 기준으로 최근 것만 유지한다 (save_last_n_steps 설정보다 우선 적용).",
    "save_state": "모델 저장 시 옵티마이저 상태 등 학습 상태(state)도 함께 저장한다. 이어서 학습할 때 유용하다.",
    "save_state_on_train_end": "학습이 끝날 때 학습 상태(state)를 저장한다.",
    "resume": "이어서 학습할 저장된 학습 상태(state) 폴더 경로.",

    # --- 학습 / 성능 ---
    "mem_eff_attn": "CrossAttention에 메모리 효율적인 attention을 사용한다.",
    "torch_compile": "torch.compile을 사용해 속도를 높인다 (PyTorch 2.0 이상 필요, 불안정할 수 있음).",
    "vae": "교체할 VAE 체크포인트 파일 또는 폴더 경로. 비워두면 베이스 모델에 내장된 VAE를 사용한다.",
    "vae_batch_size": "latent 캐싱 시 배치 크기.",
    "max_data_loader_n_workers": "DataLoader 최대 워커(프로세스) 수. 작을수록 메모리 사용은 줄고 epoch 시작이 빨라지지만 데이터 로딩은 느려진다.",
    "persistent_data_loader_workers": "DataLoader 워커를 계속 유지한다 (epoch 사이 대기시간이 줄지만 메모리를 더 쓸 수 있음).",
    "skip_cache_check": "캐시(latent/text encoder 출력) 내용 검증을 건너뛴다. 파일 존재 여부 확인은 항상 수행된다.",
    "ddp_timeout": "DDP(분산 학습) 타임아웃(분). 비워두면 accelerate 기본값을 사용한다.",
    "ddp_gradient_as_bucket_view": "DDP에서 gradient_as_bucket_view 옵션을 사용한다.",
    "ddp_static_graph": "DDP에서 static_graph 옵션을 사용한다.",
    "lowram": "메인 메모리가 적은 환경용 최적화 (예: VRAM에 모델을 바로 로드). Colab/Kaggle처럼 RAM보다 VRAM이 넉넉한 환경에 적합.",
    "highvram": "VRAM이 적은 환경을 위한 최적화를 끈다 (예: latent 캐싱마다 CUDA 캐시 정리를 하지 않음). VRAM이 넉넉할 때 사용.",
    "deepspeed": "DeepSpeed로 학습한다 (대규모 분산 학습용, 일반적인 개인 GPU 1대 학습에는 필요 없음).",
    "zero_stage": "DeepSpeed ZeRO 단계 (0/1/2/3 중 선택). deepspeed 사용 시에만 의미가 있다.",
    "offload_optimizer_device": "옵티마이저 상태를 오프로드할 장치 (none/cpu/nvme). DeepSpeed ZeRO 2·3 전용.",
    "offload_optimizer_nvme_path": "옵티마이저 상태를 오프로드할 NVMe 경로. DeepSpeed ZeRO 3 전용.",
    "offload_param_device": "파라미터를 오프로드할 장치 (none/cpu/nvme). DeepSpeed ZeRO 3 전용.",
    "offload_param_nvme_path": "파라미터를 오프로드할 NVMe 경로. DeepSpeed ZeRO 3 전용.",
    "zero3_init_flag": "대형 모델 구성 시 deepspeed.zero.Init 사용 여부. DeepSpeed ZeRO 3 전용.",
    "zero3_save_16bit_model": "16bit 모델로 저장할지 여부. DeepSpeed ZeRO 3 전용.",
    "fp16_master_weights_and_gradients": "fp16 master weight/gradient를 사용한다 (옵티마이저가 fp32 상태 유지를 지원해야 함).",
    "use_8bit_adam": "8bit AdamW 옵티마이저를 사용한다 (bitsandbytes 필요, optimizer_type=AdamW8bit과 동일한 효과).",
    "use_lion_optimizer": "Lion 옵티마이저를 사용한다 (lion-pytorch 필요).",
    "fused_backward_pass": "backward pass와 옵티마이저 step을 합쳐 VRAM 사용량을 줄인다. SDXL/SD3/FLUX 전용.",

    # --- 옵티마이저 / LR 스케줄러 ---
    "max_grad_norm": "그래디언트 클리핑 최대 norm. 0이면 클리핑을 하지 않는다.",
    "optimizer_args": "옵티마이저에 전달할 추가 인자 (예: \"weight_decay=0.01 betas=0.9,0.999\").",
    "lr_scheduler_type": "직접 지정하는 커스텀 스케줄러 모듈명.",
    "lr_scheduler_args": "LR 스케줄러에 전달할 추가 인자 (예: \"T_max=100\").",
    "lr_decay_steps": "LR 스케줄러의 감쇠 스텝 수 (정수, 기본 0) 또는 전체 학습 스텝 대비 비율 (1 미만 소수로 지정 가능).",
    "lr_scheduler_timescale": "inverse sqrt 스케줄러의 타임스케일. 기본은 num_warmup_steps와 동일.",
    "lr_scheduler_min_lr_ratio": "cosine with min lr / warmup decay 스케줄러에서, 초기 학습률 대비 최소 학습률의 비율.",

    # --- 노이즈 / 손실 함수 ---
    "noise_offset": "Noise offset을 사용한다 (켤 경우 0.1 전후 권장).",
    "noise_offset_random_strength": "noise offset에 0~noise_offset 사이 무작위 강도를 사용한다.",
    "multires_noise_iterations": "Multires noise를 사용하고 반복 횟수를 지정한다 (켤 경우 6~10 권장).",
    "ip_noise_gamma": "입력 섭동(input perturbation) noise를 사용한다. 정규화 목적이며 권장값은 0.1 전후.",
    "ip_noise_gamma_random_strength": "input perturbation noise에 0~ip_noise_gamma 사이 무작위 강도를 사용한다.",
    "multires_noise_discount": "Multires noise의 discount 값 (multires_noise_iterations를 켰을 때만 유효).",
    "adaptive_noise_scale": "'latent 평균 절댓값 × 이 값'을 noise_offset에 더한다 (기본은 비활성화).",
    "zero_terminal_snr": "noise scheduler의 beta를 보정해 terminal SNR을 0으로 강제한다.",
    "min_timestep": "U-Net 학습 시 최소 timestep (0~999, 기본 0).",
    "max_timestep": "U-Net 학습 시 최대 timestep (1~1000, 기본 1000).",
    "loss_type": "사용할 손실 함수 종류 (L1/L2/Huber/smooth L1). 기본은 L2.",
    "huber_schedule": "Huber loss의 스케줄링 방식 (constant/exponential/SNR 기반). loss_type이 huber나 smooth_l1일 때만 사용. 기본 snr.",
    "huber_c": "Huber loss의 감쇠 파라미터. huber/smooth l1 손실일 때만 사용. 기본 0.1.",
    "huber_scale": "Huber loss의 스케일 파라미터. huber/smooth l1 손실일 때만 사용. 기본 1.0.",
    "min_snr_gamma": "손실이 큰 타임스텝의 가중치를 줄이는 gamma 값. 작을수록 효과가 강하며, 논문 권장값은 5.",
    "scale_v_pred_loss_like_noise_pred": "v-prediction 손실을 noise prediction 손실과 같은 방식으로 스케일링한다.",
    "v_pred_like_loss": "이 값을 곱한 v-prediction 유사 손실을 추가로 더한다.",
    "debiased_estimation_loss": "debiased estimation loss를 사용한다 (SNR 기반 loss 가중치 보정 기법).",
    "masked_loss": "손실 계산 시 마스크를 적용한다. 데이터셋에 conditioning_data_dir이 필요하다.",

    # --- LoRA 네트워크 ---
    "cpu_offload_checkpointing": "[실험적] gradient checkpointing 시 텐서를 CPU로 오프로드한다 (U-Net/DiT 중 지원하는 경우만).",
    "no_metadata": "결과 모델 파일에 메타데이터를 저장하지 않는다.",
    "fp8_base": "베이스 모델에 fp8을 사용해 VRAM을 아낀다.",
    "fp8_base_unet": "U-Net(또는 DiT)에 fp8을 사용한다. Text Encoder는 fp16/bf16을 유지한다.",
    "base_weights": "학습 전 모델에 미리 병합할 네트워크 가중치 파일.",
    "base_weights_multiplier": "위 base_weights를 병합할 때 곱할 배율.",

    # --- 샘플 이미지 ---
    "sample_at_first": "학습 시작 전에도 샘플 이미지를 한 번 생성해본다.",

    # --- 메타데이터 ---
    "output_config": "지금 지정한 명령줄 인자들을 .toml 파일로 저장한다.",
    "metadata_title": "모델 메타데이터에 기록할 제목 (기본은 output_name).",
    "metadata_author": "모델 메타데이터에 기록할 작성자명.",
    "metadata_description": "모델 메타데이터에 기록할 설명.",
    "metadata_license": "모델 메타데이터에 기록할 라이선스.",
    "metadata_tags": "모델 메타데이터에 기록할 태그 (쉼표로 구분).",

    # --- 재개/검증 ---
    "skip_until_initial_step": "initial_step에 도달할 때까지 학습을 건너뛴다.",
    "initial_epoch": "시작 epoch 번호 (1이면 처음부터, 기본과 동일). LR 스케줄러에는 영향을 주지 않는다 (--resume 없이는 스케줄러가 0부터 시작한다).",
    "initial_step": "전체 epoch을 포함한 시작 스텝 번호 (0이면 처음부터). initial_epoch 설정보다 우선한다.",
    "validation_seed": "검증 데이터셋을 섞을 때 사용할 시드. 지정하지 않으면 학습용 seed를 그대로 사용한다.",
    "validation_split": "학습 데이터셋에서 검증용으로 떼어낼 비율.",
    "validate_every_n_steps": "N 스텝마다 검증 데이터셋으로 검증한다. 기본은 epoch마다 한 번.",
    "validate_every_n_epochs": "N epoch마다 검증 데이터셋으로 검증한다.",
    "max_validation_steps": "검증에 사용할 최대 항목 수. 기본은 검증 데이터셋 전체.",
    "cache_text_encoder_outputs_to_disk": "Text Encoder 출력을 디스크에 캐싱한다 (재학습 시 다시 계산하지 않아도 됨).",
}

# help_texts.py가 커버하지 못하는 dest가 남아 있어도 원본 일본어가 그대로 새지 않도록,
# " / 日本語" 꼬리를 잘라내는 안전망. log_translate.py의 것과 동일한 목적/패턴이다.
import re as _re

_CJK_TAIL_RE = _re.compile(r"\s*/\s*[^/]*[぀-ヿ゠-ヿ一-鿿][^/]*$")


def get_help(dest: str, fallback: str) -> str:
    """dest에 대한 한국어 설명을 반환한다. 없으면 fallback(주로 sd-scripts 원본 help)에서
    "/ 일본어" 꼬리만 제거해 최소한 일본어가 노출되지 않게 한다."""
    if dest in HELP_TEXTS:
        return HELP_TEXTS[dest]
    return _CJK_TAIL_RE.sub("", fallback)

GROUP_DESCRIPTIONS: dict[str, str] = {
    "모델": "베이스 모델 관련 옵션 (버전, 파라미터화 방식 등).",
    "LoRA 네트워크": "LoRA의 구조(랭크/알파)와 학습 강도를 결정하는 핵심 설정.",
    "데이터셋 / 캡션": "이미지 해상도, 버킷팅, 캡션 처리 방식 등 데이터 관련 설정.",
    "저장": "체크포인트 저장 방식과 주기.",
    "샘플 이미지": "학습 중 미리보기 이미지를 생성해 진행 상황을 확인하는 기능.",
    "노이즈 / 손실 함수": "학습 안정성/품질에 영향을 주는 노이즈·손실 계산 방식. 기본값 그대로 둬도 대부분 괜찮다.",
    "옵티마이저 / LR 스케줄러": "학습을 최적화하는 알고리즘과 학습률 변화 계획.",
    "학습 / 성능": "속도, VRAM 사용량, 하드웨어 활용과 관련된 설정.",
    "기타": "위 분류에 속하지 않는 그 외 옵션들.",
}
