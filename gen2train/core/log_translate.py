"""sd-scripts 학습 로그를 한국어로 번역해서 로그 콘솔에 보여준다.

sd-scripts 소스 자체는 건드리지 않는다(업스트림 갱신 시 충돌을 피하기 위함). 대신 trainer.py가
한 줄씩 이 모듈의 translate_line()을 거쳐 화면에 표시하기 전에, 아는 패턴은 한국어로 바꾸고
모르는 패턴은 "영어 / 日本語" 꼬리의 일본어 부분만 잘라내 최소한 잡음을 줄인다.

sd-scripts는 두 가지 방식으로 로그를 찍는다:
  1) logger.info()/logger.warning() - rich 콘솔 핸들러가 "타임스탬프 INFO   메시지   파일:줄" 형태로 감싼다.
  2) accelerator.print()/print() - 아무 장식 없이 메시지 그대로 출력된다.
그래서 패턴을 줄 전체(^...$)가 아니라 부분 매칭으로 찾아 그 부분만 치환해야, 두 경우 모두
(그리고 rich가 붙이는 타임스탬프/파일경로도) 안전하게 처리된다.
"""
import re

# "... / 日本語 ..." 형태에서 뒤쪽 CJK(한자/히라가나/가타카나) 꼬리를 제거하는 안전망.
# 아래 번역 규칙에 없는 문장에 최후 수단으로 적용된다.
_CJK_TAIL_RE = re.compile(r"\s*/\s*[^/]*[぀-ヿ゠-ヿ一-鿿][^/]*$")


def _rule(pattern: str, replacement: str):
    return (re.compile(pattern), replacement)


# 순서대로 검사하며 먼저 매치되는 규칙 하나만 적용한다(부분 치환이라 뒤쪽 파일:줄 등은 보존됨).
_RULES: list[tuple[re.Pattern, str]] = [
    # --- 데이터셋 준비 ---
    _rule(r"Using DreamBooth method\.", "DreamBooth 방식을 사용합니다."),
    _rule(r"Training with captions\.", "캡션을 사용해 학습합니다."),
    _rule(
        r"No data found\. Please verify arguments.*",
        "이미지를 찾지 못했습니다. train_data_dir 경로를 확인하세요"
        " (이미지가 든 폴더가 아니라, 그 폴더들의 부모 폴더를 지정해야 합니다).",
    ),
    _rule(r"prepare images\.", "이미지를 준비합니다."),
    _rule(r"loading image sizes\.", "이미지 크기 정보를 불러옵니다."),
    _rule(r"make buckets", "버킷을 생성합니다."),
    _rule(r"prepare dataset", "데이터셋을 준비합니다."),
    _rule(r"get image size from name of\s*$", "캐시 파일 이름에서 이미지 크기를 가져옵니다."),
    _rule(r"get image size from name of cache files", "캐시 파일 이름에서 이미지 크기를 가져옵니다."),
    _rule(r"set image size from cache\s*$", "캐시 파일에서 이미지 크기를 설정합니다."),
    _rule(r"set image size from cache files: (\d+)/(\d+)", r"캐시 파일에서 이미지 크기 설정: \1/\2"),
    _rule(r"found directory (.+?) contains (\d+) image files", r"폴더 \1 에서 이미지 \2장을 찾았습니다."),
    _rule(r"(\d+) train images with repeats\.", r"학습 이미지 \1장(반복 포함)."),
    _rule(r"(\d+) reg images with repeats\.", r"정규화 이미지 \1장(반복 포함)."),
    _rule(r"some of reg images are not used.*", "정규화 이미지 수가 많아 일부는 사용되지 않습니다."),
    _rule(r"no regularization images.*", "정규화 이미지가 없습니다."),
    _rule(r"caching latents with caching strategy\.", "캐싱 전략으로 latent를 캐싱합니다."),
    _rule(r"caching latents\.\.\.", "latent를 캐싱하는 중입니다..."),
    _rule(r"caching latents\.", "latent 캐싱 완료."),
    _rule(r"checking cache validity\.\.\.", "캐시 유효성을 확인하는 중입니다..."),
    _rule(r"caching Text Encoder outputs with caching strategy\.", "캐싱 전략으로 Text Encoder 출력을 캐싱합니다."),
    _rule(r"caching Text Encoder outputs\.\.\.", "Text Encoder 출력을 캐싱하는 중입니다..."),
    _rule(r"caching text encoder outputs\.\.\.", "Text Encoder 출력을 캐싱하는 중입니다..."),
    _rule(r"caching text encoder outputs\.", "Text Encoder 출력 캐싱 완료."),
    _rule(r"checking cache existence\.\.\.", "캐시 존재 여부를 확인하는 중입니다..."),
    _rule(r"no Text Encoder outputs to cache", "캐싱할 Text Encoder 출력이 없습니다."),
    _rule(
        r"No caption file found for (\d+) images.*",
        r"\1장의 이미지에서 캡션 파일을 찾지 못했습니다. 캡션 없이 학습을 계속합니다(class token이 있으면 그것을 사용합니다).",
    ),
    _rule(r"neither caption file nor class tokens are found.*", "캡션 파일도 class token도 찾지 못해 빈 캡션을 사용합니다."),
    _rule(r"npz file does not exist\. ignore npz files.*", "npz 캐시 파일이 없어 무시합니다."),
    _rule(r"some of npz file does not exist\. ignore npz files.*", "일부 npz 캐시 파일이 없어 무시합니다."),

    # --- 모델 로딩 ---
    _rule(r"preparing accelerator", "accelerator를 준비합니다."),
    _rule(r"loading model for process (\d+)/(\d+)", r"프로세스 \1/\2 에서 모델을 불러옵니다."),
    _rule(r"load StableDiffusion checkpoint: (.+)", r"Stable Diffusion 체크포인트를 불러옵니다: \1"),
    _rule(r"load Diffusers pretrained models: (.+)", r"Diffusers 사전학습 모델을 불러옵니다: \1"),
    _rule(r"model is not found as a file or in Hugging Face.*", "모델 파일을 찾을 수 없습니다. 경로/파일명을 확인하세요."),
    _rule(r"building U-Net", "U-Net을 구성합니다."),
    _rule(r"loading U-Net from checkpoint", "체크포인트에서 U-Net을 불러옵니다."),
    _rule(r"U-Net: (<All keys matched successfully>)", r"U-Net: \1 (정상적으로 로드됨)"),
    _rule(r"building text encoders", "텍스트 인코더를 구성합니다."),
    _rule(r"loading text encoders from checkpoint", "체크포인트에서 텍스트 인코더를 불러옵니다."),
    _rule(r"text encoder (\d+): (<All keys matched successfully>)", r"텍스트 인코더 \1: \2 (정상적으로 로드됨)"),
    _rule(r"loading text encoder: (.+)", r"텍스트 인코더를 불러옵니다: \1"),
    _rule(r"building VAE", "VAE를 구성합니다."),
    _rule(r"loading VAE from checkpoint", "체크포인트에서 VAE를 불러옵니다."),
    _rule(r"VAE: (<All keys matched successfully>)", r"VAE: \1 (정상적으로 로드됨)"),
    _rule(r"load VAE: (.+)", r"VAE를 불러옵니다: \1"),
    _rule(r"U-Net converted to original U-Net", "U-Net을 원본 U-Net 형식으로 변환했습니다."),
    _rule(r"additional VAE loaded", "추가 VAE를 불러왔습니다."),
    _rule(r"Enable xformers for U-Net", "U-Net에 xformers를 사용합니다."),
    _rule(r"Enable SDPA for U-Net", "U-Net에 SDPA를 사용합니다."),
    _rule(r"Enable memory efficient attention for U-Net", "U-Net에 메모리 효율 attention을 사용합니다."),
    _rule(r"move vae and unet to cpu to save memory", "메모리 절약을 위해 VAE와 U-Net을 CPU로 이동합니다."),
    _rule(r"move vae and unet back to original device", "VAE와 U-Net을 원래 장치로 되돌립니다."),
    _rule(r"prepare tokenizers", "토크나이저를 준비합니다."),
    _rule(r"try to load fp32 model", "fp32 모델을 불러오는 중입니다."),
    _rule(r"update token length: (\d+)", r"토큰 길이를 갱신합니다: \1"),
    _rule(r"accelerator device:", "accelerator 장치:"),

    # --- LoRA 네트워크 ---
    _rule(r"import network module:\s*(.+)", r"네트워크 모듈을 불러옵니다: \1"),
    _rule(
        r"create LoRA network\. base dim \(rank\): ([\d.]+), alpha: ([\d.]+)",
        r"LoRA 네트워크 생성. 기본 dim(rank): \1, alpha: \2",
    ),
    _rule(r"create LoRA network from weights", "가중치 파일로부터 LoRA 네트워크를 생성합니다."),
    _rule(r"create LoRA network from block_dims", "block_dims로부터 LoRA 네트워크를 생성합니다."),
    _rule(
        r"neuron dropout: p=(\S+), rank dropout: p=(\S+), module dropout: p=(\S+)",
        r"neuron dropout: p=\1, rank dropout: p=\2, module dropout: p=\3",
    ),
    _rule(r"create LoRA for Text Encoder (\d+):", r"Text Encoder \1 용 LoRA를 생성합니다:"),
    _rule(r"create LoRA for Text Encoder:\s*(\d+) modules\.", r"Text Encoder용 LoRA \1개 모듈을 생성했습니다."),
    _rule(r"create LoRA for Text Encoder:\s*$", "Text Encoder용 LoRA를 생성합니다:"),
    _rule(r"create LoRA for U-Net: (\d+) modules\.", r"U-Net용 LoRA \1개 모듈을 생성했습니다."),
    _rule(r"enable LoRA for text encoder: (\d+) modules", r"Text Encoder LoRA \1개 모듈을 활성화합니다."),
    _rule(r"enable LoRA for U-Net: (\d+) modules", r"U-Net LoRA \1개 모듈을 활성화합니다."),
    _rule(r"enable LoRA for text encoder(?!:)", "Text Encoder LoRA를 활성화합니다."),
    _rule(r"enable LoRA for U-Net(?! [\d:])", "U-Net LoRA를 활성화합니다."),
    _rule(r"weights are merged", "가중치가 병합되었습니다."),
    _rule(r"load network weights from (.+?): (.+)", r"기존 LoRA 가중치를 불러옵니다: \1 (\2)"),
    _rule(
        r"warning: scale_weight_norms is specified but the network does not support it.*",
        "경고: scale_weight_norms가 지정됐지만 이 네트워크는 지원하지 않습니다.",
    ),
    _rule(r"apply block learning rate.*", "계층별 학습률을 적용합니다."),
    _rule(r"NO LR skipping!", "학습률 0인 레이어를 건너뛰지 않습니다!"),

    # --- 옵티마이저 / 스케줄러 ---
    _rule(r"use (.+?) optimizer \|", r"\1 옵티마이저 사용 |"),
    _rule(r"wrap optimizer with (.+?) \|", r"\1(으)로 옵티마이저를 감쌉니다 |"),
    _rule(r"use (.+?) \|(.*) as lr_scheduler", r"\1 스케줄러 사용 |\2"),
    _rule(r"prepare optimizer, data loader etc\.", "옵티마이저/데이터로더 등을 준비합니다."),
    _rule(
        r"learning rate is too low.*?lr=(\S+)",
        r"학습률이 너무 낮은 것 같습니다(D-Adaptation/Prodigy는 보통 1.0 전후를 권장합니다): lr=\1",
    ),
    _rule(r"recommend option: lr=1\.0.*", "권장: lr=1.0"),
    _rule(r"8-bit SGD with Nesterov must be with momentum.*", "8bit SGD(Nesterov)는 momentum이 필요해 0.9로 설정합니다."),
    _rule(r"SGD with Nesterov must be with momentum.*", "SGD(Nesterov)는 momentum이 필요해 0.9로 설정합니다."),
    _rule(r"set relative_step to True because warmup_init is True.*", "warmup_init이 켜져 있어 relative_step을 True로 설정합니다."),
    _rule(r"relative_step is true", "relative_step이 켜져 있습니다."),
    _rule(r"learning rate is used as initial_lr.*", "지정한 learning rate가 initial_lr로 사용됩니다."),
    _rule(r"unet_lr and text_encoder_lr are ignored.*", "unet_lr / text_encoder_lr 값은 무시됩니다."),
    _rule(r"use adafactor_scheduler.*", "adafactor_scheduler를 사용합니다."),
    _rule(
        r"because max_grad_norm is set, clip_grad_norm is enabled.*",
        "max_grad_norm이 설정되어 clip_grad_norm이 활성화됩니다(0으로 두면 비활성화할 수 있습니다).",
    ),
    _rule(r"constant_with_warmup will be good.*", "스케줄러는 constant_with_warmup이 적합할 수 있습니다."),
    _rule(r"clip_threshold=1\.0 will be good.*", "clip_threshold는 1.0이 적합할 수 있습니다."),
    _rule(
        r"when multiple learning rates are specified with dadaptation.*?lr=(\S+)",
        r"D-Adaptation/Prodigy에 여러 학습률을 지정해도(TE, U-Net 등) 첫 번째 값만 적용됩니다: lr=\1",
    ),

    # --- 학습 시작 요약 ---
    _rule(r"running training", "학습을 시작합니다."),
    _rule(r"num train images \* repeats.*?:\s*(\d+)", r"학습 이미지 수 × 반복 횟수: \1"),
    _rule(r"num validation images \* repeats.*?:\s*(\d+)", r"검증 이미지 수 × 반복 횟수: \1"),
    _rule(r"num reg images.*?:\s*(\d+)", r"정규화 이미지 수: \1"),
    _rule(r"num batches per epoch.*?:\s*(\d+)", r"epoch당 배치 수: \1"),
    _rule(r"num epochs.*?:\s*(\d+)", r"epoch 수: \1"),
    _rule(r"batch size per device.*?:\s*(.+)", r"디바이스당 배치 크기: \1"),
    _rule(r"gradient accumulation steps.*?=\s*(\d+)", r"그래디언트 누적 스텝 수: \1"),
    _rule(r"total optimization steps.*?:\s*(\d+)", r"총 학습 스텝 수: \1"),
    _rule(
        r"override steps\. steps for (\d+) epochs is.*?:\s*(\d+)",
        r"\1 epoch 기준으로 학습 스텝 수를 재계산했습니다: \2 스텝",
    ),
    _rule(r"skipping (\d+) steps", r"\1 스텝을 건너뜁니다"),
    _rule(r"epoch is incremented\. current_epoch: (\d+), epoch: (\d+)", r"epoch가 증가했습니다. current_epoch: \1, epoch: \2"),
    _rule(r"epoch is not incremented\. current_epoch: (\d+), epoch: (\d+)", r"epoch가 증가하지 않았습니다. current_epoch: \1, epoch: \2"),
    _rule(r"text_encoder is not needed for training\. deleting to save memory\.", "학습에 텍스트 인코더가 필요 없어 메모리 절약을 위해 삭제합니다."),

    # --- 저장 ---
    _rule(r"saving checkpoint: (.+)", r"체크포인트 저장 중: \1"),
    _rule(r"removing old checkpoint: (.+)", r"이전 체크포인트 삭제: \1"),
    _rule(r"saving model: (.+)", r"모델 저장 중: \1"),
    _rule(r"removing old model: (.+)", r"이전 모델 삭제: \1"),
    _rule(r"model saved\.", "모델이 저장되었습니다."),
    _rule(r"save trained model as StableDiffusion checkpoint to (.+)", r"학습된 모델을 Stable Diffusion 체크포인트로 저장합니다: \1"),
    _rule(r"save trained model as Diffusers to (.+)", r"학습된 모델을 Diffusers 형식으로 저장합니다: \1"),
    _rule(r"saving state at epoch (\d+)", r"epoch \1 시점 학습 상태 저장 중"),
    _rule(r"saving state at step (\d+)", r"step \1 시점 학습 상태 저장 중"),
    _rule(r"saving last state\.", "마지막 학습 상태를 저장합니다."),
    _rule(r"removing old state: (.+)", r"이전 학습 상태 삭제: \1"),
    _rule(r"uploading (last )?state to huggingface\.", "Hugging Face에 학습 상태를 업로드합니다."),

    # --- 자주 보이는 경고 ---
    _rule(r"cache_latents_to_disk is enabled.*", "cache_latents_to_disk가 켜져 있어 cache_latents도 함께 켭니다."),
    _rule(r"highvram is enabled.*", "highvram 모드가 켜져 있습니다."),
    _rule(r"zero_terminal_snr is enabled, but v_parameterization is not enabled.*", "zero_terminal_snr가 켜져 있지만 v_parameterization이 꺼져 있어 학습 결과가 예상과 다를 수 있습니다."),
    _rule(r"sample_every_n_epochs is less than or equal to 0.*", "sample_every_n_epochs가 0 이하라서 비활성화됩니다."),
    _rule(r"sample_every_n_steps is less than or equal to 0.*", "sample_every_n_steps가 0 이하라서 비활성화됩니다."),
    _rule(r"v2 with clip_skip will be unexpected.*", "v2 모델에서 clip_skip을 사용하면 예상치 못한 결과가 나올 수 있습니다."),
    _rule(r"clip_skip will be unexpected.*", "SDXL 학습에서는 clip_skip이 동작하지 않습니다."),
    _rule(r"cache_text_encoder_outputs is enabled because cache_text_encoder_outputs_to_disk.*", "cache_text_encoder_outputs_to_disk가 켜져 있어 cache_text_encoder_outputs도 함께 켭니다."),
    _rule(r"latents in npz is ignored when color_aug or random_crop is True.*", "color_aug/random_crop이 켜져 있으면 npz의 latent 캐시는 무시됩니다."),
    _rule(r"ignore subset with image_dir='(.+?)': num_repeats is less than 1.*", r"num_repeats가 1 미만이라 서브셋을 무시합니다: \1"),
    _rule(r"ignore duplicated subset with image_dir='(.+?)': use the first one.*", r"중복된 서브셋이라 무시합니다(먼저 등록된 것을 사용): \1"),
    _rule(r"ignore subset with image_dir='(.+?)': no images found.*", r"이미지가 없어 서브셋을 무시합니다: \1"),
    _rule(r"illegal char in file \(not UTF-8\).*?:\s*(.+)", r"캡션 파일에 UTF-8이 아닌 문자가 있습니다: \1"),

    # --- 설정 파일 ---
    _rule(r"Config file already exists\. Aborting\.\.\..*?:\s*(.+)", r"설정 파일이 이미 존재해서 중단합니다: \1"),
    _rule(r"Saved config file.*?:\s*(.+)", r"설정 파일을 저장했습니다: \1"),
    _rule(r"Loading settings from (.+?)\.\.\.", r"설정을 불러옵니다: \1"),
    _rule(r"resume training from local state: (.+)", r"로컬 학습 상태에서 이어서 학습합니다: \1"),
    _rule(r"resume training from huggingface state: (.+)", r"Hugging Face 학습 상태에서 이어서 학습합니다: \1"),
    _rule(r"Invalid user config.*", "사용자 설정 형식이 올바르지 않은 것 같습니다."),

    # --- 데이터셋 요약 블록 (config_util.py) : 들여쓰기는 그대로 유지 ---
    _rule(r"\[Dataset (\d+)\]", r"[데이터셋 \1]"),
    _rule(r"\[Prepare dataset (\d+)\]", r"[데이터셋 \1 준비]"),
    _rule(r"\[Prepare validation dataset (\d+)\]", r"[검증 데이터셋 \1 준비]"),
    _rule(r"\[Subset (\d+) of Dataset (\d+)\]", r"[데이터셋 \2의 서브셋 \1]"),
    _rule(r"(^\s*)batch_size:", r"\1배치 크기:"),
    _rule(r"(^\s*)resolution:", r"\1해상도:"),
    _rule(r"(^\s*)resize_interpolation:", r"\1리사이즈 보간:"),
    _rule(r"(^\s*)enable_bucket:", r"\1버킷 사용:"),
    _rule(r"(^\s*)min_bucket_reso:", r"\1최소 버킷 해상도:"),
    _rule(r"(^\s*)max_bucket_reso:", r"\1최대 버킷 해상도:"),
    _rule(r"(^\s*)bucket_reso_steps:", r"\1버킷 해상도 단위:"),
    _rule(r"(^\s*)bucket_no_upscale:", r"\1버킷 업스케일 금지:"),
    _rule(r"(^\s*)image_dir:", r"\1이미지 폴더:"),
    _rule(r"(^\s*)image_count:", r"\1이미지 수:"),
    _rule(r"(^\s*)num_repeats:", r"\1반복 횟수:"),
    _rule(r"(^\s*)shuffle_caption:", r"\1캡션 셔플:"),
    _rule(r"(^\s*)keep_tokens:", r"\1keep_tokens:"),
    _rule(r"(^\s*)caption_dropout_rate:", r"\1캡션 드롭아웃 확률:"),
    _rule(r"(^\s*)caption_dropout_every_n_epochs:", r"\1캡션 드롭아웃 주기(epoch):"),
    _rule(r"(^\s*)caption_tag_dropout_rate:", r"\1태그 드롭아웃 확률:"),
    _rule(r"(^\s*)caption_prefix:", r"\1캡션 접두어:"),
    _rule(r"(^\s*)caption_suffix:", r"\1캡션 접미어:"),
    _rule(r"(^\s*)color_aug:", r"\1색상 증강:"),
    _rule(r"(^\s*)flip_aug:", r"\1좌우 반전 증강:"),
    _rule(r"(^\s*)face_crop_aug_range:", r"\1얼굴 크롭 증강 범위:"),
    _rule(r"(^\s*)random_crop:", r"\1랜덤 크롭:"),
    _rule(r"(^\s*)alpha_mask:", r"\1alpha mask:"),
    _rule(r"(^\s*)custom_attributes:", r"\1사용자 정의 속성:"),
    _rule(r"(^\s*)is_reg:", r"\1정규화 이미지 여부:"),
    _rule(r"(^\s*)class_tokens:", r"\1class token:"),
    _rule(r"(^\s*)caption_extension:", r"\1캡션 확장자:"),
    _rule(r"(^\s*)metadata_file:", r"\1메타데이터 파일:"),
    _rule(r"number of images \(including repeats\).*", "버킷별 이미지 수(반복 포함):"),
    _rule(r"mean ar error \(without repeats\): (.+)", r"평균 종횡비 오차(반복 제외): \1"),
]


def translate_line(line: str) -> str:
    for pattern, replacement in _RULES:
        if pattern.search(line):
            line = pattern.sub(replacement, line, count=1)
            break
    # 규칙이 매치된 부분만 치환하고 " / 日本語" 꼬리가 뒤에 남는 경우가 있어(예: "running training"
    # 규칙은 "running training"까지만 치환하고 뒤의 " / 学習開始"는 그대로 남음), 매치 여부와
    # 상관없이 항상 마지막에 한 번 더 CJK 꼬리 제거를 적용한다.
    return _CJK_TAIL_RE.sub("", line)
