"""ParamStore의 현재 값 + help_texts.py의 한국어 설명을 챗봇 프롬프트용 컨텍스트로 조립한다.

전체 176개 파라미터를 다 넣으면 컨텍스트가 커지고(모델의 <thought> 추론이 원래도 길다)
정작 중요한 정보가 묻히므로, (1) 기본 탭에 해당하는 핵심 파라미터는 항상 포함하고
(2) 그 외에는 기본값에서 실제로 바뀐 것만 포함한다.
"""
from . import arg_introspect, system_info
from .help_texts import get_help
from .param_store import ParamStore

# ui/basic_panel.py의 핵심 파라미터 목록과 맞춘다 (거기서 import하면 core가 ui에 의존하게 되므로 여기 따로 둔다).
CORE_DESTS = [
    "resolution",
    "train_batch_size",
    "gradient_accumulation_steps",
    "max_train_epochs",
    "learning_rate",
    "network_dim",
    "network_alpha",
    "save_every_n_epochs",
    "seed",
    "mixed_precision",
    "optimizer_type",
    "text_encoder_lr",
]

TOP_LABELS = {
    "model_type": "모델 타입",
    "trigger_word": "트리거 단어",
    "extra_tags": "추가 태그",
    "use_reg": "정규화 이미지 사용",
    "class_word": "클래스 단어",
    "repeat": "결함 이미지 반복 횟수",
    "reg_repeat": "정규화 이미지 반복 횟수",
    "output_name": "결과물 이름",
}

RECOMMEND_QUESTION = "지금 설정을 검토하고 개선하면 좋을 점이 있는지 추천해줘."


def _format_param(dest: str, value, spec) -> str:
    return f"- {spec.flag} = {value}  ({get_help(dest, spec.help)})"


def build_context(
    store: ParamStore, model_type: str, top_state: dict = None, system_snapshot: dict = None
) -> str:
    lines = []

    if system_snapshot:
        system_text = system_info.format_snapshot(system_snapshot)
        if system_text:
            lines.append(system_text)
            lines.append("")

    lines.append("[현재 Gen2Train 설정]")

    if top_state:
        top_lines = []
        for dest, label in TOP_LABELS.items():
            value = top_state.get(dest)
            if value in (None, "", False):
                continue
            top_lines.append(f"- {label}: {value}")
        if top_lines:
            lines.extend(top_lines)

    specs = {s.dest: s for s in arg_introspect.get_arg_specs(model_type)}
    values = store.as_dict()

    core_lines = []
    for dest in CORE_DESTS:
        spec = specs.get(dest)
        if spec is None:
            continue
        value = values.get(dest, spec.default)
        if value is None:
            continue
        core_lines.append(_format_param(dest, value, spec))

    changed_lines = []
    for dest, value in values.items():
        if dest in CORE_DESTS:
            continue
        spec = specs.get(dest)
        if spec is None or value is None or value == spec.default:
            continue
        changed_lines.append(_format_param(dest, value, spec))

    if core_lines:
        lines.append("")
        lines.append("[핵심 학습 파라미터]")
        lines.extend(core_lines)

    if changed_lines:
        lines.append("")
        lines.append("[기본값에서 변경된 그 외 파라미터]")
        lines.extend(changed_lines)

    return "\n".join(lines)
