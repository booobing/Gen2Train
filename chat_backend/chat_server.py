"""Gen2Train 챗봇 백엔드. venv_chat의 python으로 실행되는 상주 프로세스.

stdin에서 JSON을 한 줄씩 요청으로 받아 stdout으로 토큰을 스트리밍한다 (JSON Lines 프로토콜).
모델은 시작할 때 한 번만 로드하고, 이후 요청은 전부 이 상주 프로세스가 재사용한다.

요청: {"id": "...", "context": "...", "question": "..."}
응답: {"type": "token", "id": "...", "text": "..."}   (여러 번)
      {"type": "done", "id": "..."}
시작 시: {"type": "ready", "backend": "..."}
오류:   {"type": "error", "id": "...", "message": "..."}   (요청 단위 오류, 서버는 계속 동작)
치명적 오류(모델 로딩 실패 등): {"type": "fatal_error", "message": "..."}
"""
import json
import sys
from pathlib import Path

# Windows에서는 파이프로 연결된 stdin/stdout이 시스템 로케일(cp949 등)로 기본 설정되곤 한다.
# 그러면 부모 프로세스가 UTF-8로 보낸 한글 요청을 읽는 순간부터 깨져서(surrogateescape로
# lone surrogate가 섞여 들어옴) 이후 어떤 처리를 해도 인코딩 에러가 난다. 그래서 가장 먼저
# stdin/stdout을 명시적으로 UTF-8로 강제한다.
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

CHAT_BACKEND_DIR = Path(__file__).resolve().parent
BACKEND_CONFIG_PATH = CHAT_BACKEND_DIR / "backend_config.json"

# EXAONE-Deep 공식 권장 설정: 시스템 프롬프트를 쓰지 않고, repetition penalty는 1.0 이하로 고정.
# temperature는 파라미터 추천/QA 특성상 낮게(0.3 근처) 잡아 답변을 더 결정적으로 만든다.
GENERATION_KWARGS = dict(
    max_tokens=4096,
    temperature=0.3,
    top_p=0.95,
    repeat_penalty=1.0,
)


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def load_backend():
    config = json.loads(BACKEND_CONFIG_PATH.read_text(encoding="utf-8"))
    backend = config["backend"]
    if backend not in ("gguf-cuda", "gguf-cpu"):
        raise RuntimeError(f"지원하지 않는 backend: {backend}")

    # model_path가 절대경로면 그대로 쓰고(이전 버전과의 호환), 상대경로/파일명만 있으면
    # 이 스크립트와 같은 폴더의 models/ 아래에서 찾는다. 그래야 Gen2Train 폴더를 통째로
    # 다른 경로나 다른 PC로 옮겨도 backend_config.json을 다시 만들 필요가 없다.
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = CHAT_BACKEND_DIR / "models" / model_path
    if not model_path.exists():
        raise RuntimeError(f"모델 파일을 찾을 수 없습니다: {model_path} (setup_chat.bat을 먼저 실행하세요)")

    from llama_cpp import Llama

    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=config.get("n_gpu_layers", -1) if backend == "gguf-cuda" else 0,
        n_ctx=config.get("n_ctx", 8192),
        verbose=False,
    )
    return llm, backend


def build_prompt(context: str, question: str) -> str:
    # EXAONE-Deep 규칙: 시스템 프롬프트를 쓰지 않고, 컨텍스트+질문을 전부 user 턴 하나에 통합한다.
    parts = []
    if context.strip():
        parts.append(context.strip())
    parts.append(f"질문: {question.strip()}")
    parts.append("위 현재 설정과 파라미터 설명을 근거로 단계적으로 추론한 뒤, 최종 추천과 이유를 알려줘. 최종 답변은 반드시 한국어로 작성해줘.")
    user_content = "\n\n".join(parts)
    # GGUF에 내장된 EXAONE chat template과 동일한 형식(add_generation_prompt=True일 때)을
    # 그대로 재현한다: 시스템 턴은 비워둔 채, <thought>\n 접두사를 강제로 붙인다.
    return f"[|system|][|endofturn|]\n[|user|]{user_content}\n[|assistant|]<thought>\n"


def handle_request(llm, req: dict) -> None:
    req_id = req.get("id", "")
    try:
        prompt = build_prompt(req.get("context", ""), req.get("question", ""))
        prompt_tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
        eos_id = llm.token_eos()
        max_new_tokens = GENERATION_KWARGS["max_tokens"]

        # 토큰을 하나씩 문자열로 디코딩하면, 한글처럼 여러 바이트로 이루어진 문자가 토큰
        # 경계에서 반쪽으로 잘려 깨진 문자가 나올 수 있다. 그래서 지금까지 생성된 토큰
        # 전체를 매번 다시 통째로 디코딩하고(errors="ignore"로 아직 안 끝난 마지막 문자는
        # 자연스럽게 보류됨), 이전에 내보낸 것과의 차이만 새로 스트리밍한다.
        generated_tokens: list = []
        emitted_text = ""
        for token_id in llm.generate(
            prompt_tokens,
            top_p=GENERATION_KWARGS["top_p"],
            temp=GENERATION_KWARGS["temperature"],
            repeat_penalty=GENERATION_KWARGS["repeat_penalty"],
        ):
            if token_id == eos_id:
                break
            generated_tokens.append(token_id)
            full_text = llm.detokenize(generated_tokens).decode("utf-8", errors="ignore")
            if len(full_text) > len(emitted_text):
                emit({"type": "token", "id": req_id, "text": full_text[len(emitted_text):]})
                emitted_text = full_text
            if len(generated_tokens) >= max_new_tokens:
                break
        emit({"type": "done", "id": req_id})
    except Exception as exc:  # noqa: BLE001 - 요청 하나가 실패해도 서버 프로세스는 계속 살아있어야 한다.
        emit({"type": "error", "id": req_id, "message": str(exc)})


def main() -> None:
    try:
        llm, backend = load_backend()
    except Exception as exc:  # noqa: BLE001
        emit({"type": "fatal_error", "message": f"모델 로딩 실패: {exc}"})
        sys.exit(1)

    emit({"type": "ready", "backend": backend})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        handle_request(llm, req)


if __name__ == "__main__":
    main()
