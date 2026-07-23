"""Gen2Train 챗봇 백엔드. venv_chat의 python으로 실행되는 상주 프로세스.

stdin에서 JSON을 한 줄씩 요청으로 받아 stdout으로 토큰을 스트리밍한다 (JSON Lines 프로토콜).
모델은 시작할 때 한 번만 로드하고, 이후 요청은 전부 이 상주 프로세스가 재사용한다.

요청: {"id": "...", "context": "...", "question": "..."}
응답 (2단계로 생성):
  1단계 생각(thought) - 답변용 토큰 예산을 잠식하지 않도록 별도 예산으로 생성:
      {"type": "thought", "id": "...", "text": "..."}   (여러 번)
      {"type": "thought_done", "id": "..."}
  2단계 답변 - 1단계에서 모은 생각 전체를 컨텍스트로 붙여 생성:
      {"type": "token", "id": "...", "text": "..."}     (여러 번)
      {"type": "done", "id": "..."}
시작 시: {"type": "ready", "backend": "..."}
오류:   {"type": "error", "id": "...", "message": "..."}   (요청 단위 오류, 서버는 계속 동작)
치명적 오류(모델 로딩 실패 등): {"type": "fatal_error", "message": "..."}
"""
import json
import os
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
#
# max_thought_tokens/max_answer_tokens는 여기서 고정값으로 두지 않고 main()에서 실제 설정된
# n_ctx를 보고 동적으로 계산해 채운다(_compute_budgets 참고) - 컨텍스트 창 크기에 맞춰 최대한
# 크게 잡되 n_ctx를 넘기지 않도록 하기 위함이다. 둘을 따로 두는 이유: 하나의 예산을 생각과
# 답변이 같이 쓰면, 모델이 생각을 길게 할수록 정작 최종 답변이 예산을 다 써버려 중간에 잘리는
# 문제가 있었다. 생각은 생각대로 별도 예산을 다 쓰게 하고, 답변은 항상 자기 몫의 예산을 새로
# 보장받도록 두 단계로 나눈다 (handle_request 참고).
GENERATION_KWARGS = dict(
    max_thought_tokens=3072,
    max_answer_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    repeat_penalty=1.0,
)


def _compute_budgets(n_ctx: int) -> None:
    """생각/답변 토큰 예산을 n_ctx에 맞춰 동적으로 계산해 GENERATION_KWARGS에 채운다.

    예산이 작았던 건 모델이 그만큼만 생각/답변할 수 있어서가 아니라, 그냥 보수적으로 잡아둔
    상한이었을 뿐이다 - 컨텍스트 창(n_ctx)이 실제로 허용하는 한 최대한 크게 잡는다.
    컨텍스트(현재 설정 요약+질문)용으로 넉넉히 몫을 떼어두고, 남은 예산을 생각:답변 = 7:3으로
    나눈다(생각이 보통 답변보다 훨씬 길다). 2단계(답변)는 원본 프롬프트+생각 전체를 다시
    평가해야 하므로, reserved_for_prompt+thought+answer가 n_ctx를 넘지 않아야 한다.
    """
    reserved_for_prompt = min(4096, n_ctx // 4)
    remaining = max(n_ctx - reserved_for_prompt, 512)
    GENERATION_KWARGS["max_thought_tokens"] = int(remaining * 0.7)
    GENERATION_KWARGS["max_answer_tokens"] = remaining - GENERATION_KWARGS["max_thought_tokens"]


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

    if backend == "gguf-cuda":
        # llama-cpp-python은 빌드 시점의 CUDA 버전에 맞는 런타임 DLL(cudart64_*.dll 등)을
        # 이 프로세스 안에서 찾아야 한다. 이 PC에 다른 CUDA 버전에 의존하는 별도 프로젝트가
        # 있을 수 있으므로, 시스템 전역 CUDA_PATH/PATH는 건드리지 않고 이 프로세스 안에서만
        # os.add_dll_directory로 DLL 검색 경로를 추가한다 - 다른 프로그램에 영향이 없다.
        # 경로/버전은 절대 하드코딩하지 않고 매번 find_cuda_path()로 새로 찾는다(설치된
        # CUDA 버전이 바뀌거나 다른 PC로 옮겨도 그대로 동작하도록).
        from cuda_discovery import find_cuda_path

        cuda_path = find_cuda_path()
        if cuda_path:
            os.environ["CUDA_PATH"] = cuda_path
            cuda_bin = os.path.join(cuda_path, "bin")
            if hasattr(os, "add_dll_directory") and os.path.isdir(cuda_bin):
                os.add_dll_directory(cuda_bin)

    from llama_cpp import Llama, LLAMA_SPLIT_MODE_LAYER, LLAMA_SPLIT_MODE_NONE

    n_ctx = config.get("n_ctx", 8192)
    # n_batch/n_ubatch를 늘리고 flash_attn을 켜면 프롬프트 처리(prefill) 속도가 크게 빨라진다 -
    # 이 GPU(실측: RTX 5070)에서 512/flash_attn 끔 대비 2048/flash_attn 켬이 프롬프트 처리를
    # 약 2.5배(4800 -> 12000 tok/s) 빠르게 만드는 걸 직접 확인했다. 우리 2단계 생성 방식은
    # 답변 단계에서 (원본 프롬프트 + 생각 전체)를 다시 평가해야 하므로 이 속도가 특히 중요하다.
    # VRAM은 더 쓰지만 이 모델(2.4B, Q4_K_M)은 워낙 작아 여유가 크다. gguf-cpu 백엔드는 GPU가
    # 없으므로 이 옵션들의 이득이 없지만 넣어도 해는 없다(전부 llama.cpp가 CPU에서도 지원).
    #
    # split_mode/main_gpu: llama.cpp의 기본값(LLAMA_SPLIT_MODE_LAYER)은 보이는 CUDA 장치가
    # 여러 개면 레이어를 자동으로 전부 나눠 올린다 - 이 모델(2.4B)은 GPU 1개 VRAM에도 충분히
    # 들어가는데, 이 기본값 때문에 멀티 GPU 학습용 PC(GPU 4개 등)에서 지정한 적도 없이 VRAM이
    # 전 GPU에 조금씩 걸쳐 올라가는 문제가 있었다. 학습이 실제로 4개 GPU를 다 쓰는 상황에서는
    # 챗봇이 전 GPU에 조금씩 걸쳐있는 것보다, GPU 0 하나에만 몰아주는 쪽이 나머지 GPU들의
    # 학습용 VRAM 여유를 온전히 지켜준다.
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=config.get("n_gpu_layers", -1) if backend == "gguf-cuda" else 0,
        n_ctx=n_ctx,
        n_batch=2048,
        n_ubatch=2048,
        offload_kqv=True,
        flash_attn=True,
        main_gpu=config.get("main_gpu", 0),
        split_mode=LLAMA_SPLIT_MODE_NONE if backend == "gguf-cuda" else LLAMA_SPLIT_MODE_LAYER,
        verbose=False,
    )
    return llm, backend, n_ctx


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


def _stream_generate(llm, prompt_tokens, eos_id: int, max_new_tokens: int, stop_at: str, on_delta) -> str:
    """토큰을 하나씩 생성하며 새로 늘어난 텍스트만 on_delta로 넘긴다. 최종 생성된 전체
    텍스트를 반환한다.

    한글처럼 여러 바이트로 이루어진 문자는 토큰 경계에서 반쪽으로 잘릴 수 있다. 예전에는
    이걸 피하려고 지금까지 생성된 토큰 전체를 매번 다시 통째로 디코딩했는데, 이러면 토큰이
    늘어날수록 매 스텝 비용이 커져(O(n^2)) 생각 예산을 8600토큰까지 늘린 지금은 체감될
    정도로 느려진다. 대신 새 토큰 하나만 바이트로 디코드해 작은 버퓌에 이어붙이고, UTF-8로
    끝까지 디코드 가능한 부분만 흘려보내고 아직 안 끝난 마지막 문자의 바이트만 다음 토큰과
    합치도록 버퍼에 남겨둔다(O(n)) - llama.cpp 자체 스트리밍 예제와 같은 방식이다. 실측으로
    같은 토큰 시퀀스에서 기존 방식과 결과가 동일함을 확인했다.
    """
    token_count = 0
    raw_buffer = b""
    emitted_text = ""
    for token_id in llm.generate(
        prompt_tokens,
        top_p=GENERATION_KWARGS["top_p"],
        temp=GENERATION_KWARGS["temperature"],
        repeat_penalty=GENERATION_KWARGS["repeat_penalty"],
    ):
        if token_id == eos_id:
            break
        token_count += 1
        raw_buffer += llm.detokenize([token_id])
        try:
            delta = raw_buffer.decode("utf-8")
            raw_buffer = b""
        except UnicodeDecodeError as exc:
            delta = raw_buffer[: exc.start].decode("utf-8")
            raw_buffer = raw_buffer[exc.start :]
        if delta:
            on_delta(delta)
            emitted_text += delta
        if stop_at and stop_at in emitted_text:
            break
        if token_count >= max_new_tokens:
            break
    if raw_buffer:
        # 아직 안 끝난 멀티바이트 문자가 남아 있으면(예: 예산/중단 조건으로 문장 중간에 끊긴
        # 경우) errors="ignore"로 버려 깨진 문자가 화면에 남지 않게 한다.
        tail = raw_buffer.decode("utf-8", errors="ignore")
        if tail:
            on_delta(tail)
            emitted_text += tail
    return emitted_text


def handle_request(llm, req: dict) -> None:
    req_id = req.get("id", "")
    try:
        prompt = build_prompt(req.get("context", ""), req.get("question", ""))
        eos_id = llm.token_eos()

        # 1단계: 생각(thought)을 별도 예산(max_thought_tokens)으로 생성해 스트리밍한다.
        # </thought>가 나오면 바로 멈추고, 예산을 다 써도 </thought>가 안 나오면 거기서
        # 강제로 끊는다 - 어느 쪽이든 답변용 예산에는 손대지 않는다.
        prompt_tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
        thought_text = _stream_generate(
            llm, prompt_tokens, eos_id,
            max_new_tokens=GENERATION_KWARGS["max_thought_tokens"],
            stop_at="</thought>",
            on_delta=lambda delta: emit({"type": "thought", "id": req_id, "text": delta}),
        )
        emit({"type": "thought_done", "id": req_id})

        # 2단계: 1단계에서 모은 생각 전체를 "생각 풀"로 그대로 프롬프트에 붙이고, </thought>를
        # 강제로 닫아 모델에게 이제 최종 답변을 쓸 차례임을 알린다. 답변은 항상 자기 몫의
        # 새 예산(max_answer_tokens)을 받으므로, 생각이 아무리 길어도 답변이 중간에 잘리지
        # 않는다. llama.cpp가 이전 단계와 겹치는 프롬프트 접두사(원래 prompt + 생각)를
        # 내부적으로 재사용하므로 다시 처음부터 평가하는 비용이 크지 않다.
        thought_clean = thought_text.split("</thought>")[0].strip()
        answer_prompt = prompt + thought_clean + "\n</thought>\n\n"
        answer_prompt_tokens = llm.tokenize(answer_prompt.encode("utf-8"), add_bos=True)
        _stream_generate(
            llm, answer_prompt_tokens, eos_id,
            max_new_tokens=GENERATION_KWARGS["max_answer_tokens"],
            stop_at="",
            on_delta=lambda delta: emit({"type": "token", "id": req_id, "text": delta}),
        )
        emit({"type": "done", "id": req_id})
    except Exception as exc:  # noqa: BLE001 - 요청 하나가 실패해도 서버 프로세스는 계속 살아있어야 한다.
        emit({"type": "error", "id": req_id, "message": str(exc)})


def main() -> None:
    try:
        llm, backend, n_ctx = load_backend()
    except Exception as exc:  # noqa: BLE001
        emit({"type": "fatal_error", "message": f"모델 로딩 실패: {exc}"})
        sys.exit(1)

    _compute_budgets(n_ctx)
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
