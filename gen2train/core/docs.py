"""arg_introspect + help_texts를 엮어 백과사전 스타일의 HTML 도움말 문서를 만든다.

ui/help_panel.py의 QTextBrowser가 그대로 렌더링한다. 각 파라미터 항목마다
<a name="{dest}">앵커를 심어두어, 필드에 포커스가 갈 때 해당 위치로 자동 스크롤할 수 있게 한다.
"""
import html as html_lib

from . import arg_introspect
from .help_texts import GROUP_DESCRIPTIONS, get_help

INTRO_HTML = """
<h1>Gen2Train 설정 백과사전</h1>
<p>설정 항목을 클릭하거나 Tab으로 이동하면 이 패널이 해당 설명으로 자동 스크롤됩니다.
직접 찾아보려면 아래 목차나 위의 검색창을 사용하세요.</p>
"""


def _esc(text: str) -> str:
    return html_lib.escape(text).replace("\n", "<br/>")


def _group_anchor(group: str) -> str:
    return "group__" + html_lib.escape(group).replace(" ", "_")


def build_html(model_type: str = "sd") -> str:
    """model_type: 'sd' 또는 'sdxl'. 해당 학습 스크립트가 지원하는 전체 파라미터 문서를 만든다."""
    specs = arg_introspect.get_arg_specs(model_type)
    by_group: dict[str, list] = {}
    for spec in specs:
        by_group.setdefault(spec.group, []).append(spec)

    toc_items = []
    body_sections = []
    for group in arg_introspect.group_order():
        group_specs = by_group.get(group)
        if not group_specs:
            continue
        anchor = _group_anchor(group)
        toc_items.append(f'<li><a href="#{anchor}">{_esc(group)} ({len(group_specs)})</a></li>')

        rows = []
        for spec in group_specs:
            desc = get_help(spec.dest, spec.help or "(설명 없음)")
            rows.append(
                f'<a name="{html_lib.escape(spec.dest)}"></a>'
                f'<h3>{_esc(spec.flag)}</h3>'
                f'<p>{_esc(desc)}</p>'
            )

        body_sections.append(
            f'<a name="{anchor}"></a>'
            f'<h2>{_esc(group)}</h2>'
            f'<p style="color:#666;">{_esc(GROUP_DESCRIPTIONS.get(group, ""))}</p>'
            + "".join(rows)
        )

    toc_html = "<ul>" + "".join(toc_items) + "</ul>"
    return (
        '<html><body style="font-family:\'Segoe UI\', sans-serif; font-size:13px;">'
        + INTRO_HTML
        + "<h2>목차</h2>"
        + toc_html
        + "<hr/>"
        + "".join(body_sections)
        + "</body></html>"
    )
