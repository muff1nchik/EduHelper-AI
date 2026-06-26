"""Создаёт HTML-документацию проекта через pydoc."""

import pydoc
import sys
from pathlib import Path


def main() -> None:
    """Генерирует HTML-файлы для пакета app."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "docs" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(project_root))
    modules = ["app"] + sorted(
        f"app.{path.stem}"
        for path in (project_root / "app").glob("*.py")
        if path.name != "__init__.py"
    )

    for module_name in modules:
        html = pydoc.HTMLDoc().docmodule(__import__(module_name, fromlist=["*"]))
        file_name = f"{module_name.replace('.', '_')}.html"
        (output_dir / file_name).write_text(html, encoding="utf-8")

    for html_path in output_dir.glob("*.html"):
        _add_utf8_charset(html_path)

    print(f"HTML-документация создана: {output_dir}")


def _add_utf8_charset(path: Path) -> None:
    """Добавляет UTF-8 meta в HTML, если его ещё нет."""
    text = path.read_text(encoding="utf-8")
    if '<meta charset="utf-8">' in text.lower():
        return
    if "<head>" in text:
        text = text.replace("<head>", '<head>\n<meta charset="utf-8">', 1)
    else:
        text = f'<meta charset="utf-8">\n{text}'
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
