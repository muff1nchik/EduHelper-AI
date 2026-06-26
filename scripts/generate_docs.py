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

    print(f"HTML-документация создана: {output_dir}")


if __name__ == "__main__":
    main()
