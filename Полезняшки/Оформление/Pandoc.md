**Pandoc** — это мощный бесплатный **конвертер файлов**, который позволяет переводить документы из одного формата разметки в другой. Его часто называют «швейцарским армейским ножом» для работы с текстом.

## Главное о Pandoc:

- **Что он делает:** Он берет файл в одном формате (например, **Markdown**, **DOCX**, **HTML**) и переписывает его структуру в другой формат (например, в **PDF**, **EPUB**, **LaTeX** или обратно в DOCX).

- **Как он работает:** Программа не имеет графического интерфейса и запускается через **командную строку** (терминал).

- **Поддержка форматов:** Pandoc понимает десятки форматов данных — от простых текстовых файлов до сложных научных документов с формулами и цитатами.

- **Для кого создан:** Чаще всего им пользуются программисты, писатели, ученые и технические писатели, которые хотят писать текст в простом формате (Markdown), но сдавать работу в строго требуемом виде (DOCX или PDF).

# Преобразование markdown в pdf с помощью python

## Шаг 1. Подготовка окружения

Вам понадобятся три библиотеки. Установите их в терминале:
```bash
# Установка Python-библиотек для сборки и PDF
pip install markdown-it-py md-it-mermaid playwright

# Установка браузера для генерации PDF
playwright install chromium

# Установка CLI для PlantUML (для рендеринга схем PlantUML)
npm install -g node-plantuml
```
*(Примечание: Для работы `node-plantuml` в системе также должна быть установлена Java, так как сам PlantUML работает на ней. Если Java нет, скрипт ниже можно переписать на онлайн-рендер).*

## Шаг 2. Скрипт для сборки (`compile_vault.py`)

Создайте этот скрипт в папке с вашими файлами Markdown. Он соберет файлы, подключит библиотеки для отображения **MathJax (LaTeX)**, выполнит рендеринг **Mermaid / PlantUML** и сохранит идеальный PDF.
```python
import os
import re
import subprocess
from markdown_it import MarkdownIt
from md_it_mermaid import mermaid_plugin
from playwright.sync_api import sync_playwright

# 1. Настройки
OUTPUT_PDF = "obsidian_book.pdf"
TEMP_HTML = "temp_composite.html"

# HTML Шаблон с поддержкой стилей для оглавления, MathJax и Mermaid
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Экспорт Obsidian</title>
    <!-- Подключаем MathJax для LaTeX -->
    <script src="https://polyfill.io"></script>
    <script id="MathJax-script" async src="https://jsdelivr.net"></script>
    
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
            line-height: 1.6; 
            padding: 40px; 
            color: #1e1e1e; 
        }}
        pre {{ background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        code {{ font-family: monospace; background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
        img {{ max-width: 100%; height: auto; }}
        .page-break {{ page-break-before: always; }}
        
        /* Стили для Оглавления */
        .toc-title {{ font-size: 28px; font-weight: bold; margin-bottom: 20px; }}
        .toc-list {{ list-style: none; padding: 0; }}
        .toc-item {{ 
            display: flex; 
            justify-content: space-between; 
            align-items: flex-end; 
            margin-bottom: 8px; 
        }}
        .toc-item a {{ text-decoration: none; color: #0066cc; font-weight: 500; }}
        .toc-filler {{ 
            flex-grow: 1; 
            border-bottom: 1px dotted #aaa; 
            margin: 0 10px; 
            position: relative;
            top: -4px;
        }}
    </style>
</head>
<body>
    <!-- Блок автоматического оглавления -->
    <div class="toc-container">
        <div class="toc-title">Оглавление</div>
        <ul class="toc-list">
            {toc_content}
        </ul>
    </div>
    
    <div class="page-break"></div>

    <!-- Основной контент -->
    {content}
</body>
</html>
"""

def extract_first_h1(content, file_name):
    """Находит первый заголовок # в тексте. Если нет, берет имя файла."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return os.path.splitext(file_name)[0]

def render_plantuml(text):
    """Преобразует блоки кода plantuml в картинки"""
    def replace_puml(match):
        puml_code = match.group(1).strip()
        img_name = f"puml_{abs(hash(puml_code))}.png"
        try:
            process = subprocess.Popen(['puml', 'generate', '-o', img_name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            process.communicate(input=puml_code)
            return f"![PlantUML]({img_name})"
        except FileNotFoundError:
            # Если node-plantuml или Java не установлены, выводим заглушку, чтобы скрипт не падал
            print(f"Ошибка: Команда 'puml' не найдена. Убедитесь, что node-plantuml и Java установлены.")
            return f"\n*⚠️ Ошибка рендеринга PlantUML (Требуется Java)*\n"
    
    return re.sub(r"```plantuml\s*([\s\S]*?)\s*```", replace_puml, text)

def main():
    # Находим все .md файлы в текущей папке и сортируем их
    md_files = sorted([f for f in os.listdir('.') if f.endswith('.md')])
    
    if not md_files:
        print("В текущей директории не найдено файлов .md!")
        return

    print(f"Найдено файлов для сборки: {len(md_files)}")

    combined_md = ""
    toc_items = []

    for index, file_path in enumerate(md_files):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Получаем заголовок для оглавления
            title = extract_first_h1(content, file_path)
            anchor_id = f"chapter_{index}"
            
            # Добавляем пункт в список оглавления
            # Примечание: номера страниц в PDF расставит сам Chromium, здесь мы делаем ссылки-якоря
            toc_items.append(f'<li class="toc-item"><a href="#{anchor_id}">{title}</a><div class="toc-filler"></div></li>')
            
            # Вшиваем якорь перед заголовком в текст для кликабельности из оглавления
            content = f'<div id="{anchor_id}"></div>\n\n' + content
            
            # Обрабатываем PlantUML схемы
            content = render_plantuml(content)
            
            # Склеиваем, добавляя разрыв страницы после каждого файла
            combined_md += content + "\n\n<div class='page-break'></div>\n\n"

    # 2. Рендерим Markdown + Mermaid в HTML
    md = MarkdownIt().use(mermaid_plugin)
    raw_html = md.render(combined_md)
    
    # Собираем финальный шаблон
    toc_html = "\n".join(toc_items)
    final_html = HTML_TEMPLATE.format(toc_content=toc_html, content=raw_html)

    with open(TEMP_HTML, "w", encoding="utf-8") as f:
        f.write(final_html)

    print("Интерактивный HTML создан. Запуск Playwright для компиляции в PDF...")

    # 3. Печать в PDF через Chromium
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{os.path.abspath(TEMP_HTML)}")
        
        # Ожидаем отрисовки тяжелой графики (Mermaid скрипты и формулы MathJax)
        page.wait_for_timeout(4000) 
        
        page.pdf(
            path=OUTPUT_PDF, 
            format="A4",
            margin={"top": "20mm", "bottom": "20mm", "left": "20mm", "right": "20mm"},
            print_background=True
        )
        browser.close()

    # Чистим временный HTML
    if os.path.exists(TEMP_HTML):
        os.remove(TEMP_HTML)
        
    print(f"\nГотово! Интерактивный PDF успешно сохранен: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
```
### Как это работает:

1. Вы перечисляете файлы в массиве `INPUT_FILES`. Скрипт склеивает их, добавляя между ними разрыв страницы (`page-break`).
2. Блоки ` ```plantuml ` на лету перехватываются и компилируются утилитой `puml` в локальные PNG-картинки, а ссылки на них вставляются в текст.
3. Блоки ` ```mermaid ` парсятся плагином `md-it-mermaid` в разметку, понятную для Chromium.
4. Chromium открывает получившийся файл, загружает MathJax из сети, за 3 секунды превращает весь LaTeX код в красивые математические формулы, отрисовывает схемы и «печатает» идеальный PDF.