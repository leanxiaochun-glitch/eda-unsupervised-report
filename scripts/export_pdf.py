#!/usr/bin/env python3
"""Render a Markdown report with local images to A4 PDF using headless Chrome."""
import argparse, os, platform, shutil, subprocess
from pathlib import Path
import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "PingFang SC", "Songti SC", "Microsoft YaHei", sans-serif;
       font-size: 10.5pt; line-height: 1.75; color: #1a1a1a; background: white; }
h1 { font-size: 19pt; text-align: center; color: #1F3864; border-bottom: 2px solid #1F3864; padding-bottom: 10pt; }
h2 { font-size: 14pt; color: #1F3864; background: #F2F5FA; border-left: 4px solid #1F3864;
     padding: 5pt 9pt; margin-top: 20pt; break-after: avoid; }
h3 { font-size: 11.8pt; color: #2E4A7D; break-after: avoid; }
p { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 9.2pt; break-inside: avoid; }
th { background: #E8EDF6; color: #1F3864; border: 1px solid #B8C4DA; padding: 5pt 7pt; text-align: left; }
td { border: 1px solid #C9D2E3; padding: 4.5pt 7pt; }
tr:nth-child(even) td { background: #FAFBFD; }
blockquote { background: #F7F9FC; border-left: 3px solid #7C9BC9; padding: 8pt 12pt; break-inside: avoid; }
code { font-family: Menlo, monospace; background: #F0F2F6; font-size: 9pt; }
pre { background: #F5F6F8; border: 1px solid #D9DEE8; padding: 8pt; font-size: 8.6pt; break-inside: avoid; }
pre code { background: none; }
img { display: block; max-width: 92%; margin: 12pt auto; break-inside: avoid; }
strong { color: #14315F; }
hr { border: none; border-top: 1px solid #D5DBE6; }
"""


def find_chrome(explicit=None):
    if explicit:
        return explicit
    for name in ["google-chrome", "chromium", "chromium-browser", "chrome"]:
        path = shutil.which(name)
        if path:
            return path
    if platform.system() == "Darwin":
        for path in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                     "/Applications/Chromium.app/Contents/MacOS/Chromium",
                     "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]:
            if os.path.exists(path):
                return path
    raise FileNotFoundError("Chrome/Chromium/Edge not found; pass --chrome /path/to/browser.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--chrome", default=None)
    args = p.parse_args()
    src, dst = Path(args.input).expanduser().resolve(), Path(args.output).expanduser().resolve()
    body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
    html = f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>{src.stem}</title><style>{CSS}</style></head><body>{body}</body></html>'
    html_path = dst.parent / f".{src.stem}_print.html"
    dst.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    # Keep HTML beside Markdown so relative image paths resolve; remove after printing.
    local_html = src.parent / f".{src.stem}_print.html"
    if local_html != html_path:
        local_html.write_text(html, encoding="utf-8")
    try:
        cmd = [find_chrome(args.chrome), "--headless=new", "--disable-gpu", "--no-print-header-footer",
               f"--print-to-pdf={dst}", "--virtual-time-budget=8000", local_html.as_uri()]
        if platform.system() != "Windows":
            cmd.insert(2, "--no-sandbox")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0 or not dst.exists() or dst.stat().st_size < 1000:
            raise RuntimeError(result.stderr[-1000:] or "PDF generation failed")
    finally:
        if local_html.exists():
            local_html.unlink()
        if html_path.exists() and html_path != local_html:
            html_path.unlink()
    print(f"PDF written: {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
