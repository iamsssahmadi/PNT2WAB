"""Headless-browser smoke test of the web tool (docs/).

Serves docs/ locally, exercises the single-estimate, batch and validation tabs at desktop
and phone widths, checks for console errors, and saves screenshots.

usage: python tests/browser_check.py [screenshot_dir]
"""
import functools
import os
import http.server
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parents[1] / "docs"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())


def serve():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    httpd = serve()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    errors = []
    with sync_playwright() as p:
        exe = os.environ.get("CHROMIUM_PATH")  # optional: use an already-installed Chromium
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        for name, vp in (("desktop", dict(width=1200, height=900)), ("phone", dict(width=390, height=844))):
            page = browser.new_page(viewport=vp)
            page.on("console", lambda m: errors.append(f"{name} console {m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"{name} page error: {e}"))
            page.goto(url)
            page.wait_for_selector("#subtests tbody tr")
            assert page.locator("#subtests tbody tr").count() == 14, "expected 10 subtests + 4 group rows"
            aq = page.locator("#hero .v").inner_text()
            page.fill("#correct", "20")
            page.wait_for_timeout(100)
            aq2 = page.locator("#hero .v").inner_text()
            assert aq != aq2, "estimate did not update"
            assert page.locator("#severity-warn").is_visible(), "severity warning should show at 20/175"
            page.screenshot(path=str(OUT / f"{name}_single.png"), full_page=True)
            page.fill("#correct", "300")
            page.wait_for_timeout(100)
            assert page.locator("#single-error").is_visible(), "invalid input not flagged"
            page.fill("#correct", "120")
            hscroll = page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
            assert not hscroll, f"{name}: horizontal page scroll"
            page.click("#t-batch")
            csv = OUT / "batch.csv"
            csv.write_text("id,pnt_correct,pnt_total\nA,120,175\nB,0,175\nC,175,175\nD,abc,175\nE,80,174\n")
            page.set_input_files("#csv", str(csv))
            page.wait_for_selector("#batch-out:not([hidden])")
            summ = page.locator("#batch-summary").inner_text()
            assert summ.startswith("4 rows estimated, 1 skipped"), summ
            page.screenshot(path=str(OUT / f"{name}_batch.png"), full_page=True)
            page.click("#t-about")
            assert page.locator("#val-table tbody tr").count() == 14
            assert page.locator("#band-table tbody tr").count() == 14
            page.screenshot(path=str(OUT / f"{name}_about.png"), full_page=True)
            print(f"{name}: OK (AQ at 120/175 = {aq}, at 20/175 = {aq2}; batch: {summ})")
        browser.close()
    httpd.shutdown()
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("no console errors; screenshots in", OUT)


if __name__ == "__main__":
    main()
