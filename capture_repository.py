"""Capture the public, signed-out repository page."""

from playwright.sync_api import sync_playwright

from data import ROOT


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(
            "https://github.com/cleora88/english-darija-lstm-translation",
            wait_until="domcontentloaded",
            timeout=60000,
        )
        page.wait_for_timeout(3000)
        assert "english-darija-lstm-translation" in page.title()
        page.screenshot(path=str(ROOT / "screenshots/06_public_repository.png"))
        print(page.title())
        browser.close()


if __name__ == "__main__":
    main()
