from playwright.sync_api import sync_playwright

total = 0

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for seed in range(60, 70):
        page.goto(f"https://sanand0.github.io/tdsdata/js_table/?seed={seed}")
        page.wait_for_load_state("networkidle")

        cells = page.locator("table td").all_inner_texts()

        for c in cells:
            try:
                total += int(c)
            except ValueError:
                try:
                    total += float(c)
                except ValueError:
                    pass

    browser.close()

print(int(total))