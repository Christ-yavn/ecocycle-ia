from playwright.sync_api import sync_playwright
import time

def test_bing():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://www.bing.com/images/search?q=plastique&form=HDRSC3")
        time.sleep(3)
        images = page.locator("img").element_handles()
        print(f"Total img tags on Bing: {len(images)}")
        
        count_mimg = 0
        for img in images:
            cls = img.get_attribute("class")
            if cls and "mimg" in cls:
                count_mimg += 1
                
        print(f"Total img.mimg tags: {count_mimg}")
        browser.close()

if __name__ == "__main__":
    test_bing()
