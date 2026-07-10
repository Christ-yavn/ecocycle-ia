from playwright.sync_api import sync_playwright

def test():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://duckduckgo.com/?q=plastique&t=h_&iar=images&iax=images&ia=images")
        page.wait_for_timeout(3000)
        
        images1 = page.locator("img").element_handles()
        print(f"Total img tags: {len(images1)}")
        
        images2 = page.locator("img.tile--img__img").element_handles()
        print(f"img.tile--img__img tags: {len(images2)}")
        
        browser.close()

if __name__ == "__main__":
    test()
