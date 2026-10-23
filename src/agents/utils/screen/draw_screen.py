from pathlib import Path

from PIL import ImageGrab

def screen_shot():
    save_dir = Path(__file__).parent / "image"
    save_dir.mkdir(parents=True, exist_ok=True)
    screenshot_full = ImageGrab.grab()
    screenshot_full.save(str(save_dir) + "/full_screenshot.png")
if __name__ == '__main__':
    screen_shot()