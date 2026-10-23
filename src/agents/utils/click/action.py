import pyautogui
from .action_model import ClickParams


def click_at(params: ClickParams):
    """
    在指定坐标执行点击操作。
    """
    pyautogui.click(
        x=params.x,
        y=params.y,
        button=params.button,
        clicks=params.clicks,
        interval=params.interval
    )


def image_to_screen_coords(img_x: int, img_y: int, img_width: int, img_height: int) -> tuple[int, int]:
    """
    将图像坐标（物理像素）转换为屏幕坐标（逻辑像素）。

    Args:
        img_x, img_y: 图像中的坐标
        img_width, img_height: 图像的宽高（物理像素）

    Returns:
        (screen_x, screen_y) 屏幕逻辑坐标
    """
    screen_width, screen_height = pyautogui.size()
    scale_x = screen_width / img_width
    scale_y = screen_height / img_height
    return int(img_x * scale_x), int(img_y * scale_y)


def click_at_image_coords(
        img_x: int,
        img_y: int,
        img_width: int,
        img_height: int,
        button: str = 'left',
        clicks: int = 1,
        interval: float = 0.0
) -> None:
    """
    在图像坐标上执行点击（自动转换到屏幕坐标）。

    这是对 click_at 的便捷封装，省去手动转换步骤。
    """
    screen_x, screen_y = image_to_screen_coords(img_x, img_y, img_width, img_height)
    params = ClickParams(x=screen_x, y=screen_y, button=button, clicks=clicks, interval=interval)
    click_at(params)
