import math
from io import BytesIO
from pathlib import Path
from typing import Union, Tuple

from PIL import Image, ImageDraw, ImageFont


def draw_xyz_coordinate_system(
    draw: ImageDraw,
    center_x: int,
    center_y: int,
    radius: int,
    line_color="rgba(0, 0, 0, 100)",
    line_width=1,
):
    """
    Draw a three-dimensional XYZ coordinate system that represents
    the orientation of the entity (right-hand spiral rule).
    Args:
        draw:
        center_x:
        center_y:
        radius:
        line_color:
        line_width:

    Returns:

    """
    # 定义三个轴的颜色
    x_color = "rgba(255, 0, 0, 180)"  # 红色 - X轴
    y_color = "rgba(0, 180, 0, 180)"  # 绿色 - Y轴
    z_color = "rgba(0, 0, 255, 180)"  # 蓝色 - Z轴

    # 为增强清晰度，绘制轴时半径稍微缩小一点，留出标签空间
    axis_radius = int(radius * 0.9)

    # 绘制X轴 - 水平向右
    x_end_x = center_x + axis_radius
    x_end_y = center_y
    draw.line([(center_x, center_y), (x_end_x, x_end_y)], fill=x_color, width=line_width + 1)

    # 绘制X轴箭头
    arrow_size = int(radius * 0.05)
    draw.polygon(
        [
            (x_end_x - arrow_size, x_end_y - arrow_size),
            (x_end_x, x_end_y),
            (x_end_x - arrow_size, x_end_y + arrow_size),
        ],
        fill=x_color,
    )

    # 绘制Y轴 - 向前（从屏幕出来方向，在2D中表示为向下倾斜）
    angle_y = math.radians(135)  # 135度指向右下方，符合右手螺旋定则
    y_end_x = center_x + int(axis_radius * math.cos(angle_y))
    y_end_y = center_y + int(axis_radius * math.sin(angle_y))
    draw.line([(center_x, center_y), (y_end_x, y_end_y)], fill=y_color, width=line_width + 1)

    # 绘制Y轴箭头
    arrow_angle = angle_y + math.pi / 2  # 垂直于Y轴的角度
    arrow_x1 = y_end_x + int(arrow_size * math.cos(arrow_angle))
    arrow_y1 = y_end_y + int(arrow_size * math.sin(arrow_angle))
    arrow_x2 = y_end_x + int(arrow_size * math.cos(arrow_angle + math.pi))
    arrow_y2 = y_end_y + int(arrow_size * math.sin(arrow_angle + math.pi))
    draw.polygon([(arrow_x1, arrow_y1), (y_end_x, y_end_y), (arrow_x2, arrow_y2)], fill=y_color)

    # 绘制Z轴 - 垂直向上
    z_end_x = center_x
    z_end_y = center_y - axis_radius
    draw.line([(center_x, center_y), (z_end_x, z_end_y)], fill=z_color, width=line_width + 1)

    # 绘制Z轴箭头
    draw.polygon(
        [
            (z_end_x - arrow_size, z_end_y + arrow_size),
            (z_end_x, z_end_y),
            (z_end_x + arrow_size, z_end_y + arrow_size),
        ],
        fill=z_color,
    )

    # 添加轴标签
    try:
        font = ImageFont.truetype("arial.ttf", int(radius * 0.14))
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", int(radius * 0.14))
        except IOError:
            font = ImageFont.load_default()

    label_offset = int(radius * 0.12)

    # X轴标签位置
    x_label_x = x_end_x + label_offset
    x_label_y = x_end_y

    # Y轴标签位置
    y_label_x = y_end_x + label_offset
    y_label_y = y_end_y

    # Z轴标签位置
    z_label_x = z_end_x
    z_label_y = z_end_y - label_offset

    # 绘制轴标签
    try:
        # 使用anchor属性进行对齐
        draw.text((x_label_x, x_label_y), "X", fill=x_color, font=font, anchor="mm")
        draw.text((y_label_x, y_label_y), "Y", fill=y_color, font=font, anchor="mm")
        draw.text((z_label_x, z_label_y), "Z", fill=z_color, font=font, anchor="mm")
    except AttributeError:
        # 对于不支持anchor的老版本Pillow
        bbox = (
            draw.textbbox((0, 0), "X", font=font)
            if hasattr(draw, 'textbbox')
            else draw.textsize("X", font=font)
        )
        text_width = bbox[2] - bbox[0] if hasattr(draw, 'textbbox') else bbox[0]
        text_height = bbox[3] - bbox[1] if hasattr(draw, 'textbbox') else bbox[1]

        draw.text(
            (x_label_x - text_width // 2, x_label_y - text_height // 2),
            "X",
            fill=x_color,
            font=font,
        )
        draw.text(
            (y_label_x - text_width // 2, y_label_y - text_height // 2),
            "Y",
            fill=y_color,
            font=font,
        )
        draw.text(
            (z_label_x - text_width // 2, z_label_y - text_height // 2),
            "Z",
            fill=z_color,
            font=font,
        )

    # 绘制中心点
    dot_radius = max(3, int(radius * 0.03))
    draw.ellipse(
        (
            center_x - dot_radius,
            center_y - dot_radius,
            center_x + dot_radius,
            center_y + dot_radius,
        ),
        fill="rgba(0, 0, 0, 200)",
    )

    # 绘制辅助网格/圆环，增强3D效果
    grid_color = "rgba(200, 200, 200, 100)"  # 淡灰色，半透明

    # XY平面上的网格圆
    for r in [radius * 0.33, radius * 0.66]:
        r = int(r)
        draw.ellipse(
            (center_x - r, center_y - r, center_x + r, center_y + r), outline=grid_color, width=1
        )


def create_comparison_image(
    image_source: Union[str, bytes, Path], reference_width: int = None
) -> Tuple[Image.Image, Image.Image]:
    """
    Create a display image with coordinate system from a wide-format input image.

    The input image should contain one or two lines of images:
    - First row: N consecutive 200 pixel wide subgraphs.
    - Line 2: A reference subgraph (assumed to be at the beginning).

    Args:
        image_source: The path to the image file or the image's bytes object.
        reference_width: The width of the reference diagram. If not specified, use the same width as the subgraph (200 pixels).

    Returns:
        Tuple containing two PIL Image objects: (array_image, reference_image)
        - array_image: Layout image containing Array subgraphs
        - reference_image: Reference image or space-holding blank image

    Raises:
        FileNotFoundError: If the image path is invalid.
        ValueError: If the image format is invalid or the size is incompatible.
        Exception: Other image processing errors.
    """
    try:
        if isinstance(image_source, (str, Path)):
            img = Image.open(image_source)
        elif isinstance(image_source, bytes):
            img = Image.open(BytesIO(image_source))
        else:
            raise TypeError("image_source 必须是路径 (str/Path) 或 bytes")

        img = img.convert("RGBA")  # 确保使用支持透明度的 RGBA 模式

    except FileNotFoundError:
        print(f"错误：找不到文件 {image_source}")
        raise
    except Exception as e:
        print(f"打开或处理图像时出错: {e}")
        raise

    original_width, original_height = img.size
    sub_image_width = 200
    sub_image_height = original_height // 2  # 仍然按两行计算，但只使用第一行

    # 如果未指定参考图宽度，使用与子图相同的宽度
    if reference_width is None:
        reference_width = sub_image_width

    if original_height % 2 != 0:
        print("警告：图像高度不是偶数，裁切可能不精确。")

    # 根据用户描述，第一行包含 N 个 200px 宽的子图
    # 我们假设 `original_width` 至少是 `sub_image_width` 的倍数
    num_sub_images = original_width // sub_image_width
    if num_sub_images == 0:
        raise ValueError("计算出的子图数量为零，请检查图像宽度和子图宽度。")
    if original_width % sub_image_width != 0:
        print(
            f"警告：图像总宽度 {original_width} 不能被子图宽度 {sub_image_width} 整除。将处理 {num_sub_images} 个完整子图。"
        )

    # --- 裁切子图 ---
    array_images = []
    for i in range(num_sub_images):
        left = i * sub_image_width
        top = 0
        right = left + sub_image_width
        bottom = sub_image_height
        box = (left, top, right, bottom)
        try:
            sub_img = img.crop(box)
            array_images.append(sub_img)
        except Exception as e:
            print(f"裁切顶部子图 {i} 时出错: {e}")
            raise ValueError(f"无法裁切顶部子图 {i}")

    # 裁切参考图像 (假定它是第二行的前 reference_width 像素)
    if original_height > sub_image_height:
        ref_box = (0, sub_image_height, min(reference_width, original_width), original_height)
        try:
            reference_image = img.crop(ref_box)
        except Exception as e:
            print(f"裁切参考图像时出错: {e}")
            # 出错时创建一个空白的参考图像
            reference_image = Image.new(
                'RGBA', (reference_width, sub_image_height), (255, 255, 255, 0)
            )
    else:
        # 如果图像高度不够，创建一个空白的参考图像
        reference_image = Image.new('RGBA', (reference_width, sub_image_height), (255, 255, 255, 0))

    # --- 准备新画布 ---
    spacing = 15
    border_width = 1
    label_width = 50
    index_height = 40
    font_size_labels = 20
    font_size_indices = 15
    coord_line_color = "rgba(0, 0, 0, 100)"  # 改为半透明黑色
    coord_line_width = 1  # 改为更细的线

    # 计算总宽度和高度（现在只有一行）
    total_width = label_width + spacing + num_sub_images * (sub_image_width + spacing)
    total_height = spacing + sub_image_height + spacing + index_height  # 只有一行Array + 索引高度

    # 创建白色背景 (RGBA)
    new_img = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(new_img)

    # --- 加载字体 ---
    try:
        # 尝试使用常见字体，或者提供 .ttf 文件路径
        font_labels = ImageFont.truetype("arial.ttf", font_size_labels)
        font_indices = ImageFont.truetype("arial.ttf", font_size_indices)
    except IOError:
        print("警告：找不到 Arial 字体，将使用默认字体。标签可能看起来不同。")
        try:
            # 尝试其他常见字体 (例如 DejaVu Sans)
            font_labels = ImageFont.truetype("DejaVuSans.ttf", font_size_labels)
            font_indices = ImageFont.truetype("DejaVuSans.ttf", font_size_indices)
        except IOError:
            font_labels = ImageFont.load_default()
            font_indices = ImageFont.load_default()

    # --- 绘制布局 ---
    for i in range(num_sub_images):
        # 计算当前列图像的左上角坐标
        img_x = label_width + spacing + i * (sub_image_width + spacing)
        array_y = spacing

        # 粘贴 Array 图像
        new_img.paste(array_images[i], (img_x, array_y))
        # 绘制 Array 图像边框
        draw.rectangle(
            (
                img_x - border_width,
                array_y - border_width,
                img_x + sub_image_width,
                array_y + sub_image_height,
            ),
            outline="black",
            width=border_width,
        )

        # 在 Array 图像上绘制坐标系
        center_x = img_x + sub_image_width // 2
        center_y = array_y + sub_image_height // 2
        # 让半径更大，接近填满子图，并考虑线宽
        radius = int(
            min(sub_image_width, sub_image_height) / 2.0 - coord_line_width - 20
        )  # 额外减小半径，为轴标签留出空间
        draw_xyz_coordinate_system(
            draw, center_x, center_y, radius, coord_line_color, coord_line_width
        )

        # 绘制索引标签（直接放在Array图像下方）
        index_text = str(i)
        index_x = img_x + sub_image_width // 2
        index_y = array_y + sub_image_height + spacing // 2  # 放置在Array图下方

        # 使用 text 方法的 anchor 参数进行居中对齐 (Pillow >= 8.0.0)
        try:
            # anchor="mt" 表示中上对齐
            draw.text((index_x, index_y), index_text, fill="black", font=font_indices, anchor="mt")
        except AttributeError:
            # Pillow 旧版本的回退方案
            bbox = (
                draw.textbbox((0, 0), index_text, font=font_indices)
                if hasattr(draw, 'textbbox')
                else draw.textsize(index_text, font=font_indices)
            )
            text_width = bbox[2] - bbox[0] if hasattr(draw, 'textbbox') else bbox[0]
            draw.text(
                (index_x - text_width // 2, index_y), index_text, fill="black", font=font_indices
            )

    # --- 绘制行标签 ---
    label_array_x = spacing
    label_array_y = spacing + sub_image_height // 2

    # 使用 anchor="mm" 进行中心对齐
    try:
        draw.text(
            (label_array_x + label_width // 2 - 5, label_array_y),
            "Array",
            fill="black",
            font=font_labels,
            anchor="mm",
        )
    except AttributeError:
        # Pillow 旧版本的回退方案
        bbox_a = (
            draw.textbbox((0, 0), "Array", font=font_labels)
            if hasattr(draw, 'textbbox')
            else draw.textsize("Array", font=font_labels)
        )
        text_width_a = bbox_a[2] - bbox_a[0] if hasattr(draw, 'textbbox') else bbox_a[0]
        text_height_a = bbox_a[3] - bbox_a[1] if hasattr(draw, 'textbbox') else bbox_a[1]
        draw.text(
            (label_array_x + (label_width - text_width_a) // 2, label_array_y - text_height_a // 2),
            "Array",
            fill="black",
            font=font_labels,
        )

    # --- 为参考图像单独创建一个带有坐标系的图像 ---
    ref_img_with_axis = Image.new('RGBA', (reference_width, sub_image_height), (255, 255, 255, 255))
    ref_draw = ImageDraw.Draw(ref_img_with_axis)

    # 粘贴参考图像
    ref_img_with_axis.paste(reference_image, (0, 0))

    # 在参考图像上绘制坐标系
    ref_center_x = reference_width // 2
    ref_center_y = sub_image_height // 2
    ref_radius = int(min(reference_width, sub_image_height) / 2.0 - coord_line_width - 20)
    draw_xyz_coordinate_system(
        ref_draw, ref_center_x, ref_center_y, ref_radius, coord_line_color, coord_line_width
    )

    # 绘制边框
    ref_draw.rectangle(
        (0, 0, reference_width - 1, sub_image_height - 1), outline="black", width=border_width
    )

    # 把处理过的参考图像赋值回去
    reference_image = ref_img_with_axis

    return new_img, reference_image
