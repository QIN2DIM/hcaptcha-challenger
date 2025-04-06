from pathlib import Path

from hcaptcha_challenger.helper.create_comparison_image import create_comparison_image

dataset_dir = Path("funcaptcha/compare_entity_orientation")


def test_create_comparison_image():
    for input_path in dataset_dir.glob("*"):
        if "cmp" in input_path.name:
            continue

        array_filename = f"{input_path.stem}_cmp.png"
        array_path = input_path.parent.joinpath(array_filename)

        # --- 调用核心函数 ---
        print("正在生成图像...")
        array_image, _ = create_comparison_image(input_path, 135)

        # --- 保存结果图像 ---
        array_image.save(array_path)
        print(f"Array 图已保存到: {array_path}")
