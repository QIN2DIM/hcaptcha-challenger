import matplotlib.pyplot as plt

from hcaptcha_challenger.helper.create_coordinate_grid import create_coordinate_grid


def test_create_coordinate_grid():
    image_path = "challenge_view/image_label_area_select/2.png"
    output_path = "challenge_view/image_label_area_select/grid_divisions_2.png"
    bbox = {'x': 98, 'y': 30, 'width': 500, 'height': 430}

    result = create_coordinate_grid(image_path, bbox)

    # Display result
    # plt.figure(figsize=(10, 10))
    # plt.imshow(result)
    # plt.axis('off')

    # Save
    plt.imsave(output_path, result)
