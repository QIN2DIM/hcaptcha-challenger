import cv2

from hcaptcha_challenger.helper.rasterization import overlay_grid_on_image

prompt = """
Identify the correct area to drag the element into for optimal proximity to its matching puzzle piece's gap.

Solve the challenge, use [0,0] ~ [4,4] to locate 25grid, output the coordinates of the correct answer as json.
"""


def test_overlay_grid_on_image():
    try:
        real_image = cv2.imread("challenge_view/image_drag_drop/20250331223605022127.png")
        s = real_image.shape

        # Define a bounding box
        bbox = ((int(s[0] * 0.03), int(s[1] * 0.2)), (int(s[0] * 0.85), int(s[1] * 0.83)))

        # Create grid references with different division levels
        divisions = 2

        # Overlay grid on the sample image
        result_image = overlay_grid_on_image(real_image, bbox, divisions)

        # Display the result
        cv2.imshow(f"Grid with {divisions} divisions", result_image)
        cv2.waitKey(0)

        # Save the result
        cv2.imwrite(f"grid_divisions_{divisions}.jpg", result_image)

        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error occurred: {e}")
