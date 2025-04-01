import cv2

from hcaptcha_challenger.helper.mark_element_point import mark_points_on_image


def test_mark_element_point():
    image_path = "challenge_view/image_drag_drop/1.png"
    coordinates = [(553, 553)]

    try:
        # Replace with your image path
        result_image = mark_points_on_image(image_path, coordinates, output_path="marked_image.jpg")

        # Display the result
        cv2.imshow("Marked Image", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Marked {len(coordinates)} points on the image and saved as 'marked_image.jpg'")
    except Exception as e:
        print(f"Error occurred: {e}")
