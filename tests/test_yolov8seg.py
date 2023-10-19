import hcaptcha_challenger as solver
from hcaptcha_challenger.onnx.modelhub import ModelHub

solver.install(upgrade=True)


def test_yolov8seg_ash():
    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    ash = "please click on the object that appears only once"
    for focus_name, classes in modelhub.lookup_ash_of_war(ash):
        assert "appears_only_once" in focus_name
        assert "_yolov8" in focus_name
        assert "-seg" in focus_name
        assert ".onnx" in focus_name
