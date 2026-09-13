"""Weight-free regression of the actual LeRobot image transform and serving recipe."""
import hashlib
import json
from pathlib import Path
import numpy as np
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.processor_groot import _transform_n1_7_image_for_vlm_albumentations as transform
from policy.backends.lerobot.models.groot import EXPECTED_TAG, serving_preprocessor_overrides
from policy.backends.lerobot.features import CAMERA_KEYS, FRAME_HEIGHT, FRAME_WIDTH

FIXTURE = Path(__file__).parents[1] / "fixtures/groot-so101"
FIELDS = ("image_crop_size", "image_target_size", "shortest_image_edge", "crop_fraction", "letter_box_transform")


def served_recipe():
    config = GrootConfig(base_model_path=str(FIXTURE), embodiment_tag=EXPECTED_TAG, model_params_fp32=False)
    config.input_features = {f"observation.images.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, FRAME_HEIGHT, FRAME_WIDTH)) for c in CAMERA_KEYS}
    config.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    config.device = "cpu"
    pre, _ = make_pre_post_processors(config, pretrained_path=str(FIXTURE),
        preprocessor_overrides={"device_processor": {"device": "cpu"}, "rename_observations_processor": {"rename_map": {}}, **serving_preprocessor_overrides()},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    step = next(s for s in pre.steps if hasattr(s, "letter_box_transform"))
    return {k: getattr(step, k) for k in FIELDS}


def test_serving_geometry_matches_recorded_native_bytes():
    recipe = served_recipe()
    declared = json.loads((FIXTURE / "processor_config.json").read_text())["processor_kwargs"]
    assert declared["letter_box_transform"] is False and recipe["letter_box_transform"] is True
    assert all(recipe[k] == declared[k] for k in FIELDS if k != "letter_box_transform")
    frame = np.random.RandomState(0).randint(0, 255, (480, 640, 3), dtype=np.uint8)
    served = transform(frame, **recipe)
    assert served.shape == (256, 256, 3) and served.dtype == np.uint8
    # Historical native eval result on the identical deterministic RGB input.
    assert hashlib.sha256(served.tobytes()).hexdigest() == "c30150ec8d9d7ccb648aade0588aed2d18a356ab7f984a510dc57bb6c485927f"
    unpatched = transform(frame, **dict(recipe, letter_box_transform=False))
    assert unpatched.shape == (256, 340, 3)
    assert np.array_equal(served, transform(frame, **recipe))


def test_square_placeholder_resize_is_not_an_equivalent_preprocessor():
    import cv2
    recipe = served_recipe()
    frame = np.random.RandomState(0).randint(0, 255, (480, 640, 3), dtype=np.uint8)
    actual = transform(frame, **recipe)
    resized = transform(cv2.resize(frame, (224, 224)), **recipe)
    assert actual.shape == resized.shape
    assert not np.array_equal(actual, resized)
    assert np.array_equal(actual, transform(frame, **dict(recipe, image_crop_size=[999, 999])))
