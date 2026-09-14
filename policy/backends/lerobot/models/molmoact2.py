"""Pinned molmoact2 loading and processor configuration."""

from pathlib import Path


def load(runtime):
    from lerobot.configs import PolicyFeature, FeatureType
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors
    config = MolmoAct2Config(
        checkpoint_path=runtime.snapshot, device="cuda", model_dtype="bfloat16",
        norm_tag="so100_so101_molmoact2", action_mode="continuous",
        inference_action_mode="continuous", normalize_gripper=True,
        enable_inference_cuda_graph=False, num_inference_steps=10,
        chunk_size=30, n_action_steps=30,
        image_keys=["observation.images.front", "observation.images.wrist"],
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
    )
    runtime.policy = MolmoAct2Policy(config)
    runtime.pre, runtime.post = make_molmoact2_pre_post_processors(config)
