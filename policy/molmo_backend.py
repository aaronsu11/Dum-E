"""MolmoAct 2 SO101 adapter; physical limits belong to the attended runner."""
from policy.http_backend import SO101HTTPPolicyBackend

class MolmoPolicyBackend(SO101HTTPPolicyBackend):
    def __init__(self, port=18081, language_instruction=None):
        super().__init__(port, language_instruction, profile="molmoact2-so101")
