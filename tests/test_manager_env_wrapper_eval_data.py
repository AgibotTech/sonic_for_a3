import ast
from pathlib import Path
from types import SimpleNamespace

WRAPPER_PATH = (
    Path(__file__).parents[1]
    / "gear_sonic"
    / "envs"
    / "wrapper"
    / "manager_env_wrapper.py"
)


def _load_get_env_data_without_isaac_dependencies():
    tree = ast.parse(WRAPPER_PATH.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ManagerEnvWrapper"
    )
    method = next(
        node
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_env_data"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(WRAPPER_PATH), "exec"), namespace)
    return namespace["get_env_data"]


def test_eval_position_fields_preserve_reference_and_prediction_semantics():
    get_env_data = _load_get_env_data_without_isaac_dependencies()
    reference_positions = object()
    robot_positions = object()
    wrapper = SimpleNamespace(
        motion_command=SimpleNamespace(
            body_pos_w=reference_positions,
            robot_body_pos_w=robot_positions,
        )
    )

    assert get_env_data(wrapper, "ref_body_pos_extend") is reference_positions
    assert get_env_data(wrapper, "rigid_body_pos_extend") is robot_positions


def test_eval_data_unknown_key_delegates_to_wrapped_environment():
    get_env_data = _load_get_env_data_without_isaac_dependencies()

    class StubEnv:
        def get_env_data(self, key):
            return ("delegated", key)

    wrapper = SimpleNamespace(motion_command=SimpleNamespace(), env=StubEnv())

    assert get_env_data(wrapper, "other") == ("delegated", "other")
