import importlib.util
import unittest
import sys
import types
from pathlib import Path
from types import SimpleNamespace


_USING_TORCH_STUB = False


def _import_torch_or_stub():
    global _USING_TORCH_STUB
    try:
        import torch
        return torch
    except Exception:
        _USING_TORCH_STUB = True
        for module_name in list(sys.modules):
            if module_name == "torch" or module_name.startswith("torch."):
                sys.modules.pop(module_name, None)

        torch_stub = types.ModuleType("torch")
        nn_stub = types.ModuleType("torch.nn")
        functional_stub = types.ModuleType("torch.nn.functional")

        class Module:
            def __init__(self, *args, **kwargs):
                pass

        class Identity(Module):
            pass

        class Function:
            pass

        nn_stub.Module = Module
        nn_stub.Identity = Identity
        nn_stub.Sequential = object
        nn_stub.ModuleDict = dict
        nn_stub.Embedding = object
        nn_stub.Parameter = object
        torch_stub.Tensor = object
        torch_stub.device = object
        torch_stub.nn = nn_stub
        torch_stub.autograd = types.SimpleNamespace(Function=Function)

        sys.modules["torch"] = torch_stub
        sys.modules["torch.nn"] = nn_stub
        sys.modules["torch.nn.functional"] = functional_stub
        return torch_stub


torch = _import_torch_or_stub()


def _load_afsr_method_without_package_init():
    repo_root = Path(__file__).resolve().parents[1]
    methods_pkg = types.ModuleType("methods")
    methods_pkg.__path__ = [str(repo_root / "methods")]
    sys.modules["methods"] = methods_pkg

    spec = importlib.util.spec_from_file_location(
        "methods.wrappers",
        repo_root / "methods" / "wrappers.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["methods.wrappers"] = module
    spec.loader.exec_module(module)
    return module.AFSRMethod


if _USING_TORCH_STUB:
    AFSRMethod = _load_afsr_method_without_package_init()
    for module_name in [
        "methods",
        "methods.wrappers",
        "methods.augmentations",
        "methods.common",
        "methods.mce",
        "models.afrl_components",
        "torch",
        "torch.nn",
        "torch.nn.functional",
    ]:
        sys.modules.pop(module_name, None)
else:
    from methods.wrappers import AFSRMethod


class AFSRCandidateTests(unittest.TestCase):
    def test_multi_hop_candidates_accept_scored_similarity_entries(self):
        config = SimpleNamespace(hidden_units=4, num_items=6, seed=42)
        resources = {
            "similarity_candidates": {
                1: [(2, 0.9), (3, 0.8)],
                2: [(4, 0.7)],
                3: [(5, 0.6)],
            }
        }
        method = AFSRMethod(torch.nn.Identity(), config, resources)

        self.assertEqual(method._multi_hop_candidates(1, hops=2), [2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
