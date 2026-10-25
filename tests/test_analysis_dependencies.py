import subprocess
import sys


def test_analysis_imports_without_model_or_plot_dependencies():
    code = """
import sys
from importlib.abc import MetaPathFinder
class NoModels(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'olmo_core', 'transformers', 'vllm', 'matplotlib'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, NoModels())
from src.analysis.measurements import router_cosines
from src.analysis.specialization import specialist_weight_share
from src.analysis.kmeans_errors import boundary_score
import numpy as np
assert np.array_equal(router_cosines(np.eye(3)), np.zeros(3))
assert specialist_weight_share(np.array([[2, 1]]), np.array([[3., 1.]]), 1) == .75
assert boundary_score(np.arange(18.).reshape(9, 1, 2)).shape == (1, 2)
"""
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
