"""知识库向量可视化的投影数学（api/knowledge.py 的纯函数部分）。

_pca_2d 的契约是"坐标严格落在 [0,1]、X/Y 同比例尺（簇形状不被拉伸）"。
这两个性质靠的是"用全局包围盒中点平移 + 全局跨度缩放"，而不是"按轴均值居中"——
后者在某一轴方差远大于另一轴时会把点画到画布外。这里用几组极端数据把这个性质钉住。
"""

import numpy as np
import pytest

from src.api.knowledge import _parse_vector, _pca_2d


def test_parse_vector_handles_spaces_and_signs():
    assert _parse_vector("[0.1, 0.2,-0.3]") == [0.1, 0.2, -0.3]
    assert _parse_vector("[]") == []


def test_pca_maps_to_full_unit_range():
    """非退化数据下，全局最小/最大坐标应正好落在 0 和 1。"""
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(40, 8))

    coords = _pca_2d(vectors)

    assert coords.shape == (40, 2)
    assert coords.min() == pytest.approx(0.0, abs=1e-9)
    assert coords.max() == pytest.approx(1.0, abs=1e-9)


def test_pca_stays_inside_unit_square_on_anisotropic_data():
    """各轴方差差异极大时也不能越界——这是"全局跨度缩放"相对"按轴缩放"的意义。"""
    rng = np.random.default_rng(7)
    scales = np.array([1.0, 1.0, 1.0, 1.0, 200.0, 200.0, 200.0, 200.0])
    vectors = rng.normal(size=(60, 8)) * scales

    coords = _pca_2d(vectors)

    assert coords.min() >= 0.0
    assert coords.max() <= 1.0


def test_pca_handles_single_row():
    # n < 2 时没有可投影的方向，全部落在画布中心
    coords = _pca_2d(np.array([[1.0, 2.0]]))

    assert np.allclose(coords, np.full((1, 2), 0.5))


def test_pca_handles_single_dimension():
    # d < 2 时无法取两个主成分
    coords = _pca_2d(np.array([[1.0], [2.0], [3.0]]))

    assert np.allclose(coords, np.full((3, 2), 0.5))


def test_pca_handles_zero_variance():
    """所有向量完全相同时跨度为 0，不能除零。"""
    coords = _pca_2d(np.tile(np.array([[1.0, 2.0, 3.0]]), (5, 1)))

    assert np.allclose(coords, np.full((5, 2), 0.5))


def test_pca_keeps_two_separated_clusters_apart():
    """相距很远的两个簇，投影后仍应明显分开，而不是被压成一团。

    这里刻意不拿 0.5 当分界：该数据只有 1 个有效方差方向，第二主成分退化，
    它的坐标会整体落在 0.5 上（无法区分两簇），所以只断言"存在某个方向把它们拉开"。
    """
    vectors = np.vstack([np.zeros((10, 4)), np.full((10, 4), 100.0)])

    coords = _pca_2d(vectors)

    gap = np.abs(coords[:10].mean(axis=0) - coords[10:].mean(axis=0))
    assert gap.max() > 0.5
