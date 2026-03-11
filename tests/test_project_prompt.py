import os
from unittest.mock import patch

from axon.main import AxonBrain, AxonConfig


def test_should_recommend_project_true_when_default_and_no_projects():
    with patch("axon.projects.list_projects", return_value=[]):
        config = AxonConfig()
        brain = AxonBrain(config)
        # By default brain starts in 'default' project
        assert brain._active_project == "default"
        assert brain.should_recommend_project() is True


def test_should_recommend_project_false_when_not_default():
    with patch("axon.projects.list_projects", return_value=[]):
        config = AxonConfig()
        brain = AxonBrain(config)
        brain._active_project = "my-project"
        assert brain.should_recommend_project() is False


def test_should_recommend_project_false_when_projects_exist():
    with patch("axon.projects.list_projects", return_value=[{"name": "p1"}]):
        config = AxonConfig()
        brain = AxonBrain(config)
        assert brain.should_recommend_project() is False


def test_wsl_path_resolution_legacy_defaults():
    # Legacy defaults should be forced to home projects/default
    config = AxonConfig(vector_store_path="./chroma_data", bm25_path="./bm25_index")
    home = os.path.join(os.path.expanduser("~"), ".axon")
    assert config.vector_store_path == os.path.join(home, "projects", "default", "chroma_data")
    assert config.bm25_path == os.path.join(home, "projects", "default", "bm25_index")
    assert config.projects_root == os.path.join(home, "projects")


def test_wsl_path_resolution_mnt_c_redirect():
    # If running in linux and path is on /mnt/, it should redirect if it looks like default
    with patch("sys.platform", "linux"):
        with patch("os.name", "posix"):
            # Test path containing 'axon'
            config = AxonConfig(vector_store_path="/mnt/c/dev/axon/chroma_data")
            home = os.path.join(os.path.expanduser("~"), ".axon")
            expected = os.path.join(home, "projects", "default", "chroma_data")
            assert config.vector_store_path == expected

            # Test path NOT containing 'axon' or 'studio_brain' - should stay as is
            config = AxonConfig(vector_store_path="/mnt/d/my_custom_store")
            assert config.vector_store_path == "/mnt/d/my_custom_store"
