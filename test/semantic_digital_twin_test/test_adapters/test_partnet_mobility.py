from semantic_digital_twin.adapters.partnet_mobility_dataset.loader import (
    PartNetMobilityDatasetLoader,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)


def test_loader(rclpy_node):

    loader = PartNetMobilityDatasetLoader()
    world = loader.load()
    VizMarkerPublisher(node=rclpy_node, _world=world).with_tf_publisher()


def test_semantics_extraction():
    loader = PartNetMobilityDatasetLoader()
    loader._create_python_file_with_semantic_annotations_from_dataset()
