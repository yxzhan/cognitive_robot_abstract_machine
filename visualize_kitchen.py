import rclpy
import threading
import time
from semantic_digital_twin.predetermined_maps.kitchen_environment import KitchenEnvironment
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher

def main():
    rclpy.init()

    print("Building kitchen environment...")
    world = KitchenEnvironment().get_world()
    print(f"World built with {len(world.bodies)} bodies.")

    node = rclpy.create_node("kitchen_visualizer")
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    tf_publisher = TFPublisher(_world=world, node=node)
    viz_publisher = VizMarkerPublisher(_world=world, node=node)

    print("Markers are being published.")
    print("In rviz2: Set 'Fixed Frame' to 'root' and add 'MarkerArray' for '/semantic_digital_twin/markers'.")
    print("Press Ctrl+C to stop.")

    try:
        while rclpy.ok():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
