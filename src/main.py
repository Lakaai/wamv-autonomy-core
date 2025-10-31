import rclpy
import logging
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from src.gaussian import Gaussian
from tf2_msgs.msg import TFMessage


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

class Subscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            TFMessage,
            '/wamv/pose',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg: TFMessage):
        for transform in msg.transforms:
            t = transform.transform.translation
            r = transform.transform.rotation
            self.get_logger().info(
                f"Frame {transform.header.frame_id} → {transform.child_frame_id} "
                f"pos=({t.x:.2f}, {t.y:.2f}, {t.z:.2f})"
                f"rot=({r.x:.2f}, {r.y:.2f}, {r.z:.2f}, {r.w:.2f})"
            )


def main(args=None):
    density = Gaussian.from_moment(np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    logging.info(density.mean)
    logging.info(density.covariance)
    
    rclpy.init(args=args)
    
    # # minimal_publisher = MinimalPublisher()
  
    # # rclpy.spin(minimal_publisher)

    # # Destroy the node explicitly
    # # (optional - otherwise it will be done automatically
    # # when the garbage collector destroys the node object)
    # # minimal_publisher.destroy_node()

    minimal_subscriber = Subscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
