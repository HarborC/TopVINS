#ifndef OV_MSCKF_FOXGLOVEVISUALIZER_H
#define OV_MSCKF_FOXGLOVEVISUALIZER_H

#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <Eigen/Eigen>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <foxglove_warp/visualizer.hpp>

namespace ov_core {
class YamlParser;
struct CameraData;
}

namespace ov_msckf {

class Simulator;
class VioManager;

class FoxgloveVisualizer {
public:
  FoxgloveVisualizer(std::shared_ptr<ros::NodeHandle> nh, std::shared_ptr<VioManager> app,
                     std::shared_ptr<Simulator> sim = nullptr);

  void setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser);
  void visualize();
  void visualize_odometry(double timestamp);
  void visualize_final();

  void callback_inertial(const sensor_msgs::Imu::ConstPtr &msg);
  void callback_monocular(const sensor_msgs::ImageConstPtr &msg, int cam_id);
  void callback_stereo(const sensor_msgs::ImageConstPtr &left, const sensor_msgs::ImageConstPtr &right,
                       int left_id, int right_id);

protected:
  void publish_state();
  void publish_images();
  void publish_features();
  void publish_cameras();

  std::shared_ptr<ros::NodeHandle> _nh;
  std::shared_ptr<VioManager> _app;
  std::shared_ptr<Simulator> _sim;
  std::unique_ptr<foxglove_warp::Visualizer> _viz;

  std::atomic<bool> thread_update_running{false};
  std::deque<ov_core::CameraData> camera_queue;
  std::mutex camera_queue_mtx;
  std::map<int, double> camera_last_timestamp;

  using StereoSyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;
  std::vector<ros::Subscriber> subs_cam;
  std::vector<std::shared_ptr<message_filters::Synchronizer<StereoSyncPolicy>>> sync_cam;
  std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> sync_subs_cam;
  ros::Subscriber sub_imu;

  unsigned int poses_seq_imu = 0;
  std::vector<Eigen::Matrix4f> poses_imu;
  std::deque<std::pair<double, Eigen::Matrix4f>> poses_imu_odom;
  double last_visualization_timestamp = 0;
  double last_visualization_timestamp_image = 0;
  bool start_time_set = false;
  boost::posix_time::ptime rT1, rT2;
};

} // namespace ov_msckf

#endif // OV_MSCKF_FOXGLOVEVISUALIZER_H
