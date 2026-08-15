#include "FoxgloveVisualizer.h"

#include "core/VioManager.h"
#include "sim/Simulator.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "utils/dataset_reader.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include <cv_bridge/cv_bridge.h>

using namespace ov_core;
using namespace ov_type;

namespace ov_msckf {
namespace {

constexpr const char *kWorldFrame = "global";
constexpr const char *kImuFrame = "imu";

Eigen::Matrix4f pose_from_jpl(const Eigen::Vector4d &quaternion, const Eigen::Vector3d &position) {
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  pose.block<3, 3>(0, 0) = Eigen::Quaterniond(quaternion(3), quaternion(0), quaternion(1), quaternion(2)).toRotationMatrix().cast<float>();
  pose.block<3, 1>(0, 3) = position.cast<float>();
  return pose;
}

void publish_cloud(foxglove_warp::Visualizer &visualizer, const std::string &topic, int64_t timestamp,
                   const std::vector<Eigen::Vector3d> &features, const std::vector<uint8_t> &color) {
  std::vector<std::vector<float>> points;
  points.reserve(features.size());
  for (const auto &feature : features) {
    points.push_back({static_cast<float>(feature.x()), static_cast<float>(feature.y()), static_cast<float>(feature.z())});
  }
  visualizer.showPointCloud(topic, timestamp, points, {color}, kWorldFrame);
}

} // namespace

FoxgloveVisualizer::FoxgloveVisualizer(std::shared_ptr<ros::NodeHandle> nh, std::shared_ptr<VioManager> app,
                                       std::shared_ptr<Simulator> sim)
    : _nh(std::move(nh)), _app(std::move(app)), _sim(std::move(sim)) {
  int server_port = 8765;
  _nh->param<int>("foxglove_port", server_port, server_port);
  if (server_port < 1 || server_port > 65535) {
    throw std::invalid_argument("foxglove_port must be between 1 and 65535");
  }

  foxglove_warp::ServerOptions options;
  options.port = static_cast<uint16_t>(server_port);
  options.name = "TopVINS";
  _viz = std::make_unique<foxglove_warp::Visualizer>(std::move(options));
  PRINT_INFO("Foxglove WebSocket server listening on port %u\n", _viz->port());

  if (_app->get_params().use_multi_threading_pubs) {
    std::thread([this] {
      ros::Rate loop_rate(20);
      while (ros::ok()) {
        publish_images();
        loop_rate.sleep();
      }
    }).detach();
  }
}

void FoxgloveVisualizer::setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser) {
  assert(parser != nullptr);

  std::string topic_imu;
  _nh->param<std::string>("topic_imu", topic_imu, "/imu0");
  parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
  sub_imu = _nh->subscribe(topic_imu, 1000, &FoxgloveVisualizer::callback_inertial, this);

  const int camera_count = _app->get_params().state_options.num_cameras;
  if (camera_count == 2) {
    std::string left_topic, right_topic;
    _nh->param<std::string>("topic_camera0", left_topic, "/cam0/image_raw");
    _nh->param<std::string>("topic_camera1", right_topic, "/cam1/image_raw");
    parser->parse_external("relative_config_imucam", "cam0", "rostopic", left_topic);
    parser->parse_external("relative_config_imucam", "cam1", "rostopic", right_topic);
    auto left_subscriber = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(*_nh, left_topic, 1);
    auto right_subscriber = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(*_nh, right_topic, 1);
    auto synchronizer = std::make_shared<message_filters::Synchronizer<StereoSyncPolicy>>(StereoSyncPolicy(10), *left_subscriber, *right_subscriber);
    synchronizer->registerCallback(boost::bind(&FoxgloveVisualizer::callback_stereo, this, _1, _2, 0, 1));
    sync_cam.push_back(synchronizer);
    sync_subs_cam.push_back(left_subscriber);
    sync_subs_cam.push_back(right_subscriber);
  } else {
    for (int camera_id = 0; camera_id < camera_count; ++camera_id) {
      std::string topic;
      _nh->param<std::string>("topic_camera" + std::to_string(camera_id), topic,
                              "/cam" + std::to_string(camera_id) + "/image_raw");
      parser->parse_external("relative_config_imucam", "cam" + std::to_string(camera_id), "rostopic", topic);
      subs_cam.push_back(_nh->subscribe<sensor_msgs::Image>(
          topic, 10, boost::bind(&FoxgloveVisualizer::callback_monocular, this, _1, camera_id)));
    }
  }
}

void FoxgloveVisualizer::callback_inertial(const sensor_msgs::Imu::ConstPtr &msg) {
  ImuData measurement;
  measurement.timestamp = msg->header.stamp.toSec();
  measurement.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  measurement.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
  _app->feed_measurement_imu(measurement);
  visualize_odometry(measurement.timestamp);

  if (thread_update_running.exchange(true)) {
    return;
  }
  std::thread update_thread([this, timestamp = measurement.timestamp] {
    std::lock_guard<std::mutex> lock(camera_queue_mtx);
    std::map<int, bool> unique_camera_ids;
    for (const auto &camera : camera_queue) {
      unique_camera_ids[camera.sensor_ids.at(0)] = true;
    }
    const size_t required_cameras = _app->get_params().state_options.num_cameras == 2 ? 1 : _app->get_params().state_options.num_cameras;
    if (unique_camera_ids.size() == required_cameras) {
      const double imu_time_in_camera_clock = timestamp - _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
      while (!camera_queue.empty() && camera_queue.front().timestamp < imu_time_in_camera_clock) {
        _app->feed_measurement_camera(camera_queue.front());
        visualize();
        camera_queue.pop_front();
      }
    }
    thread_update_running = false;
  });
  if (_app->get_params().use_multi_threading_subs) {
    update_thread.detach();
  } else {
    update_thread.join();
  }
}

void FoxgloveVisualizer::callback_monocular(const sensor_msgs::ImageConstPtr &msg, int cam_id) {
  const double timestamp = msg->header.stamp.toSec();
  const double time_delta = 1.0 / _app->get_params().track_frequency;
  if (camera_last_timestamp.count(cam_id) && timestamp < camera_last_timestamp.at(cam_id) + time_delta) {
    return;
  }
  camera_last_timestamp[cam_id] = timestamp;

  cv_bridge::CvImageConstPtr image;
  try {
    image = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
  } catch (const cv_bridge::Exception &exception) {
    PRINT_ERROR("cv_bridge exception: %s\n", exception.what());
    return;
  }

  CameraData measurement;
  measurement.timestamp = timestamp;
  measurement.sensor_ids.push_back(cam_id);
  measurement.images.push_back(image->image.clone());
  measurement.masks.push_back(_app->get_params().use_mask ? _app->get_params().masks.at(cam_id)
                                                            : cv::Mat::zeros(image->image.rows, image->image.cols, CV_8UC1));
  std::lock_guard<std::mutex> lock(camera_queue_mtx);
  camera_queue.push_back(std::move(measurement));
  std::sort(camera_queue.begin(), camera_queue.end());
}

void FoxgloveVisualizer::callback_stereo(const sensor_msgs::ImageConstPtr &left, const sensor_msgs::ImageConstPtr &right,
                                         int left_id, int right_id) {
  const double timestamp = left->header.stamp.toSec();
  const double time_delta = 1.0 / _app->get_params().track_frequency;
  if (camera_last_timestamp.count(left_id) && timestamp < camera_last_timestamp.at(left_id) + time_delta) {
    return;
  }
  camera_last_timestamp[left_id] = timestamp;

  cv_bridge::CvImageConstPtr left_image;
  cv_bridge::CvImageConstPtr right_image;
  try {
    left_image = cv_bridge::toCvShare(left, sensor_msgs::image_encodings::MONO8);
    right_image = cv_bridge::toCvShare(right, sensor_msgs::image_encodings::MONO8);
  } catch (const cv_bridge::Exception &exception) {
    PRINT_ERROR("cv_bridge exception: %s\n", exception.what());
    return;
  }

  CameraData measurement;
  measurement.timestamp = timestamp;
  measurement.sensor_ids = {left_id, right_id};
  measurement.images = {left_image->image.clone(), right_image->image.clone()};
  if (_app->get_params().use_mask) {
    measurement.masks = {_app->get_params().masks.at(left_id), _app->get_params().masks.at(right_id)};
  } else {
    measurement.masks = {cv::Mat::zeros(left_image->image.rows, left_image->image.cols, CV_8UC1),
                         cv::Mat::zeros(right_image->image.rows, right_image->image.cols, CV_8UC1)};
  }
  std::lock_guard<std::mutex> lock(camera_queue_mtx);
  camera_queue.push_back(std::move(measurement));
  std::sort(camera_queue.begin(), camera_queue.end());
}

void FoxgloveVisualizer::visualize_odometry(double timestamp) {
  if (!_app->initialized()) {
    return;
  }
  Eigen::Matrix<double, 13, 1> propagated = Eigen::Matrix<double, 13, 1>::Zero();
  Eigen::Matrix<double, 12, 12> covariance = Eigen::Matrix<double, 12, 12>::Zero();
  if (!_app->get_propagator()->fast_state_propagate(_app->get_state(), timestamp, propagated, covariance)) {
    return;
  }
  const auto pose = pose_from_jpl(propagated.head<4>(), propagated.segment<3>(4));
  const int64_t timestamp_usec = static_cast<int64_t>(timestamp * 1e6);
  _viz->showPose("/topvins/transform/global_to_imu", timestamp_usec, pose, kWorldFrame, kImuFrame);
  _viz->publishOdometry("/topvins/odometry", timestamp_usec, kWorldFrame, kImuFrame, pose, propagated.segment<3>(7).cast<float>());

  poses_imu_odom.emplace_back(timestamp, pose);
  while (!poses_imu_odom.empty() && poses_imu_odom.front().first < timestamp - 10.0) {
    poses_imu_odom.pop_front();
  }
  std::vector<Eigen::Matrix4f> path;
  path.reserve(std::min<size_t>(poses_imu_odom.size(), 200));
  const size_t stride = std::max<size_t>(1, (poses_imu_odom.size() + 199) / 200);
  for (size_t index = 0; index < poses_imu_odom.size(); index += stride) {
    path.push_back(poses_imu_odom[index].second);
  }
  _viz->showPath("/topvins/path/odometry_10s", timestamp_usec, path, kWorldFrame);
}

void FoxgloveVisualizer::visualize() {
  if (last_visualization_timestamp == _app->get_state()->_timestamp && _app->initialized()) {
    return;
  }
  last_visualization_timestamp = _app->get_state()->_timestamp;
  if (!_app->get_params().use_multi_threading_pubs) {
    publish_images();
  }
  if (!_app->initialized()) {
    return;
  }
  if (!start_time_set) {
    rT1 = boost::posix_time::microsec_clock::local_time();
    start_time_set = true;
  }
  publish_state();
  publish_features();
  publish_cameras();
}

void FoxgloveVisualizer::publish_state() {
  const auto state = _app->get_state();
  const double timestamp = state->_timestamp + state->_calib_dt_CAMtoIMU->value()(0);
  const int64_t timestamp_usec = static_cast<int64_t>(timestamp * 1e6);
  const auto pose = pose_from_jpl(state->_imu->quat(), state->_imu->pos());
  _viz->showPose("/topvins/transform/state_imu", timestamp_usec, pose, kWorldFrame, kImuFrame);
  _viz->publishOdometry("/topvins/state", timestamp_usec, kWorldFrame, kImuFrame, pose, state->_imu->vel().cast<float>());
  _viz->publishVector3("/topvins/state/gyro_bias", timestamp_usec, state->_imu->bias_g().cast<float>());
  _viz->publishVector3("/topvins/state/accel_bias", timestamp_usec, state->_imu->bias_a().cast<float>());

  poses_imu.push_back(pose);
  std::vector<Eigen::Matrix4f> path;
  const size_t stride = std::max<size_t>(1, (poses_imu.size() + 16383) / 16384);
  path.reserve((poses_imu.size() + stride - 1) / stride);
  for (size_t index = 0; index < poses_imu.size(); index += stride) {
    path.push_back(poses_imu[index]);
  }
  _viz->showPath("/topvins/path/state", timestamp_usec, path, kWorldFrame);
  ++poses_seq_imu;
}

void FoxgloveVisualizer::publish_features() {
  const int64_t timestamp_usec = static_cast<int64_t>(_app->get_state()->_timestamp * 1e6);
  publish_cloud(*_viz, "/topvins/points/msckf", timestamp_usec, _app->get_good_features_MSCKF(), {255, 0, 0, 255});
  publish_cloud(*_viz, "/topvins/points/slam", timestamp_usec, _app->get_features_SLAM(), {0, 255, 0, 255});
  publish_cloud(*_viz, "/topvins/points/aruco", timestamp_usec, _app->get_features_ARUCO(), {0, 0, 255, 255});
  if (_sim != nullptr) {
    publish_cloud(*_viz, "/topvins/points/simulation", timestamp_usec, _sim->get_map_vec(), {255, 255, 255, 255});
  }
}

void FoxgloveVisualizer::publish_cameras() {
  const auto state = _app->get_state();
  const int64_t timestamp_usec = static_cast<int64_t>(state->_timestamp * 1e6);
  for (const auto &calibration : state->_calib_IMUtoCAM) {
    const int camera_id = calibration.first;
    const auto &imu_to_camera = calibration.second;
    Eigen::Matrix4f camera_to_imu = Eigen::Matrix4f::Identity();
    camera_to_imu.block<3, 3>(0, 0) = imu_to_camera->Rot().transpose().cast<float>();
    camera_to_imu.block<3, 1>(0, 3) = (-imu_to_camera->Rot().transpose() * imu_to_camera->pos()).cast<float>();
    const std::string camera_frame = "cam" + std::to_string(camera_id);
    _viz->showPose("/topvins/transform/imu_to_" + camera_frame, timestamp_usec, camera_to_imu, kImuFrame, camera_frame);

    const auto &camera = state->_cam_intrinsics_cameras.at(camera_id);
    if (camera) {
      const cv::Matx33d matrix = camera->get_K();
      Eigen::Matrix3f intrinsics = Eigen::Matrix3f::Identity();
      intrinsics(0, 0) = static_cast<float>(matrix(0, 0));
      intrinsics(1, 1) = static_cast<float>(matrix(1, 1));
      intrinsics(0, 2) = static_cast<float>(matrix(0, 2));
      intrinsics(1, 2) = static_cast<float>(matrix(1, 2));
      Eigen::Matrix<float, 3, 4> projection = Eigen::Matrix<float, 3, 4>::Zero();
      projection.block<3, 3>(0, 0) = intrinsics;
      _viz->showCameraCalibration("/topvins/camera/" + camera_frame + "/calibration", timestamp_usec, camera_frame,
                                   intrinsics, camera->w(), camera->h(), projection);
    }
  }
}

void FoxgloveVisualizer::publish_images() {
  if (_app->get_state() == nullptr ||
      (last_visualization_timestamp_image == _app->get_state()->_timestamp && _app->initialized())) {
    return;
  }
  last_visualization_timestamp_image = _app->get_state()->_timestamp;
  const cv::Mat image = _app->get_historical_viz_image();
  if (!image.empty()) {
    _viz->showImage("/topvins/image/tracks", static_cast<int64_t>(_app->get_state()->_timestamp * 1e6), image, "cam0");
  }
}

void FoxgloveVisualizer::visualize_final() {
  if (!start_time_set) {
    return;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();
  PRINT_INFO("TIME: %.3f seconds\n", (rT2 - rT1).total_microseconds() * 1e-6);
}

} // namespace ov_msckf
