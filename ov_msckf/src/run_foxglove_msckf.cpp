/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */

#include <memory>

#include <ros/ros.h>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/print.h"
#include "visualizer/FoxgloveVisualizer.h"

using namespace ov_msckf;

int main(int argc, char **argv) {
  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  ros::init(argc, argv, "run_foxglove_msckf");
  auto node_handle = std::make_shared<ros::NodeHandle>("~");
  node_handle->param<std::string>("config_path", config_path, config_path);

  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  parser->set_node_handler(node_handle);

  std::string verbosity = "DEBUG";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  VioManagerOptions parameters;
  parameters.print_and_load(parser);
  parameters.use_multi_threading_subs = true;
  auto system = std::make_shared<VioManager>(parameters);
  auto visualizer = std::make_shared<FoxgloveVisualizer>(node_handle, system);
  visualizer->setup_subscribers(parser);

  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    return EXIT_FAILURE;
  }

  ros::AsyncSpinner spinner(0);
  spinner.start();
  ros::waitForShutdown();
  visualizer->visualize_final();
  ros::shutdown();
  return EXIT_SUCCESS;
}
