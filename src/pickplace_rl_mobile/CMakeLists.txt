cmake_minimum_required(VERSION 3.8)
project(pickplace_rl_mobile)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  urdf
  config
  worlds
  DESTINATION share/${PROJECT_NAME}/
)

# Install executable scripts
install(PROGRAMS
  pickplace_rl_mobile/train_rl.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME train_rl
)
install(PROGRAMS
  pickplace_rl_mobile/test_policy.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_policy
)
install(PROGRAMS
  pickplace_rl_mobile/demo_pick_place.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME demo_pick_place
)
install(PROGRAMS
  pickplace_rl_mobile/perception_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME perception_node
)
install(PROGRAMS
  pickplace_rl_mobile/manip_rl_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME manip_rl_node
)
install(PROGRAMS
  pickplace_rl_mobile/wbc_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME wbc_node
)
install(PROGRAMS
  pickplace_rl_mobile/safety_guard.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME safety_guard
)
install(PROGRAMS
  pickplace_rl_mobile/smart_pick_place.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME smart_pick_place
)
install(PROGRAMS
  pickplace_rl_mobile/vla_action_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME vla_action_node
)
install(PROGRAMS
  pickplace_rl_mobile/vla_vision_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME vla_vision_node
)
install(PROGRAMS
  pickplace_rl_mobile/vla_language_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME vla_language_node
)
install(PROGRAMS
  pickplace_rl_mobile/vla_coordinator_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME vla_coordinator_node
)

# Install config files




if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()