find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_executable(resistor src/main.cpp)
target_link_libraries(resistor PUBLIC Eigen3::Eigen)
