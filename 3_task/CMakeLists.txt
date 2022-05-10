file(GLOB SRC
        src/*.cpp)

add_executable(solver ${SRC})
target_include_directories(solver PUBLIC include)
target_link_libraries(solver PUBLIC Eigen3::Eigen)
