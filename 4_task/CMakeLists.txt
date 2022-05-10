add_executable(image_reader src/main.cpp)
target_include_directories(image_reader PUBLIC ${OpenCV_INCLUDE_DIR})
target_link_libraries(image_reader PUBLIC ${OpenCV_LIBRARIES})
