add_library(special_points scr/frame.cpp)
target_include_directories(special_points PUBLIC include
        PUBLIC ${OpenCV_INCLUDE_DIR})
target_link_libraries(special_points PUBLIC ${OpenCV_LIBRARIES})

add_executable(special_points_main main.cpp)
target_link_libraries(special_points_main PUBLIC special_points)
