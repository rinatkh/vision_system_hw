#unit-tests
add_executable(test_matrix src/test_matrix.cpp)
target_link_libraries(test_matrix matrix gtest)