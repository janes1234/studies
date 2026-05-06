cmake_minimum_required(VERSION 3.13)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(example example.cc)
target_link_libraries(example emdw)

add_executable(week1 week1.cc)
target_link_libraries(week1 emdw)

add_executable(week2 week2.cc)
target_link_libraries(week2 emdw)

add_executable(week2Q2 week2Q2.cc)
target_link_libraries(week2Q2 emdw)

add_executable(week2Q3 week2Q3.cc)
target_link_libraries(week2Q3 emdw)

add_executable(week3Q1 week3Q1.cc)
target_link_libraries(week3Q1 emdw)

add_executable(week3Q2 week3Q2.cc)
target_link_libraries(week3Q2 emdw)
