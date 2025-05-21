cmake_minimum_required(VERSION 3.10)

# 정책 설정
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

project(voxel_grid LANGUAGES CXX)

# 의존성
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs)
find_package(PCL REQUIRED)

# PCL 설정
add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

# 소스 파일 목록 (visualize.cpp, main.cpp 삭제)
set(SOURCES
    src/voxel_grid.cpp
    # src/visualize.cpp   # 삭제됨
    # src/main.cpp        # 삭제됨
)

# 실행 파일 생성
add_executable(voxel_grid_node ${SOURCES})

# 헤더 포함 경로
target_include_directories(voxel_grid_node PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 라이브러리 링크
target_link_libraries(voxel_grid_node
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
)

# 설치 설정
install(TARGETS voxel_grid_node DESTINATION bin)
