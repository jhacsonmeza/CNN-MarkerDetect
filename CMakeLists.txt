cmake_minimum_required(VERSION 3.10)
project(marker_detect)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision QUIET)

if(TorchVision_FOUND)
	add_executable(train train.cpp utils.cpp)
	target_link_libraries(train ${OpenCV_LIBS} ${TORCH_LIBRARIES} TorchVision::TorchVision)
	
	add_executable(realtime realtime.cpp)
	target_link_libraries(realtime ${OpenCV_LIBS} ${TORCH_LIBRARIES} TorchVision::TorchVision)
else()
	add_executable(train train.cpp utils.cpp model/resnet.cpp)
	target_link_libraries(train ${OpenCV_LIBS} ${TORCH_LIBRARIES})
	
	add_executable(realtime realtime.cpp model/resnet.cpp)
	target_link_libraries(realtime ${OpenCV_LIBS} ${TORCH_LIBRARIES})
endif()
