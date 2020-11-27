cmake_minimum_required(VERSION 3.10)
project(data_augmentation)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED HINTS)
find_package(TorchVision QUIET HINTS)

if(TorchVision_FOUND)
	add_executable(train train.cpp utils.cpp)
	target_link_libraries(train ${OpenCV_LIBS} ${TORCH_LIBRARIES} TorchVision::TorchVision)
else()
	add_executable(train train.cpp utils.cpp model/resnet.cpp)
	target_link_libraries(train ${OpenCV_LIBS} ${TORCH_LIBRARIES})
endif()
