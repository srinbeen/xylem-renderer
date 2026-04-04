cmake_minimum_required(VERSION 3.31)
project(Xylem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(DONUT_SHADERS_OUTPUT_SUBDIR "shaders/framework")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/debug")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/release")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(DONUT_SHADERS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${DONUT_SHADERS_OUTPUT_SUBDIR}")

# Set Donut options before adding the subdirectory
set(DONUT_WITH_DX12 ON CACHE BOOL "" FORCE)
set(DONUT_WITH_DX11 OFF CACHE BOOL "" FORCE)
set(DONUT_WITH_VULKAN OFF CACHE BOOL "" FORCE)


set(DONUT_FORCE_DISCRETE_GPU ON CACHE BOOL "" FORCE)

set(DONUT_WITH_AFTERMATH ON CACHE BOOL "" FORCE)
set(AFTERMATH_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/aftermath-src")


if (NVRHI_WITH_DX12)
    set(DONUT_D3D_AGILITY_SDK_URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.717.1-preview")
    set(DONUT_D3D_AGILITY_SDK_FETCH_DIR "" CACHE STRING "" FORCE)
    include("${CMAKE_CURRENT_SOURCE_DIR}/donut/cmake/FetchAgilitySDK.cmake")

    file(COPY
        "${DONUT_D3D_AGILITY_SDK_PATH}/build/native/bin/x64/D3D12Core.dll"
        "${DONUT_D3D_AGILITY_SDK_PATH}/build/native/bin/x64/D3D12SDKLayers.dll"
        DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/D3D12/")
endif()

# Include the Donut framework
add_subdirectory(Donut)

set(source_folder src)

include(donut/compileshaders.cmake)
file(GLOB shaders "${source_folder}/shaders/*.hlsl")
file(GLOB sources "${source_folder}/*.cpp")

donut_compile_shaders_all_platforms(
    TARGET ${PROJECT_NAME}_shaders
    PROJECT_NAME ${PROJECT_NAME}
    FOLDER ${PROJECT_NAME}
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/${source_folder}/shaders/shaders.cfg
    OUTPUT_BASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/custom
    SHADER_MODEL 6_5
)

add_executable(${PROJECT_NAME} WIN32 ${sources}
    Donut/thirdparty/imgui/misc/cpp/imgui_stdlib.cpp
)

target_compile_definitions(Xylem PRIVATE DONUT_SHADERS_OUTPUT_SUBDIR="${DONUT_SHADERS_OUTPUT_SUBDIR}")

if (DONUT_WITH_AFTERMATH)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${AFTERMATH_RUNTIME_LIBRARY}"
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}"
)
endif()

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_shaders)
target_link_libraries(${PROJECT_NAME} PRIVATE donut_app donut_render donut_engine donut_core
    "${AFTERMATH_SDK_DIR}/lib/x64/GFSDK_Aftermath_Lib.x64.lib"
)
target_include_directories(${PROJECT_NAME} PRIVATE 
    "${source_folder}/include"
    "${DONUT_D3D_AGILITY_SDK_PATH}/build/native/include"
    "${AFTERMATH_SDK_DIR}/include"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${AFTERMATH_SDK_DIR}/lib/x64/GFSDK_Aftermath_Lib.x64.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER ${PROJECT_NAME})

