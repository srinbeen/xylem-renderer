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
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/relDeb")
set(DONUT_SHADERS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${DONUT_SHADERS_OUTPUT_SUBDIR}")

# Set Donut options before adding the subdirectory
set(DONUT_WITH_DX12 ON CACHE BOOL "" FORCE)
set(DONUT_WITH_DX11 OFF CACHE BOOL "" FORCE)
set(DONUT_WITH_VULKAN OFF CACHE BOOL "" FORCE)


set(DONUT_FORCE_DISCRETE_GPU ON CACHE BOOL "" FORCE)

set(DONUT_WITH_AFTERMATH OFF CACHE BOOL "" FORCE)

if (DONUT_WITH_AFTERMATH)
    set(AFTERMATH_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/aftermath-src")
endif()

if (NVRHI_WITH_DX12)
    set(DONUT_D3D_AGILITY_SDK_URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.717.1-preview")
    set(DONUT_D3D_AGILITY_SDK_FETCH_DIR "" CACHE STRING "" FORCE)
    include("${CMAKE_CURRENT_SOURCE_DIR}/donut/cmake/FetchAgilitySDK.cmake")

    file(COPY
        "${DONUT_D3D_AGILITY_SDK_PATH}/build/native/bin/x64/D3D12Core.dll"
        "${DONUT_D3D_AGILITY_SDK_PATH}/build/native/bin/x64/D3D12SDKLayers.dll"
        DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/D3D12/")
endif()

set(${PROJECT_NAME}_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(${PROJECT_NAME}_INCLUDE_DIR "${${PROJECT_NAME}_SOURCE_DIR}/include")
# for 32 bit indices
add_compile_definitions(IMGUI_USER_CONFIG="${${PROJECT_NAME}_INCLUDE_DIR}/xylem_imgui_config.h")

# Include the Donut framework
add_subdirectory(Donut)
# Include meshoptimizer
add_subdirectory(meshoptimizer)

include(donut/compileshaders.cmake)
# file(GLOB SHADERS "${SOURCE_DIR}/shaders/*.hlsl")
file(GLOB ${PROJECT_NAME}_SOURCES CONFIGURE_DEPENDS "${${PROJECT_NAME}_SOURCE_DIR}/*.cpp")

set(SHADER_REG_SPEC "${${PROJECT_NAME}_SOURCE_DIR}/shaders/shader_register_map.json")
set(SHADER_REG_GEN_CMAKE "${CMAKE_CURRENT_SOURCE_DIR}/generate_shader_register_map.cmake")
set(SHADER_REG_CPP "${${PROJECT_NAME}_INCLUDE_DIR}/shaders/ShaderRegisterMap.hpp")
set(SHADER_REG_HLSL "${${PROJECT_NAME}_SOURCE_DIR}/shaders/ShaderRegisterMap.hlsli")

add_custom_command(
    OUTPUT "${SHADER_REG_CPP}" "${SHADER_REG_HLSL}"
    COMMAND "${CMAKE_COMMAND}"
        "-DSHADER_REG_SPEC=${SHADER_REG_SPEC}"
        "-DSHADER_REG_CPP=${SHADER_REG_CPP}"
        "-DSHADER_REG_HLSL=${SHADER_REG_HLSL}"
        -P "${SHADER_REG_GEN_CMAKE}"
    DEPENDS "${SHADER_REG_SPEC}" "${SHADER_REG_GEN_CMAKE}"
    COMMENT "Generating coupled shader register maps (C++ + HLSL)"
    VERBATIM
)

add_custom_target(${PROJECT_NAME}_shader_registers
    DEPENDS "${SHADER_REG_CPP}" "${SHADER_REG_HLSL}"
)

donut_compile_shaders_all_platforms(
    TARGET ${PROJECT_NAME}_shaders
    PROJECT_NAME ${PROJECT_NAME}
    FOLDER ${PROJECT_NAME}
    CONFIG ${${PROJECT_NAME}_SOURCE_DIR}/shaders/shaders.cfg
    OUTPUT_BASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/custom
    SHADER_MODEL 6_5
)

add_executable(${PROJECT_NAME} WIN32 ${${PROJECT_NAME}_SOURCES}
    Donut/thirdparty/imgui/misc/cpp/imgui_stdlib.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE DONUT_SHADERS_OUTPUT_SUBDIR="${DONUT_SHADERS_OUTPUT_SUBDIR}")

if (DONUT_WITH_AFTERMATH)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${AFTERMATH_RUNTIME_LIBRARY}"
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}"
)
endif()

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_shader_registers ${PROJECT_NAME}_shaders)
add_dependencies(${PROJECT_NAME}_shaders ${PROJECT_NAME}_shader_registers)
target_link_libraries(${PROJECT_NAME} PRIVATE donut_app donut_render donut_engine donut_core meshoptimizer)
if (DONUT_WITH_AFTERMATH)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "${AFTERMATH_SDK_DIR}/lib/x64/GFSDK_Aftermath_Lib.x64.lib"
    )
endif()

target_include_directories(${PROJECT_NAME} PRIVATE 
    "${${PROJECT_NAME}_INCLUDE_DIR}"
    "${DONUT_D3D_AGILITY_SDK_PATH}/build/native/include"
)
if (DONUT_WITH_AFTERMATH)
    target_include_directories(${PROJECT_NAME} PRIVATE
        "${AFTERMATH_SDK_DIR}/include"
    )
endif()

if (DONUT_WITH_AFTERMATH)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${AFTERMATH_SDK_DIR}/lib/x64/GFSDK_Aftermath_Lib.x64.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    )
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER ${PROJECT_NAME})

