cmake_minimum_required(VERSION 3.10)
project(Hybrid-ANNS VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 如果没有指定构建类型，则设置为Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -mavx2 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# 支持OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 查找BLAS/LAPACK库
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# 查找Faiss库
find_package(faiss QUIET)

# Faiss构建选项
option(BUILD_FAISS_FROM_SOURCE "Build Faiss from source if not found" ON)

# 如果没有找到系统安装的Faiss，则尝试从源码构建
if(NOT faiss_FOUND AND BUILD_FAISS_FROM_SOURCE)

    set(FAISS_INSTALL_DIR ${CMAKE_BINARY_DIR}/faiss/install)
    set(FAISS_INCLUDE_DIR ${FAISS_INSTALL_DIR}/include)
    set(FAISS_LIBRARY_DIR ${FAISS_INSTALL_DIR}/lib)
    set(FAISS_LIBRARY ${FAISS_LIBRARY_DIR}/libfaiss.a)

    # 确保目录存在
    file(MAKE_DIRECTORY ${FAISS_INCLUDE_DIR})
    file(MAKE_DIRECTORY ${FAISS_LIBRARY_DIR})

    include(ExternalProject)
    
    # 设置Faiss源码下载和构建
    ExternalProject_Add(
        faiss_external
        GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
        GIT_TAG v1.10.0  # 指定一个稳定的版本
        PREFIX ${CMAKE_BINARY_DIR}/faiss
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/faiss/install
            -DCMAKE_BUILD_TYPE=Release
            -DFAISS_ENABLE_GPU=OFF
            -DFAISS_ENABLE_PYTHON=OFF
            -DBUILD_TESTING=OFF
            -DFAISS_OPT_LEVEL=generic  # 使用通用优化级别，以增加兼容性
        BUILD_BYPRODUCTS
            ${CMAKE_BINARY_DIR}/faiss/install/lib/libfaiss.a
    )
    

    # 设置Faiss的包含目录和库目录
    set(FAISS_INCLUDE_DIR ${CMAKE_BINARY_DIR}/faiss/install/include)
    set(FAISS_LIBRARY ${CMAKE_BINARY_DIR}/faiss/install/lib/libfaiss.a)
    
    # 添加自定义目标
    add_library(faiss STATIC IMPORTED)
    set_target_properties(faiss PROPERTIES
        IMPORTED_LOCATION ${FAISS_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${FAISS_INCLUDE_DIR}
    )
    
    # 添加依赖关系
    add_dependencies(faiss faiss_external)
    
    message(STATUS "Will build Faiss from source")
else()
    message(STATUS "Using system-installed Faiss")
endif()

# 添加包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/scr_range)
if(NOT faiss_FOUND AND BUILD_FAISS_FROM_SOURCE)
    include_directories(${FAISS_INCLUDE_DIR})
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 确保输出目录存在
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/label)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/range)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/index)


# 设置公共头文件，分类整理
set(COMMON_HEADERS_LABEL
    src/FileReader.h
    src/HybridSearchExperiment.h
)

set(ONE_ATTR_HEADERS
    src/DataStructures_OneAttr.h
)

set(THREE_ATTR_HEADERS
    src/DataStructures_ThreeAttr.h
)

set(COMMON_HEADERS_RANGE
    scr_range/FileReader.h
    scr_range/DataStructures.h
    scr_range/HybridSearchExperiment.h
)

# 编译src目录中的所有cc文件到bin/label目录
file(GLOB SRC_CC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
foreach(src_file ${SRC_CC_FILES})
    # 获取文件名（不含扩展名）
    get_filename_component(target_name ${src_file} NAME_WE)
    
    # 添加可执行文件
    if(${target_name} MATCHES "oneattr")
        add_executable(${target_name} ${src_file} ${COMMON_HEADERS_LABEL} ${ONE_ATTR_HEADERS})
        target_compile_definitions(${target_name} PRIVATE USE_ONE_ATTR)
    elseif(${target_name} MATCHES "threeattr")
        add_executable(${target_name} ${src_file} ${COMMON_HEADERS_LABEL} ${THREE_ATTR_HEADERS})
        target_compile_definitions(${target_name} PRIVATE USE_THREE_ATTR)
    else()
        # 默认使用单属性结构
        add_executable(${target_name} ${src_file} ${COMMON_HEADERS_LABEL} ${ONE_ATTR_HEADERS})
        target_compile_definitions(${target_name} PRIVATE USE_ONE_ATTR)
    endif()
    
    # 设置输出目录
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/label"
    )
    
    # 链接必要的库
    target_link_libraries(${target_name} 
        faiss
        OpenMP::OpenMP_CXX
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
    )
endforeach()

# 编译scr_range目录中的所有cc文件到bin/range目录
file(GLOB RANGE_CC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/scr_range/*.cc")
foreach(range_file ${RANGE_CC_FILES})
    # 获取文件名（不含扩展名）
    get_filename_component(target_name ${range_file} NAME_WE)
    
    # 添加前缀或后缀以避免与src目录中的目标名称冲突
    set(unique_target_name "${target_name}_range")
    
    # 添加可执行文件 - 只使用scr_range/DataStructures.h
    add_executable(${unique_target_name} ${range_file} ${COMMON_HEADERS_RANGE})
    
    # 设置输出目录
    set_target_properties(${unique_target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/range"
    )
    
    # 链接必要的库
    target_link_libraries(${unique_target_name}
        faiss
        OpenMP::OpenMP_CXX
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
    )
endforeach()


# 安装从src和scr_range编译的可执行文件
file(GLOB SRC_EXECUTABLES "${CMAKE_BINARY_DIR}/bin/label/*")
file(GLOB RANGE_EXECUTABLES "${CMAKE_BINARY_DIR}/bin/range/*")
install(PROGRAMS ${SRC_EXECUTABLES} DESTINATION bin/label)
install(PROGRAMS ${RANGE_EXECUTABLES} DESTINATION bin/range)


# 安装README文件
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    ${CMAKE_CURRENT_BINARY_DIR}/README.md
    COPYONLY
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md DESTINATION .)

# 生成 compile_commands.json 以支持代码分析工具
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)