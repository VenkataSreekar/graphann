cmake_minimum_required(VERSION 3.14)
project(graphann LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags: Added -ffast-math to allow AVX/SIMD auto-vectorization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Library with shared sources
add_library(graphann_lib STATIC
    src/distance.cpp
    src/io_utils.cpp
    src/vamana_index.cpp
)
target_include_directories(graphann_lib PUBLIC include)
target_link_libraries(graphann_lib PUBLIC OpenMP::OpenMP_CXX)

# Build index executable
add_executable(build_index src/build_index.cpp)
target_link_libraries(build_index PRIVATE graphann_lib)

# Search index executable
add_executable(search_index src/search_index.cpp)
target_link_libraries(search_index PRIVATE graphann_lib)

# ==========================================================================
# Vamana-JL library  (src_jl/)  — JL dimensionality reduction variant
# ==========================================================================
add_library(graphann_jl_lib STATIC
    src_jl/vamana_index_jl.cpp
    src/distance.cpp
    src/io_utils.cpp          
)

# Tell the JL library to look in the include/ folder for .h files
target_include_directories(graphann_jl_lib PUBLIC include)
target_link_libraries(graphann_jl_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(build_index_jl  src_jl/build_index_jl.cpp)
add_executable(search_index_jl src_jl/search_index_jl.cpp)

target_link_libraries(build_index_jl  PRIVATE graphann_jl_lib)
target_link_libraries(search_index_jl PRIVATE graphann_jl_lib)
# ─────────────────────────────────────────────────────────────────────────────
# IVRG patch — append ONLY these lines to the BOTTOM of your CMakeLists.txt.
# This adds three new targets: ivrg_index (lib), build_ivrg, search_ivrg.
# Do NOT change anything above — the existing graphann_lib and rlg targets
# are completely unaffected.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PASTE EVERYTHING BELOW to the BOTTOM of your existing CMakeLists.txt.
# Do NOT change anything above this block.
# Files in src/ and src_jl/ are completely unaffected.
# ─────────────────────────────────────────────────────────────────────────────

# ── RLG  (src2/) ──────────────────────────────────────────────────────────────
add_library(rlg_index STATIC src2/rlg_index.cpp)
target_include_directories(rlg_index PUBLIC include)

# This line is the fix! It tells RLG where load_fbin and compute_l2sq are.
target_link_libraries(rlg_index PUBLIC graphann_lib) 

if(OpenMP_CXX_FOUND)
    target_link_libraries(rlg_index PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(build_rlg src2/build_rlg.cpp)
target_link_libraries(build_rlg PRIVATE rlg_index)

add_executable(search_rlg src2/search_rlg.cpp)
target_link_libraries(search_rlg PRIVATE rlg_index)

# ── IVRG (src_ivrg/) ──────────────────────────────────────────────────────────
add_library(ivrg_index STATIC src_ivrg/ivrg_index.cpp)
target_include_directories(ivrg_index PUBLIC include)

# Same fix here for IVRG!
target_link_libraries(ivrg_index PUBLIC graphann_lib)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ivrg_index PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(build_ivrg src_ivrg/build_ivrg.cpp)
target_link_libraries(build_ivrg PRIVATE ivrg_index)

add_executable(search_ivrg src_ivrg/search_ivrg.cpp)
target_link_libraries(search_ivrg PRIVATE ivrg_index)