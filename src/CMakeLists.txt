include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(IR)

add_executable(practice_mlir)
target_sources(practice_mlir PRIVATE main.cpp)
target_link_libraries(practice_mlir PRIVATE MLIRMyDialect)
