cmake_minimum_required(VERSION 3.12)
project(Thesis)

add_executable(minigrad "minigrad.c")
target_link_libraries(minigrad "-framework Accelerate")
add_executable(blas "blas.c")
target_link_libraries(blas "-framework Accelerate")
add_executable(dbml "dbml.c")
# add_test(NAME test COMMAND dbml)
set_source_files_properties("kernels.metal" PROPERTIES LANGUAGE METAL)
add_executable(metal "metal.m" "kernels.metal" "metal_common.h")
# FIXME: xcode does not like external symbols. How do I tell Cmake to generate
# an Xcode project were it tels the metal compiler to include source information?
add_custom_command(TARGET metal POST_BUILD
	COMMAND xcrun ARGS -sdk macosx metal -frecord-sources=flat "${CMAKE_SOURCE_DIR}/kernels.metal"
	BYPRODUCTS default.metallibsym)
set_target_properties(metal PROPERTIES XCODE_SCHEME_ENVIRONMENT "MTL_CAPTURE_ENABLED=1")
target_link_libraries(metal "-framework Metal" "-framework MetalPerformanceShaders")

add_executable(metal_test "metal_test.m")
target_link_libraries(metal_test "-framework Metal")

add_custom_target(digest SOURCES digest.md)

find_package(Python REQUIRED)
add_test(NAME Pytorch COMMAND ${Python_EXECUTABLE} pytorch.py)

# return()

# https://stackoverflow.com/a/34704693
enable_testing()
find_package(XCTest REQUIRED)
add_library(FrameworkExample SHARED dbml.c)
target_include_directories(FrameworkExample PUBLIC "${XCTest_INCLUDE_DIRS}")
target_link_libraries(FrameworkExample "${XCTest_LIBRARIES}")
set_target_properties(FrameworkExample PROPERTIES FRAMEWORK TRUE)
# To avoid codesign shenanigans
# https://stackoverflow.com/a/54296008
# https://discourse.cmake.org/t/code-signing-macos-application/7275
# https://gitlab.kitware.com/cmake/cmake/-/issues/19879#note_887513
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "")
xctest_add_bundle(FrameworkExampleTests FrameworkExample test.m)