cmake_minimum_required(VERSION 3.4...3.18)
project(basic_ml LANGUAGES CXX)
add_compile_options(-O3 -Wall -Wextra -Wpedantic -Werror)
# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 20)

# EXTRA INCLUDE PKGs
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(pybind11 REQUIRED)
include_directories(${PYBIND11_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

pybind11_add_module(basic_ml python/module.cpp)

# TESTING SUITE
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)
# source: https://cliutils.gitlab.io/modern-cmake/chapters/testing/googletest.html
macro(package_add_test TESTNAME)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

package_add_test(test_rpca tests/test_rpca.cpp)
target_link_libraries(test_rpca Eigen3::Eigen)
# VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(basic_ml
                           PRIVATE VERSION_INFO=${VERSION_INFO})