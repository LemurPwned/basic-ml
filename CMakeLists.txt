cmake_minimum_required(VERSION 3.4...3.18)
project(basic_ml LANGUAGES CXX)

find_package(pybind11 REQUIRED)
pybind11_add_module(basic_ml python/module.cpp)

# VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(basic_ml
                           PRIVATE VERSION_INFO=${VERSION_INFO})