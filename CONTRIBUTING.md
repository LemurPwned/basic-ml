# Development setup

Currently, the project requires following dependencies:

- CMake
- Eigen3
- Pybind11
- gtest (optional), for testing
- emscripten (optional), for javascript bindings

## Installation

### Mac OS X

```bash
brew install cmake eigen pybind11
```

### Ubuntu

```bash
sudo apt-get install cmake libeigen3-dev
python3 -m pip install pybind11
```

### Windows

Best to use chocolatey to install dependencies:

```bash
choco install cmake eigen
python3 -m pip install pybind11
```

## Development

To build a project (including the Python bindings), you can run:

```
python3 -m pip install -e .
```

This will install the Python package in editable mode, so that any changes to the source code will be reflected in the installed package. If the code was made in C++, it requires a reinstall, if it was made in Python, it will be reflected immediately.

Any C++ code is in [`python`](./python) folder.
Bindings code for:

- Python is in [`python/module.cpp`](./python/module.cpp) file
- Javascript is in [`javascript`](./javascript/module.cpp)

Python package code can be placed in [`basic_ml`](./basic_ml/src/) folder.

### Building emscripten bindings

To build emscripten bindings, you need to have emscripten installed.

```bash
python3 -m pip install emscripten
```

Then, you can use the provided `Makefile` to build the bindings

```bash
make jsbind
```
