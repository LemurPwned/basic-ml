LIBNAME="libml"
SRC="./javascript"
CPP_SRC="./python"
INCLUDE_PATH="/usr/local/include"
LVL="-O3"

all: cpp jsbind
library: cpp
    emcc -lembind $(LVL) -o $(SRC)/$(LIBNAME).js -Wl,--whole-archive $(SRC)/$(LIBNAME).a -Wl,--no-whole-archive
cpp:
    emcc -c -std=c++17 $(LVL) -I$(INCLUDE_PATH) $(SRC)/module.cpp -o $(SRC)/$(LIBNAME).o
    emar rcs $(SRC)/$(LIBNAME).a $(SRC)/$(LIBNAME).o
jsbind: cpp
    emcc --bind $(LIB_PATH) module.bc -o module.js

.PHONY: clean
clean:
    rm -f $(SRC)/$(LIBNAME).o $(SRC)/$(LIBNAME).a $(SRC)/$(LIBNAME).js

.PHONY: cmake
cmake:
    rm -rf ./build
    cmake -S . -B build
    cmake --build build
    cd build && ctest --rerun-failed --output-on-failure
.PHONY: docker
docker:
    docker build -t ml .

.PHONY: install
install:
    rm -rf build/ *.so
    arch -arm64 python3 -m pip install --no-cache-dir --ignore-installed --force-reinstall -e .
    python -c "from basic_ml.tracker import ByteTracker"
    python -c "from basic_ml.vis import annotate_frame"
