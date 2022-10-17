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
