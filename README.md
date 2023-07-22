# Basic ML library

![](assets/main_image.png)

<!-- [![Gitter][gitter-badge]][gitter-link]

| Badge      | status                                                       |
| ---------- | ------------------------------------------------------------ |
| pip builds | [![Pip Actions Status][actions-pip-badge]][actions-pip-link] |

[actions-pip-link]: https://github.com/pybind/basic_ml/actions?query=workflow%3A%22Pip
[actions-pip-badge]: https://github.com/pybind/basic_ml/workflows/Pip/badge.svg -->

A library of basic machine learning algorithms -- written in C++, with Python and JS bindings.
The purpose of this library is to provide a simple, easy to use, and fast implementation of basic machine learning algorithms that can be used both in the web and in Python code.

## Installation

To set up install install Pybind11:

```
python3 -m pip install pybind11[global]
```

and then run the following command:

```bash
python3 -m pip install basic_ml
```

### Macs with M1

```bash
arch -arm64 python3 -m pip install basic_ml
```

## Contents

- [RPCA](python/rpca.hpp) Robust PCA
- [Tracker](python/tracker.hpp) Basic IOU tracker -- read [docs](docs/tracker.md)

## Usage

```python
import basic_ml
```

## Building with emscripten

After installing the emscripten package, you should set the envs:

```bash
cd ~/loc-to-sdk/emsdk && source emsdk_env.sh
```

Then, you can use `make library` Makefile command.

## Demo usage

Launching demo requires building the webasm libraries first.
To launch demo best is to install live server with:

```bash
npm install -g live-server
```

and then launch it:

```bash
live-server demo/
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on dev setup, and the process for submitting pull requests to us.

## Authors

- _Initial work_ - [LemurPwned](www.github.com/LemurPwned)
