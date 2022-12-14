# Basic ML library

![](assets/main_image.png)

<!-- [![Gitter][gitter-badge]][gitter-link]

| Badge      | status                                                       |
| ---------- | ------------------------------------------------------------ |
| pip builds | [![Pip Actions Status][actions-pip-badge]][actions-pip-link] |

[actions-pip-link]: https://github.com/pybind/basic_ml/actions?query=workflow%3A%22Pip
[actions-pip-badge]: https://github.com/pybind/basic_ml/workflows/Pip/badge.svg -->

A library of basic machine learning algorithms -- written in C++, with Python bindings.

## Installation

To set up install install Pybind11:

```
python3 -m pip install pybind11[global]
```

and then run the following command:

```bash
python3 -m pip install basic_ml
```

## Contents

- [RPCA](python/rpca.hpp) Robust PCA
- [Tracker](python/tracker.hpp) Basic IOU tracker -- read [docs](docs/tracker.md)

## Usage

```python
import basic_ml
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on dev setup, and the process for submitting pull requests to us.

## Authors

- _Initial work_ - [LemurPwned](www.github.com/LemurPwned)
