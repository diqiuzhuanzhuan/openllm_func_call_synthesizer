# openllm_func_call_synthesizer

![PyPI version](https://img.shields.io/pypi/v/openllm_func_call_synthesizer.svg)
[![Documentation Status](https://readthedocs.org/projects/openllm_func_call_synthesizer/badge/?version=latest)](https://openllm_func_call_synthesizer.readthedocs.io/en/latest/?version=latest)

A tool for generating synthetic function call datasets for Large Language Models (LLMs).

* PyPI package: https://pypi.org/project/openllm_func_call_synthesizer/
* Free software: MIT License
* Documentation: https://openllm_func_call_synthesizer.readthedocs.io.

## Features

- Generate synthetic function call datasets for LLM training and evaluation
- Flexible configuration via YAML and Hydra
- CLI interface powered by Typer & Rich
- Utility functions for dataset manipulation
- Extensible and easy to integrate into your own pipeline

## Installation

You can install the stable release from PyPI:

```sh
uv add openllm_func_call_synthesizer
# or
pip install openllm_func_call_synthesizer
```

For more installation options, see [docs/installation.md](docs/installation.md).

## Usage

Basic usage:

```python
import openllm_func_call_synthesizer
# 具体API和CLI用法请参考文档
```

命令行用法：

```sh
openllm_func_call_synthesizer --help
```

详细用法见 [docs/usage.md](docs/usage.md)。

## Quickstart

```sh
openllm_func_call_synthesizer
# 或
python -m openllm_func_call_synthesizer
```

## Testing

Run tests with pytest:

```sh
pytest
```

## Contributing

Welcome to contribute！Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Links

- [Documentation](https://openllm_func_call_synthesizer.readthedocs.io)
- [PyPI](https://pypi.org/project/openllm_func_call_synthesizer/)
- [GitHub](https://github.com/diqiuzhuanzhuan/openllm_func_call_synthesizer)

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
