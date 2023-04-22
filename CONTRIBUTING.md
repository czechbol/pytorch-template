# Contributing to the project
Feel free to contribute any kind of function or enhancement.

As long as the enhancements are sensible and pass [pre-commit](https://pre-commit.com/#quick-start), I have no issues with merging.
## Install development dependencies
In addition to the requirements setup in [README.md](README.md#requirements),
you also need to install the development requirements and set up pre-commit:
```bash
$ python3.10 -m pip install -r requirements-dev.txt
$ pre-commit install
# Optionally run the pre-commit checks for all files
$ pre-commit run --all
```
Pre-commit will automatically check your commits and will prevent you from
commiting changes that violate the checks

## TODOs

- [ ] Fix all typing hints
- [ ] Multiple optimizers
- [ ] Support more tensorboard functions
- [x] Using fixed random seed
- [x] Support pytorch native tensorboard
- [x] `tensorboardX` logger support
- [x] Configurable logging layout, checkpoint naming
- [x] Iteration-based training (instead of epoch-based)
- [x] Adding command line option for fine-tuning
