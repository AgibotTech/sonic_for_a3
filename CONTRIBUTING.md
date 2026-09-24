# Contributing to SONIC for A3

We welcome contributions from the community! Here's how to get started.

## Reporting Issues

- Search [existing issues](https://github.com/AgibotTech/sonic_for_a3/issues) first
- [Open an issue](https://github.com/AgibotTech/sonic_for_a3/issues/new) with a clear description, error messages, and steps to reproduce
- For security vulnerabilities, follow [the security policy](SECURITY.md) instead
- Include your Python version, OS, GPU, and Isaac Lab version

## Pull Requests

1. Fork [AgibotTech/sonic_for_a3](https://github.com/AgibotTech/sonic_for_a3)
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes
4. Run the pre-flight check: `python check_environment.py`
5. Commit and push to your fork
6. Open a [pull request](https://github.com/AgibotTech/sonic_for_a3/pulls) against `main`

### Guidelines

- Keep PRs focused on a single change
- Follow existing code style (no linter is enforced, but be consistent)
- Update documentation if your change affects user-facing behavior
- Add yourself to the PR description if you'd like credit

## Development Setup

See the [Installation Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html)
for setting up the training environment.

## Questions

For A3-specific questions, use [this repository's Issues](https://github.com/AgibotTech/sonic_for_a3/issues). This fork is
maintained independently; the upstream GEAR team does not maintain this A3
adaptation.

If an issue is confirmed to reproduce in unmodified upstream SONIC, report it
through the [upstream issue tracker](https://github.com/NVlabs/GR00T-WholeBodyControl/issues)
with an upstream reproduction. Keep A3 adaptation issues and feature requests here.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache 2.0 License](LICENSE).
