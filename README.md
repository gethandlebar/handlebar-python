# Handlebar Python SDK

Python SDKs for implementing [Handlebar], the agentic governance runtime.\
Supports popular agent frameworks, such as google-adk.

## About

## Project structure

This repository is a monorepo containing installable packages
for different JS/TS agent building frameworks.

- [`handlebar-google-adk`](./packages/google-adk) - Handlebar for Google ADK
- [`handlebar-core`](./packages/core) - contains framework-agnostic logic for building rules.

For each package,
check out its specific README.

Frameworks coming soon:
- langchain (+ for python, not in this repository)
- crew
- openai
- ...and your favourite (open an issue requesting it and we'll add it to the roadmap)

## How-to

1. Sign-up for a [Handlebar account][handlebar-login]
1. Install the appropriate Handlebar framework for your agents and configure it (refer to the `./packages` here for more details)
1. Create an API key for your org and configure your project with `HANDLEBAR_API_KEY` environment variable/secret
1. ... and that's it! Handlebar is working to monitor and enforce rules on your agents

**N.b.: Developer docs are coming soon.**

## Roadmap

Immediate updates:
- [ ] Other agent frameworks + python frameworks
- [ ] Agent escalations (e.g. human-in-the-loop)

Handlebar is in early development. We have a lot of functionality planned,
but need your feedback on what you need to help you build better agents.

- Please feel free to [open an issue](https://github.com/gethandlebar/handlebar-python/issues/new) if you have any feedback or suggestions
- or [join our Discord][discord_invite] to talk to us directly

## Contributing

We welcome contributions from the community: bug reports, feedback, feature requests

## Developing

**Requirements**
- `mise` (mise-en-place)

**Initial setup**
- `mise i` to install uv dependencies
- `uv sync --all-packages` to sync the local packages

### Code Commands

[`mise.toml`](./mise.toml) provides shortcuts for code maintenance tasks:

- `mise r format` to run ruff formatter
- `mise r lint` to run ruff linter
- `mise r types` to run `ty` type checker on the packages

## License

These SDKs defined under [`packages/`](./packages/)
are currently licensed under the Apache License 2.0.
You’re free to use and distribute it in accordance with that license.

[handlebar]: https://www.gethandlebar.com
[handlebar-login]: https://app.gethandlebar.com/login
[discord_invite]: https://discord.gg/Q6xwvccg
