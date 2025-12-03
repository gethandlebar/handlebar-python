# Handlebar Core

Core primitives for running [Handlebar] governance
for AI agents built in Python.
This codebase should typically be used in conjunction
with a framework-specific Handlebar SDK,
such as [`google-adk`][handlebar-google].

_Note: This package in early development and the interface is subject to change._

## Features

Short-term roadmap:

- [X] Rule engine for allow/block tools, based on:
  - [X] user category
  - [X] rule category
- [X] Tool ordering + execution time checks
- [X] custom checks for rules (numeric tracking; boolean evaluation)
- [X] Audit telemetry + consumers
- [ ] Agent lockdown + human-in-the-loop actions

### Roadmap

Handlebar is in early development. We have a lot of functionality planned,
but need your feedback on what you need to help you build better agents.

- Please feel free to [open an issue](https://github.com/gethandlebar/handlebar-python/issues/new) if you have any feedback or suggestions
- or [join our Discord][discord_invite] to talk to us directly

## Getting started

The core package should be used alongside a framework-specific Handlebar SDK,
such as [google-adk](https://github.com/gethandlebar/handlebar-python/blob/main/packages/google-adk/).
Refer to that package's README for more information.

`handlebar-core` exposes core primitives for building rules and a governance runtime.
In particular, it defines "rules" to enforcing tool-use behaviour based on information like
a tool's category, the user on who's behalf the agent is acting, and tool use parameters.

**N.b. Our developer docs are incoming.**

## Contributing

We welcome contributions from the community: bug reports, feedback, feature requests

## About Handlebar

Find out more at https://gethandlebar.com

[handlebar]: https://gethandlebar.com
[discord_invite]: https://discord.gg/Q6xwvccg
