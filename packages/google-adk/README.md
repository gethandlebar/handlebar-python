# Handlebar for google-adk

Add [Handlebar] runtime checks, controls and governance into the Google ADK.
This package provides a `HandlebarRunner` wrapper around the default `Runner` class, to provide agent governance in a single-line code change.
The Handlebar runner collects audit logs of your agents' runs,
tool and model use,
and enforces rules around agent behaviour you've configured
on the [Handlebar platform][handlebar_platform].

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

Install the package in your codebase

```bash
pip install handlebar_google_adk
```

The package provides a runner, `HandlebarRunner`,
which is a drop-in replacement for the default google-adk `Runner` class

```diff
- from google.adk.runners import Runner;
+ from handlebar_google_adk import HandlebarRunner;

- runner = Runner(
+ runner = HandlebarRunner(
    agent=agent,
    app_name=handlebar_app_name,
    session_service=session_service,
+   handlebar_api_key=handlebar_api_key,
)
```

On the [Handlebar platform][handlebar_platform]
create an api key for your org
and ensure you've set the environment variable in your codebase as
`HANDLEBAR_API_KEY=<your api key>`

By default, the Handlebar runner enforces no governance rules,
but does collect agent telemetry which can be exported to analyse your agent's behaviour.
You can configure rules on tool use and behaviour
for Handlebar to enforce at runtime on the [platform][handlebar_platform]

### Audit logs

The Handlebar runner collects audit logs for key events
in the lifetime of an agent run,
such as tool usage
and agent rule evaluations
(see below for more on Handlebar rules).
With your Handlebar API key configured,
the [platform][handlebar_platform]
will automatically ingest the logs created by the runner,
which you can view on the dashboard.

## Contributing

We welcome contributions from the community: bug reports, feedback, feature requests.
Please refer to [CONTRIBUTING.md][root_contributing]
for ways you can help,
and guidelines.

## About Handlebar

Find out more at https://gethandlebar.com

[handlebar]: https://gethandlebar.com
[handlebar_platform]: https://app.gethandlebar.com
[discord_invite]: https://discord.gg/Q6xwvccg
[lens_repo]: https://github.com/gethandlebar/lens
