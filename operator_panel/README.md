# Operator Panel core

This package is repository-agnostic. It owns only localhost HTTP serving,
static assets, exclusive subprocess supervision, guarded terminal input, and
create-only configuration storage. It has no Galaxea, ROS, camera, model, or
tracked-config imports.

A consuming repository implements `PanelAdapter` to provide its catalog,
strict configuration validators, and argv-only workflow launches. Child
processes call `operator_panel.protocol.announce_input()` immediately before an
interactive prompt; the panel will accept one input and lock the buttons until
the next announcement.

Consumers implement the five `PanelAdapter` methods and pass the adapter to
`serve_operator_panel(adapter, bind="127.0.0.1", port=...)`. Workflow forms,
select options, cameras, and configuration kinds come from the adapter's JSON
catalog.
