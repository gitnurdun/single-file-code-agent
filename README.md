# Single-File Code Agent

A small interactive coding agent that asks a local llama.cpp server for a runnable, single-file deliverable. It parses the model response into file, shell, read, list, ask, and finish actions. The `sfc` launcher starts the agent in interactive mode.

## Requirements

- Python 3.9 or newer
- `openai` and `langchain-core` Python packages
- A running OpenAI-compatible local chat-completions server; the default endpoint is `http://127.0.0.1:8081/v1`
- Bash for the launcher; Conda is optional

Install the Python packages in your chosen environment, start the local model server, then run from this directory:

```bash
./sfc --workspace ./workspace
```

You can also provide a task directly:

```bash
./sfc --workspace ./workspace "Write a Python script that summarizes a CSV file"
```

The launcher tries to activate the Conda environment named `pytorch_env`. Override that name with `SFC_ENV`, or set `SFC_PYTHON` to select the Python executable. The agent defaults to model name `local` and the endpoint above; use `--base-url` and `--model` to change them.

## Agent options

The Python entry point supports `--workspace`, `--output`, repeatable `--reference` files, `--max-steps`, `--test`, `--allow-ask`, and `--no-verify`. By default it verifies a finished Python deliverable by running it. Interactive commands include `/show`, `/run`, `/test`, `/reset`, and `/exit`.

## Safety

The agent confines its direct file read/write actions to the selected workspace. Its shell action runs commands with the host shell and is not sandboxed; the model can request arbitrary commands. Final verification also runs the generated deliverable. Use a disposable workspace, review commands and code before execution, and do not point it at sensitive files.

The `examples/` directory preserves two adjacent matrix-inverse samples. They are separate from the agent and are not verified as part of this publication.

No license file was supplied, so this repository does not add a license.
