<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# ROCm CLI Assistant Skill

Use this skill when answering ROCm CLI local assistant questions.

## Status And Running Questions

- Inspect before answering. For "is X running", "what is running", status, or port questions, call a read-only tool first.
- Use `services list --all` for vLLM, Lemonade, qwen, and general local model servers.
- Use `comfyui status` or `port_status` for ComfyUI and port 8188.
- Interpret `running_state=running` as running, `running_state=starting` as starting, `running_state=not_running` as not running, and no matching row as unknown or not managed by ROCm CLI.
- Treat `localhost` and `127.0.0.1` as the same loopback endpoint.

## Setup And Install

- `active_runtime_status=ready` means ROCm CLI has an active registered runtime, which may be a managed ROCm/TheRock install or an adopted read-only system SDK. Use `rocm_command` args `["runtimes","list"]` to distinguish them. `legacy_rocm_status=not_detected` only means no global system ROCm install was found.
- If the user asks to install ROCm/TheRock, require an explicit install folder or use the guided folder picker. Preserve the exact path with `--prefix`.
- After a non-TUI SDK install succeeds, tell the user it installed successfully and to run `rocm help`.
- On Linux or WSL, if `examine` reports a candidate system ROCm install and `rocm_command` args `["runtimes","list"]` show no suitable registered runtime, offer `rocm_command` args `["runtimes","adopt-system"]`. Add `--root PATH` when the detected SDK is not the intended one, and add `--activate` only when it should become the default. The command validates the SDK before writing the registry. Because successful adoption changes local state, the structured tool call requires approval before the command runs; it never writes into the SDK tree. Do not offer `adopt-system` on native Windows.
- Adopted system runtimes remain owned by the OS package manager. `rocm update` reports them as not applicable; after an OS-side SDK update, refresh the registry with `adopt-system --replace`. `rocm runtimes uninstall` only unregisters the record. They have no managed Python and cannot back a vLLM engine install; vLLM needs a managed `rocm install sdk` runtime, while Lemonade manages its own environment.

## Engines And Assistant

- vLLM and Lemonade are the supported serving engines.
- The built-in assistant is fixed to qwen (`Qwen3-4B-Instruct-2507-GGUF`) served by Lemonade with `gpu_required`. Do not switch it to vLLM.
- Installing an engine and running a model server are different states. Inspect them separately with `rocm_command` args `["engines","list"]` and `["services","list","--all"]`.
- For general model serving, preserve an explicit engine choice. Otherwise omit `--engine` and let ROCm CLI select the configured default, then a compatible host-GPU preference, then the model recipe preference, and finally the platform default. It does not retry another engine after a failure.
- `rocm serve` accepts `--gpu auto|<index>`. `auto` prefers a GPU that looks idle from `amd-smi` VRAM telemetry and rocm-cli service records, then the GPU with the most free memory; an index pins one GPU. Serving one model across multiple GPUs is not supported. Always use `gpu_required`; a busy, unavailable, or invalid GPU must fail without CPU fallback.
- On native Windows, vLLM serving and installation are unsupported; tell the user to use WSL/Linux for that ROCm GPU engine and do not suggest CPU fallback.

## ComfyUI

- After ComfyUI install completes, say it is installed and offer to start it. Do not say ComfyUI finished as if the running app completed.
- After ComfyUI starts, include the URL and the models folder when the tool output provides them.
