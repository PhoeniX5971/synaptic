# Implementation Brief: Codex-Style Streaming File Modification Support

## Goal

Add support for live, Codex-style file modification previews in our agent framework/projects.

The system should allow an LLM coding agent to show files being modified dynamically in the frontend while the response is still streaming.

This should not rely on waiting for a full `write_file` tool call to finish. Instead, implement a host-side protocol where the model streams an **edit proposal**, the runtime incrementally parses that proposal into file-change snapshots/diffs, the UI renders those snapshots live, and only later the runtime applies the edit to disk.

This should be implemented either:

1. as a reusable feature in the core agent framework, or
2. as a project-local integration that can later be extracted into the framework.

---

# Background / Reasoning

Codex-style UIs do not appear to simply wait for a tool call to finish and then show the result. OpenAI’s Codex app-server docs describe a rich-client protocol with streamed agent events, approvals, and file-change items. The app-server is explicitly described as powering rich clients like the Codex VS Code extension, including “approvals, and streamed agent events.” ([OpenAI Developers][1])

For file changes, Codex exposes a `fileChange` item shaped like:

```ts
{
  id: string;
  changes: Array<{
    path: string;
    kind: string;
    diff: string;
  }>;
  status: "inProgress" | "completed" | "failed" | "declined";
}
```

The docs describe `fileChange` as proposed edits where each change includes `{ path, kind, diff }`. ([OpenAI Developers][1])

Codex also separates command-output streaming from file-change streaming. It has events like `item/commandExecution/outputDelta` for stdout/stderr and `item/fileChange/outputDelta` for patch/tool output. ([OpenAI Developers][1])

The important bit: file-change approval order starts with `item/started`, which emits a `fileChange` item with proposed changes and `status: "inProgress"`, then requests approval, then completes with final status. ([OpenAI Developers][1])

So the architecture is:

```text
model stream
   ↓
edit proposal stream
   ↓
host parses proposal into diffs
   ↓
frontend renders live file changes
   ↓
user/runtime approves
   ↓
executor applies patch
   ↓
final fileChange status emitted
```

Also, streamed tool calls are possible. OpenAI’s Responses API can emit `response.function_call_arguments.delta` events as function-call arguments are generated, followed by `response.function_call_arguments.done`. ([OpenAI Developers][2])

Structured outputs can also be streamed and partially parsed; OpenAI’s structured output docs show `content.delta` events with a running snapshot and parsed partial object, and they also show streaming parsed function-call arguments. ([OpenAI Developers][3])

Therefore, do **not** assume tool calls are always “one final block.” The problem is not that tool calls cannot stream. The problem is that a naive `write_file(path, content)` tool gives the UI a bad semantic stream. It streams raw arguments, not clean file-change events.

---

# Desired Architecture

## Core principle

Separate **proposal** from **execution**.

Bad:

```text
LLM calls write_file(path, content)
→ runtime waits for full tool call
→ file is written
→ frontend sees final result
```

Good:

```text
LLM streams edit proposal
→ runtime parses partial proposal
→ runtime emits fileChange/diff events
→ frontend previews live changes
→ runtime applies patch only after completion/approval
```

---

# Components To Implement

## 1. Edit Proposal Tool / Channel

Add a dedicated edit proposal mechanism.

This can be implemented as one of these:

### Option A: Function tool with JSON arguments

Recommended for structured, small-to-medium edits.

Tool name:

```ts
propose_file_edits;
```

Schema:

```ts
type ProposeFileEditsArgs = {
  edits: FileEditProposal[];
};

type FileEditProposal = {
  path: string;
  kind: "create" | "modify" | "delete" | "rename";
  old_path?: string;
  language?: string;
  summary?: string;
  diff?: string;
  operations?: EditOperation[];
};

type EditOperation =
  | {
      type: "replace_range";
      start_line: number;
      end_line: number;
      replacement: string;
    }
  | {
      type: "insert_after";
      line: number;
      content: string;
    }
  | {
      type: "delete_range";
      start_line: number;
      end_line: number;
    };
```

Notes:

- `diff` is preferred for multi-line code edits.
- `operations` are useful for precise editor integrations.
- Do **not** write files from this tool directly.
- This tool creates a proposal artifact only.

### Option B: Custom text tool / patch DSL

Recommended for larger code edits.

Tool name:

```ts
propose_patch;
```

Input format:

```diff
*** Begin Patch
*** Update File: src/example.py
@@
-old line
+new line
*** Add File: src/new_file.py
+print("hello")
*** Delete File: src/old_file.py
*** End Patch
```

This is probably closest to the Codex-style approach because patches are naturally streamable and diffable.

OpenAI’s function-calling docs mention custom tools that accept arbitrary string input instead of forcing everything into JSON, useful when JSON wrapping is unnecessary or when applying a custom grammar. ([OpenAI Developers][2])

For code edits, this is cleaner than giant JSON blobs.

---

## 2. Stream Collector

Implement a stream collector that listens to model stream events and tracks tool/function argument deltas.

For OpenAI Responses-style APIs, consume:

```text
response.output_item.added
response.function_call_arguments.delta
response.function_call_arguments.done
response.output_item.done
```

The docs show that function-call arguments stream as repeated `response.function_call_arguments.delta` chunks. ([OpenAI Developers][2])

Pseudo-code:

```ts
type ActiveToolCall = {
  id: string;
  name: string;
  argumentsBuffer: string;
};

const activeCalls = new Map<string, ActiveToolCall>();

for await (const event of modelStream) {
  switch (event.type) {
    case "response.output_item.added": {
      if (event.item.type === "function_call") {
        activeCalls.set(event.item.id, {
          id: event.item.id,
          name: event.item.name,
          argumentsBuffer: "",
        });

        emitAgentEvent({
          type: "toolCall.started",
          toolCallId: event.item.id,
          name: event.item.name,
        });
      }
      break;
    }

    case "response.function_call_arguments.delta": {
      const call = activeCalls.get(event.item_id);
      if (!call) break;

      call.argumentsBuffer += event.delta;

      emitAgentEvent({
        type: "toolCall.argumentsDelta",
        toolCallId: event.item_id,
        name: call.name,
        delta: event.delta,
        snapshot: call.argumentsBuffer,
      });

      handlePossibleEditProposalDelta(call);
      break;
    }

    case "response.function_call_arguments.done": {
      const call = activeCalls.get(event.item_id);
      if (!call) break;

      call.argumentsBuffer = event.arguments;

      emitAgentEvent({
        type: "toolCall.argumentsDone",
        toolCallId: event.item_id,
        name: call.name,
        arguments: event.arguments,
      });

      finalizeEditProposal(call);
      break;
    }
  }
}
```

---

## 3. Incremental Edit Parser

Implement an incremental parser for edit proposals.

This parser should accept a growing buffer and emit stable file-change snapshots.

### For patch DSL

Input:

```ts
class StreamingPatchParser {
  push(delta: string): PatchSnapshot;
  finalize(): PatchProposal;
}
```

Output:

```ts
type PatchSnapshot = {
  proposalId: string;
  status: "inProgress";
  changes: FileChangeSnapshot[];
  parseState: "partial" | "valid_so_far" | "invalid";
  warnings?: string[];
};

type FileChangeSnapshot = {
  path: string;
  kind: "create" | "modify" | "delete" | "rename";
  diff: string;
  complete: boolean;
};
```

The parser does not need to perfectly parse incomplete patches at every token. It only needs to emit the latest safe snapshot.

Example behavior:

```text
delta: "*** Update File: src/a.py"
→ emit fileChange started for src/a.py

delta: "@@\n-old\n+new"
→ emit updated diff snapshot

delta: "*** End Patch"
→ mark proposal parse-complete
```

### For JSON tool args

Use a tolerant partial JSON parser.

Important: Do not require full valid JSON before showing anything.

Possible strategies:

1. parse fully when valid;
2. if invalid, recover string fields manually;
3. use SDK parsed partial events when available;
4. use a simpler line-based DSL instead of JSON for better streaming.

For JSON, the minimal viable version can simply aggregate deltas and only emit snapshots when the buffer becomes valid JSON. That will still be better than waiting for tool execution, but not as “live” as patch parsing.

---

## 4. Internal Agent Event Protocol

Add framework-level events.

This is the main thing your frontend/IDE integration consumes.

```ts
type AgentEvent =
  | AgentMessageDelta
  | ToolCallStarted
  | ToolCallArgumentsDelta
  | FileChangeStarted
  | FileChangeUpdated
  | FileChangeApprovalRequested
  | FileChangeCompleted
  | CommandOutputDelta
  | ErrorEvent;

type FileChangeStarted = {
  type: "fileChange.started";
  id: string;
  turnId: string;
  status: "inProgress";
  changes: FileChangeSnapshot[];
};

type FileChangeUpdated = {
  type: "fileChange.updated";
  id: string;
  turnId: string;
  status: "inProgress";
  changes: FileChangeSnapshot[];
  aggregatedDiff: string;
};

type FileChangeApprovalRequested = {
  type: "fileChange.approvalRequested";
  id: string;
  turnId: string;
  changes: FileChangeSnapshot[];
  reason?: string;
};

type FileChangeCompleted = {
  type: "fileChange.completed";
  id: string;
  turnId: string;
  status: "completed" | "failed" | "declined";
  changes: FileChangeSnapshot[];
  error?: string;
};
```

This mirrors the useful part of Codex’s file-change concept without copying its entire protocol.

Codex’s app-server model has `fileChange` items with `{id, changes, status}`, where changes include path/kind/diff. ([OpenAI Developers][1])

---

## 5. Diff Renderer / Frontend Contract

Frontend should **not** wait for disk writes.

Frontend subscribes to agent events:

```ts
agent.on("fileChange.started", renderNewFileChangeCard);
agent.on("fileChange.updated", updateDiffView);
agent.on("fileChange.approvalRequested", showApprovalButtons);
agent.on("fileChange.completed", finalizeStatus);
```

UI behavior:

```text
fileChange.started
→ show file card with "editing..."

fileChange.updated
→ update diff view live

fileChange.approvalRequested
→ show approve/reject/apply buttons

fileChange.completed
→ mark as applied / failed / declined
```

The UI should treat these as **proposed changes**, not guaranteed disk state.

Use labels like:

```text
Proposed changes
Generating diff...
Ready to apply
Applied
Rejected
Failed
```

Do not show “written” until the apply step succeeds.

---

## 6. Apply Engine

Implement a separate executor:

```ts
interface PatchExecutor {
  validate(proposal: PatchProposal): ValidationResult;
  apply(proposal: PatchProposal): ApplyResult;
  rollback(applyId: string): RollbackResult;
}
```

Requirements:

- Validate paths are inside workspace root.
- Prevent path traversal like `../../`.
- Prevent editing denied files unless explicitly allowed.
- Compute pre-apply checksums.
- Apply patch atomically when possible.
- Store rollback info.
- Emit final status.

Pseudo-flow:

```ts
async function finalizeEditProposal(call: ActiveToolCall) {
  const proposal = parser.finalize(call.argumentsBuffer);

  emit({
    type: "fileChange.approvalRequested",
    id: proposal.id,
    turnId,
    changes: proposal.changes,
  });

  const decision = await approvalPolicy.decide(proposal);

  if (decision === "decline") {
    emit({
      type: "fileChange.completed",
      id: proposal.id,
      status: "declined",
      changes: proposal.changes,
    });
    return;
  }

  const validation = patchExecutor.validate(proposal);

  if (!validation.ok) {
    emit({
      type: "fileChange.completed",
      id: proposal.id,
      status: "failed",
      changes: proposal.changes,
      error: validation.error,
    });
    return;
  }

  const result = await patchExecutor.apply(proposal);

  emit({
    type: "fileChange.completed",
    id: proposal.id,
    status: result.ok ? "completed" : "failed",
    changes: proposal.changes,
    error: result.error,
  });
}
```

---

# Recommended Modes

## Mode 1: Preview-only mode

Use when frontend wants to show changes but not auto-write.

```text
model streams proposal
→ frontend renders diff
→ user approves manually
→ apply engine writes files
```

Best for IDE integrations.

## Mode 2: Auto-apply mode

Use for internal agent tasks where approval is unnecessary.

```text
model streams proposal
→ frontend renders live diff
→ runtime auto-applies after valid final patch
```

Still keep proposal and execution separate.

## Mode 3: No-preview fallback

Use for providers that do not expose tool-call argument deltas.

```text
model emits full tool call
→ parse final proposal
→ render final diff
→ apply
```

This preserves correctness even without live streaming.

---

# Provider Support Strategy

The framework should expose a normalized stream interface.

```ts
type NormalizedModelEvent =
  | {
      type: "text.delta";
      delta: string;
    }
  | {
      type: "tool_call.started";
      id: string;
      name: string;
    }
  | {
      type: "tool_call.arguments.delta";
      id: string;
      name: string;
      delta: string;
      snapshot: string;
    }
  | {
      type: "tool_call.arguments.done";
      id: string;
      name: string;
      arguments: string;
    };
```

Then each provider adapter maps its native events to this.

OpenAI Responses API maps naturally because it has streamed function-call argument deltas. ([OpenAI Developers][2])

Other providers may need fallbacks:

```ts
capabilities = {
  textStreaming: true,
  toolArgumentStreaming: false,
  structuredOutputStreaming: false,
  customToolTextStreaming: false,
};
```

If `toolArgumentStreaming` is false, the framework should still work, just without live diff updates.

---

# Agent Prompt / System Instructions

Add this to coding agents that support the feature:

```md
## File editing protocol

When modifying files, do not directly write full files unless explicitly required.

Prefer proposing edits through the available edit proposal tool.

Use the `propose_patch` tool for multi-file or code edits. Emit a patch in the expected patch format.

The patch is a proposal. The runtime will preview, validate, and apply it.

Rules:

- Keep patches minimal and focused.
- Do not include unrelated formatting changes.
- Use separate file sections for each changed file.
- Never modify files outside the workspace.
- For new files, use Add File.
- For existing files, use Update File.
- For deleted files, use Delete File.
- Do not claim a file was changed until the runtime confirms the patch was applied.
```

For JSON edit mode:

```md
## File editing protocol

When editing files, use `propose_file_edits`.

Do not use `write_file` for normal edits.

Each edit must include:

- path
- kind
- summary
- either a unified diff or structured operations

The runtime will parse, preview, validate, approve, and apply the edits.
```

---

# Tool Definitions

## Patch tool

```ts
const proposePatchTool = {
  type: "function",
  name: "propose_patch",
  description:
    "Propose file changes as a patch. This does not write files. The runtime previews and applies the patch after validation.",
  parameters: {
    type: "object",
    additionalProperties: false,
    required: ["patch", "summary"],
    properties: {
      summary: {
        type: "string",
        description: "Brief summary of the intended changes.",
      },
      patch: {
        type: "string",
        description: "Patch text in the supported patch format.",
      },
    },
  },
};
```

This works, but note: JSON string escaping can make streamed parsing annoying.

Better for custom tool support:

```ts
const proposePatchCustomTool = {
  type: "custom",
  name: "propose_patch",
  description:
    "Propose file changes as a patch. Input must be patch text only. The runtime previews and applies it after validation.",
};
```

## JSON edit tool

```ts
const proposeFileEditsTool = {
  type: "function",
  name: "propose_file_edits",
  description:
    "Propose structured file edits. This does not write files. The runtime previews, validates, and applies the edits.",
  parameters: {
    type: "object",
    additionalProperties: false,
    required: ["edits"],
    properties: {
      edits: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          required: ["path", "kind", "summary"],
          properties: {
            path: { type: "string" },
            old_path: { type: "string" },
            kind: {
              type: "string",
              enum: ["create", "modify", "delete", "rename"],
            },
            language: { type: "string" },
            summary: { type: "string" },
            diff: { type: "string" },
          },
        },
      },
    },
  },
};
```

---

# Minimal MVP

Implement this first:

```text
1. Add propose_patch tool.
2. Collect streamed tool argument deltas.
3. Buffer patch text.
4. On each delta, try parsing patch sections.
5. Emit fileChange.updated events with best-effort diffs.
6. On done, validate final patch.
7. Apply patch.
8. Emit fileChange.completed.
9. Frontend renders diff cards from fileChange events.
```

MVP does not need perfect incremental parsing.

It can start with:

```text
- show raw patch text live
- generate diff preview once full patch is valid
- later improve to per-file live diff parsing
```

But architect the event system correctly from day one.

---

# Important Anti-Patterns

Avoid this:

```ts
write_file(path, fullContent);
```

as the main editing path.

It is simple but bad for live UX because:

- the UI receives a growing full file blob, not semantic edits;
- diffing partial file content is noisy;
- approval is awkward;
- large files waste tokens;
- failed partial generation is hard to reason about.

Avoid this too:

```text
model writes directly to disk during streaming
```

That makes rollback, approval, and security painful.

Preferred:

```text
proposal first, apply later
```

---

# Security Requirements

The apply engine must enforce:

```text
- workspace root boundary
- no path traversal
- no symlink escape unless explicitly allowed
- max patch size
- max files changed
- denied file patterns
- optional approval for destructive changes
- rollback metadata
```

Suggested denied patterns:

```text
.env
.env.*
**/secrets/**
**/.ssh/**
**/.gnupg/**
**/id_rsa
**/id_ed25519
```

For your framework, allow projects to configure this.

---

# Testing Requirements

Add tests for:

```text
1. streamed patch creates fileChange.started
2. streamed patch updates fileChange.updated
3. final valid patch applies correctly
4. invalid patch emits failed status
5. path traversal is rejected
6. delete file requires approval when policy says so
7. multiple files are grouped into one proposal
8. provider without tool deltas falls back to final-only preview
9. partial malformed patch does not crash parser
10. rollback works after failed partial apply
```

---

# Recommended Final Design

Use this as the long-term framework abstraction:

```text
Model Adapter
  ↓ normalized stream events
Streaming Tool Call Collector
  ↓ edit proposal deltas
Edit Proposal Parser
  ↓ fileChange events
Agent Event Bus
  ↓
Frontend Diff Renderer

After proposal completion:
  ↓
Validation / Approval
  ↓
Patch Executor
  ↓
Final fileChange.completed event
```

In code-ish form:

```ts
class AgentRuntime {
  modelAdapter: ModelAdapter;
  eventBus: AgentEventBus;
  editProposalManager: EditProposalManager;
  patchExecutor: PatchExecutor;

  async run(input: AgentInput) {
    const stream = this.modelAdapter.stream(input);

    for await (const event of stream) {
      const normalized = normalize(event);

      this.eventBus.emit(normalized);

      if (isEditProposalEvent(normalized)) {
        await this.editProposalManager.handle(normalized);
      }
    }
  }
}
```

---

# Final Recommendation

Build this as a **framework feature**, not a one-off project hack.

Name it something like:

```text
Streaming Edit Proposals
```

or:

```text
Live Patch Preview
```

Expose it as a capability:

```ts
agentCapabilities = {
  liveEditPreview: true,
  patchProposalTool: true,
  requiresApprovalForFileChanges: true,
};
```

And keep this design rule sacred:

```text
The model proposes edits.
The runtime owns files.
The frontend renders proposals.
The executor applies changes.
```

That separation is the whole magic. Codex-looking live edits are not “the model writing files live.” They are a clean event pipeline around streamed edit intent.

[1]: https://developers.openai.com/codex/app-server "App Server – Codex | OpenAI Developers"
[2]: https://developers.openai.com/api/docs/guides/function-calling "Function calling | OpenAI API"
[3]: https://developers.openai.com/api/docs/guides/structured-outputs "Structured model outputs | OpenAI API"
