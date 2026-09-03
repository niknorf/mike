/**
 * Regression test for: the ask_inputs pause lost its class through the AI SDK
 * adapter (reproduced on dd91a85).
 *
 * ask_inputs ends an assistant turn by throwing AssistantStreamAskInputsPause
 * from inside runTools. Under the AI SDK loop that throw happens inside the
 * tool `execute` → ToolExecutionBatcher rejects → the SDK emits a `tool-error`
 * part. aiSdk.ts used to wrap that in `new Error(message)`, so streaming.ts's
 * `err instanceof AssistantStreamAskInputsPause` could never be true and every
 * ask_inputs turn died on the generic error path; meanwhile the SDK's step
 * loop kept running and fired a second model request behind the dead stream.
 *
 * On pristine dd91a85 the caller received `Error("Waiting for user input.")`
 * (name "Error", not the thrown instance) and `model.doStreamCalls.length`
 * was 2 after the throw.
 */
import { describe, expect, it } from "vitest";

import { streamAiSdk } from "../../lib/llm/aiSdk";
import {
  AssistantStreamAskInputsPause,
  callStep,
  config,
  makeModel,
  textStep,
  tick,
  TOOLS,
} from "../../lib/llm/__tests__/mockLanguageModel";

const base = {
  model: "m",
  systemPrompt: "S",
  messages: [{ role: "user" as const, content: "go" }],
  tools: TOOLS,
  maxIterations: 5,
};

describe("AssistantStreamAskInputsPause identity through streamAiSdk", () => {
  it("the pause thrown inside runTools reaches the caller as the very same instance", async () => {
    const model = makeModel([
      callStep("c1", "read_document", { doc_id: "doc-0" }),
      textStep("must never be requested"),
    ]);
    const pause = new AssistantStreamAskInputsPause();
    let caught: unknown;
    try {
      await streamAiSdk(
        {
          ...base,
          runTools: async () => {
            throw pause;
          },
        },
        config(model),
      );
    } catch (e) {
      caught = e;
    }
    expect(caught).toBe(pause);
    expect(caught).toBeInstanceOf(AssistantStreamAskInputsPause);
    expect((caught as Error).name).toBe("AssistantStreamAskInputsPause");
    expect((caught as Error).message).toBe("Waiting for user input.");
  });

  it("no background model step runs after the pause", async () => {
    const model = makeModel([
      callStep("c1", "read_document", { doc_id: "doc-0" }),
      textStep("must never be requested"),
    ]);
    await streamAiSdk(
      {
        ...base,
        runTools: async () => {
          throw new AssistantStreamAskInputsPause();
        },
      },
      config(model),
    ).catch(() => undefined);
    await tick();
    await tick();
    expect(model.doStreamCalls.length).toBe(1);
  });
});
