import test from "node:test";
import assert from "node:assert/strict";
import { Worker } from "node:worker_threads";

function browserWorker() {
  const source = `
    import { parentPort } from 'node:worker_threads';
    globalThis.self = { postMessage: (data) => parentPort.postMessage(data) };
    await import(${JSON.stringify(new URL("../web/engine/worker.js", import.meta.url).href)});
    parentPort.on('message', (data) => self.onmessage({data}));
    parentPort.postMessage({type:'ready'});
  `;
  return new Worker(
    new URL(`data:text/javascript,${encodeURIComponent(source)}`),
    { type: "module" },
  );
}

function until(worker, predicate) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error("Worker response timeout"));
    }, 10000);
    const onMessage = (message) => {
      if (predicate(message)) {
        cleanup();
        resolve(message);
      }
    };
    const onError = (error) => {
      cleanup();
      reject(error);
    };
    function cleanup() {
      clearTimeout(timer);
      worker.off("message", onMessage);
      worker.off("error", onError);
    }
    worker.on("message", onMessage);
    worker.on("error", onError);
  });
}

test("browser worker posts progress and complete auditable results", async () => {
  const worker = browserWorker();
  try {
    await until(worker, (message) => message.type === "ready");
    const result = until(worker, (message) => message.type === "result");
    worker.postMessage({
      id: "smoke",
      type: "compare",
      config: { steps: 3 },
      options: { seeds: 2 },
    });
    const message = await result;
    assert.equal(message.id, "smoke");
    assert.equal(message.result.rows.length, 6);
    assert.ok(
      message.result.rows.every(({ ledgerError }) => ledgerError < 1e-7),
    );
  } finally {
    await worker.terminate();
  }
});

test("browser worker accepts cancellation within the first long session", async () => {
  const worker = browserWorker();
  try {
    await until(worker, (message) => message.type === "ready");
    const progress = until(worker, (message) => message.type === "progress");
    const cancelled = until(worker, (message) => message.type === "cancelled");
    worker.postMessage({
      id: "cancel",
      type: "compare",
      config: { steps: 120, policy: "R2B" },
      options: { seeds: 200, policies: ["R2B"] },
    });
    const update = await progress;
    assert.equal(update.completed, 0);
    worker.postMessage({ type: "cancel" });
    const final = await cancelled;
    assert.equal(final.id, "cancel");
    assert.equal(final.completed, 0);
  } finally {
    await worker.terminate();
  }
});

test("browser worker rejects unsupported batch sizes visibly", async () => {
  const worker = browserWorker();
  try {
    await until(worker, (message) => message.type === "ready");
    const error = until(worker, (message) => message.type === "error");
    worker.postMessage({
      id: "invalid",
      type: "compare",
      config: {},
      options: { seeds: 201 },
    });
    assert.match((await error).message, /200/);
  } finally {
    await worker.terminate();
  }
});

test("browser worker preserves a failed path and aborts rather than excluding its seed", async () => {
  const worker = browserWorker();
  try {
    await until(worker, (message) => message.type === "ready");
    const error = until(worker, (message) => message.type === "error");
    worker.postMessage({
      id: "failed-path",
      type: "compare",
      config: {
        policy: "B3",
        steps: 120,
        initialPrice: 50,
        initialBasis: 10,
        basisPersistence: 1.02,
        basisVolatility: 2,
        priceVolatility: 0,
        quoteEdge: 1,
      },
      options: { seeds: [2, 3], policies: ["B3"] },
    });
    const message = await error;
    assert.equal(message.failure.seed, 2);
    assert.equal(message.failure.step, 92);
    assert.equal(message.comparison.completed, 0);
    assert.equal(message.comparison.aborted, true);
    assert.equal(message.comparison.total, 2);
    assert.ok(message.failure.partialSnapshot.rfqHistory.length > 0);
  } finally {
    await worker.terminate();
  }
});
