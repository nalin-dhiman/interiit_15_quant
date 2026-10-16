import {
  advanceComparison,
  createComparison,
  summarizeComparison,
} from "./index.js";

let active = null;

self.onmessage = async ({ data }) => {
  if (data.type === "cancel") {
    if (active) active.cancelled = true;
    return;
  }
  if (data.type !== "compare") return;
  if (active) active.cancelled = true;
  const job = { id: data.id, cancelled: false };
  active = job;
  let state = null;
  try {
    state = createComparison(data.config, data.options);
    while (!state.done && !job.cancelled) {
      // Yield within sessions so expensive configurations can still be cancelled.
      const progress = advanceComparison(state, 2);
      self.postMessage({ id: job.id, type: "progress", ...progress });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    if (job.cancelled)
      self.postMessage({
        id: job.id,
        type: "cancelled",
        completed: state.completed,
        total: state.total,
      });
    else
      self.postMessage({
        id: job.id,
        type: "result",
        result: summarizeComparison(state),
      });
  } catch (error) {
    self.postMessage({
      id: job.id,
      type: "error",
      message: error instanceof Error ? error.message : String(error),
      failure: error?.failure ?? {
        stage: "configuration",
        config: data.config,
        options: data.options,
      },
      comparison: state
        ? {
            config: state.config,
            seeds: state.seeds,
            policies: state.policies,
            completed: state.completed,
            total: state.total,
            rows: state.rows,
            aborted: true,
          }
        : null,
    });
  } finally {
    if (active === job) active = null;
  }
};
