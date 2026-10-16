# Optional organiser callback adapters

These are Python research adapters for a separately installed development SDK.
They do not run in the browser and are not needed for the NumPy tests.
The third-party SDK and organiser data are not redistributed here.

- `sdk_dealer.py` contains conventional inventory controls and the earlier
  frozen-book continuation comparison.
- `boundary_dealer.py` waits for resolved RFQs, reconciled inventory and
  confirmed exchange executions before admitting the next batch.
- `resetmm.basis_baseline` adds the conventional public-basis control and the
  matched scalar continuation candidate.
- `dealer_scaffold.py` is a minimal disabled-by-default interface example.

The browser engine is a separate synthetic implementation. Local callback
conformance does not certify any external deployment. The adapters retain
their conservative failure handling and response adaptation stays disabled.
