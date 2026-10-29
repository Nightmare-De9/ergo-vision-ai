# Verification record

This record is intentionally conservative. The application source, feature-contract tests, and uploaded TFLite asset are present in this repository. Actual webcam inference must be exercised in a modern browser with network access and camera permission.

The assembly environment's `npm install` timed out while retrieving packages, so the full Vite build and an actual webcam/TFLite inference session could not be run here. The independent landmark utility passes a standalone strict TypeScript compile. This is not a substitute for the runtime checks below.

Checks intended for the project:

```bash
npm test
npm run typecheck
npm run build
```

Known unverified items until those commands and a live browser session are run:

- Compatibility of the supplied TFLite file with the LiteRT.js WASM runtime in the target browser.
- Its exact tensor dtype, input tensor shape, and output layout.
- Whether the browser feature preprocessing exactly matches the training helper.
- Live camera behavior on each target device/browser and the real model's classification quality.
- The app samples at most 2,400 recent probability readings for memory/export; summary counts cover all readings in the session.

No accuracy, validation metric, or medical/ergonomic outcome is claimed by this project.
