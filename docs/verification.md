# Verification record

This record is intentionally conservative. The application source, feature-contract tests, and uploaded TFLite asset are present in this repository. Actual webcam inference must be exercised in a modern browser with network access and camera permission.

The automated checks below pass in the repository after installing the declared dependencies. A browser smoke check also confirmed that the app loads without console errors before camera permission is requested. This is not a substitute for a live webcam/TFLite inference session.

Checks intended for the project:

```bash
npm test
npm run typecheck
npm run build
```

Latest local results:

- `npm test` — 5 tests passed.
- `npm run typecheck` — passed.
- `npm run build` — passed.

Known unverified items until a live browser session with camera permission is run:

- Compatibility of the supplied TFLite file with the LiteRT.js WASM runtime in the target browser.
- Its exact tensor dtype, input tensor shape, and output layout.
- Whether the browser feature preprocessing exactly matches the training helper.
- Live camera behavior on each target device/browser and the real model's classification quality.
- The app samples at most 2,400 recent probability readings for memory/export; summary counts cover all readings in the session.

No accuracy, validation metric, or medical/ergonomic outcome is claimed by this project.
