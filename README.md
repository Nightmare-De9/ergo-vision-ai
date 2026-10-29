# ErgoVision

**Posture, with perspective.** A minimal, browser-first posture-awareness web app for the CBSE Class XII AI project (2026–27).

ErgoVision uses the supplied TensorFlow Lite classifier to label a 132-feature pose vector as `correct` or `incorrect`. MediaPipe estimates 33 body landmarks from the webcam. The camera preview, landmark processing, and classifier inference run in the browser; the app does not upload or save camera frames.

> **Educational prototype, not a medical device.** The model's probability is not model accuracy, a validated ergonomic score, or medical advice. Its results depend on the training data and the exact preprocessing used during training. Move naturally and take breaks regardless of its output.

## What's included

- A responsive, Apple-inspired single-purpose interface, not a generic admin dashboard.
- User-initiated camera access, live pose overlay, and explicit stop control.
- The supplied `sitting_posture_classifier.tflite` at `frontend/public/models/sitting_posture_classifier.tflite`.
- The accompanying model contract at `frontend/public/models/labels.json`.
- Browser-side TFLite inference through Google's LiteRT.js runtime, with an adjustable decision threshold (default `0.50`).
- Session timer, model-reading trace, a gentle sustained-state nudge, and user-triggered JSON export.
- Unit tests for the 33-landmark to 132-feature contract.

## Run locally

You need Node.js 20+ and npm. Because browsers restrict webcam access, use `localhost` during development or deploy behind HTTPS.

```bash
npm install
npm run dev
```

Open the local URL Vite prints, normally `http://localhost:5173`, then press **Start session** and allow camera access. The app loads the MediaPipe Web runtime and pose model from the official public CDN, while your supplied TFLite classifier is served from the app itself.

To verify the project:

```bash
npm test
npm run typecheck
npm run build
```

To preview a production build:

```bash
npm run preview
```

## Model contract and preprocessing

The supplied labels document describes these inputs:

- 33 MediaPipe pose landmarks in landmark-index order.
- Four values per landmark: hip-relative `x`, hip-relative `y`, hip-relative `z`, and visibility.
- 132 float features total.
- Class `0` is `incorrect`; class `1` is `correct`.
- The output is described as `probability_correct`; the default decision threshold is `0.5`.

`frontend/src/utils/landmarks.ts` implements the preprocessing as written in that contract: it uses the average of the left and right hip landmark coordinates as the reference origin, subtracts that origin from each landmark's x/y/z, and then appends visibility. **This needs to match the training function exactly.** The provided metadata describes the transform but does not include the function body, so compare this implementation with the original `landmarks_to_features()` used to generate the TFLite model before interpreting its scores as meaningful. If the training helper used a different hip origin, coordinate scale, missing-landmark handling, or feature order, update this adapter.

The app assumes the model exposes a scalar probability of the `correct` class. It also accepts a two-value class-probability output and reads index 1. The code validates the feature count, but the model file's tensor dtype, exact output layout, and inference behavior should still be tested in a browser with the training script and Keras model available. Only the TFLite file and the JSON contract were supplied in this workspace; the Keras file was not included and is not required for on-device inference when TFLite works.

## Privacy and network behavior

- The browser does not request camera access until the user presses **Start session**.
- Webcam pixels are processed in-browser and are never included in the exported session JSON.
- The optional JSON export contains the threshold, duration, counts, and up to the latest 2,400 sampled model probabilities only; it never includes camera frames.
- The MediaPipe Web runtime and pose-estimation model are fetched from their public CDN/hosting locations when the app starts. This requires an internet connection on first use.
- Do not use a public/shared computer for a sensitive camera session. Stop the session when finished.

## Design and limitations

The app reports the model's binary label and probability; it does not infer a medical diagnosis or claim validated ergonomic safety. `correct` and `incorrect` describe the supplied model's classes. The adjustable threshold changes the decision boundary but does not improve the underlying model or prove that the probability is calibrated.

The optional notebook `notebooks/ergonomic_ml_training.ipynb` is a **separate six-angle feature experiment** from the earlier project scaffold. It does not recreate this 132-feature TFLite model. Do not combine its outputs with the supplied TFLite classifier without training and verifying a matching feature contract.

## Repository layout

```text
ergo-vision-ai/
├── frontend/
│   ├── public/models/
│   │   ├── sitting_posture_classifier.tflite
│   │   └── labels.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── services/poseLandmarker.ts
│   │   ├── services/postureModel.ts
│   │   ├── utils/landmarks.ts
│   │   └── styles/global.css
│   └── tests/landmarks.test.ts
├── models/README.md
├── docs/verification.md
├── package.json
├── tsconfig.json
├── vite.config.ts
└── index.html
```
