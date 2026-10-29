# Supplied posture classifier

The app uses the supplied model at `frontend/public/models/sitting_posture_classifier.tflite`. Its label/feature contract is stored next to it at `frontend/public/models/labels.json`.

Expected input: `[1, 132]`, float values formed from 33 MediaPipe landmarks × `(x - mean hip x, y - mean hip y, z - mean hip z, visibility)`. Output is assumed to be a scalar probability for class `1` (`correct`); if the model returns class probabilities, the app reads the second value.

**Important:** the JSON contract names `landmarks_to_features()` but does not include its source. Confirm the exact transform and input tensor type against the original training notebook/webcam helper. This repo cannot confirm the model's learned quality or performance based on the model binary alone. The Keras file was mentioned but was not provided in the workspace.
