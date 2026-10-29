import { loadAndCompile, loadLiteRt, Tensor } from '@litertjs/core';

export type PostureClass = 'correct' | 'incorrect';

export type PosturePrediction = {
  probabilityCorrect: number;
  probabilityIncorrect: number;
  label: PostureClass;
};

type LiteRtTensor = InstanceType<typeof Tensor>;
type TensorDetails = { shape?: number[]; dtype?: string; name?: string };
type ModelContract = {
  '0': string;
  '1': string;
  input_features: number;
  landmark_count: number;
  values_per_landmark: string[];
  probability_output: string;
  decision_threshold: number;
  preprocessing: string;
};

type BrowserLiteRtModel = {
  run: (inputs: LiteRtTensor | LiteRtTensor[]) => Promise<LiteRtTensor[]>;
  getInputDetails: () => TensorDetails[];
  getOutputDetails: () => TensorDetails[];
  delete: () => void;
};

const MODEL_URL = `${import.meta.env.BASE_URL}models/sitting_posture_classifier.tflite`;
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/';
let runtimePromise: Promise<void> | null = null;
let contractPromise: Promise<ModelContract> | null = null;
let modelPromise: Promise<BrowserLiteRtModel> | null = null;

export async function loadPostureModel(): Promise<BrowserLiteRtModel> {
  if (!contractPromise) {
    contractPromise = fetch(`${import.meta.env.BASE_URL}models/labels.json`)
      .then(async (response) => {
        if (!response.ok) throw new Error('The model labels file could not be loaded.');
        return response.json() as Promise<ModelContract>;
      })
      .then((contract) => {
        if (contract.input_features !== 132 || contract.landmark_count !== 33) {
          throw new Error('The labels file does not match the expected 33-landmark / 132-feature contract.');
        }
        if (contract.probability_output !== 'probability_correct') {
          throw new Error('The labels file does not describe a probability_correct model output.');
        }
        if (contract['0'] !== 'incorrect' || contract['1'] !== 'correct') {
          throw new Error('The labels file must map class 0 to incorrect and class 1 to correct.');
        }
        return contract;
      })
      .catch((error: unknown) => {
        contractPromise = null;
        throw error;
      });
  }
  if (!runtimePromise) {
    runtimePromise = loadLiteRt(WASM_URL).then(() => undefined).catch((error: unknown) => {
      runtimePromise = null;
      throw error;
    });
  }
  if (!modelPromise) {
    modelPromise = (async () => {
      await contractPromise;
      await runtimePromise;
      const loaded = await loadAndCompile(MODEL_URL, { accelerator: 'wasm' });
      const model = loaded as unknown as BrowserLiteRtModel;
      const inputs = model.getInputDetails();
      if (inputs.length !== 1) {
        model.delete();
        throw new Error(`Expected one TFLite input tensor; found ${inputs.length}.`);
      }

      const shape = inputs[0].shape ?? [];
      
      const validInputShape =
        (shape.length === 1 && shape[0] === 132) ||
        (shape.length === 2 &&
          (shape[0] === -1 || shape[0] === 1) &&
          shape[1] === 132);
      
      if (!validInputShape) {
        model.delete();
        throw new Error(
          `Unsupported TFLite input shape [${shape.join(', ')}]. ` +
          'Expected [132], [1, 132], or [-1, 132].'
        );
      }

      const inputDtype = inputs[0].dtype?.toLowerCase();
      if (inputDtype && !inputDtype.includes('float32')) {
        model.delete();
        throw new Error(`This browser adapter expects float32 model input; the file reports ${inputs[0].dtype}.`);
      }
      const outputs = model.getOutputDetails();
      if (outputs.length !== 1) {
        model.delete();
        throw new Error(`Expected one TFLite output tensor; found ${outputs.length}.`);
      }
      const outputShape = outputs[0].shape ?? [];
      
      // -1 is a dynamic batch dimension in TFLite.
      const validOutputShape =
        outputShape.length === 0 ||
        (outputShape.length === 1 &&
          (outputShape[0] === 1 || outputShape[0] === 2)) ||
        (outputShape.length === 2 &&
          (outputShape[0] === -1 || outputShape[0] === 1) &&
          (outputShape[1] === 1 || outputShape[1] === 2));
      
      if (!validOutputShape) {
        model.delete();
        throw new Error(
          `Unsupported TFLite output shape [${outputShape.join(', ')}]. ` +
          'Expected a scalar, [1], [2], [1, 1], [1, 2], [-1, 1], or [-1, 2].'
        );
      }
      const outputDtype = outputs[0].dtype?.toLowerCase();
      if (outputDtype && !outputDtype.includes('float32')) {
        model.delete();
        throw new Error(`This browser adapter expects float32 model output; the file reports ${outputs[0].dtype}.`);
      }
      return model;
    })().catch((error: unknown) => {
      modelPromise = null;
      throw error;
    });
  }
  return modelPromise;
}

export async function predictPosture(
  model: BrowserLiteRtModel,
  features: number[],
  threshold = 0.5,
): Promise<PosturePrediction> {
  if (features.length !== 132) throw new Error(`Expected 132 inputs; received ${features.length}.`);
  const input = new Tensor(new Float32Array(features), [1, 132]);
  let outputs: LiteRtTensor[] = [];
  try {
    outputs = await model.run([input]);
    const output = outputs[0];
    if (!output) throw new Error('The TFLite model returned no output tensor.');
const values = Array.from(await output.data(), Number);

if (values.length !== 1 && values.length !== 2) {
  throw new Error(
    `Expected one probability or two class probabilities; received ${values.length} values.`
  );
}

if (values.some((value) => !Number.isFinite(value))) {
  throw new Error('The model output contained a non-finite number.');
}

// One output = probability_correct.
// Two outputs = [probability_incorrect, probability_correct].
const probability = values.length === 2 ? values[1] : values[0];

if (probability < 0 || probability > 1) {
  throw new Error(
    `Expected a probability between 0 and 1; received ${probability}.`
  );
} finally {
    input.delete();
    outputs.forEach((output) => output.delete());
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
