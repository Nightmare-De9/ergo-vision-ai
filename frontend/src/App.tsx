import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Camera,
  Check,
  CheckCircle2,
  CircleHelp,
  Clock3,
  Focus,
  LockKeyhole,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Square,
  Target,
  X,
} from 'lucide-react';
import { createPoseDetector, drawPose, type Landmark, type PoseDetector } from './services/poseLandmarker';
import {
  loadPostureModel,
  predictPosture,
  type PosturePrediction,
} from './services/postureModel';
import { landmarksToFeatures } from './utils/landmarks';

const DEFAULT_THRESHOLD = 0.5;
const INFERENCE_INTERVAL_MS = 220;
const NUDGE_AFTER_MS = 3_200;

type SessionPhase = 'ready' | 'loading' | 'live' | 'stopped' | 'error';
type Sample = { time: number; probabilityCorrect: number; label: 'correct' | 'incorrect' };

type SessionExport = {
  product: 'ErgoVision';
  exportedAt: string;
  note: string;
  model: { inputFeatures: 132; threshold: number; source: 'local TFLite' };
  session: {
    durationSeconds: number;
    classifiedSamples: number;
    classifiedAsCorrect: number;
    classifiedAsIncorrect: number;
    percentSamplesClassifiedAsCorrect: number | null;
  };
  sampleRowsIncluded: number;
  samples: Sample[];
};

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectorRef = useRef<PoseDetector | null>(null);
  const modelRef = useRef<Awaited<ReturnType<typeof loadPostureModel>> | null>(null);
  const animationRef = useRef<number | null>(null);
  const generationRef = useRef(0);
  const runningRef = useRef(false);
  const lastInferenceRef = useRef(0);
  const badSinceRef = useRef<number | null>(null);
  const sampleRef = useRef<Sample[]>([]);
  const sessionStartRef = useRef<number | null>(null);
  const correctCountRef = useRef(0);
  const totalSamplesRef = useRef(0);
  const inferenceBusyRef = useRef(false);
  const sessionEndRef = useRef<number | null>(null);
  const thresholdRef = useRef(DEFAULT_THRESHOLD);

  const [phase, setPhase] = useState<SessionPhase>('ready');
  const [errorMessage, setErrorMessage] = useState('');
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  thresholdRef.current = threshold;
  const [prediction, setPrediction] = useState<PosturePrediction | null>(null);
  const [poseFound, setPoseFound] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [classifiedSamples, setClassifiedSamples] = useState(0);
  const [correctSamples, setCorrectSamples] = useState(0);
  const [history, setHistory] = useState<Sample[]>([]);
  const [nudge, setNudge] = useState(false);
  const [showModelInfo, setShowModelInfo] = useState(false);

  const isLive = phase === 'live';
  const percentCorrect = classifiedSamples ? Math.round((correctSamples / classifiedSamples) * 100) : null;

  const stopTracks = useCallback(() => {
    generationRef.current += 1;
    runningRef.current = false;
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    detectorRef.current?.close();
    detectorRef.current = null;
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
    const canvas = canvasRef.current;
    if (canvas) {
      const context = canvas.getContext('2d');
      context?.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  useEffect(() => () => stopTracks(), [stopTracks]);

  useEffect(() => {
    if (!isLive) return undefined;
    const timer = window.setInterval(() => {
      const started = sessionStartRef.current;
      if (started !== null) setElapsedSeconds(Math.floor((Date.now() - started) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [isLive]);

  const resetSession = useCallback(() => {
    sampleRef.current = [];
    correctCountRef.current = 0;
    totalSamplesRef.current = 0;
    inferenceBusyRef.current = false;
    badSinceRef.current = null;
    lastInferenceRef.current = 0;
    sessionStartRef.current = null;
    sessionEndRef.current = null;
    setPrediction(null);
    setPoseFound(false);
    setElapsedSeconds(0);
    setClassifiedSamples(0);
    setCorrectSamples(0);
    setHistory([]);
    setNudge(false);
  }, []);

  const startSession = useCallback(async () => {
    const attemptId = ++generationRef.current;
    setErrorMessage('');
    setPhase('loading');
    resetSession();

    if (!navigator.mediaDevices?.getUserMedia) {
      setPhase('error');
      setErrorMessage('This browser cannot open a camera here. Use a current browser on localhost or an HTTPS site.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error('The camera preview could not be mounted.');
      video.srcObject = stream;
      await video.play();

      const [detector, model] = await Promise.all([createPoseDetector(), loadPostureModel()]);
      if (generationRef.current !== attemptId) { detector.close(); return; }
      detectorRef.current = detector;
      modelRef.current = model;
      sessionStartRef.current = Date.now();
      runningRef.current = true;
      setPhase('live');

      const processFrame = () => {
        if (!runningRef.current || generationRef.current !== attemptId) return;
        const liveVideo = videoRef.current;
        const canvas = canvasRef.current;
        const poseDetector = detectorRef.current;
        const postureModel = modelRef.current;
        if (liveVideo && canvas && poseDetector && postureModel && liveVideo.readyState >= 2) {
          const now = performance.now();
          let landmarks: Landmark[] | undefined;
          try {
            const result = poseDetector.detectForVideo(liveVideo, now);
            landmarks = result.landmarks?.[0];
          } catch (poseError) {
            setErrorMessage(poseError instanceof Error ? poseError.message : 'Pose detection stopped unexpectedly.');
            stopTracks();
            setPhase('error');
            return;
          }
          setPoseFound(Boolean(landmarks?.length));

          if (landmarks?.length) {
            const recentSample = sampleRef.current[sampleRef.current.length - 1];
            drawPose(canvas, liveVideo, landmarks, recentSample?.label === 'incorrect');
            if (now - lastInferenceRef.current >= INFERENCE_INTERVAL_MS && !inferenceBusyRef.current) {
              lastInferenceRef.current = now;
              inferenceBusyRef.current = true;
              const frameGeneration = attemptId;
              const features = landmarksToFeatures(landmarks);
              void (async () => {
                try {
                  const nextPrediction = await predictPosture(postureModel, features, thresholdRef.current);
                  if (!runningRef.current || generationRef.current !== frameGeneration) return;
                  const sample: Sample = {
                    time: Date.now() - (sessionStartRef.current ?? Date.now()),
                    probabilityCorrect: nextPrediction.probabilityCorrect,
                    label: nextPrediction.label,
                  };
                  sampleRef.current = [...sampleRef.current, sample].slice(-2400);
                  totalSamplesRef.current += 1;
                  setPrediction(nextPrediction);
                  setClassifiedSamples(totalSamplesRef.current);
                  if (nextPrediction.label === 'correct') {
                    correctCountRef.current += 1;
                    badSinceRef.current = null;
                    setNudge(false);
                  } else {
                    if (badSinceRef.current === null) badSinceRef.current = performance.now();
                    if (performance.now() - badSinceRef.current >= NUDGE_AFTER_MS) setNudge(true);
                  }
                  setCorrectSamples(correctCountRef.current);
                  setHistory(sampleRef.current.slice(-48));
                } catch (modelError) {
                  if (runningRef.current && generationRef.current === frameGeneration) setErrorMessage(modelError instanceof Error ? modelError.message : 'The classifier could not read this pose.');
                } finally {
                  if (generationRef.current === frameGeneration) inferenceBusyRef.current = false;
                }
              })();
            }
          } else {
            drawPose(canvas, liveVideo, undefined);
            badSinceRef.current = null;
            setNudge(false);
          }
        }
        animationRef.current = requestAnimationFrame(processFrame);
      };
      animationRef.current = requestAnimationFrame(processFrame);
    } catch (error) {
      stopTracks();
      setPhase('error');
      if (error instanceof DOMException && error.name === 'NotAllowedError') {
        setErrorMessage('Camera access was blocked. Allow camera access in your browser settings, then try again.');
      } else if (error instanceof DOMException && error.name === 'NotFoundError') {
        setErrorMessage('No camera was found. Connect a camera and try again.');
      } else {
        setErrorMessage(error instanceof Error ? error.message : 'The live models could not be started. Check your connection and try again.');
      }
    }
  }, [resetSession, stopTracks]);

  const stopSession = useCallback(() => {
    sessionEndRef.current = Date.now();
    if (sessionStartRef.current !== null) setElapsedSeconds(Math.floor((sessionEndRef.current - sessionStartRef.current) / 1000));
    stopTracks();
    setPhase('stopped');
    setPoseFound(false);
    setNudge(false);
  }, [stopTracks]);

  const exportSession = useCallback(() => {
    const endTime = sessionEndRef.current ?? Date.now();
    const duration = sessionStartRef.current === null ? elapsedSeconds : Math.floor((endTime - sessionStartRef.current) / 1000);
    const data: SessionExport = {
      product: 'ErgoVision',
      exportedAt: new Date().toISOString(),
      note: 'Model classifications are experimental posture-awareness signals, not clinical or medical advice. Raw camera frames are not included.',
      model: { inputFeatures: 132, threshold, source: 'local TFLite' },
      session: {
        durationSeconds: Math.max(0, duration),
        classifiedSamples,
        classifiedAsCorrect: correctSamples,
        classifiedAsIncorrect: Math.max(0, classifiedSamples - correctSamples),
        percentSamplesClassifiedAsCorrect: percentCorrect,
      },
      sampleRowsIncluded: sampleRef.current.length,
      samples: sampleRef.current,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `ergovision-session-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [classifiedSamples, correctSamples, elapsedSeconds, percentCorrect, threshold]);

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60).toString().padStart(2, '0');
    const rest = (seconds % 60).toString().padStart(2, '0');
    return `${minutes}:${rest}`;
  };

  const verdictTitle = !isLive
    ? phase === 'loading' ? 'Preparing your space.' : phase === 'error' ? 'We hit a small snag.' : phase === 'stopped' ? 'Session paused.' : 'Ready when you are.'
    : !poseFound ? 'Finding your frame.'
      : !prediction ? 'Reading your posture.'
        : prediction.label === 'correct' ? 'Looking good.' : 'A small reset.';

  const statusCopy = !isLive
    ? phase === 'loading' ? 'Warming up the camera and on-device models.' : phase === 'error' ? 'Your camera was not started. Nothing is being recorded.' : phase === 'stopped' ? 'Your summary stays here until you start another session.' : 'Start a session to see a live, model-based posture signal.'
    : !poseFound ? 'Bring your head, shoulders and hips into view.'
      : !prediction ? 'The model is preparing its first reading.'
        : prediction.label === 'correct'
          ? 'The model currently recognises the “correct” class. Natural movement is still important.'
          : 'The model currently recognises the “incorrect” class. Try a gentle, comfortable adjustment.';

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="ErgoVision home">
          <span className="brand-mark"><Focus size={19} strokeWidth={1.8} /></span>
          <span className="brand-name">ergovision<span>.</span></span>
        </a>
        <nav className="header-links" aria-label="Main navigation">
          <a href="#how-it-works">How it works</a>
          <a href="#privacy">Privacy</a>
        </nav>
        <button className="export-button" type="button" onClick={exportSession} disabled={classifiedSamples === 0}>
          <ArrowDownToLine size={15} /> <span>Export session</span>
        </button>
      </header>

      <main id="top" className="page-content">
        <section className="hero-row" aria-labelledby="hero-title">
          <div>
            <div className="eyebrow"><span className="eyebrow-dot" /> POSTURE AWARENESS, REIMAGINED</div>
            <h1 id="hero-title">Posture, with<br className="desktop-break" /> perspective<span className="title-period">.</span></h1>
            <p className="hero-description">A quieter way to notice how you sit. A little awareness, a gentle nudge, and more room to focus on what matters.</p>
          </div>
          <div className="hero-note">
            <div className="hero-note-icon"><Sparkles size={18} strokeWidth={1.7} /></div>
            <p><strong>Small changes add up.</strong><br />This is a guide, not a grade.</p>
          </div>
        </section>

        <section className="session-workspace" aria-label="Live posture session">
          <div className="camera-column">
            <div className="camera-heading">
              <div>
                <span className="section-kicker">YOUR SPACE</span>
                <h2>Live session</h2>
              </div>
              <div className={`connection-pill ${isLive ? 'connected' : ''}`}>
                <span className="connection-dot" />
                {isLive ? 'Live' : phase === 'loading' ? 'Connecting' : 'Camera off'}
              </div>
            </div>

            <div className={`camera-stage ${isLive ? 'camera-stage-live' : ''}`}>
              <video ref={videoRef} className="camera-video" playsInline muted aria-label="Live camera preview" />
              <canvas ref={canvasRef} className="pose-canvas" aria-hidden="true" />
              {!isLive && (
                <div className="camera-empty-state">
                  <div className="camera-orbit orbit-one" />
                  <div className="camera-orbit orbit-two" />
                  <div className="camera-empty-icon"><Camera size={27} strokeWidth={1.4} /></div>
                  <p className="camera-empty-title">Make yourself comfortable.</p>
                  <p className="camera-empty-copy">Position your head, shoulders and hips in frame.</p>
                  <button className="primary-button" type="button" onClick={startSession} disabled={phase === 'loading'}>
                    {phase === 'loading' ? <span className="button-spinner" /> : <Play size={15} fill="currentColor" />}
                    {phase === 'loading' ? 'Getting things ready' : phase === 'stopped' ? 'Start a new session' : 'Start session'}
                    {phase !== 'loading' && <ArrowRight size={15} />}
                  </button>
                </div>
              )}
              {isLive && (
                <>
                  <div className="camera-live-label"><span className="live-pulse" /> CAMERA PREVIEW</div>
                  <div className="camera-corner-tag"><Activity size={14} /> ON-DEVICE ANALYSIS</div>
                  <div className="camera-bottom-label">
                    <span className="view-status-dot" />
                    {poseFound ? 'Pose detected' : 'Looking for your pose'}
                  </div>
                  <button className="stop-button" type="button" onClick={stopSession} aria-label="Stop camera session"><Square size={13} fill="currentColor" /> Stop</button>
                </>
              )}
              {phase === 'error' && <div className="camera-error"><X size={15} /> {errorMessage}</div>}
            </div>
            <div className="camera-caption">
              <span><LockKeyhole size={14} /> Your video stays on this device.</span>
              <span className="camera-caption-right">No uploads. No account. Just awareness.</span>
            </div>
          </div>

          <aside className="insight-column" aria-label="Posture reading">
            <div className="insight-topline">
              <span className="section-kicker">THE MOMENT, RIGHT NOW</span>
              <span className="signal-icon"><Activity size={16} /></span>
            </div>
            <div className={`verdict ${isLive && prediction?.label === 'correct' ? 'verdict-correct' : ''} ${isLive && prediction?.label === 'incorrect' ? 'verdict-adjust' : ''}`}>
              <div className="verdict-icon">
                {isLive && prediction?.label === 'correct' ? <CheckCircle2 size={23} strokeWidth={1.7} /> : isLive && prediction?.label === 'incorrect' ? <RotateCcw size={23} strokeWidth={1.7} /> : phase === 'error' ? <CircleHelp size={23} strokeWidth={1.7} /> : <Target size={23} strokeWidth={1.7} />}
              </div>
              <div className="verdict-copy">
                <h2>{verdictTitle}</h2>
                <p>{statusCopy}</p>
              </div>
            </div>

            {nudge && isLive && prediction?.label === 'incorrect' && (
              <div className="gentle-nudge"><Sparkles size={15} /><span>Gentle reminder: give your shoulders a little room and find a comfortable, neutral position.</span></div>
            )}

            <div className="model-readout">
              <div className="readout-heading">
                <span>Probability of “correct”</span>
                <span className="readout-value">{prediction && isLive && poseFound ? `${Math.round(prediction.probabilityCorrect * 100)}%` : '—'}</span>
              </div>
              <div className="probability-track" role="progressbar" aria-label="Probability of correct posture" aria-valuemin={0} aria-valuemax={100} aria-valuenow={prediction && isLive && poseFound ? Math.round(prediction.probabilityCorrect * 100) : 0}>
                <span style={{ width: `${prediction && isLive && poseFound ? prediction.probabilityCorrect * 100 : 0}%` }} />
              </div>
              <div className="readout-foot"><span>Model output</span><span>Decision point <strong>{Math.round(threshold * 100)}%</strong></span></div>
            </div>

            <div className="threshold-control">
              <div className="threshold-label-row">
                <label htmlFor="threshold">Decision threshold</label>
                <output htmlFor="threshold">{threshold.toFixed(2)}</output>
              </div>
              <input id="threshold" type="range" min="0.1" max="0.9" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
              <div className="range-captions"><span>More sensitive</span><span>More selective</span></div>
            </div>

            <div className="model-footnote">
              <ShieldCheck size={15} />
              <p>Powered by your 132-feature TFLite classifier. Its output is a model estimate, not a medical assessment.</p>
            </div>
            {errorMessage && phase === 'live' && <div className="inline-error">{errorMessage}</div>}
            <button className="text-button model-info-toggle" type="button" onClick={() => setShowModelInfo((current) => !current)} aria-expanded={showModelInfo}>
              What the model is looking at <span>{showModelInfo ? '−' : '+'}</span>
            </button>
            {showModelInfo && (
              <div className="model-info-panel">
                The model receives 33 pose landmarks, each represented by hip-relative x, y and z coordinates plus visibility. That becomes 132 numeric features. ErgoVision does not upload the camera feed; the classifier runs in your browser after the model assets load.
              </div>
            )}
          </aside>
        </section>

        <section className="session-summary" aria-label="Session summary">
          <div className="summary-intro">
            <span className="section-kicker">A LITTLE CONTEXT</span>
            <h2>Stay in the flow.</h2>
            <p>Notice patterns without turning posture into another thing to worry about.</p>
          </div>
          <div className="summary-metric">
            <span className="metric-icon"><Clock3 size={16} /></span>
            <span className="metric-label">Time in session</span>
            <strong>{formatDuration(elapsedSeconds)}</strong>
            <span className="metric-subtext">{isLive ? 'In progress' : phase === 'stopped' ? 'Session stopped' : 'Ready to begin'}</span>
          </div>
          <div className="summary-metric">
            <span className="metric-icon"><Activity size={16} /></span>
            <span className="metric-label">Classified samples</span>
            <strong>{classifiedSamples.toLocaleString('en-US')}</strong>
            <span className="metric-subtext">Model readings, not frames stored</span>
          </div>
          <div className="summary-metric summary-metric-last">
            <span className="metric-icon"><Check size={16} /></span>
            <span className="metric-label">Read as “correct”</span>
            <strong>{percentCorrect === null ? '—' : `${percentCorrect}%`}</strong>
            <span className="metric-subtext">{percentCorrect === null ? 'No readings yet' : `${correctSamples} of ${classifiedSamples} readings`}</span>
          </div>
        </section>

        <section className="history-section" aria-label="Recent model readings">
          <div className="history-header">
            <div>
              <span className="section-kicker">OVER THE SESSION</span>
              <h2>Your recent rhythm</h2>
            </div>
            <span className="history-legend"><i /> Correct class <i className="legend-amber" /> Incorrect class</span>
          </div>
          <div className={`history-track ${history.length ? '' : 'history-track-empty'}`} aria-label="Recent readings visualised from left to right">
            {history.length ? history.map((sample, index) => (
              <span
                className={`history-bar ${sample.label === 'correct' ? 'bar-correct' : 'bar-incorrect'}`}
                key={`${sample.time}-${index}`}
                title={`${sample.label}: ${Math.round(sample.probabilityCorrect * 100)}% probability correct`}
                style={{ height: `${18 + sample.probabilityCorrect * 82}%` }}
              />
            )) : <span className="history-empty-message">Your session trace will appear here once the model has a reading.</span>}
          </div>
          <div className="history-footer"><span>Earlier</span><span>Latest</span></div>
        </section>

        <section id="how-it-works" className="how-section">
          <div className="how-heading">
            <span className="section-kicker">MADE TO STAY OUT OF YOUR WAY</span>
            <h2>Thoughtful by design.</h2>
          </div>
          <div className="how-steps">
            <article className="how-step">
              <span className="step-number">01</span>
              <h3>Look in, not away.</h3>
              <p>MediaPipe finds your body landmarks locally from the live preview. Camera access starts only when you ask for it.</p>
            </article>
            <article className="how-step">
              <span className="step-number">02</span>
              <h3>A signal, not a score.</h3>
              <p>Your supplied TFLite model classifies the landmark features. The probability and threshold are shown plainly.</p>
            </article>
            <article className="how-step">
              <span className="step-number">03</span>
              <h3>Keep it gentle.</h3>
              <p>Small prompts encourage a comfortable reset, never a perfect pose or a diagnosis. Move naturally and take breaks.</p>
            </article>
          </div>
        </section>

        <section id="privacy" className="privacy-strip">
          <div className="privacy-emblem"><LockKeyhole size={18} /></div>
          <div><h3>Your camera. Your device.</h3><p>Video is processed in your browser and is not sent to an ErgoVision server. Only a session summary is exported, and only when you choose to download it.</p></div>
          <button className="text-button privacy-export" type="button" onClick={exportSession} disabled={classifiedSamples === 0}>Export summary <ArrowRight size={14} /></button>
        </section>

        <footer className="site-footer">
          <a className="footer-brand" href="#top">ergovision<span>.</span></a>
          <p>Built for awareness, not perfection.</p>
          <span>Educational prototype · Not medical advice</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
