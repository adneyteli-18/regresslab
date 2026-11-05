import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import { Activity, BarChart3, BookOpen, Check, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Database, Download, FileText, GraduationCap, History, Info, Lightbulb, Menu, Play, RotateCcw, Scissors, Search, Settings2, Target, Trash2, Upload, X } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import diabetes from "./data/diabetes.json";
import { metrics, olsFit, predict, standardize } from "./lib/math";
import { downloadCsv, formatNumber, formatPercent } from "./lib/format";
import { clearHistory, deleteHistory, loadHistory, saveHistory } from "./lib/storage";
import Simulation from "./Simulation";

const pages = [
  ["aim", "Aim", Target], ["theory", "Theory", BookOpen], ["procedure", "Procedure", FileText],
  ["data", "Data Preparation", Database], ["simulation", "Simulation", Settings2],
  ["exercise", "Exercise", GraduationCap], ["references", "References", BookOpen], ["history", "Experiment History", History],
];
const pageMap = Object.fromEntries(pages.map(([id, label]) => [id, label]));
const featureNames = diabetes.features;
const initialConfig = { features: ["bmi", "s5"], target: "target", testPercent: 20, seed: 42, standardize: true };

function useSection() {
  const readSection = () => {
    const querySection = new URLSearchParams(window.location.search).get("section");
    if (querySection && pageMap[querySection]) return querySection;
    const pathSection = window.location.pathname.replace(/^\/+|\/+$/g, "");
    return pageMap[pathSection] ? pathSection : "aim";
  };
  const [section, setSection] = useState(readSection);
  useEffect(() => {
    const handlePopState = () => setSection(readSection());
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);
  const navigate = (next) => {
    const url = `?section=${next}`;
    window.history.pushState({}, "", url);
    setSection(next);
  };
  return [section, navigate];
}

function useDataset() {
  const [dataset, setDataset] = useState(diabetes);
  const [uploadedName, setUploadedName] = useState("");
  const upload = (file) => new Promise((resolve, reject) => {
    if (!file || file.size > 10 * 1024 * 1024) return reject(new Error("Upload a CSV or TXT file smaller than 10 MB."));
    Papa.parse(file, { header: true, skipEmptyLines: true, dynamicTyping: false, transformHeader: (header) => header.trim(), complete: ({ data, meta }) => {
      if (!meta.fields?.length || data.length < 3) return reject(new Error("The CSV must have a header and at least 3 data rows."));
      const parseNumber = (value) => value === null || value === undefined || String(value).trim() === "" ? null : Number(String(value).trim());
      const numeric = meta.fields.filter((field) => data.filter((row) => String(row[field] ?? "").trim() !== "").filter((row) => Number.isFinite(parseNumber(row[field]))).length >= data.length - 1);
      if (numeric.length < 2) return reject(new Error("Select a CSV with at least one numeric feature and one numeric target column."));
      const validRows = [];
      let missingRows = 0;
      let invalidRows = 0;
      data.forEach((row) => {
        const values = numeric.map((field) => parseNumber(row[field]));
        if (values.every((value) => value === null)) return;
        if (values.some((value) => value === null)) { missingRows += 1; return; }
        if (values.some((value) => !Number.isFinite(value))) { invalidRows += 1; return; }
        validRows.push(Object.fromEntries(numeric.map((field, index) => [field, values[index]])));
      });
      if (validRows.length < 3) return reject(new Error(`Only ${validRows.length} valid rows remain; at least 3 complete rows are required.`));
      const next = { name: file.name, description: "Uploaded CSV · numeric rows processed locally in this browser.", target: numeric[numeric.length - 1], features: numeric.slice(0, -1), rows: validRows, quality: { sourceRows: data.length, validRows: validRows.length, missingRows, invalidRows } };
      setDataset(next); setUploadedName(file.name); resolve(next);
    }, error: () => reject(new Error("Could not parse this file.")) });
  });
  return { dataset, setDataset, upload, uploadedName, reset: () => { setDataset(diabetes); setUploadedName(""); } };
}

function getSplit(dataset, config) {
  const rows = dataset.rows.filter((row) => config.features.every((feature) => Number.isFinite(Number(row[feature])) && Number.isFinite(Number(row[config.target]))));
  if (config.features.length < 1) throw new Error("Select at least one input feature.");
  if (rows.length < config.features.length + 2) throw new Error("The dataset is too small for the selected features.");
  const seed = config.seed;
  const indexed = rows.map((row, index) => ({ row, index }));
  let state = seed >>> 0;
  for (let i = indexed.length - 1; i > 0; i -= 1) { state = (state * 1664525 + 1013904223) >>> 0; const j = state % (i + 1); [indexed[i], indexed[j]] = [indexed[j], indexed[i]]; }
  const testCount = Math.max(1, Math.round(rows.length * config.testPercent / 100));
  const testRows = indexed.slice(0, testCount).map((item) => item.row);
  const trainRows = indexed.slice(testCount).map((item) => item.row);
  const trainX = trainRows.map((row) => config.features.map((feature) => Number(row[feature])));
  const testX = testRows.map((row) => config.features.map((feature) => Number(row[feature])));
  const trainY = trainRows.map((row) => Number(row[config.target]));
  const testY = testRows.map((row) => Number(row[config.target]));
  const rawMeans = trainX[0].map((_, column) => average(trainX.map((row) => row[column])));
  const rawStds = rawMeans.map((value, column) => Math.sqrt(average(trainX.map((row) => (row[column] - value) ** 2))) || 1);
  const scale = config.standardize ? standardize(trainX, testX) : { train: trainX, test: testX };
  return { trainRows, testRows, trainX: scale.train, testX: scale.test, trainY, testY, means: rawMeans, stds: rawStds };
}

function fitRun(dataset, config) {
  const split = getSplit(dataset, config);
  const model = olsFit(split.trainX, split.trainY);
  const trainPred = predict(split.trainX, model);
  const testPred = predict(split.testX, model);
  const baselinePred = split.testY.map(() => average(split.trainY));
  return { split, model, trainPred, testPred, baselinePred, trainMetrics: metrics(split.trainY, trainPred, config.features.length), testMetrics: metrics(split.testY, testPred, config.features.length), baselineMetrics: metrics(split.testY, baselinePred, 0) };
}

function Card({ title, eyebrow, children, className = "", id }) {
  return <section id={id} className={`card ${className}`}>{eyebrow && <div className="card-eyebrow">{eyebrow}</div>}{title && <h2>{title}</h2>}{children}</section>;
}
function Button({ children, variant = "primary", icon: Icon, onClick, disabled, type = "button" }) {
  return <button type={type} className={`button ${variant}`} onClick={onClick} disabled={disabled}>{Icon && <Icon size={16} />}{children}</button>;
}
function PageHeader({ title, subtitle }) {
  return <header className="page-header"><div className="eyebrow">LINEAR REGRESSION · VIRTUAL LAB</div><h1>{title}</h1><p>{subtitle}</p></header>;
}
function Metric({ label, value, compact = false, caption }) { return <div className={`metric ${compact ? "compact" : ""}`}><span>{label}</span><strong>{typeof value === "number" ? formatNumber(value) : value}</strong>{caption && <small>{caption}</small>}</div>; }
function Alert({ type = "info", children }) { return <div className={`alert ${type}`}>{children}</div>; }
function EmptyState({ message, onClick }) { return <div className="empty"><Database size={26} /><p>{message}</p>{onClick && <Button onClick={onClick}>Go to Data Preparation</Button>}</div>; }
function ChartBox({ title, children }) { return <Card title={title} className="chart-card"><div className="chart">{children}</div></Card>; }
function PageNavigation({ section, navigate }) {
  const index = pages.findIndex(([id]) => id === section);
  return <div className="page-navigation">{index > 0 ? <Button variant="secondary" icon={ChevronLeft} onClick={() => navigate(pages[index - 1][0])}>{pages[index - 1][1]}</Button> : <span />}{index < pages.length - 1 ? <Button variant="secondary" onClick={() => navigate(pages[index + 1][0])}>{pages[index + 1][1]} <ChevronRight size={15} /></Button> : <span />}</div>;
}
function BackToTop() { return <button className="back-top" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })} aria-label="Back to top"><ChevronDown size={16} /></button>; }

function Sidebar({ section, navigate, open, onClose }) {
  return <aside className={`sidebar ${open ? "open" : ""}`}><div className="brand"><div className="logo"><Lightbulb size={21} /></div><div><strong>RegressLab</strong><small>VIRTUAL LAB</small></div></div><div className="sidebar-label">LINEAR REGRESSION</div><nav>{pages.map(([id, label, Icon], index) => <button key={id} className={section === id ? "active" : ""} onClick={() => { navigate(id); onClose(); }}><Icon size={17} /><span>{label}</span><b>{String(index + 1).padStart(2, "0")}</b></button>)}</nav><div className="sidebar-footer"><i />Runs locally in your browser</div></aside>;
}
function TopBar({ section, onMenu }) {
  return <><div className="topbar"><button className="menu-button" onClick={onMenu} aria-label="Open navigation"><Menu size={20} /></button><div className="breadcrumb">RegressLab <span>/</span> <b>{pageMap[section]}</b></div><div className="top-caption">LINEAR REGRESSION · VIRTUAL LAB</div></div></>;
}
function StatusStrip({ dataset, split, historyCount }) {
  return <div className="status-strip"><span><i />Local interactive lab</span><span>{dataset.rows.length} dataset rows</span><span>{split ? `${split.trainY.length} train / ${split.testY.length} test` : "No split created"}</span><span>{historyCount} saved experiments</span></div>;
}

function Aim({ navigate }) {
  return <><PageHeader title="Aim" subtitle="Explore how a straight-line model learns the relationship between features and a continuous target." /><Card className="aim-card"><div className="aim-icon"><Target size={25} /></div><div><div className="card-eyebrow">PURPOSE</div><h2>Linear Regression</h2><p className="lead">To understand how a linear regression model estimates a continuous value by fitting a line or hyperplane that minimises squared prediction error.</p><div className="outcomes"><h3>By the end of this lab, you can</h3>{["Explain slope, intercept, and coefficients in a regression equation.", "Fit a model with the Ordinary Least Squares Normal Equation.", "Evaluate a model using R², MAE, MSE and RMSE.", "Interpret a genuine best-fit line and actual-versus-predicted graph.", "Explain train-test splitting and generalisation."].map((item) => <p key={item}><Check size={16} />{item}</p>)}</div><Button icon={ChevronRight} onClick={() => navigate("theory")}>Read the theory</Button></div></Card></>;
}

function Theory() {
  return <><PageHeader title="Theory" subtitle="Understand Ordinary Least Squares, model evaluation, and responsible interpretation." /><div className="theory-layout"><div><Card eyebrow="INTRODUCTION" title="What is Linear Regression?"><p>Linear Regression predicts a continuous target from one or more numeric features. A simple model uses one feature; a multiple model uses several.</p></Card><div className="two-col"><Card id="model" title="The model"><p>A prediction is a weighted combination of the features:</p><Formula label="MULTIPLE LINEAR REGRESSION">ŷ = β₀ + β₁x₁ + β₂x₂ + … + βₚxₚ</Formula><p>β₀ is the intercept, β values are coefficients, and the residual is e = y − ŷ.</p></Card><Card id="normal" title="Normal Equation"><Formula label="ORDINARY LEAST SQUARES">β = argmin Σ(yᵢ − ŷᵢ)²</Formula><p>RegressLab solves the least-squares system with a numerically stable QR decomposition. It rejects constant or linearly dependent features instead of silently regularising them.</p><Formula label="SIMPLE REGRESSION">β₁ = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)² &nbsp;&nbsp; β₀ = ȳ − β₁x̄</Formula></Card></div><Card id="worked" title="Worked numeric example"><p>For points (1, 2), (2, 3), (3, 5), (4, 4), (5, 6), x̄ = 3 and ȳ = 4. The slope is β₁ = 9 / 10 = 0.9 and the intercept is β₀ = 4 − 0.9 × 3 = 1.3.</p><Formula label="FITTED LINE">ŷ = 1.3 + 0.9x</Formula><p>For x = 6, the prediction is 6.7. A residual is the observed target minus its prediction.</p></Card><Card id="metrics" title="Evaluation metrics"><div className="formula-grid"><Formula label="R-SQUARED">R² = 1 − SSres / SStot</Formula><Formula label="MEAN ABSOLUTE ERROR">MAE = (1/n) Σ |yᵢ − ŷᵢ|</Formula><Formula label="MEAN SQUARED ERROR">MSE = (1/n) Σ (yᵢ − ŷᵢ)²</Formula><Formula label="ROOT MSE">RMSE = √MSE</Formula></div><p>R² can be negative when the model performs worse than predicting the training mean. MAE and RMSE are in target units.</p></Card><Card id="considerations" title="Important considerations"><div className="considerations"><p><b>Data split:</b> Fit preprocessing on training rows only.</p><p><b>Linearity:</b> A patterned residual would suggest limitations of a straight-line model.</p><p><b>Generalisation:</b> Test results estimate performance on unseen rows.</p><p><b>Multicollinearity:</b> Dependent predictors make individual coefficients unreliable.</p><p><b>Interpretation:</b> Association is not proof of causation.</p></div></Card><Card title="Quick glossary"><div className="glossary">{["Target — the value being predicted.", "Feature — an input predictor.", "Coefficient — a learned weight.", "Intercept — prediction when inputs are zero.", "Residual — actual minus predicted.", "Baseline — prediction using the training target mean."].map((item) => <span key={item}>{item}</span>)}</div></Card></div><aside className="theory-toc"><strong>On this page</strong><a href="#model">The model</a><a href="#normal">Normal Equation</a><a href="#metrics">Evaluation metrics</a><a href="#worked">Worked example</a><a href="#considerations">Important considerations</a></aside></div></>;
}
function Formula({ label, children }) { return <div className="formula"><small>{label}</small><code>{children}</code></div>; }

function Procedure({ navigate }) {
  const steps = [["Inspect data", "Load a built-in dataset or upload a CSV; review rows, features and data-quality counts."], ["Prepare data", "Choose a numeric target and features, then create a reproducible train/test split."], ["Fit preprocessing", "Optionally standardise with training statistics only to avoid leakage."], ["Configure model", "Use the Ordinary Least Squares Normal Equation."], ["Train and predict", "Fit the model on training data and predict held-out rows or a query point."], ["Inspect and save", "Review metrics, the two model-generated graphs, coefficients, and interpretation."]];
  return <><PageHeader title="Procedure" subtitle="Follow six repeatable steps from raw data to a validated regression model." /><Card title="Workflow"><div className="timeline">{steps.map(([title, text], index) => <div className="timeline-item" key={title}><span>{String(index + 1).padStart(2, "0")}</span><div><h3>{title}</h3><p>{text}</p></div></div>)}</div><Button icon={ChevronRight} onClick={() => navigate("data")}>Open data preparation</Button></Card></>;
}

function DataPreparation({ datasetState, config, setConfig, onSplit, navigate }) {
  const { dataset, setDataset, upload, reset } = datasetState;
  const [error, setError] = useState("");
  const numeric = [...(dataset.features || Object.keys(dataset.rows[0] || {}).filter((key) => dataset.rows.some((row) => typeof row[key] === "number"))), dataset.target].filter((value, index, list) => list.indexOf(value) === index);
  const missing = numeric.filter((feature) => dataset.rows.some((row) => !Number.isFinite(Number(row[feature])))).length;
  const correlations = numeric.filter((f) => f !== config.target).slice(0, 8).map((feature) => ({ feature, value: correlation(dataset.rows, feature, config.target) }));
  const handleUpload = (event) => upload(event.target.files[0]).then((next) => setConfig((old) => ({ ...old, target: next.target, features: next.features, split: null }))).catch((reason) => setError(reason.message));
  const createSplit = () => { if (!config.features.length) return setError("Select at least one feature."); try { setError(""); onSplit(); } catch (reason) { setError(reason.message); } };
  return <><PageHeader title="Data Preparation" subtitle="Inspect the dataset, choose predictors, and create a reproducible training and testing split." />{error && <Alert type="error">{error}</Alert>}{dataset.quality && <Alert type={dataset.quality.missingRows || dataset.quality.invalidRows ? "error" : "success"}>Upload audit: {dataset.quality.validRows} valid rows, {dataset.quality.missingRows} rows with missing values, {dataset.quality.invalidRows} rows with invalid values. Invalid rows are not used for training.</Alert>}<div className="summary-grid"><Metric label="Rows" value={dataset.rows.length} /><Metric label="Numeric features" value={numeric.length - 1} /><Metric label="Target" value={config.target} /><Metric label="Missing values" value={missing} /></div><div className="two-col prep-grid"><Card eyebrow="STEP 1 · INSPECT DATA" title="Dataset"><div className="dataset-select"><label>Choose a dataset<select value={dataset.name} onChange={(event) => { if (event.target.value === diabetes.name) setDataset(diabetes); }}><option value={diabetes.name}>Diabetes (442 rows, 10 predictors)</option></select></label><label className="upload-button"><Upload size={15} /> Upload CSV<input type="file" accept=".csv,.txt" onChange={handleUpload} /></label></div><p>{dataset.description}</p><div className="table-wrap"><table><thead><tr>{[...numeric.slice(0, 5), config.target].map((name) => <th key={name}>{name}</th>)}</tr></thead><tbody>{dataset.rows.slice(0, 8).map((row, index) => <tr key={index}>{[...numeric.slice(0, 5), config.target].map((name) => <td key={name}>{formatNumber(row[name])}</td>)}</tr>)}</tbody></table></div><h3>Features used by the model</h3><div className="chip-list">{numeric.filter((feature) => feature !== config.target).map((feature) => <label className={config.features.includes(feature) ? "chip selected" : "chip"} key={feature}><input type="checkbox" checked={config.features.includes(feature)} onChange={() => setConfig((old) => ({ ...old, features: old.features.includes(feature) ? old.features.filter((item) => item !== feature) : [...old.features, feature] }))} />{feature}</label>)}</div><div className="inline-actions"><Button variant="ghost" onClick={() => setConfig((old) => ({ ...old, features: numeric.filter((feature) => feature !== config.target) }))}>Select all</Button><Button variant="ghost" onClick={() => setConfig((old) => ({ ...old, features: [] }))}>Clear</Button><Button variant="ghost" icon={X} onClick={() => { reset(); setError(""); }}>Clear dataset</Button></div></Card><Card eyebrow="STEP 2 · PREPARE DATA" title="Train / test split"><label>Target column<select value={config.target} onChange={(event) => setConfig((old) => ({ ...old, target: event.target.value, features: old.features.filter((feature) => feature !== event.target.value) }))}>{numeric.map((name) => <option key={name}>{name}</option>)}</select></label><label>Test set <b>{config.testPercent}%</b><input type="range" min="10" max="50" value={config.testPercent} onChange={(event) => setConfig((old) => ({ ...old, testPercent: Number(event.target.value) }))} /></label><label>Random seed<input type="number" value={config.seed} onChange={(event) => setConfig((old) => ({ ...old, seed: Number(event.target.value) }))} /></label><label className="check-row"><input type="checkbox" checked={config.standardize} onChange={(event) => setConfig((old) => ({ ...old, standardize: event.target.checked }))} />Standardize features<span>Use statistics fitted on training rows only.</span></label><Button icon={Scissors} onClick={createSplit}>Create train/test split</Button></Card></div>{config.split && <Card title="Split ready"><div className="summary-grid"><Metric label="Training" value={config.split.trainY.length} /><Metric label="Testing" value={config.split.testY.length} /><Metric label="Excluded rows" value={dataset.rows.length - config.split.trainY.length - config.split.testY.length} /><Metric label="Seed" value={config.seed} /></div><Alert type="success">Correct scaler fitted on training data only. The split is deterministic and ready for Simulation.</Alert></Card>}<div className="two-col"><ChartBox title="Target distribution"><ResponsiveContainer width="100%" height={230}><BarChart data={histogram(dataset.rows.map((row) => Number(row[config.target])))}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="label" /><YAxis /><Tooltip /><Bar dataKey="count" fill="#4f6bed" /></BarChart></ResponsiveContainer></ChartBox><ChartBox title="Feature correlation with target"><ResponsiveContainer width="100%" height={230}><BarChart layout="vertical" data={correlations}><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" domain={[-1, 1]} /><YAxis dataKey="feature" type="category" width={80} /><Tooltip /><Bar dataKey="value" fill="#1fa971" /></BarChart></ResponsiveContainer></ChartBox></div>{config.split && <Button icon={ChevronRight} onClick={() => navigate("simulation")}>Open simulation</Button>}</>;
}
function correlation(rows, a, b) { const x = rows.map((row) => Number(row[a])); const y = rows.map((row) => Number(row[b])); const mx = average(x); const my = average(y); const numerator = x.reduce((sum, value, i) => sum + (value - mx) * (y[i] - my), 0); const denominator = Math.sqrt(x.reduce((sum, value) => sum + (value - mx) ** 2, 0) * y.reduce((sum, value) => sum + (value - my) ** 2, 0)); return denominator ? numerator / denominator : 0; }
function histogram(values) { const min = Math.min(...values); const max = Math.max(...values); const width = (max - min || 1) / 8; return Array.from({ length: 8 }, (_, index) => ({ label: `${(min + width * index).toFixed(0)}`, count: values.filter((value) => value >= min + width * index && value < min + width * (index + 1)).length })); }
function average(values) { return values.reduce((sum, value) => sum + value, 0) / values.length; }
function targetStats(values) {
  const averageValue = average(values);
  return { mean: averageValue, std: Math.sqrt(average(values.map((value) => (value - averageValue) ** 2))) };
}

function originalCoefficients(run, config) {
  const coefficients = config.features.map((_, index) => config.standardize ? run.model.coefficients[index] / run.split.stds[index] : run.model.coefficients[index]);
  const intercept = config.standardize ? run.model.intercept - run.model.coefficients.reduce((sum, value, index) => sum + value * run.split.means[index] / run.split.stds[index], 0) : run.model.intercept;
  return { coefficients, intercept };
}

function InfoMetric({ label, value, caption, formula, meaning }) {
  return <div className="metric metric-info"><div className="metric-label">{label}<span className="tooltip-icon" title={`${formula} — ${meaning}`}><Info size={12} /></span></div><strong>{value}</strong>{caption && <small>{caption}</small>}</div>;
}

function StepBadge({ number, label }) { return <div className="step-badge"><b>{number}</b><span>{label}</span></div>; }

const questions = [["What does Linear Regression predict?", ["A continuous value", "A cluster", "A class only", "A text label"], 0, "Regression predicts numerical continuous outcomes."], ["What is a residual?", ["Actual minus predicted", "The intercept", "A feature name", "The dataset size"], 0, "Residual = y − ŷ."], ["Which metric is in the target's units?", ["RMSE", "R²", "MSE", "Adjusted R²"], 0, "Taking the square root returns RMSE to the target scale."], ["What does OLS minimise?", ["Sum of squared residuals", "Number of features", "A coefficient", "Test percentage"], 0, "Ordinary Least Squares minimises squared errors."], ["What is data leakage?", ["Using test information during fitting", "A missing label", "A small dataset", "A low R²"], 0, "The scaler must be fitted on training rows only."], ["What does a negative R² mean?", ["Worse than the mean baseline", "Perfect fit", "No target", "No residuals"], 0, "A negative R² indicates performance below the target-mean baseline."]];
function Exercise({ run }) { const [index, setIndex] = useState(0); const [answers, setAnswers] = useState({}); const [submitted, setSubmitted] = useState(false); const [score, setScore] = useState(null); const item = questions[index]; const submit = () => { setSubmitted(true); setScore(questions.reduce((sum, question, i) => sum + (answers[i] === question[2] ? 1 : 0), 0)); }; return <><PageHeader title="Exercise" subtitle="Check your understanding of Linear Regression with immediate feedback." /><Card title={`Question ${index + 1} of ${questions.length}`}><div className="progress"><span style={{ width: `${((index + 1) / questions.length) * 100}%` }} /></div><h3>{item[0]}</h3><div className="answers">{item[1].map((option, optionIndex) => <label key={option} className={answers[index] === optionIndex ? "answer selected" : "answer"}><input type="radio" name={`question-${index}`} checked={answers[index] === optionIndex} onChange={() => setAnswers((old) => ({ ...old, [index]: optionIndex }))} />{option}</label>)}</div>{submitted && <Alert type={answers[index] === item[2] ? "success" : "error"}>{answers[index] === item[2] ? "Correct." : `Correct answer: ${item[1][item[2]]}.`} {item[3]}</Alert>}<div className="quiz-actions"><Button variant="secondary" icon={ChevronLeft} disabled={index === 0} onClick={() => setIndex(index - 1)}>Previous</Button>{index < questions.length - 1 ? <Button icon={ChevronRight} onClick={() => { setSubmitted(false); setIndex(index + 1); }}>Next</Button> : <Button onClick={submit}>Submit quiz</Button>}</div>{score !== null && <div className="score">Score: {score} / {questions.length}</div>}</Card>{run && <Alert>Your current test R² is {run.testMetrics.r2.toFixed(3)}. Use it when interpreting your model.</Alert>}</>; }

function References() { const refs = [["Scikit-learn LinearRegression", "Reference for ordinary least-squares regression.", "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"], ["Diabetes dataset", "Dataset description and loading documentation.", "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html"], ["An Introduction to Statistical Learning", "Accessible textbook on statistical learning.", "https://www.statlearning.com/"], ["Pandas", "DataFrame and data preparation reference.", "https://pandas.pydata.org/docs/"], ["Recharts", "Chart component documentation used by the lab.", "https://recharts.org/en-US/"]]; return <><PageHeader title="References" subtitle="Reliable documentation and textbooks for further study." /><div className="reference-grid">{refs.map(([title, description, url]) => <Card title={title} key={title}><p>{description}</p><a className="external" href={url} target="_blank" rel="noreferrer">Open reference <ChevronRight size={15} /></a></Card>)}</div></>; }
function HistoryPage() { const [records, setRecords] = useState(loadHistory()); return <><PageHeader title="Experiment History" subtitle="Review saved browser-local experiments and compare measured performance." />{records.length === 0 ? <EmptyState message="No saved experiments yet. Train and save a run from Simulation." /> : <Card><div className="table-wrap"><table><thead><tr><th>Timestamp</th><th>Dataset</th><th>Target</th><th>Features</th><th>Method</th><th>Split</th><th>Test R²</th><th>Test RMSE</th><th>Action</th></tr></thead><tbody>{records.map((record, index) => <tr key={`${record.timestamp}-${index}`}><td>{record.timestamp}</td><td>{record.dataset}</td><td>{record.target}</td><td>{record.features}</td><td>{record.method}</td><td>{record.split || "—"}</td><td>{formatNumber(record.testMetrics?.r2 ?? record.r2)}</td><td>{formatNumber(record.testMetrics?.rmse ?? record.rmse)}</td><td><button className="icon-button" onClick={() => setRecords(deleteHistory(index))} aria-label="Delete experiment"><Trash2 size={16} /></button></td></tr>)}</tbody></table></div><Button variant="secondary" icon={Trash2} onClick={() => setRecords(clearHistory())}>Clear history</Button></Card>}</>; }

export default function App() {
  const [section, navigate] = useSection();
  const datasetState = useDataset();
  const [config, setConfigState] = useState(() => {
    try { return { ...initialConfig, ...JSON.parse(localStorage.getItem("regresslab-config") || "{}") }; } catch { return initialConfig; }
  });
  const [run, setRun] = useState(null);
  const [historyCount, setHistoryCount] = useState(loadHistory().length);
  const setConfig = (updater) => setConfigState((old) => {
    const next = typeof updater === "function" ? updater(old) : updater;
    const changed = ["target", "testPercent", "seed", "standardize"].some((key) => old[key] !== next[key]) || JSON.stringify(old.features) !== JSON.stringify(next.features);
    if (changed) { setRun(null); return { ...next, split: null }; }
    return next;
  });
  useEffect(() => { localStorage.setItem("regresslab-config", JSON.stringify(config)); }, [config]);
  useEffect(() => {
    const available = [...datasetState.dataset.features, datasetState.dataset.target];
    const validFeatures = config.features.filter((feature) => datasetState.dataset.features.includes(feature) && feature !== config.target);
    if (!available.includes(config.target) || validFeatures.length !== config.features.length || config.split) {
      setConfigState((old) => ({ ...old, target: available.includes(old.target) ? old.target : datasetState.dataset.target, features: validFeatures.length ? validFeatures : datasetState.dataset.features.slice(0, 2), split: null }));
      setRun(null);
    }
  }, [datasetState.dataset, config.target]);
  const onSplit = () => setConfigState((old) => ({ ...old, split: getSplit(datasetState.dataset, old) }));
  const onSave = (result) => { const next = saveHistory({ timestamp: new Date().toLocaleString(), dataset: datasetState.dataset.name, target: config.target, features: config.features.join(", "), method: "Normal Equation", split: `${result.split.trainY.length}/${result.split.testY.length}`, seed: config.seed, trainMetrics: result.trainMetrics, testMetrics: result.testMetrics, coefficients: result.model.coefficients, intercept: result.model.intercept }); setHistoryCount(next.length); };
  const page = section === "aim" ? <Aim navigate={navigate} /> : section === "theory" ? <Theory /> : section === "procedure" ? <Procedure navigate={navigate} /> : section === "data" ? <DataPreparation datasetState={datasetState} config={config} setConfig={setConfig} onSplit={onSplit} navigate={navigate} /> : section === "simulation" ? <Simulation dataset={datasetState.dataset} config={config} setConfig={setConfig} navigate={navigate} run={run} setRun={setRun} onSave={onSave} /> : section === "exercise" ? <Exercise run={run} /> : section === "references" ? <References /> : <HistoryPage />;
  const [drawer, setDrawer] = useState(false);
  return <div className="app-shell"><Sidebar section={section} navigate={navigate} open={drawer} onClose={() => setDrawer(false)} /><div className="main-shell"><TopBar section={section} onMenu={() => setDrawer(true)} /><StatusStrip dataset={datasetState.dataset} split={config.split} historyCount={historyCount} /><main>{page}<PageNavigation section={section} navigate={navigate} /></main><BackToTop /></div>{drawer && <button className="drawer-backdrop" onClick={() => setDrawer(false)} aria-label="Close navigation" />}</div>;
}
