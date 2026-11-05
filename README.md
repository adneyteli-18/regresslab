# RegressLab — Linear Regression Virtual Lab

RegressLab is a browser-based React and Vite virtual laboratory for learning
and experimenting with Linear Regression. All model fitting, evaluation, and
visualisation run locally in the browser.

Live deployment: **[Add the Vercel URL here]**

## Features

- Academic virtual-lab sections: Aim, Theory, Procedure, Data Preparation,
  Simulation, Exercise, References, and Experiment History.
- Built-in Scikit-learn Diabetes dataset (`src/data/diabetes.json`).
- CSV upload with numeric-column detection.
- Deterministic train/test splitting with optional training-only standardisation.
- Ordinary Least Squares Normal Equation training with a numerically stable QR solver.
- Actual MAE, MSE, RMSE, R², adjusted R², predictions, coefficients, residuals,
  and responsive Recharts visualisations.
- Browser-local experiment history and responsive navigation.
- Vercel SPA rewrite support for direct links to any section.

## Requirements

- Node.js 18 or newer
- npm

## Development

From the project directory:

```powershell
npm.cmd install
npm.cmd run dev
```

Open the local URL printed by Vite.

## Production build and preview

```powershell
npm.cmd test
npm.cmd run build
npm.cmd run preview
```

`npm run preview` serves the generated `dist/` build locally so production
assets and built-in dataset loading can be checked before deployment.

## Vercel deployment

Import this project into Vercel. Vercel detects Vite automatically:

- Build command: `npm run build`
- Output directory: `dist`
- Install command: `npm install`

The included `vercel.json` rewrites application routes to `index.html`, which
keeps deep links such as `/?section=simulation` working after deployment.

The original Streamlit implementation remains in `legacy-streamlit/` for
reference and is not part of the Vercel build.
