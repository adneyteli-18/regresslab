export function mean(values) {
  if (!values.length) return NaN;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function transpose(matrix) {
  if (!matrix.length) return [];
  return matrix[0].map((_, column) => matrix.map((row) => row[column]));
}

export function predict(rows, model) {
  return rows.map((row) => model.intercept + row.reduce((sum, value, index) => sum + value * model.coefficients[index], 0));
}

function assertFiniteMatrix(rows, targets) {
  if (!Array.isArray(rows) || !rows.length || !Array.isArray(rows[0]) || !targets.length || rows.length !== targets.length) {
    throw new Error("Training data must contain matching non-empty feature and target rows.");
  }
  const width = rows[0].length;
  if (!width || rows.some((row) => row.length !== width || row.some((value) => !Number.isFinite(value))) || targets.some((value) => !Number.isFinite(value))) {
    throw new Error("Training data contains missing or non-numeric values.");
  }
  if (rows.length <= width) throw new Error("At least one more observation than feature is required.");
}

function qrSolve(matrix, vector, tolerance = 1e-10) {
  const rowCount = matrix.length;
  const columnCount = matrix[0].length;
  const q = Array.from({ length: columnCount }, () => Array(rowCount).fill(0));
  const r = Array.from({ length: columnCount }, () => Array(columnCount).fill(0));
  for (let column = 0; column < columnCount; column += 1) {
    const vectorColumn = matrix.map((row) => row[column]);
    for (let previous = 0; previous < column; previous += 1) {
      r[previous][column] = q[previous].reduce((sum, value, index) => sum + value * vectorColumn[index], 0);
      for (let row = 0; row < rowCount; row += 1) vectorColumn[row] -= r[previous][column] * q[previous][row];
    }
    const norm = Math.sqrt(vectorColumn.reduce((sum, value) => sum + value * value, 0));
    if (!Number.isFinite(norm) || norm <= tolerance) throw new Error("The selected features are constant or linearly dependent.");
    r[column][column] = norm;
    q[column] = vectorColumn.map((value) => value / norm);
  }
  const qtY = q.map((basis) => basis.reduce((sum, value, index) => sum + value * vector[index], 0));
  const result = Array(columnCount).fill(0);
  for (let row = columnCount - 1; row >= 0; row -= 1) {
    result[row] = (qtY[row] - r[row].slice(row + 1).reduce((sum, value, index) => sum + value * result[row + index + 1], 0)) / r[row][row];
  }
  return result;
}

export function olsFit(rows, targets, fitIntercept = true) {
  assertFiniteMatrix(rows, targets);
  const design = fitIntercept ? rows.map((row) => [1, ...row]) : rows;
  if (design.length <= design[0].length) throw new Error("Insufficient observations for the selected number of parameters.");
  const coefficients = qrSolve(design, targets);
  return { intercept: fitIntercept ? coefficients[0] : 0, coefficients: fitIntercept ? coefficients.slice(1) : coefficients };
}

export function metrics(actual, predicted, featureCount = 0) {
  if (!actual.length || actual.length !== predicted.length || actual.some((value) => !Number.isFinite(value)) || predicted.some((value) => !Number.isFinite(value))) {
    throw new Error("Metrics require matching finite actual and predicted values.");
  }
  const errors = actual.map((value, index) => value - predicted[index]);
  const mse = mean(errors.map((error) => error ** 2));
  const mae = mean(errors.map((error) => Math.abs(error)));
  const targetMean = mean(actual);
  const total = actual.reduce((sum, value) => sum + (value - targetMean) ** 2, 0);
  const residual = errors.reduce((sum, error) => sum + error ** 2, 0);
  const r2 = total === 0 ? (residual === 0 ? 1 : NaN) : 1 - residual / total;
  const n = actual.length;
  const adjustedR2 = n - featureCount - 1 > 0 && Number.isFinite(r2) ? 1 - (1 - r2) * (n - 1) / (n - featureCount - 1) : NaN;
  return { r2, adjustedR2, mse, rmse: Math.sqrt(mse), mae };
}

export function standardize(train, test) {
  if (!train.length || !test) throw new Error("Training and testing rows are required for standardization.");
  const means = train[0].map((_, column) => mean(train.map((row) => row[column])));
  const stds = means.map((average, column) => Math.sqrt(mean(train.map((row) => (row[column] - average) ** 2))));
  const transform = (rows) => rows.map((row) => row.map((value, column) => (value - means[column]) / (stds[column] || 1)));
  return { train: transform(train), test: transform(test), means, stds: stds.map((value) => value || 1) };
}
