export function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function transpose(matrix) {
  return matrix[0].map((_, column) => matrix.map((row) => row[column]));
}

export function multiply(left, right) {
  return left.map((row) => right[0].map((_, column) => row.reduce((sum, value, index) => sum + value * right[index][column], 0)));
}

export function solveLinearSystem(matrix, vector) {
  return gaussianSolve(matrix, vector);
}

export function olsFit(rows, targets, fitIntercept = true) {
  const design = fitIntercept ? rows.map((row) => [1, ...row]) : rows;
  const xt = transpose(design);
  const xtx = multiply(xt, design);
  const xty = multiply(xt, targets.map((value) => [value])).map((row) => row[0]);
  let coefficients;
  try {
    coefficients = gaussianSolve(xtx, xty);
  } catch {
    const ridge = xtx.map((row, i) => row.map((value, j) => value + (i === j ? 1e-8 : 0)));
    coefficients = gaussianSolve(ridge, xty);
  }
  return { intercept: fitIntercept ? coefficients[0] : 0, coefficients: fitIntercept ? coefficients.slice(1) : coefficients };
}

export function gaussianSolve(matrix, vector) {
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < a.length; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < a.length; row += 1) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    if (Math.abs(a[pivot][col]) < 1e-12) throw new Error("Singular matrix");
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let row = col + 1; row < a.length; row += 1) {
      const factor = a[row][col] / a[col][col];
      for (let j = col; j <= a.length; j += 1) a[row][j] -= factor * a[col][j];
    }
  }
  const result = Array(a.length).fill(0);
  for (let row = a.length - 1; row >= 0; row -= 1) {
    result[row] = (a[row][a.length] - a[row].slice(row + 1, a.length).reduce((sum, value, index) => sum + value * result[row + index + 1], 0)) / a[row][row];
  }
  return result;
}

export function predict(rows, model) {
  return rows.map((row) => model.intercept + row.reduce((sum, value, index) => sum + value * model.coefficients[index], 0));
}

export function metrics(actual, predicted, featureCount) {
  const errors = actual.map((value, index) => value - predicted[index]);
  const mse = mean(errors.map((error) => error ** 2));
  const mae = mean(errors.map((error) => Math.abs(error)));
  const targetMean = mean(actual);
  const total = actual.reduce((sum, value) => sum + (value - targetMean) ** 2, 0);
  const residual = errors.reduce((sum, error) => sum + error ** 2, 0);
  const r2 = total === 0 ? 0 : 1 - residual / total;
  const n = actual.length;
  const adjusted = n - featureCount - 1 > 0 ? 1 - (1 - r2) * (n - 1) / (n - featureCount - 1) : r2;
  return { r2, adjustedR2: adjusted, mse, rmse: Math.sqrt(mse), mae };
}

export function standardize(train, test) {
  const means = train[0].map((_, column) => mean(train.map((row) => row[column])));
  const stds = means.map((average, column) => Math.sqrt(mean(train.map((row) => (row[column] - average) ** 2))) || 1);
  const transform = (rows) => rows.map((row) => row.map((value, column) => (value - means[column]) / stds[column]));
  return { train: transform(train), test: transform(test), means, stds };
}

export function gradientDescent(rows, targets, rate = 0.05, epochs = 500) {
  const design = rows.map((row) => [1, ...row]);
  const weights = Array(design[0].length).fill(0);
  const loss = [];
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const predictions = design.map((row) => row.reduce((sum, value, index) => sum + value * weights[index], 0));
    const errors = predictions.map((value, index) => value - targets[index]);
    loss.push(mean(errors.map((error) => error ** 2)));
    for (let column = 0; column < weights.length; column += 1) weights[column] -= rate * 2 / rows.length * design.reduce((sum, row, index) => sum + errors[index] * row[column], 0);
  }
  return { intercept: weights[0], coefficients: weights.slice(1), loss };
}
