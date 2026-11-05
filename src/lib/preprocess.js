function isNumeric(value) {
  return value !== "" && value !== null && value !== undefined && Number.isFinite(Number(value));
}

export function fitPreprocessor(rows, features) {
  const definitions = features.map((name) => {
    const values = rows.map((row) => row[name]);
    const numeric = values.every(isNumeric);
    if (numeric) {
      const mean = values.reduce((sum, value) => sum + Number(value), 0) / values.length;
      const std = Math.sqrt(values.reduce((sum, value) => sum + (Number(value) - mean) ** 2, 0) / values.length) || 1;
      return { name, type: "numeric", mean, std, labels: [name] };
    }
    const categories = [...new Set(values.map((value) => String(value)))].sort();
    return { name, type: "categorical", categories, labels: categories.slice(1).map((category) => `${name}=${category}`) };
  });
  const labels = definitions.flatMap((definition) => definition.labels);
  const transform = (inputRows, standardize = true) => inputRows.map((row) => definitions.flatMap((definition) => {
    if (definition.type === "numeric") {
      const value = Number(row[definition.name]);
      return [standardize ? (value - definition.mean) / definition.std : value];
    }
    return definition.categories.slice(1).map((category) => String(row[definition.name]) === category ? 1 : 0);
  }));
  return { definitions, labels, transform };
}

export function validateRows(rows, features, target) {
  return rows.filter((row) => features.every((feature) => row[feature] !== undefined && row[feature] !== "") && isNumeric(row[target]));
}
