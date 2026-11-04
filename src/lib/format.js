export function formatNumber(value, significantDigits = 4) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (number === 0) return "0";
  const absolute = Math.abs(number);
  if (absolute < 1e-4 || absolute >= 1e6) return number.toExponential(significantDigits - 1);
  return new Intl.NumberFormat("en-US", {
    maximumSignificantDigits: significantDigits,
    useGrouping: true,
  }).format(number);
}

export function formatPercent(value, significantDigits = 4) {
  return `${formatNumber(Number(value) * 100, significantDigits)}%`;
}

export function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function downloadCsv(filename, rows) {
  const content = rows.map((row) => row.map(csvCell).join(",")).join("\n");
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
