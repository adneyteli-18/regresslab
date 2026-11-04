const KEY = "regresslab-history";
export function loadHistory() {
  try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch { return []; }
}
export function saveHistory(record) {
  const next = [...loadHistory(), record];
  localStorage.setItem(KEY, JSON.stringify(next));
  return next;
}
export function deleteHistory(index) {
  const next = loadHistory().filter((_, itemIndex) => itemIndex !== index);
  localStorage.setItem(KEY, JSON.stringify(next));
  return next;
}
export function clearHistory() { localStorage.removeItem(KEY); return []; }
