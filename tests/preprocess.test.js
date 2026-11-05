import { describe, expect, it } from "vitest";
import { fitPreprocessor, validateRows } from "../src/lib/preprocess";

describe("student preprocessing", () => {
  const rows = [
    { age: 15, internet: "yes", G3: 10 },
    { age: 17, internet: "no", G3: 12 },
    { age: 16, internet: "yes", G3: 14 },
  ];

  it("detects numeric and categorical predictors and encodes categories", () => {
    const preprocessor = fitPreprocessor(rows, ["age", "internet"]);
    expect(preprocessor.labels).toEqual(["age", "internet=yes"]);
    expect(preprocessor.transform(rows, false)).toEqual([[15, 1], [17, 0], [16, 1]]);
  });

  it("learns numeric scaling from training rows only", () => {
    const preprocessor = fitPreprocessor(rows.slice(0, 2), ["age"]);
    expect(preprocessor.transform([{ age: 15 }], true)[0][0]).toBeCloseTo(-1);
    expect(preprocessor.transform([{ age: 100 }], true)[0][0]).toBeGreaterThan(1);
  });

  it("rejects incomplete target or predictor rows", () => {
    expect(validateRows([...rows, { age: "", internet: "yes", G3: 9 }], ["age", "internet"], "G3")).toHaveLength(3);
  });
});
