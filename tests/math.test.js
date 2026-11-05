import { describe, expect, it } from "vitest";
import { metrics, olsFit, predict } from "../src/lib/math";

describe("regression math", () => {
  it("solves a simple linear relationship with OLS", () => {
    const model = olsFit([[1], [2], [3], [4]], [3, 5, 7, 9]);
    expect(model.intercept).toBeCloseTo(1, 8);
    expect(model.coefficients[0]).toBeCloseTo(2, 8);
    expect(predict([[5]], model)[0]).toBeCloseTo(11, 8);
  });

  it("computes perfect-fit metrics", () => {
    expect(metrics([1, 2, 3], [1, 2, 3], 1)).toMatchObject({ r2: 1, mse: 0, rmse: 0, mae: 0 });
  });

  it("solves multiple regression without regularisation", () => {
    const model = olsFit([[1, 2], [2, 1], [3, 4], [4, 3], [5, 6]], [8, 7, 14, 13, 20]);
    expect(model.intercept).toBeCloseTo(3, 8);
    expect(model.coefficients[0]).toBeCloseTo(1, 8);
    expect(model.coefficients[1]).toBeCloseTo(2, 8);
  });

  it("rejects constant and singular features", () => {
    expect(() => olsFit([[1], [1], [1], [1]], [1, 2, 3, 4])).toThrow(/constant|dependent/i);
    expect(() => olsFit([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], [3, 5, 7, 9, 11])).toThrow(/constant|dependent/i);
  });

  it("preserves negative R-squared and handles a constant target", () => {
    expect(metrics([1, 2, 3], [5, 5, 5], 1).r2).toBeLessThan(0);
    expect(metrics([2, 2, 2], [2, 2, 2], 1).r2).toBe(1);
  });
});
