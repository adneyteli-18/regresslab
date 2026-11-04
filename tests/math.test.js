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
});
