// Types + fetch helpers for the artifacts written by scripts/export_web_artifacts.py.

export type Method = {
  k: number;
  labels: number[] | null;
  groups: number[] | null;
  folds: number[][]; // [fold][sample] -> 0 unused, 1 train, 2 test, 3 purge
  blurb: string;
};

export type FoldsData = {
  n: number;
  legend: { unused: number; train: number; test: number; purge: number };
  methods: Record<string, Method>;
};

export type Headline = {
  leakage_noise: { leaky: number; correct: number; n_features: number; n_rows: number; note: string };
  group_leak: {
    random_r2: number; group_r2: number; random_mae: number; group_mae: number;
    baseline_mae: number; note: string;
  };
  time_leak: { shuffled_r2: number; timeseries_r2: number; note: string };
  nested: { non_nested: number; nested: number; nested_std: number; note: string };
};

export async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`failed to load ${path}`);
  return res.json() as Promise<T>;
}
