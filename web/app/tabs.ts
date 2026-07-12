// Single source of truth for the visualizer tabs. Each entry is one concept.
export type CvTab = {
  id: string;
  title: string;
  badge: string;   // shown in the panel chip
  tagline: string;
  help: string;    // shown in the panel's "?" tooltip
};

export const CV_TABS: CvTab[] = [
  {
    id: "explorer", title: "Fold Explorer", badge: "5 splitters",
    tagline: "Every splitter answers the same question — which rows train, which rows test — differently. Pick one and watch the real scikit-learn fold layout, column by column.",
    help: "A demo strip of samples (each column) split into folds (each row). Blue = trained on, red = held out to test. Switch splitters to see how structure — classes, groups, time — changes the split.",
  },
  {
    id: "leakage", title: "Leakage", badge: "German Credit",
    tagline: "One leaky feature-selection step, run on all the data before splitting, manufactures 0.82 accuracy out of features that are 100% random noise.",
    help: "Data leakage is letting the model — or your choices about it — peek at the test rows. Here, selecting features on the full dataset before CV leaks the answer, so pure noise looks predictive.",
  },
  {
    id: "groups", title: "Group Leakage", badge: "Parkinsons",
    tagline: "The same voice model reads as R² 0.91 with a random split and −0.57 once GroupKFold keeps each patient whole — it's actually worse than predicting the mean on unseen patients.",
    help: "When one entity (a patient) has many rows, a random split puts some of their rows in train and some in test, so the model memorizes the person. GroupKFold keeps every entity entirely on one side.",
  },
  {
    id: "time", title: "Time Leakage", badge: "Bike Sharing",
    tagline: "Shuffling a time series lets the model train on the future to predict the past — a free +0.15 R² that evaporates the moment TimeSeriesSplit only ever trains on earlier hours.",
    help: "Time-ordered data must never be shuffled: a fair test only uses the past to predict the future. Shuffled KFold leaks future rows into training and inflates the score.",
  },
  {
    id: "nested", title: "Nested CV", badge: "German Credit",
    tagline: "The best_score_ you tuned to is a selection score, not a generalization estimate. Nested CV wraps tuning inside an outer loop to report what the model actually does on fresh data.",
    help: "Tuning many configurations and reporting the best CV score is optimistic — you picked the luckiest one. Nested CV re-tunes inside each outer fold, so the reported score never saw the choice that produced it.",
  },
  {
    id: "oof", title: "Out-of-Fold", badge: "German Credit",
    tagline: "cross_val_predict gives every row one prediction — made while it sat in the held-out fold. Stitch them together and you can build an honest ROC curve and confusion matrix with no separate test set.",
    help: "Out-of-fold prediction assigns each row the prediction from the fold where it was held out. Concatenated, these let you evaluate the whole dataset at once — the leak-free way to draw ROC / PR curves and confusion matrices.",
  },
  {
    id: "about", title: "About", badge: "the series",
    tagline: "What cross-validation is, what each notebook proves, and the golden rules underneath all of it.",
    help: "A plain-language guide to the four notebooks and the ideas this visualizer illustrates.",
  },
];
