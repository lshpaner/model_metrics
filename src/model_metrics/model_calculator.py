import pandas as pd
import numpy as np
from tqdm import tqdm
import shap


class ModelCalculator:
    def __init__(self, model_dict, outcomes, top_n=3):
        """
        Initialize the ModelCalculator class.

        Parameters:
        -----------
        model_dict : dict
            A dictionary where keys are outcomes and values are trained models.

        outcomes : list
            A list of outcome names to evaluate.

        top_n : int, optional (default=3)
            The number of top SHAP features to extract for each row.
        """
        self.model_dict = model_dict
        self.outcomes = outcomes
        self.top_n = top_n

    def generate_predictions(
        self,
        X_test,
        y_test,
        calculate_shap=False,
        use_coefficients=False,
        include_contributions=False,
        subset_results=False,
        global_shap=False,
        global_coefficients=False,
    ):
        """
        Generate a prediction DataFrame with metrics, SHAP values, and
        coefficients.

        This function calculates predictions, metrics, SHAP values, and
        coefficients for the provided test data. It supports both row-wise and
        global-level computations.

        Parameters:
        -----------
        X_test : pd.DataFrame or dict
            The test dataset. Can be:
            - A single DataFrame used for all outcomes, or
            - A dictionary where keys are outcomes and values are corresponding
              DataFrames.

        y_test : pd.DataFrame
            A DataFrame containing true labels for each outcome as columns.

        calculate_shap : bool, optional (default=False)
            Whether to calculate SHAP values for each row in the dataset.

        use_coefficients : bool, optional (default=False)
            Whether to calculate coefficients for each row in the dataset.

        include_contributions : bool, optional (default=False)
            If True, include the contribution values alongside feature names
            for SHAP or coefficient calculations. For SHAP, returns a dict of
            {feature: shap_value} per row. For coefficients, returns a dict of
            {feature: contribution} per row. In both cases the top-N entries
            by absolute value are retained.

        subset_results : bool, optional (default=False)
            If True, returns only ``TP``, ``FN``, ``FP``, ``TN``, and
            ``y_pred_proba``. Includes top-N SHAP or coefficient features if
            ``calculate_shap`` or ``use_coefficients`` is enabled. Ignored for
            global SHAP or coefficients.

        global_shap : bool, optional (default=False)
            Whether to compute global SHAP values across the entire dataset.
            If enabled, returns a DataFrame with global SHAP values. SHAP
            values are computed in a single batched call rather than row-by-row
            for efficiency.

        global_coefficients : bool, optional (default=False)
            Whether to compute global coefficients for the model. If enabled,
            returns a DataFrame with global coefficients.

        Returns:
        --------
        pd.DataFrame
            - If ``global_shap`` is True: Returns a DataFrame with global SHAP
              values.
            - If ``global_coefficients`` is True: Returns a DataFrame with
              global coefficients.
            - Otherwise: Returns a DataFrame containing row-wise predictions,
              metrics, and top-n features or coefficients.

        Raises:
        -------
        ValueError
            If conflicting parameters are set to True simultaneously:
            - ``calculate_shap`` and ``use_coefficients``
            - ``global_shap`` and ``global_coefficients``
            - ``global_coefficients`` and ``subset_results``
            - ``global_shap`` and ``subset_results``
        """

        # Define conflicting conditions and their error messages
        conflicts = [
            (
                calculate_shap and use_coefficients,
                "Both 'calculate_shap' and 'use_coefficients' cannot be True "
                "simultaneously.",
            ),
            (
                global_coefficients and global_shap,
                "Both 'global_shap' and 'global_coefficients' cannot be True "
                "simultaneously.",
            ),
            (
                global_coefficients and subset_results,
                "Both 'global_coefficients' and 'subset_results' cannot be True "
                "simultaneously.",
            ),
            (
                global_shap and subset_results,
                "Both 'global_shap' and 'subset_results' cannot be True "
                "simultaneously.",
            ),
        ]

        for condition, message in conflicts:
            if condition:
                raise ValueError(message)

        results = []

        for outcome in self.outcomes:
            print(f"Running metrics for {outcome}...")

            # Retrieve the trained model
            model = self.model_dict["model"].get(outcome)
            if model is None:
                print(f"Model for outcome '{outcome}' not found. Skipping...")
                continue
            estimator = self._extract_final_model(model)

            # Extract X_test and y_test for this outcome
            X_test_m = X_test[outcome] if isinstance(X_test, dict) else X_test.copy()
            y_test_m = y_test[outcome].copy()

            # Handle Global Coefficients
            if global_coefficients:
                print("Calculating global coefficients...")
                global_coeff_df = self._calculate_coefficients(
                    estimator, X_test_m, global_coefficients=True
                )
                print(f"Global Coefficients for {outcome}:\n{global_coeff_df}")
                return global_coeff_df

            # Handle Global SHAP Values
            if global_shap:
                print("Calculating global SHAP values...")
                global_shap_df = self._calculate_shap_values(
                    estimator, X_test_m, global_shap=True
                )
                print(f"Global SHAP Values for {outcome}:\n{global_shap_df}")
                return global_shap_df

            # Row-wise predictions
            if hasattr(model, "threshold"):
                threshold = list(model.threshold.values())[0]
                print(f"Using optimal threshold {threshold}")
                y_pred_proba = estimator.predict_proba(X_test_m)[:, 1]
                y_pred = (y_pred_proba > threshold).astype(int)
            else:
                y_pred = estimator.predict(X_test_m)

            try:
                y_pred_proba = estimator.predict_proba(X_test_m)[:, 1]
            except AttributeError:
                y_pred_proba = [0.5] * len(y_pred)

            # Add metrics
            X_test_with_metrics = self._add_metrics(
                X_test_m.copy(), y_test_m, y_pred, y_pred_proba, outcome
            )

            # Per-row SHAP
            if calculate_shap:
                print("Calculating SHAP values per row...")
                shap_values = self._calculate_shap_values(
                    estimator, X_test_m, include_contributions
                )
                X_test_with_metrics[f"top_{self.top_n}_features"] = shap_values

            # Per-row Coefficients
            if use_coefficients:
                print("Calculating coefficients per row...")
                coeff_values = self._calculate_coefficients(
                    estimator,
                    X_test_m,
                    include_contributions=include_contributions,
                )
                if isinstance(coeff_values, pd.DataFrame):
                    X_test_with_metrics[f"top_{self.top_n}_coefficients"] = (
                        coeff_values.apply(lambda row: row.to_dict(), axis=1)
                    )
                else:
                    X_test_with_metrics[f"top_{self.top_n}_coefficients"] = coeff_values

            results.append(X_test_with_metrics)

        # Combine row-wise results
        results_df = pd.concat(results, axis=0)

        # Subset results only for row-wise data
        if subset_results:
            print("Subset results only applies to row-wise outputs.")

            subset_cols = ["TP", "FN", "FP", "TN", "y_pred_proba"]

            if calculate_shap:
                top_features_col = f"top_{self.top_n}_features"
                if top_features_col in results_df.columns:
                    subset_cols.append(top_features_col)
            elif use_coefficients:
                top_coefficients_col = f"top_{self.top_n}_coefficients"
                if top_coefficients_col in results_df.columns:
                    subset_cols.append(top_coefficients_col)

            existing_cols = [col for col in subset_cols if col in results_df.columns]
            results_df = results_df[existing_cols]

        print(f"Shape of results_df: {results_df.shape}")
        return results_df

    def _extract_final_model(self, model):
        """
        Extract the final estimator from a model or pipeline.

        Parameters:
        -----------
        model : object
            The model or pipeline to extract the final estimator from.

        Returns:
        --------
        final_model : object
            The final estimator object.

        Raises:
        -------
        ValueError
            If the model or pipeline structure is unsupported.
        """
        try:
            # Unwrap plain dict wrappers (e.g. {"model": <Model>})
            if isinstance(model, dict):
                if "model" in model:
                    model = model["model"]
                else:
                    raise ValueError(
                        f"Dict model has no 'model' key. Keys found: {list(model.keys())}"
                    )

            if hasattr(model, "steps"):
                final_model = model.steps[-1][1]
                print(
                    f"Pipeline detected. Extracted model: {type(final_model).__name__}"
                )
                return final_model

            if hasattr(model, "estimator"):
                print(
                    f"Wrapped model detected. Extracted estimator: "
                    f"{type(model.estimator).__name__}"
                )
                return model.estimator

            if hasattr(model, "model"):
                print(
                    f"Wrapped model detected. Extracted model: "
                    f"{type(model.model).__name__}"
                )
                return model.model

            if (
                hasattr(model, "predict_proba")
                or hasattr(model, "decision_function")
                or hasattr(model, "predict")
            ):
                print(f"Standalone model detected: {type(model).__name__}")
                return model

            raise ValueError("Unsupported model or pipeline structure.")

        except Exception as e:
            raise ValueError(f"Error extracting final model: {e}")

    def _add_metrics(
        self,
        X_test_m,
        y_test_m,
        y_pred,
        y_pred_proba,
        outcome,
    ):
        """
        Add performance metrics to the DataFrame.

        Parameters:
        -----------
        X_test_m : pd.DataFrame
            The feature matrix.

        y_test_m : pd.Series or pd.DataFrame
            The true labels. Single-column DataFrames are squeezed to a Series
            automatically.

        y_pred : array-like
            The predicted labels.

        y_pred_proba : array-like
            The predicted probabilities.

        outcome : str
            The name of the outcome.

        Returns:
        --------
        X_test_m : pd.DataFrame
            The feature matrix with TP, FN, FP, TN, y_pred_proba, and Outcome
            columns added.
        """
        metrics_df = X_test_m.copy()

        # FIX: use isinstance instead of type() ==
        if isinstance(y_test_m, pd.DataFrame):
            y_test_m = y_test_m.squeeze(axis=1)

        metrics_df["TP"] = ((y_test_m == 1) & (y_pred == 1)).astype(int)
        metrics_df["FN"] = ((y_test_m == 1) & (y_pred == 0)).astype(int)
        metrics_df["FP"] = ((y_test_m == 0) & (y_pred == 1)).astype(int)
        metrics_df["TN"] = ((y_test_m == 0) & (y_pred == 0)).astype(int)
        metrics_df["y_pred_proba"] = y_pred_proba
        metrics_df["Outcome"] = outcome
        return metrics_df

    def _make_predict_proba_wrapper(self, model, feature_names):
        """
        Wrap predict_proba to ensure the input always arrives as a named
        DataFrame. SHAP internally converts data to numpy before calling the
        model function; this wrapper re-attaches the original column names so
        pipeline steps like StandardScaler that were fitted with named data
        never receive an unnamed array.
        """

        def wrapped(X):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=feature_names)
            return model.predict_proba(X)

        return wrapped

    def _get_shap_explainer(self, model, X_transformed):
        """
        Select and initialize the appropriate SHAP explainer based on model
        type, using explicit explainer classes to avoid auto-detection warnings.

        Parameters:
        -----------
        model : object
            The unwrapped final estimator.

        X_transformed : pd.DataFrame
            The numeric feature matrix used as the background dataset.

        Returns:
        --------
        shap.Explainer subclass
            The appropriate SHAP explainer instance.
        """
        # Reject models without predict_proba before attempting any explainer
        if not hasattr(model, "predict_proba"):
            raise ValueError("Unsupported model or pipeline structure.")

        if hasattr(model, "tree_") or hasattr(model, "estimators_"):
            return shap.TreeExplainer(model)

        if hasattr(model, "coef_"):
            return shap.LinearExplainer(model, X_transformed)

        wrapped = self._make_predict_proba_wrapper(model, X_transformed.columns)
        return shap.KernelExplainer(wrapped, X_transformed)

    def _calculate_shap_values(
        self,
        model,
        X_test_m,
        include_contributions=False,
        global_shap=False,
        sample_size=None,
    ):
        """
        Calculate SHAP values for a given model or pipeline.

        Dynamically handles standalone models and pipelines by extracting the
        final estimator. The appropriate SHAP explainer is selected explicitly
        based on model type to avoid auto-detection warnings. Global SHAP
        values are computed in a single batched call for efficiency rather
        than iterating row-by-row.

        Parameters:
        -----------
        model : object
            The trained model or pipeline.

        X_test_m : pd.DataFrame
            The feature matrix.

        include_contributions : bool, optional (default=False)
            If True, row-wise output is a list of dicts mapping each feature
            to its SHAP value (top-N by absolute value). If False, returns a
            list of feature-name lists (top-N by absolute value).

        global_shap : bool, optional (default=False)
            If True, compute a single global SHAP summary across all rows and
            return a sorted DataFrame of feature/SHAP-value pairs.

        sample_size : int or None, optional (default=None)
            If set and ``global_shap`` is True, randomly sample this many
            rows before computing global SHAP values.

        Returns:
        --------
        list or pd.DataFrame
            - Row-wise mode: list (one entry per row) of either feature-name
            lists or feature-to-value dicts depending on
            ``include_contributions``.
            - Global mode: pd.DataFrame with columns
            ``["Feature", "SHAP Value"]`` sorted by absolute SHAP value
            descending.
        """
        if X_test_m.index.name is not None:
            X_test_m = X_test_m.reset_index(drop=True)
        X_test_m = X_test_m.select_dtypes(include=["number"])

        # Unwrap dict/pipeline wrappers before resolving the SHAP callable
        model = self._extract_final_model(model)
        if hasattr(model, "steps"):
            model = model.steps[-1][1]
            print(
                f"Pipeline unwrapped in SHAP. "
                f"Final estimator: {type(model).__name__}"
            )

        if not (hasattr(model, "predict_proba") or hasattr(model, "predict")):
            raise ValueError(
                "Model must implement predict_proba or predict to use SHAP "
                "explainability."
            )

        if global_shap and sample_size is not None and sample_size < len(X_test_m):
            print(f"Sampling {sample_size} rows from X_test_m for global SHAP...")
            X_test_m = X_test_m.sample(n=sample_size, random_state=42)

        X_transformed = X_test_m

        # Select the right explainer explicitly to avoid auto-detection warnings
        explainer = self._get_shap_explainer(model, X_transformed)

        # Input to pass to the explainer — linear models need numpy arrays to
        # avoid sklearn "feature names" warnings at call time
        X_input = X_transformed.values if hasattr(model, "coef_") else X_transformed

        # Global SHAP — single batched call
        if global_shap:
            print("Computing global SHAP values (batched)...")
            try:
                shap_output = explainer(X_input)
                shap_values = (
                    shap_output.values
                    if hasattr(shap_output, "values")
                    else np.array(shap_output)
                )

                if shap_values.ndim == 3:
                    # Multi-class: average over classes then over rows
                    global_shap_values = shap_values.mean(axis=2).mean(axis=0)
                elif shap_values.ndim == 2:
                    global_shap_values = shap_values.mean(axis=0)
                else:
                    raise ValueError(
                        f"Unexpected SHAP values shape: {shap_values.shape}"
                    )

                shap_df = pd.DataFrame(
                    {
                        "Feature": X_test_m.columns,
                        "SHAP Value": global_shap_values,
                    }
                ).sort_values(by="SHAP Value", key=abs, ascending=False)
                return shap_df.reset_index(drop=True)

            except KeyboardInterrupt:
                print(
                    "\nKeyboardInterrupt detected. "
                    "Returning empty global SHAP DataFrame."
                )
                return pd.DataFrame({"Feature": [], "SHAP Value": []})

        # Row-wise SHAP — single batched call
        try:
            shap_output = explainer(X_input)

            if isinstance(shap_output, np.ndarray):
                shap_values = shap_output
            elif hasattr(shap_output, "values"):
                shap_values = shap_output.values
            elif isinstance(shap_output, list):
                shap_values = np.array(shap_output)
            else:
                raise ValueError(f"Unexpected SHAP output type: {type(shap_output)}")

            shap_values = np.array(shap_values)

            if shap_values.ndim not in {2, 3}:
                raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

        except KeyboardInterrupt:
            print(
                "\nKeyboardInterrupt detected. " "Returning results calculated so far."
            )
            return []

        # Build per-row output — consistent between SHAP and coefficient paths
        if include_contributions:
            # Return top-N {feature: shap_value} dicts sorted by abs value
            return [
                dict(
                    sorted(
                        zip(
                            X_test_m.columns,
                            (
                                shap_row.flatten()
                                if hasattr(shap_row, "flatten")
                                else shap_row
                            ),
                        ),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[: self.top_n]
                )
                for shap_row in shap_values
            ]

        # Return top-N feature name lists
        return [
            [
                feature
                for feature, _ in sorted(
                    zip(
                        X_test_m.columns,
                        (
                            shap_row.flatten()
                            if hasattr(shap_row, "flatten")
                            else shap_row
                        ),
                    ),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[: self.top_n]
            ]
            for shap_row in shap_values
        ]

    def _calculate_coefficients(
        self, model, X_test_m, include_contributions=False, global_coefficients=False
    ):
        """
        Calculate coefficients or feature importances for a given model.

        For linear models (those with ``coef_``), returns either a global
        sorted coefficient DataFrame or per-row contribution dicts/lists.
        For tree-based models (those with ``feature_importances_``), always
        returns a global sorted importance DataFrame.

        Parameters:
        -----------
        model : object
            The trained model or pipeline.

        X_test_m : pd.DataFrame
            The feature matrix.

        include_contributions : bool, optional (default=False)
            For linear models in row-wise mode: if True, return a list of
            top-N ``{feature: contribution}`` dicts per row (consistent with
            the SHAP ``include_contributions`` path). If False, return a list
            of top-N feature-name lists.

        global_coefficients : bool, optional (default=False)
            If True, return a sorted DataFrame of global coefficients rather
            than per-row values.

        Returns:
        --------
        list or pd.DataFrame
            - Global mode: pd.DataFrame with columns ``["Feature",
            "Coefficient"]`` or ``["Feature", "Importance"]``.
            - Row-wise mode (linear models): list of top-N dicts or feature
            lists depending on ``include_contributions``.

        Raises:
        -------
        ValueError
            If the model does not support ``coef_`` or
            ``feature_importances_``.
        """
        final_model = self._extract_final_model(model)

        # If the extracted model is still a Pipeline, unwrap to its final step
        if hasattr(final_model, "steps"):
            final_model = final_model.steps[-1][1]
            print(
                f"Pipeline unwrapped in coefficients. "
                f"Final estimator: {type(final_model).__name__}"
            )

        print(final_model)

        if hasattr(final_model, "coef_"):
            coefficients = final_model.coef_
            if coefficients.ndim > 1:
                coefficients = coefficients[0]
            feature_names = getattr(
                final_model, "feature_names_in_", X_test_m.columns[: len(coefficients)]
            )

            if global_coefficients:
                return pd.DataFrame(
                    {"Feature": feature_names, "Coefficient": coefficients}
                ).sort_values(by="Coefficient", key=abs, ascending=False)

            contributions = X_test_m[feature_names].values * coefficients

            if include_contributions:
                return [
                    dict(
                        sorted(
                            zip(feature_names, row),
                            key=lambda x: abs(x[1]),
                            reverse=True,
                        )[: self.top_n]
                    )
                    for row in contributions
                ]

            return [
                [
                    feature
                    for feature, _ in sorted(
                        zip(feature_names, row),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[: self.top_n]
                ]
                for row in contributions
            ]

        if hasattr(final_model, "feature_importances_"):
            feature_importances = final_model.feature_importances_
            feature_names = X_test_m.columns[: len(feature_importances)]
            return pd.DataFrame(
                {"Feature": feature_names, "Importance": feature_importances}
            ).sort_values(by="Importance", key=abs, ascending=False)

        raise ValueError(
            f"Model {type(final_model).__name__} does not support coefficients "
            f"or feature importances."
        )
