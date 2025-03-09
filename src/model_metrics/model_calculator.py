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
        global_shap=False,  # Global SHAP extraction
        global_coefficients=False,  # Global coefficient extraction
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
            for SHAP or coefficient calculations.

        subset_results : bool, optional (default=False)
            If True, returns only `TP`, `FN`, `FP`, `TN`, and `y_pred_proba`.
            Includes top-N SHAP or coefficient features if `calculate_shap` or
            `use_coefficients` is enabled. Ignored for global SHAP or coefficients.

        global_shap : bool, optional (default=False)
            Whether to compute global SHAP values across the entire dataset.
            If enabled, returns a DataFrame with global SHAP values.

        global_coefficients : bool, optional (default=False)
            Whether to compute global coefficients for the model. If enabled,
            returns a DataFrame with global coefficients.

        Returns:
        --------
        pd.DataFrame
            - If `global_shap` is True: Returns a DataFrame with global SHAP values.
            - If `global_coefficients` is True: Returns a DataFrame with global
              coefficients.
            - Otherwise: Returns a DataFrame containing row-wise predictions,
              metrics, and top-n features or coefficients.

        Raises:
        -------
        ValueError
            If conflicting parameters are set to True simultaneously:
            - `calculate_shap` and `use_coefficients`
            - `global_shap` and `global_coefficients`
            - `global_coefficients` and `subset_results`
            - `global_shap` and `subset_results`

        Notes:
        ------
        - When `global_shap` or `global_coefficients` is enabled, the computation
          immediately stops after generating the corresponding global-level output,
          and no row-wise computations are performed.
        - The `subset_results` flag only applies to row-wise outputs and has no
          effect when `global_shap`
          or `global_coefficients` is enabled.
        """

        ## Define conflicting conditions and their error messages
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

        ## Check for conflicts
        for condition, message in conflicts:
            if condition:
                raise ValueError(message)

        results = []

        for outcome in self.outcomes:
            print(f"Running metrics for {outcome}...")

            ## Retrieve the trained model
            model = self.model_dict["model"].get(outcome)
            if model is None:
                print(f"Model for outcome '{outcome}' not found. Skipping...")
                continue

            if hasattr(model, "estimator"):
                estimator = model.estimator
            else:
                estimator = model

            ## Extract X_test and y_test
            X_test_m = X_test[outcome] if isinstance(X_test, dict) else X_test.copy()
            y_test_m = y_test[outcome].copy()

            ## Handle Global Coefficients
            if global_coefficients:
                print("Calculating global coefficients...")
                global_coeff_df = self._calculate_coefficients(
                    estimator, X_test_m, global_coefficients=True
                )
                print(f"Global Coefficients for {outcome}:\n{global_coeff_df}")
                return global_coeff_df  # Immediately return the global
                # coefficients DataFrame

            ## Handle Global SHAP Values
            if global_shap:
                print("Calculating global SHAP values...")
                global_shap_df = self._calculate_shap_values(
                    estimator, X_test_m, global_shap=True
                )
                print(f"Global SHAP Values for {outcome}:\n{global_shap_df}")
                return global_shap_df  # Immediately return the global SHAP
                # DataFrame

            ## Row-wise Calculations
            if hasattr(model, "threshold"):
                threshold = list(model.threshold.values())[0]
                print(f"Using optimal threshold {threshold}")
                y_pred_proba = model.predict_proba(X_test_m)[:, 1]
                y_pred = 1 * (y_pred_proba > threshold)
            else:
                y_pred = model.predict(X_test_m)

            try:
                y_pred_proba = model.predict_proba(X_test_m)[:, 1]
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
                    # Convert DataFrame to a list of tuples (or dicts) for each row
                    X_test_with_metrics[f"top_{self.top_n}_coefficients"] = (
                        coeff_values.apply(lambda row: row.to_dict(), axis=1)
                    )
                else:
                    X_test_with_metrics[f"top_{self.top_n}_coefficients"] = coeff_values

            results.append(X_test_with_metrics)

        ## Combine row-wise results
        results_df = pd.concat(results, axis=0)

        ## Subset results only for row-wise data
        if subset_results:
            print("Subset results only applies to row-wise outputs.")

            # Default subset columns
            subset_cols = ["TP", "FN", "FP", "TN", "y_pred_proba"]

            # Only add SHAP or coefficient columns if they were actually computed
            if calculate_shap:
                top_features_col = f"top_{self.top_n}_features"
                if top_features_col in results_df.columns:
                    subset_cols.append(top_features_col)

            elif use_coefficients:
                top_coefficients_col = f"top_{self.top_n}_coefficients"
                if top_coefficients_col in results_df.columns:
                    subset_cols.append(top_coefficients_col)

            # Select only the columns that exist
            existing_cols = [col for col in subset_cols if col in results_df.columns]
            results_df = results_df[existing_cols]

        print(f"Shape of results_df: {results_df.shape}")
        return results_df

    def _extract_final_model(self, model):
        """
        Extracts the final estimator from a model or pipeline.

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
            # Standalone model

            if hasattr(model, "steps"):
                pipeline = model
                if hasattr(pipeline, "estimator"):
                    final_model = pipeline.estimator
                else:
                    final_model = pipeline.steps[-1][1]  # Access the last step
                print(
                    f"Pipeline detected. Extracted model: {type(final_model).__name__}"
                )
                return final_model
            elif (
                hasattr(model, "predict_proba")
                or hasattr(model, "decision_function")
                or hasattr(model, "predict")
            ):
                print("Standalone model detected.")
                return model
            # Pipeline logic
            # Custom handling for non-callable models
            elif hasattr(model, "model"):
                # Handle wrapped models
                return model.model
            else:
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

        y_test_m : pd.Series
            The true labels.

        y_pred : array-like
            The predicted labels.

        y_pred_proba : array-like
            The predicted probabilities.

        outcome : str
            The name of the outcome.

        Returns:
        --------
        X_test_m : pd.DataFrame
            The feature matrix with metrics added.
        """
        metrics_df = X_test_m.copy()
        if type(y_test_m) == pd.DataFrame:
            y_test_m = y_test_m.squeeze(axis=0)
        metrics_df["TP"] = ((y_test_m == 1) & (y_pred == 1)).astype(int)
        metrics_df["FN"] = ((y_test_m == 1) & (y_pred == 0)).astype(int)
        metrics_df["FP"] = ((y_test_m == 0) & (y_pred == 1)).astype(int)
        metrics_df["TN"] = ((y_test_m == 0) & (y_pred == 0)).astype(int)
        metrics_df["y_pred_proba"] = y_pred_proba
        metrics_df["Outcome"] = outcome
        return metrics_df

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
        final estimator.
        """
        if X_test_m.index.name is not None:
            X_test_m = X_test_m.reset_index(drop=True)
        X_test_m = X_test_m.select_dtypes(include=["number"])

        if global_shap and sample_size is not None and sample_size < len(X_test_m):
            print(f"Sampling {sample_size} rows from X_test_m for global SHAP...")
            X_test_m = X_test_m.sample(n=sample_size, random_state=42)

        # Dynamically handle pipelines and standalone models
        try:
            if hasattr(model, "predict_proba"):
                # Standalone model
                final_model = model.predict_proba
                X_transformed = X_test_m

            else:
                raise ValueError("Unsupported model or pipeline structure.")
        except Exception as e:
            raise ValueError(f"Error extracting model or transforming data: {e}")

        # Initialize SHAP Explainer
        try:
            explainer = shap.Explainer(final_model, X_transformed)
        except Exception as e:
            print(f"Error initializing SHAP explainer with default method: {e}")
            # Fallback to KernelExplainer
            try:
                explainer = shap.KernelExplainer(
                    final_model.predict_proba, X_transformed
                )
                print("Using KernelExplainer as a fallback.")
            except Exception as kernel_error:
                raise ValueError(
                    f"Unable to initialize SHAP explainer for the provided model. "
                    f"Default error: {e}, KernelExplainer error: {kernel_error}"
                )

        # Global SHAP calculation
        if global_shap:
            print("Computing global SHAP values...")
            shap_values_accumulated = []
            try:
                for row_batch in tqdm(
                    X_transformed.itertuples(index=False, name=None),
                    total=len(X_transformed),
                    desc="Global SHAP Progress",
                ):
                    row = pd.DataFrame([row_batch], columns=X_test_m.columns)
                    shap_values = explainer(row).values
                    shap_values_accumulated.append(shap_values)

                shap_values = np.vstack(shap_values_accumulated)

                if len(shap_values.shape) == 3:  # Multi-class SHAP values
                    global_shap_values = shap_values.mean(axis=0).mean(axis=1)
                elif len(shap_values.shape) == 2:  # Single-class SHAP values
                    global_shap_values = shap_values.mean(axis=0)
                else:
                    raise ValueError(
                        f"Unexpected SHAP values shape: {shap_values.shape}"
                    )

                shap_df = pd.DataFrame(
                    {"Feature": X_test_m.columns, "SHAP Value": global_shap_values}
                ).sort_values(by="SHAP Value", key=abs, ascending=False)
                return shap_df.reset_index(drop=True)

            except KeyboardInterrupt:
                print(
                    "\nKeyboardInterrupt detected! Returning partial global SHAP values..."
                )
                if shap_values_accumulated:
                    shap_values = np.vstack(shap_values_accumulated)
                    global_shap_values = shap_values.mean(axis=0).mean(axis=1)
                    shap_df = pd.DataFrame(
                        {"Feature": X_test_m.columns, "SHAP Value": global_shap_values}
                    ).sort_values(by="SHAP Value", key=abs, ascending=False)
                    return shap_df.reset_index(drop=True)
                else:
                    return pd.DataFrame({"Feature": [], "SHAP Value": []})

        # Row-wise SHAP calculation
        shap_values = []
        try:
            shap_output = explainer(
                pd.DataFrame(X_transformed, columns=X_test_m.columns)
            )

            # Ensure SHAP values are always a NumPy array
            if isinstance(shap_output, np.ndarray):
                shap_values = shap_output  # Use as is if already an array
            elif hasattr(shap_output, "values"):
                shap_values = shap_output.values  # Extract values from SHAP Explanation
            elif isinstance(shap_output, list):
                shap_values = np.array(shap_output)  # Convert list to NumPy array
            else:
                raise ValueError(f"Unexpected SHAP output type: {type(shap_output)}")

            # Convert to NumPy array before checking shape
            shap_values = np.array(shap_values)

            # Now check for unexpected shapes
            if len(shap_values.shape) not in {2, 3}:
                raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Returning results calculated so far.")
            return shap_values

        # Extract top-n SHAP features per row
        top_shap_features = [
            [
                feature
                for feature, shap_value in sorted(
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

        if include_contributions:
            return [
                {feature: value for feature, value in zip(X_test_m.columns, shap_row)}
                for shap_row in shap_values
            ]

        return top_shap_features

    def _calculate_coefficients(
        self, model, X_test_m, include_contributions=False, global_coefficients=False
    ):
        final_model = self._extract_final_model(model)
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
            return [
                {
                    feature: value
                    for feature, value in sorted(
                        zip(feature_names, row), key=lambda x: abs(x[1]), reverse=True
                    )[: self.top_n]
                }
                for row in contributions
            ]

        if hasattr(final_model, "feature_importances_"):
            feature_importances = final_model.feature_importances_
            feature_names = X_test_m.columns[: len(feature_importances)]
            return pd.DataFrame(
                {"Feature": feature_names, "Importance": feature_importances}
            ).sort_values(by="Importance", key=abs, ascending=False)

        raise ValueError(f"Model {type(final_model)} does not support coefficients.")
