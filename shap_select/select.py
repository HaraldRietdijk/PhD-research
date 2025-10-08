from __future__ import annotations
from typing import Any, Tuple, List, Dict, Union

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import scipy.stats as stats

# Utilities

def _is_tree_model(model) -> bool:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)):
        return True
    if isinstance(model, AdaBoostClassifier):
        base = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
        return (base is None) or isinstance(base, (DecisionTreeClassifier, DecisionTreeRegressor))
    return False

def _ensure_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cast a numeric, fill NaN with median (or 0 if all NaN).
    """
    Xn = X.apply(pd.to_numeric, errors="coerce")
    med = Xn.median(numeric_only=True)
    # riempi col mediano; se una colonna è tutta NaN, ultimo fallback a zero
    Xn = Xn.fillna(med).fillna(0.0)
    # garantisci float64 per compatibilità con np.isfinite
    return Xn.astype(np.float64)


def _score_function(model):
    """
    Function to pass to SHAP Explainer for non-tree models.
    Look for predict_proba, then decision_function, otherwise predict.
    """
    def f(X: np.ndarray):
        if hasattr(model, "predict_proba"):
            P = model.predict_proba(X)
            if isinstance(P, np.ndarray) and P.ndim == 2 and P.shape[1] == 2:
                return P[:, 1]
            return P
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        return model.predict(X)
    return f


def _make_masker(X: pd.DataFrame, max_background: int = 200):
    """
    Return a masker for SHAP Explainer.
    """
    if len(X) > max_background:
        Xb = X.sample(max_background, random_state=0)
    else:
        Xb = X
    Xb_num = _ensure_numeric_df(Xb)
    return shap.maskers.Independent(Xb_num.values)


def _get_explainer(model, X: pd.DataFrame):
    """
    Tries to return a SHAP Explainer for the model and data.
    Prefers TreeExplainer for tree-based models, otherwise
    falls back to a generic Explainer with a masker and a scoring function.
    """
    from sklearn.ensemble import AdaBoostClassifier
    # AdaBoost: vai diretto su Permutation (TreeExplainer spesso non lo supporta)
    if isinstance(model, AdaBoostClassifier):
        return shap.Explainer(_score_function(model), masker=_make_masker(X), algorithm="permutation")

    if _is_tree_model(model):
        try:
            return shap.TreeExplainer(model, feature_perturbation="interventional")
        except Exception:
            pass  # fallback sotto

    return shap.Explainer(_score_function(model), masker=_make_masker(X), algorithm="permutation")

# Output normalization

def _normalize_shap_output_any(
    sv_or_vals: Union[shap._explanation.Explanation, np.ndarray, List[np.ndarray], List[pd.DataFrame]],
    feature_names: List[str],
    index,
    task: str | None,
    classes: List | None
) -> Union[pd.DataFrame, Dict[Any, pd.DataFrame]]:
    """
    Manages various SHAP output formats and normalizes them to DataFrame(s).
    If binary classification, returns a single DataFrame with the difference
    between the two classes' SHAP values.
    If multi-class, returns a dict of DataFrames, one per class.
    If regression, returns a single DataFrame.
    """
    vals = getattr(sv_or_vals, "values", sv_or_vals)

    # List of DataFrames or ndarrays
    if isinstance(vals, (list, tuple)) and len(vals) > 0 and isinstance(vals[0], (np.ndarray, pd.DataFrame)):
        mats = [v.values if isinstance(v, pd.DataFrame) else v for v in vals]
        all_2d = all((m.ndim == 2 for m in mats))
        if not all_2d:
            try:
                vals = np.stack(mats, axis=2)
            except Exception:
                # take first as fallback
                return pd.DataFrame(mats[0], columns=feature_names, index=index)
        else:
            n_classes = len(mats)
            if (task == "binary") and (n_classes == 2):
                arr = mats[1] - mats[0]
                return pd.DataFrame(arr, columns=feature_names, index=index)
            if classes is None:
                classes = list(range(n_classes))
            out: Dict[Any, pd.DataFrame] = {}
            for i, c in enumerate(classes):
                out[c] = pd.DataFrame(mats[i], columns=feature_names, index=index)
            return out

    # ndarray
    if isinstance(vals, np.ndarray):
        if vals.ndim == 2:
            return pd.DataFrame(vals, columns=feature_names, index=index)
        if vals.ndim == 3:
            n_classes = vals.shape[2]
            if (task == "binary") and (n_classes == 2):
                arr = vals[:, :, 1] - vals[:, :, 0]
                return pd.DataFrame(arr, columns=feature_names, index=index)
            if classes is None:
                classes = list(range(n_classes))
            out: Dict[Any, pd.DataFrame] = {}
            for i, c in enumerate(classes):
                out[c] = pd.DataFrame(vals[:, :, i], columns=feature_names, index=index)
            return out

    # fallback
    try:
        return pd.DataFrame(vals, columns=feature_names, index=index)
    except Exception:
        return pd.DataFrame(np.asarray(vals), columns=feature_names, index=index)


# Creation of SHAP features Modified to fit all models

def create_shap_features(
    model: Any,
    validation_df: pd.DataFrame,
    classes: List | None = None,
    task: str | None = None,
) -> Union[pd.DataFrame, Dict[Any, pd.DataFrame]]:
    # numeric dataframe no changes to original df
    X_num = _ensure_numeric_df(validation_df)

    # explainer + shap values
    explainer = _get_explainer(model, X_num)
    sv = explainer(X_num)

    # normalize output
    return _normalize_shap_output_any(
        sv,
        feature_names=list(validation_df.columns),
        index=validation_df.index,
        task=task,
        classes=classes
    )


# Significance testing

def binary_classifier_significance(
    shap_features: pd.DataFrame, target: pd.Series, alpha: float
) -> pd.DataFrame:

    shap_features_with_constant = sm.add_constant(shap_features)

    alpha_in_loop = alpha
    for _ in range(10):
        try:
            logit_model = sm.Logit(target, shap_features_with_constant)
            result = logit_model.fit_regularized(disp=False, alpha=alpha_in_loop)
            break
        except np.linalg.LinAlgError:
            alpha_in_loop *= 5
        except Exception as ex:
            raise RuntimeError(ex)
    else:
        raise RuntimeError("Logistic regression failed to converge after maximum retries.")

    sf = result.summary2().tables[1]
    df = pd.DataFrame(
        {
            "feature name": sf.index,
            "coefficient": sf["Coef."],
            "stderr": sf["Std.Err."],
            "stat.significance": sf["P>|z|"],
            "t-value": sf["Coef."] / sf["Std.Err."],
        }
    ).reset_index(drop=True)
    df["closeness to 1.0"] = (df["coefficient"] - 1.0).abs()
    return df.loc[df["feature name"] != "const", :]


def multi_classifier_significance(
    shap_features: Dict[Any, pd.DataFrame],
    target: pd.Series,
    alpha: float,
    return_individual_significances: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[pd.DataFrame]]]:
    sig_dfs = []
    for cls, feature_df in shap_features.items():
        bin_target = (target == cls).astype(int)
        sig_df = binary_classifier_significance(feature_df, bin_target, alpha)
        sig_dfs.append(sig_df)

    combined = pd.concat(sig_dfs)
    max_sig = (
        combined.groupby("feature name", as_index=False)
        .agg({"t-value": "max", "closeness to 1.0": "min", "coefficient": "max"})
        .reset_index(drop=True)
    )
    # Correction for number of classes
    max_sig["stat.significance"] = max_sig["t-value"].apply(
        lambda x: len(shap_features) * (1 - stats.norm.cdf(x))
    )
    if return_individual_significances:
        return max_sig, sig_dfs
    return max_sig


def regression_significance(
    shap_features: pd.DataFrame, target: pd.Series, alpha: float
) -> pd.DataFrame:
    ols_model = sm.OLS(target, shap_features)
    result = ols_model.fit_regularized(alpha=alpha, refit=True)
    sf = result.summary2().tables[1]
    df = pd.DataFrame(
        {
            "feature name": sf.index,
            "coefficient": sf["Coef."],
            "stderr": sf["Std.Err."],
            "stat.significance": sf["P>|t|"],
            "t-value": sf["Coef."] / sf["Std.Err."],
        }
    ).reset_index(drop=True)
    df["closeness to 1.0"] = (df["coefficient"] - 1.0).abs()
    return df

# shap_features to significance

def shap_features_to_significance(
    shap_features: Union[pd.DataFrame, Dict[Any, pd.DataFrame], List[pd.DataFrame]],
    target: pd.Series,
    task: str,
    alpha: float,
) -> pd.DataFrame:
    # Defensive patch for binary: ensure ALWAYS a 2D DataFrame
    if task == "binary":
        if isinstance(shap_features, (list, tuple)) and len(shap_features) > 0:
            if len(shap_features) == 2:
                a = shap_features[0].values if isinstance(shap_features[0], pd.DataFrame) else np.asarray(shap_features[0])
                b = shap_features[1].values if isinstance(shap_features[1], pd.DataFrame) else np.asarray(shap_features[1])
                shap_features = pd.DataFrame(b - a, index=target.index)
            else:
                first = shap_features[0]
                shap_features = first if isinstance(first, pd.DataFrame) else pd.DataFrame(first, index=target.index)
        if isinstance(shap_features, dict):
            vals = list(shap_features.values())
            if len(vals) == 2:
                shap_features = vals[1] - vals[0]
            else:
                shap_features = vals[0]
            if not isinstance(shap_features, pd.DataFrame):
                shap_features = pd.DataFrame(shap_features, index=target.index)

    if task == "regression":
        out = regression_significance(shap_features, target, alpha)
    elif task == "binary":
        out = binary_classifier_significance(shap_features, target, alpha)
    elif task == "multiclass":
        out = multi_classifier_significance(shap_features, target, alpha)
    else:
        raise ValueError("`task` must be 'regression', 'binary', or 'multiclass'.")

    return out.sort_values(by="t-value", ascending=False).reset_index(drop=True)


# Iterative SHAP feature reduction

def iterative_shap_feature_reduction(
    shap_features: Union[pd.DataFrame, Dict[Any, pd.DataFrame]],
    target: pd.Series,
    task: str,
    alpha: float = 1e-6,
) -> pd.DataFrame:
    collected_rows: List[Dict[str, Any]] = []
    current = shap_features

    while True:
        sig_df = shap_features_to_significance(current, target, task, alpha)

        if sig_df["t-value"].isna().all():
            collected_rows.extend(sig_df.to_dict("records"))
            break

        min_row = sig_df.loc[sig_df["t-value"].idxmin()]
        collected_rows.append(min_row.to_dict())

        feat = min_row["feature name"]
        if isinstance(current, pd.DataFrame):
            current = current.drop(columns=[feat], errors="ignore")
            if current.shape[1] == 0:
                break
        else:
            current = {k: v.drop(columns=[feat], errors="ignore") for k, v in current.items()}
            any_df = next(iter(current.values()))
            if any_df.shape[1] == 0:
                break

    result_df = pd.DataFrame(collected_rows).sort_values(by="t-value", ascending=False).reset_index(drop=True)
    return result_df



# Main API: shap_select

def shap_select(
    tree_model: Any,
    validation_df: pd.DataFrame,
    target: Union[pd.Series, str],
    feature_names: List[str] | None = None,
    task: str | None = None,
    threshold: float = 0.05,
    return_extended_data: bool = False,
    alpha: float = 1e-6,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.DataFrame, Dict[Any, pd.DataFrame]]]]:
    # Target handling
    if isinstance(target, str):
        target = validation_df[target]

    if feature_names is None:
        feature_names = list(validation_df.columns)

    # Infer task
    if task is None:
        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 10:
            task = "regression"
        elif target.nunique() == 2:
            task = "binary"
        else:
            task = "multiclass"

    # Compute SHAP features 
    if task == "multiclass":
        classes = sorted(list(target.unique()))
        shap_features = create_shap_features(tree_model, validation_df[feature_names], classes=classes, task=task)
    else:
        shap_features = create_shap_features(tree_model, validation_df[feature_names], classes=None, task=task)

    # Significance + ablation
    significance_df = iterative_shap_feature_reduction(shap_features, target, task, alpha)

    # Selection based on p-value threshold "stat.significance" (and negative sign on t-value)
    significance_df["selected"] = (significance_df["stat.significance"] < threshold).astype(int)
    significance_df.loc[significance_df["t-value"] < 0, "selected"] = -1

    if return_extended_data:
        return significance_df, shap_features

    return significance_df[["feature name", "t-value", "stat.significance", "coefficient", "selected"]]
