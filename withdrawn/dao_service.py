from .dao import DAO

def predict(input_smiles):
    model = DAO()
    predicted_class, withdrawn_probability, pvalue_approved, pvalue_withdrawn = model.predict(input_smiles)
    return predicted_class, withdrawn_probability, pvalue_approved, pvalue_withdrawn


def complementary_model(input_smiles):
    model = DAO()
    complementary_prediction, complementary_tasks, shap_values = model.complementary_model(input_smiles)
    return complementary_prediction, complementary_tasks, shap_values


def explain(input_smiles):
    model = DAO()
    bond_importance, feature_importance = model.explain(input_smiles)
    return bond_importance, feature_importance
