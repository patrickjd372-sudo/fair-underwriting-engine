import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def generate_synthetic_underwriting_data(n_samples=10000, random_state=42):
    """
    Simulates an enterprise insurance underwriting dataset with built-in 
    demographic attributes to test for algorithmic bias.
    """
    np.random.seed(random_state)
    
    age = np.random.randint(18, 70, size=n_samples)
    credit_score = np.random.randint(500, 850, size=n_samples)
    annual_income = np.random.normal(60000, 20000, size=n_samples)
    
    # Protected demographic feature (e.g., binary representation for testing disparity)
    demographic_group = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Introduce synthetic bias to mimic real-world historical lending/underwriting disparities
    risk_score_latent = (
        (850 - credit_score) * 0.005 + 
        (70 - age) * 0.02 - 
        (annual_income * 0.00001) + 
        (demographic_group * 0.35) +  # Historical bias artifact injection
        np.random.normal(0, 0.5, size=n_samples)
    )
    
    # Convert to binary default/approval risk label (1 = High Risk/Denial, 0 = Approved)
    default_probability = 1 / (1 + np.exp(-risk_score_latent))
    target = np.random.binomial(1, default_probability)
    
    df = pd.DataFrame({
        'age': age,
        'credit_score': credit_score,
        'annual_income': annual_income,
        'demographic_group': demographic_group,
        'risk_label': target
    })
    return df

def evaluate_demographic_parity(df, predictions, sensitive_col='demographic_group'):
    """
    Evaluates selection rates across demographic groups to detect bias.
    """
    temp_df = df.copy()
    temp_df['prediction'] = predictions
    
    parity_table = temp_df.groupby(sensitive_col)['prediction'].mean().reset_index()
    parity_table.rename(columns={'prediction': 'approval_or_positive_rate'}, inplace=True)
    return parity_table

if __name__ == "__main__":
    print("Initializing Enterprise Underwriting Pipeline & Bias Audit Framework...")
    
    # 1. Load Data
    data = generate_synthetic_underwriting_data(n_samples=15000)
    
    # 2. Define Features & Target
    X = data[['age', 'credit_score', 'annual_income', 'demographic_group']]
    y = data['risk_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Baseline Machine Learning Model (Random Forest Classifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    print("\n--- Model Performance Metrics ---")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
    
    # 4. Audit Fairness and Disparity Across Demographics
    print("\n--- Demographic Parity Audit (Pre-Mitigation) ---")
    X_test_evaluated = X_test.copy()
    disparity_report = evaluate_demographic_parity(X_test_evaluated, preds)
    print(disparity_report)
    
    print("\nPipeline execution complete. Ready for cloud deployment via AWS SageMaker.")
