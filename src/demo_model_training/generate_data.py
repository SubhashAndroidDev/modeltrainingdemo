import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000

data = pd.DataFrame({
    "age": np.random.randint(21, 61, N),

    "monthly_income": np.random.randint(
        18000, 250001, N
    ),

    "employment_years": np.random.randint(
        0, 21, N
    ),

    "credit_score": np.random.randint(
        550, 851, N
    ),

    "existing_emi": np.random.randint(
        0, 50001, N
    ),

    "loan_amount": np.random.randint(
        50000, 2000001, N
    ),

    "loan_term_months": np.random.choice(
        [12, 24, 36, 48, 60, 72, 84],
        N
    ),

    "employment_type": np.random.choice(
        ["Salaried", "Self-employed"],
        N
    ),

    "city_tier": np.random.choice(
        ["Tier 1", "Tier 2", "Tier 3"],
        N
    ),
})


# --------------------------------------------------
# Create a synthetic approval pattern
# --------------------------------------------------

score = (
    (data["credit_score"] - 650) * 0.025
    + (data["monthly_income"] / 50000)
    + (data["employment_years"] * 0.08)
    - (data["existing_emi"] / 30000)
    - (data["loan_amount"] / 1000000)
)

# Employment advantage
score += np.where(
    data["employment_type"] == "Salaried",
    0.5,
    0
)

# Convert score into probability
probability = 1 / (1 + np.exp(-score))

data["loan_approved"] = (
    np.random.random(N) < probability
).astype(int)


# Save
data.to_csv(
    "data/loan_data.csv",
    index=False
)

print("Dataset created successfully!")
print(data.head())
print("\nShape:", data.shape)
print("\nApproval distribution:")
print(data["loan_approved"].value_counts())