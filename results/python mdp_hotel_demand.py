import pandas as pd
import numpy as np


# ---------------------------------------------------------
# 1. Load Hotel Demand Dataset
# ---------------------------------------------------------
DATA_PATH = "hotel_bookings.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)


# ---------------------------------------------------------
# 2. Define Demand States
# ---------------------------------------------------------
# ADR (Average Daily Rate) and number of guests are used
# to create simple demand categories.

if "adr" not in df.columns:
    raise ValueError("Column 'adr' is required in the dataset.")

df["total_guests"] = (
    df["adults"].fillna(0)
    + df["children"].fillna(0)
    + df["babies"].fillna(0)
)

# Demand score based on guests and booking status
df["demand_score"] = (
    df["total_guests"] * 0.5
    + df["adr"].fillna(df["adr"].median()) * 0.5
)

# Divide demand into 3 states
df["demand_state"] = pd.qcut(
    df["demand_score"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop"
)

print("\nDemand State Distribution:")
print(df["demand_state"].value_counts())


# ---------------------------------------------------------
# 3. Define Actions
# ---------------------------------------------------------
# Actions represent dynamic pricing decisions.

actions = {
    0: "Decrease Price",
    1: "Keep Price",
    2: "Increase Price"
}

print("\nActions:")
for key, value in actions.items():
    print(key, ":", value)


# ---------------------------------------------------------
# 4. Reward Function
# ---------------------------------------------------------
def calculate_reward(state, action, adr, is_cancelled):
    """
    Reward is based on demand, price action and cancellation.

    High demand:
        Increase price -> positive reward
        Decrease price -> lower reward

    Low demand:
        Decrease price -> positive reward
        Increase price -> negative reward

    Cancellation:
        Additional penalty
    """

    reward = 0

    if state == "High":
        if action == 2:       # Increase Price
            reward = adr * 0.20
        elif action == 1:     # Keep Price
            reward = adr * 0.10
        else:                 # Decrease Price
            reward = -adr * 0.05

    elif state == "Medium":
        if action == 2:
            reward = adr * 0.10
        elif action == 1:
            reward = adr * 0.08
        else:
            reward = adr * 0.05

    else:  # Low demand
        if action == 0:
            reward = adr * 0.15
        elif action == 1:
            reward = adr * 0.05
        else:
            reward = -adr * 0.10

    # Cancellation penalty
    if is_cancelled == 1:
        reward -= adr * 0.20

    return reward


# ---------------------------------------------------------
# 5. Calculate Rewards
# ---------------------------------------------------------
df["reward"] = df.apply(
    lambda row: calculate_reward(
        row["demand_state"],
        1,  # Example action: Keep Price
        row["adr"],
        row["is_canceled"]
    ),
    axis=1
)

print("\nAverage Reward:", df["reward"].mean())


# ---------------------------------------------------------
# 6. Create Transition Probabilities
# ---------------------------------------------------------
# State transition is calculated from consecutive records.

states = ["Low", "Medium", "High"]

transition_counts = pd.DataFrame(
    0,
    index=states,
    columns=states,
    dtype=float
)

for i in range(len(df) - 1):
    current_state = str(df.iloc[i]["demand_state"])
    next_state = str(df.iloc[i + 1]["demand_state"])

    if current_state in states and next_state in states:
        transition_counts.loc[current_state, next_state] += 1

# Convert counts to probabilities
transition_probabilities = transition_counts.div(
    transition_counts.sum(axis=1).replace(0, 1),
    axis=0
)

print("\nTransition Probability Matrix:")
print(transition_probabilities)


# ---------------------------------------------------------
# 7. Value Iteration
# ---------------------------------------------------------
# Value Iteration finds the best action (policy) for each state.

gamma = 0.90       # Discount factor
theta = 0.0001     # Convergence threshold

state_index = {state: i for i, state in enumerate(states)}

# Average ADR and cancellation rate for each state
state_data = {}

for state in states:
    subset = df[df["demand_state"].astype(str) == state]

    if len(subset) > 0:
        state_data[state] = {
            "adr": subset["adr"].mean(),
            "cancel_rate": subset["is_canceled"].mean()
        }
    else:
        state_data[state] = {
            "adr": df["adr"].mean(),
            "cancel_rate": df["is_canceled"].mean()
        }


def expected_reward(state, action):
    data = state_data[state]

    return calculate_reward(
        state,
        action,
        data["adr"],
        1 if data["cancel_rate"] >= 0.5 else 0
    )


V = np.zeros(len(states))

while True:
    delta = 0

    for state in states:
        i = state_index[state]

        action_values = []

        for action in actions:
            immediate_reward = expected_reward(state, action)

            future_value = np.dot(
                transition_probabilities.loc[state].values,
                V
            )

            value = immediate_reward + gamma * future_value
            action_values.append(value)

        new_value = max(action_values)

        delta = max(delta, abs(new_value - V[i]))
        V[i] = new_value

    if delta < theta:
        break


# ---------------------------------------------------------
# 8. Extract Optimal Policy
# ---------------------------------------------------------
policy = {}

for state in states:
    action_values = {}

    for action in actions:
        immediate_reward = expected_reward(state, action)

        future_value = np.dot(
            transition_probabilities.loc[state].values,
            V
        )

        action_values[action] = (
            immediate_reward + gamma * future_value
        )

    best_action = max(action_values, key=action_values.get)
    policy[state] = actions[best_action]


# ---------------------------------------------------------
# 9. Display MDP Results
# ---------------------------------------------------------
print("\nValue Function:")
for state, value in zip(states, V):
    print(f"{state}: {value:.2f}")

print("\nOptimal Policy:")
for state, action in policy.items():
    print(f"{state} Demand -> {action}")


# ---------------------------------------------------------
# 10. Save Results
# ---------------------------------------------------------
policy_df = pd.DataFrame(
    list(policy.items()),
    columns=["Demand_State", "Optimal_Action"]
)

policy_df.to_csv("mdp_policy_results.csv", index=False)

transition_probabilities.to_csv(
    "mdp_transition_matrix.csv"
)

print("\nMDP results saved successfully:")
print("1. mdp_policy_results.csv")
print("2. mdp_transition_matrix.csv")