import math

# ReLU: Turns negative numbers into 0. Keeps positive numbers exactly as they are.
def relu(x):
    return max(0, x)

# Sigmoid: Squishes any number down to a decimal between 0 and 1.
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# --- Testing them out ---

print("--- Testing ReLU ---")
print(f"Input -5  -> Output: {relu(-5)}")   # Becomes 0
print(f"Input  12 -> Output: {relu(12)}")   # Stays 12

print("\n--- Testing Sigmoid ---")
print(f"Input -4  -> Output: {sigmoid(-4):.4f}")  # Close to 0 (unlikely)
print(f"Input  0   -> Output: {sigmoid(0):.4f}")   # Exactly 0.5 (50/50 split)
print(f"Input  4   -> Output: {sigmoid(4):.4f}")   # Close to 1 (highly likely)