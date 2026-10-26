import numpy as np

def matinv(A):
    # Convert the input list to a numpy array
    A_np = np.array(A)
    
    # Calculate the inverse using numpy's linalg.inv function
    inverse = np.linalg.inv(A_np)
    
    return inverse

# Example usage
A = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

try:
    inverse = matinv(A)
    print("Inverse of A:\n", inverse)
except np.linalg.LinAlgError as e:
    print("Error:", e)