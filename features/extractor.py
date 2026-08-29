import numpy as np

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

def extract_feature(landmarks: np.ndarray) -> np.ndarray:
    assert landmarks.shape == (21,3), f"Expected shape (21,3), got {landmarks.shape}"

    wrist = landmarks[WRIST_IDX]
    middle_mcp = landmarks[MIDDLE_MCP_IDX]

    centered = landmarks - wrist  # shape (21,3)

    scale = np.linalg.norm(middle_mcp - wrist)
    scale_ref = max(scale, 1e-6)  # avoid division by zero

    normalized = centered / scale_ref  # shape (21,3)

    return normalized.flatten().astype(np.float32)  # shape (63,)

if __name__ == "__main__":
    # Example usage:
    dummy_landmarks = np.random.rand(21, 3).astype(np.float32)
    feature_vector = extract_feature(dummy_landmarks)
    print("Extracted feature vector shape:", feature_vector.shape)