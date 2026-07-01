import tensorflow as tf
import numpy as np
from pathlib import Path
from spectraflow_net import SpectraFlowNet

def main():
    data_dir = Path('sentinel2_triplet')
    
    frame_a_path = data_dir / 'frame_a.npy'
    frame_c_path = data_dir / 'frame_c.npy'
    
    if not frame_a_path.exists() or not frame_c_path.exists():
        print("Error: Could not find frame_a.npy or frame_c.npy in sentinel2_triplet/")
        return

    frame_a_raw = np.load(frame_a_path)
    frame_c_raw = np.load(frame_c_path)
    
    frame_a = tf.expand_dims(tf.convert_to_tensor(frame_a_raw, dtype=tf.float32), axis=0)
    frame_c = tf.expand_dims(tf.convert_to_tensor(frame_c_raw, dtype=tf.float32), axis=0)
    
    model = SpectraFlowNet(feature_dim=256)
    
    features_a, features_c = model(frame_a, frame_c, tf.constant([35.0]), tf.constant([40.0]), training=False)
    
    print("Integration Successful!")
    print(f"Feature output shape: {features_a.shape}")

if __name__ == "__main__":
    main()