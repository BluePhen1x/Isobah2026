import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from spectraflow_net import SpectraFlowNet

def visualize():
    data_dir = Path('sentinel2_triplet')
    
    frame_a = np.load(data_dir / 'frame_a.npy')
    frame_c = np.load(data_dir / 'frame_c.npy')
    
    input_a = tf.expand_dims(tf.convert_to_tensor(frame_a, dtype=tf.float32), axis=0)
    input_c = tf.expand_dims(tf.convert_to_tensor(frame_c, dtype=tf.float32), axis=0)
    
    model = SpectraFlowNet()
    
    theta_0 = tf.constant([35.0])
    theta_t = tf.constant([40.0])
    
    output = model(input_a, input_c, theta_0, theta_t, training=False)
    
    if isinstance(output, (list, tuple)):
        prediction = output[0]
    else:
        prediction = output
        
    pred_img = tf.squeeze(prediction).numpy()
    
    rgb = pred_img[..., [2, 1, 0]]
    rgb = np.clip(rgb * 3, 0, 1)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Input Frame A")
    plt.imshow(np.clip(frame_a[..., [2, 1, 0]] * 3, 0, 1))
    
    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.imshow(rgb)
    
    plt.subplot(1, 3, 3)
    plt.title("Input Frame C")
    plt.imshow(np.clip(frame_c[..., [2, 1, 0]] * 3, 0, 1))
    
    plt.show()

if __name__ == "__main__":
    visualize()