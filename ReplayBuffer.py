import threading
import random
import pickle

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.lock = threading.Lock()
    def push(self, frames, custom_features, action, reward, next_frames, next_custom_features, done, old_log_probs):
        with self.lock:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
            self.buffer.append((frames, custom_features, action, reward, next_frames, next_custom_features, done, old_log_probs))
            
    def sample(self, batch_size):
        with self.lock:
            if len(self.buffer) < batch_size:
                return []
            return random.sample(self.buffer, batch_size)
            
    def save(self, filename):
        with self.lock:
            with open(filename, 'wb') as f:
                pickle.dump(self.buffer, f)
                
    def load(self, filename):
        with self.lock:
            with open(filename, 'rb') as f:
                self.buffer = pickle.load(f)
