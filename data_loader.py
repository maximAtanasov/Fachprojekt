# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

def load_train(strip):
    file = f"data/np/strip_{strip}_train.npz"
    if os.path.exists(file):
        data = np.load(file)
        features = data['features']
        labels = data['labels']
        if features.shape[1] != 150:
            raise Exception(f"Unerwartete Datengröße beim laden von {file}, evtl. alte Daten?")
        return features, labels
    else:
        data = pd.read_csv(f"data/train/strip_{strip}_train.csv")
        features = []
        labels = []
        for idx, frame in data.groupby(['frame_number', 'run_number']):
            frame = frame.fillna(-100)
            if len(frame.index) < 15:
                frame = frame.set_index("node_id").reindex(pd.Index(np.arange(1, 16), name="node_id")).reset_index().interpolate()
                
            f = frame[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'r']].to_numpy(dtype=np.float32).flatten()
            features.append(f)
            labels.append(frame['near'].values[0])

        features = np.array(features)
        labels = np.array(labels)

        os.makedirs(os.path.dirname(file), exist_ok=True)
        np.savez_compressed(file, features=features, labels=labels)
        return features, labels

def load_vicon_train(strip):
    file = f"data/np/strip_{strip}_train_vicon.npz"
    if os.path.exists(file):
        data = np.load(file)
        features = data['features']
        labels = data['labels']
        if features.shape[1] != 150:
            raise Exception(f"Unerwartete Datengröße beim laden von {file}, evtl. alte Daten?")
        return features, labels
    else:
        data = pd.read_csv(f"data/train/strip_{strip}_train.csv")
        features = []
        labels = []
        for idx, frame in data.groupby(['frame_number', 'run_number']):
            if frame['near'].values[0] == 0:
                continue
            frame = frame.fillna(-100)
            if len(frame.index) < 15:
                frame = frame.set_index("node_id").reindex(pd.Index(np.arange(1, 16), name="node_id")).reset_index().interpolate()
            f = frame[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'r']].to_numpy(dtype=np.float32).flatten()
            features.append(f)
            labels.append(frame[['vicon_x', 'vicon_y']].to_numpy(dtype=np.float32).flatten()[:2])
        features = np.array(features)
        labels = np.array(labels)

        os.makedirs(os.path.dirname(file), exist_ok=True)
        np.savez_compressed(file, features=features, labels=labels)
        return features, labels

def load_test(strip):
    file = f"data/np/strip_{strip}_test.npz"
    if os.path.exists(file):
        data = np.load(file)
        f = data['frames']
        if f.shape[1] != 150:
            raise Exception(f"Unerwartete Datengröße beim laden von {file}, evtl. alte Daten?")
        return f
    else:
        df = pd.read_csv(f"data/test/strip_{strip}_test_no_labels.csv")
        frames = []
        for idx , frame in df.groupby(['frame_number']):
            frame = frame.fillna(-100)
            if len(frame.index) < 15:
                frame = frame.set_index("node_id").reindex(pd.Index(np.arange(1, 16), name="node_id")).reset_index().interpolate()
            f = frame[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'r']].to_numpy(dtype=np.float32).flatten()
            frames.append(f)

        frames = np.array(frames)

        os.makedirs(os.path.dirname(file), exist_ok=True)
        np.savez_compressed(file, frames=frames)
        return frames

def load_test_all():
    frames = []
    for i in range(1, 24):
        strip = load_test(i)
        for f in strip:
            frames.append(f)
    return frames
