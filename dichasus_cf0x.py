import tensorflow as tf
import numpy as np
import json

PATHS = [
    {
        "tfrecords" : "dataset/dichasus-cf02.tfrecords",
        "offsets" : "dataset/reftx-offsets-dichasus-cf02.json"
    },
    {
        "tfrecords" : "dataset/dichasus-cf03.tfrecords",
        "offsets" : "dataset/reftx-offsets-dichasus-cf03.json"
    },
    {
        "tfrecords" : "dataset/dichasus-cf04.tfrecords",
        "offsets" : "dataset/reftx-offsets-dichasus-cf04.json"
    }
]

spec = None

antenna_assignments = []
antenna_count = 0

with open("dataset/spec.json") as specfile:
    spec = json.load(specfile)
    for antenna in spec["antennas"]:
        antenna_count = antenna_count + sum([len(row) for row in antenna["assignments"]])
        antenna_assignments.append(antenna["assignments"])

def load_calibrate(path, offset_path, td = False):
    offsets = None
    with open(offset_path, "r") as offsetfile:
        offsets = json.load(offsetfile)

    def record_parse_function(proto):
        record = tf.io.parse_single_example(
            proto,
            {
                "cfo": tf.io.FixedLenFeature([], tf.string, default_value = ""),
                "csi": tf.io.FixedLenFeature([], tf.string, default_value = ""),
                "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value = ""),
                "snr": tf.io.FixedLenFeature([], tf.string, default_value = ""),
                "time": tf.io.FixedLenFeature([], tf.float32, default_value = 0),
            },
        )

        cfo = tf.ensure_shape(tf.io.parse_tensor(record["cfo"], out_type = tf.float32), (antenna_count))

        csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (antenna_count, 1024, 2))
        csi = tf.complex(csi[:, :, 0], csi[:, :, 1])
        csi = tf.signal.fftshift(csi, axes=1)
        
        if td == True:
            incr = tf.cast(tf.math.angle(tf.math.reduce_sum(csi[:,1:] * tf.math.conj(csi[:,:-1]))), tf.complex64)
            csi = csi * tf.exp(-1.0j * incr * tf.cast(tf.range(csi.shape[-1]), tf.complex64))[tf.newaxis,:]

        position = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3))

        snr = tf.ensure_shape(tf.io.parse_tensor(record["snr"], out_type = tf.float32), (32))
        
        time = tf.ensure_shape(record["time"], ())

        return cfo, csi, position[:2], snr, time

    def apply_calibration(cfo, csi, pos, snr, time):
        sto_offset = tf.tensordot(tf.constant(offsets["sto"]), 2 * np.pi * tf.range(tf.shape(csi)[1], dtype = np.float32) / tf.cast(tf.shape(csi)[1], np.float32), axes = 0)
        cpo_offset = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(tf.shape(csi)[1], dtype = np.float32), axes = 0)
        csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))

        return cfo, csi, pos, snr, time

    def order_by_antenna_assignments(cfo, csi, pos, snr, time):
        csi = tf.stack([[tf.gather(csi, antenna_indices) for antenna_indices in array] for array in antenna_assignments])
        snr = tf.stack([[tf.gather(snr, antenna_indices) for antenna_indices in array] for array in antenna_assignments])
        return cfo, csi, pos, snr, time

    
    dset = tf.data.TFRecordDataset(path)
    dset = dset.map(record_parse_function, num_parallel_calls = tf.data.AUTOTUNE)
    dset = dset.map(apply_calibration, num_parallel_calls = tf.data.AUTOTUNE)
    dset = dset.map(order_by_antenna_assignments, num_parallel_calls = tf.data.AUTOTUNE)

    return dset

def average_subcarriers(chunksize = 32):
    def average_subcarriers_func(cfo, csi, pos, snr, time):
        return cfo, csi[:, :, :, ::chunksize], pos, snr, time
    return average_subcarriers_func

def csi_time_domain(cfo, csi, pos, snr, time):
    csi = tf.signal.fftshift(tf.signal.ifft(tf.signal.fftshift(csi, axes = -1)), axes = -1)
    return csi

def cut_out_taps(tap_start, tap_stop):
    def cut_out_taps_func(csi):
        return csi[:,:,:,tap_start:tap_stop]
    return cut_out_taps_func

# frequency domain
full_dataset_freq_domain = load_calibrate(PATHS[0]["tfrecords"], PATHS[0]["offsets"], td = False)

for path in PATHS[1:]:
    full_dataset_freq_domain = full_dataset_freq_domain.concatenate(load_calibrate(path["tfrecords"], path["offsets"], td = False))

full_dataset_freq_domain = full_dataset_freq_domain.map(average_subcarriers(chunksize = 32), num_parallel_calls = tf.data.AUTOTUNE)

# time domain
full_dataset_time_domain = load_calibrate(PATHS[0]["tfrecords"], PATHS[0]["offsets"], td = True)

for path in PATHS[1:]:
    full_dataset_time_domain = full_dataset_time_domain.concatenate(load_calibrate(path["tfrecords"], path["offsets"], td = True))

full_dataset_time_domain = full_dataset_time_domain.map(csi_time_domain, num_parallel_calls = tf.data.AUTOTUNE)
full_dataset_time_domain = full_dataset_time_domain.map(cut_out_taps(507, 520), num_parallel_calls = tf.data.AUTOTUNE)
