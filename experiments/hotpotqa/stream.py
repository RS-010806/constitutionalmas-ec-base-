import json


def stream_json_array(path: str):
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        in_array = False
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk

            idx = 0
            if not in_array:
                while idx < len(buf) and buf[idx].isspace():
                    idx += 1
                if idx < len(buf) and buf[idx] == "[":
                    in_array = True
                    idx += 1
                buf = buf[idx:]
                idx = 0

            while True:
                while idx < len(buf) and buf[idx].isspace():
                    idx += 1
                if idx >= len(buf):
                    break
                if buf[idx] == ",":
                    idx += 1
                    continue
                if buf[idx] == "]":
                    return

                try:
                    obj, next_idx = decoder.raw_decode(buf, idx)
                except json.JSONDecodeError:
                    break
                yield obj
                idx = next_idx

            buf = buf[idx:]


def reservoir_sample(stream, k: int, seed: int = 0):
    import random

    rng = random.Random(seed)
    sample = []
    seen = 0
    for item in stream:
        seen += 1
        if len(sample) < k:
            sample.append(item)
            continue
        j = rng.randint(1, seen)
        if j <= k:
            sample[j - 1] = item
    return sample

