from glob import glob
import sys
from pathlib import Path
import numpy as np
import subprocess
from multiprocessing.pool import ThreadPool


def run_thread(prgm):
    subprocess.Popen(prgm).wait()


folders = sys.argv[1]
target_folder = sys.argv[2]

with open(folders, 'r') as file:
    lines = file.readlines()

for line in lines:
    pool = ThreadPool(4)
    pcs = []
    print("Doing line", line)
    curr_folder = line[:-1]
    images = sorted(glob(curr_folder + "/imgs/*"))
    unaries = sorted(glob(curr_folder + "/unaries/*"))
    gt = np.genfromtxt(curr_folder + "/gt_label.csv", delimiter=",")[:, :2]
    gt = np.clip((gt - [448, 28]) / [1024, 1024], 0, 1)
    wh = 30 / 1024
    Path(f"yolo/{target_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"yolo/{target_folder}_2").mkdir(parents=True, exist_ok=True)
    count = 0
    for (i, image) in enumerate(images):
        # if count == 100:
        #     break
        image_target = image.replace("/", "-").replace("-imgs-", "-").replace(".png", "")
        pcs.append(
            [
                "/usr/bin/convert", f"{image}", "-gravity", "Center", "-crop",
                "1024x1024+0+0", "+repage", f"yolo/{target_folder}/{image_target}.png"
            ]
        )
        with open(f"yolo/{target_folder}/{image_target}.txt", "w") as f:
            f.write("0 {} {} {} {}".format(gt[i, 0], gt[i, 1], wh, wh))
        count += 1

    count = 0
    for unary in unaries:
        # if count == 100:
        #     break
        unary_target = unary.replace("/", "-").replace("-unaries-", "-").replace(".png", "")
        pcs.append(
            [
                "/usr/bin/convert", f"{unary}", "-gravity", "Center", "-crop",
                "819x819+0+0", "+repage", "-resize", "1024x1024",
                f"yolo/{target_folder}_2/{unary_target}.png"
            ]
        )
        count += 1

    for pc in pcs:
        pool.apply_async(run_thread, (pc, ))
    pool.close()
    pool.join()
