import cv2
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform.rotation import Rotation as R
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataDir', type=Path, help='data dir')
    args = parser.parse_args()

    poses = None
    with (args.dataDir / 'pose.txt').open() as fp:
        buf = fp.read()
        print(buf)
        poses = [json.loads(jline) for jline in buf.split('\n')]

    for iimg in range(len(poses)):
        buf_ = '%03d.png' % iimg

        print(f'loading : {buf_}')
        img = cv2.imread(str(args.dataDir / buf_))
        pnt = 4 * 4
        objp = np.zeros((4, pnt)).astype(np.float)

        for i in range(4):
            for j in range(4):
                objp[0, i * 4 + j] = 0.4 * j - 1 + 0.4
                objp[1, i * 4 + j] = 0.4 * i - 1 + 0.4
                objp[2, i * 4 + j] = 0
                objp[3, i * 4 + j] = 1

        K = np.zeros((3, 3)).astype(np.float)
        K[0, 0] = 3840 / 2
        K[1, 1] = 3840 / 2
        K[0, 2] = 512 - 0.5
        K[1, 2] = 512 - 0.5
        K[2, 2] = 1
        c = R.from_rotvec([np.pi, 0, 0]).as_dcm()
        p = poses[iimg]
        r = R.from_quat([p['qx'], p['qy'], p['qz'], p['qw']]).as_dcm()
        r = r @ c
        t = np.array([p['x'], p['y'], p['z']]).astype(np.float)
        t = - r.T @ t

        D = np.zeros((3, 4), dtype=np.float)
        D[0:3, 0:3] = r.T
        D[0:3, 3] = t

        pts = K @ D @ objp

        for i in range(16):
            pts[0, i] = pts[0, i] / pts[2, i]
            pts[1, i] = pts[1, i] / pts[2, i]

        imgPoints = pts[0:2, :]
        np.savetxt(str(args.dataDir / ('ans_%03d.txt' % iimg)), imgPoints.T)
        for i in range(imgPoints.shape[1]):
            pt = np.round(imgPoints[:, i]).astype(np.int32)
            img = cv2.drawMarker(img, (pt[0], pt[1]), color=(0, 255, 0))

        cv2.imshow('show', img)
        cv2.imwrite(str(args.dataDir / ('ans_%03d.png' % iimg)), img)
        while ord('q') != cv2.waitKey():
            pass
