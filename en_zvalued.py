import os
import re
import numpy as np
from scipy import stats
import cv2

def main():
    base_dir = '/Users/inabary/shuron/tilezval/h_2014-07859-12/hyper'
    save_dir = './savetest2'

    publish(base_dir, save_dir)


def publish(base_dir, save_dir):
    regx = re.compile(r'^\..+')
    imnames = [n for n in os.listdir(base_dir) if not regx.search(n)]
    imlen = len(imnames)

    imgsheet = None
    for i, fn in enumerate(imnames):
        fn_path = os.path.join(base_dir, fn)
        if imgsheet is not None:
            im = cv2.imread(fn_path)
            imgsheet = np.concatenate((imgsheet, im))
        else:
            imgsheet = cv2.imread(fn_path)

        print('\r結合中…　{}/{}'.format(i + 1, imlen), end='')
    print("")

    imgsheet = np.reshape(imgsheet, (300 * 300 * imlen, 3))

    imgsheet_zscore = stats.zscore(imgsheet, axis=0, ddof=1)
    imgsheet_normalized = (np.clip(imgsheet_zscore, -2, 2) + 2) / 4


    zscored_picts = [
        (pic * 255).astype(np.uint8)
        for pic in (
            np.reshape(arr, (300, 300, 3))
            for arr in np.split(imgsheet_normalized, imlen, axis=0)
        )
    ]

    for i in range(imlen):
        print('\r保存しています…　{}/{}'.format(i + 1, imlen), end='')
        cv2.imwrite(os.path.join(save_dir, 'z-{}'.format(imnames[i])) ,zscored_picts[i])

    print('終了 {}'.format(base_dir))

if __name__ == '__main__':
    main()
