import os
import re
import numpy as np
from scipy import stats
import cv2

def main():
    base_dir = '/home/bioinfo/ml/data/hyper_mutation/merged/not_hyper_mu'
    save_dir = '/media/bioinfo/fatdata/tumor_tiles_zvalue/manual/non-hyper'

    ENV = {
        "use_tumor": False,
    }

    regx = re.compile(r'^\..+')
    targets = [n for n in os.listdir(base_dir) if not regx.search(n)]

    for t in targets:
        _out = os.path.join(save_dir, t)
        _source = os.path.join(base_dir, t)

        if ENV["use_tumor"]:
            _source = os.path.join(_source, "tumor")

        if not os.path.exists(_out):
            os.mkdir(_out)


        if ENV["use_tumor"]:
            _out = os.path.join(_out, "tumor")
            if not os.path.exists(_out):
                os.mkdir(_out)

        print("{} -> {}".format(_source, _out))


        publish(_source, _out)


def publish(base_dir, save_dir):
    regx = re.compile(r'^\..+')
    imnames = [n for n in os.listdir(base_dir) if not regx.search(n)]
    imlen = len(imnames)

    n_trashed = 0
    imgsheet = None
    for i, fn in enumerate(imnames):
        fn_path = os.path.join(base_dir, fn)
        if imgsheet is not None:
            im = cv2.imread(fn_path)
            if im.shape != (300, 300, 3):
                imlen -= 1
                n_trashed += 1
                continue

            try:
                imgsheet = np.concatenate((imgsheet, im))
            except:
                print("err")
                return
        else:
            imgsheet = cv2.imread(fn_path)

        print('\r結合中…　{}/{}'.format(i + 1, imlen), end='')
    print("")

    if n_trashed:
        print("{} 枚のタイルがサイズ不備により削除された".format(n_trashed))

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

    print("")
    print('終了 {}'.format(base_dir))
    print("")

if __name__ == '__main__':
    main()
