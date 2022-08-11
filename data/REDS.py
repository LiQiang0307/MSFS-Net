import os


def move(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    if not os.path.exists(os.path.join(dst, 'blur')):
        os.mkdir(os.path.join(dst, 'blur'))
    if not os.path.exists(os.path.join(dst, 'sharp')):
        os.mkdir(os.path.join(dst, 'sharp'))

    folders = os.listdir(src)
    # print(folders)
    cnt = 0
    for f in folders:
        image_names = os.listdir(os.path.join(src, f))
        print(image_names)

        for i in image_names:
            # os.rename(os.path.join(src, f, i), os.path.join(dst, 'blur', f + '_' + i))
            os.rename(os.path.join(src, f, i), os.path.join(dst, 'sharp', f + '_' + i))
            cnt += 1
    print('%d images are moved' % cnt)

if __name__ =='__main__':
    # 24000
    src='/media/zhangyn/新加卷/reds/REDS/train/train_sharp'
    dst='/media/zhangyn/新加卷/reds/REDSTEST/train'
    move(src,dst)