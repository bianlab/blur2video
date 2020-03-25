import cv2
from os import listdir
from os.path import join


def getVideo(folder, exp_name, image_shape=(256, 256)):
    fps = 4
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(folder+exp_name+'.avi', fourcc, fps, image_shape, False)
    image_names = [join(folder, x) for x in listdir(folder)]
    for item in image_names:
        if item.endswith('.png'):
            frame = cv2.cvtColor(cv2.imread(item), cv2.COLOR_BGR2GRAY)
            # frame = cv2.imread(item)
            videoWriter.write(frame)
    videoWriter.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    exp_name = ''
    folder = ''
    getVideo(folder, exp_name)

