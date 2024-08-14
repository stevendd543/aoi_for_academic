import argparse
import image_process 
import shutil
import os 

def parse_args():
    parser = argparse.ArgumentParser(description="anomal detection system")
    parser.add_argument('--num_epochs', type = int, default = 100, help = 'number of epochs for training')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'number of batch size')
    parser.add_argument('--oip', type = str, default = "dataset\\origin\\normal", help = 'folder path of original image ')
    parser.add_argument('--op', type = str, help = 'al : auxiliary label \n ag : augmentation \n sg : segmentaion \n cp : crop \n rt : rotate \n tr : training \n tt : test')
    parser.add_argument('--stride', type = int, default = 2, help = 'Sample stride')
    parser.add_argument('--sip', type = str, default = 'segmentation', help = 'folder path of segmentation')
    parser.add_argument('--text', type = str, default = ' ', help = 'setting saving text file name')
    parser.add_argument('--srcdes', type = str, default = 'dataset/origin/normal', help = 'any saving path and src path')
    # for detection ---------------------------------------------------------------------------------------------------
    parser.add_argument('--thr', type=float, default=0.78, help="threshold of classfication.")
    parser.add_argument('--resunet', type=str, default="weights\\resunet.pth", help="weight path of segmentation")
    parser.add_argument('--detector', type=str, default="weights\\detector2.pth", help="weight path of detector")
    parser.add_argument('--src', type=str, default = "D:\\Users\\THCHENAQ\\Desktop\\bad2", help="source of detection")
    parser.add_argument('--des', type=str, default = "D:\\Users\\THCHENAQ\\Desktop\\r4", help="saving path of result")
    parser.add_argument('--gpu', type=bool, default=False, help="execution with GPU")
    args = parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    imp = image_process.ImageProcessor(args)
    if args.op == 'al': # label tool
        imp.label(args.text)
    elif args.op == 'sg': # segmentation
        imp.segmentation()
    elif args.op == 'det': # detection
        imp = image_process.ImageProcessor(None)
        detector = detect.Detector(args)
        model = detector.create_system()
        resunet = detect.create_resunet()
        model.eval()
        print("start detecting...")
        for idx,file_name in enumerate(os.listdir(args.src)):
            print(f"{idx}-th detection : {file_name}")
            detector.detect_and_save(args.src,file_name,args.des)
        print("stop detecting...")
    else:
        print("Error: Please input correct operation!")
