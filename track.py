# limit the number of cpus used by high performance libraries
from datetime import datetime
import dis
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import numpy as numpy
import math
import configparser

cnfg = configparser.ConfigParser()

with open('configuration/config.ini', encoding='utf-8') as f:
    cnfg.read_file(f)

roi = [int(cnfg['region_of_interest']['roi_0']),int(cnfg['region_of_interest']['roi_1']),int(cnfg['region_of_interest']['roi_2']),int(cnfg['region_of_interest']['roi_3'])]

point_matrix1 = cnfg['line_indicator']['point_matrix1'].replace('[','').replace(']','').replace('(','').replace(')','').split(', ')
point_matrix1 = [int(i) for i in point_matrix1]
point_matrix2 = cnfg['line_indicator']['point_matrix2'].replace('[','').replace(']','').replace('(','').replace(')','').split(', ')
point_matrix2 = [int(i) for i in point_matrix2]
point_matrix3 = cnfg['line_indicator']['point_matrix3'].replace('[','').replace(']','').replace('(','').replace(')','').split(', ')
point_matrix3 = [int(i) for i in point_matrix3]
point_matrix4 = cnfg['line_indicator']['point_matrix1'].replace('[','').replace(']','').replace('(','').replace(')','').split(', ')
point_matrix4 = [int(i) for i in point_matrix4]

point_matrix= [point_matrix1, point_matrix2, point_matrix3, point_matrix4]

detection_time=int(cnfg['lane_times']['detection_time'])

# lane_time = detection_time
lane_times=[10,10,8,10]

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def searchForId(matrix, v_id):
    i=0
    if numpy.array(matrix).size !=0 :
        for row in matrix:
            if row[0] == v_id:
                print(row[0])
                return i
            i=i+1
    return -1

def lineFromPoints(point):

    a = point[3] - point[1]
    b = point[0] - point[2]
    c = a*(point[0]) + b*(point[1])
 
    if(b < 0):
        return a,b,c
    else:
        return a,b,c

def shortestDistance(px, py, a, b, c):
      
    distance = abs((a * px + b * py + c)) / (math.sqrt(a * a + b * b))
    return distance


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(opt, cur_time, index):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source[index].startswith(
        'rtsp') or source[index].startswith('http') or source[index].endswith('.txt')

    device = select_device(opt.device)
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )

    # Initialize
    
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()
        print('permits video showingggggggggggggggggggggggggggggggggg')
        print(show_vid)

    # Dataloader
    if webcam:
        #show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source[index], img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source[index], img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    distance_arr = [[]]

    # extract what is in between the last '/' and last '.'
    txt_file_name = source[0].split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0


    end_time=cur_time + detection_time

    isfirst=1
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                if isfirst==1:
                    print('New Time--------------------', outputs)
                    if len(outputs) > 0:
                        timeEstimations=[]
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            pointX = output[0]+((output[2] - output[0])/2)
                            pointY = output[1]+((output[3] - output[1])/2)
                            pA,pB,pC = lineFromPoints(point_matrix[index])
                            distance = shortestDistance(pointX,pointY,pA, pB, pC)

                            twheel_speed = float(cnfg['vehicle_speed_and_counter']['3wheel_speed'])
                            bus_speed = float(cnfg['vehicle_speed_and_counter']['bus_speed'])
                            lorry_speed = float(cnfg['vehicle_speed_and_counter']['lorry_speed'])
                            van_speed = float(cnfg['vehicle_speed_and_counter']['van_speed'])
                            jeep_speed = float(cnfg['vehicle_speed_and_counter']['jeep_speed'])
                            car_speed = float(cnfg['vehicle_speed_and_counter']['car_speed'])
                            motorbike_speed = float(cnfg['vehicle_speed_and_counter']['motorbike_speed'])
                            truck_speed = float(cnfg['vehicle_speed_and_counter']['truck_speed'])
            
                            if(output[5]==0):
                                timeEstimations.append(distance/twheel_speed)
                            elif(output[5]==1):
                                timeEstimations.append(distance/bus_speed)
                            elif(output[5]==2):
                                timeEstimations.append(distance/lorry_speed)
                            elif(output[5]==3):
                                timeEstimations.append(distance/van_speed)
                            elif(output[5]==4):
                                timeEstimations.append(distance/jeep_speed)
                            elif(output[5]==5):
                                timeEstimations.append(distance/car_speed)
                            elif(output[5]==6):
                                timeEstimations.append(distance/motorbike_speed)
                            else:
                                timeEstimations.append(distance/truck_speed)

                        lane_time=max(timeEstimations)

                        print('New Time--------------------', lane_time)
                        print('distance--------------------', distance)

                            # cnfg['lane_times']['lane_times'] = str(lane_times)
                            # with open('configuration/config.ini', 'w', encoding='utf-8') as f:
                            #     cnfg.write(f)
                            # with open('configuration/config.ini', encoding='utf-8') as f:
                            #     cnfg.read_file(f)
                            # lane_times = cnfg['lane_times']['lane_times'].replace('[','').replace(']','').split(',')
                            # lane_times = [int(i) for i in lane_times]
                        end_time=cur_time + lane_times[index] - detection_time
                    
                        isfirst+=1
               
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        
                        id_index = searchForId(distance_arr,output[4])

                        pointX = output[0]+((output[2] - output[0])/2)
                        pointY = output[1]+((output[3] - output[1])/2)

                        pA,pB,pC = lineFromPoints(point_matrix[index])
                        distance = shortestDistance(pointX,pointY,pA, pB, pC)

                        if numpy.array(distance_arr).size ==0 :
                            distance_arr[0]=[output[4], output[5], time.time(), pointX, pointY, time.time(), pointX, pointY]
                            
                        else:
                            if (id_index >= 0):
                                distance_arr [id_index][5]= time.time()
                                distance_arr [id_index][6]= pointX
                                distance_arr [id_index][7]= pointY
                            else:
                                distance_arr.append([output[4], output[5], time.time(), pointX, pointY,time.time(), pointX, pointY])

                        cv2.circle(im0,(int(pointX),int(pointY)),3,(0,255,0),cv2.FILLED)

                        cv2.line(im0, (int(point_matrix[index][0]),int(point_matrix[index][1])), (int(point_matrix[index][1]),int(point_matrix[index][2])), (255,0,0), 5)


                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration


            # Save results (image with detections)
            save_vid=True
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(roi[2])
                        h = int(roi[3])
                    else:  # stream
                        fps, w, h = 30, int(roi[2]), int(roi[3])

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
            
        if (time.time() >= end_time) :
            break
    


    if numpy.array(distance_arr).size !=0 :
        twheel_speed = float(cnfg['vehicle_speed_and_counter']['3wheel_speed'])
        twheel_count = int(cnfg['vehicle_speed_and_counter']['3wheel_count']) 
        bus_speed = float(cnfg['vehicle_speed_and_counter']['bus_speed'])
        bus_count = int(cnfg['vehicle_speed_and_counter']['bus_count']) 
        lorry_speed = float(cnfg['vehicle_speed_and_counter']['lorry_speed'])
        lorry_count = int(cnfg['vehicle_speed_and_counter']['lorry_count']) 
        van_speed = float(cnfg['vehicle_speed_and_counter']['van_speed'])
        van_count = int(cnfg['vehicle_speed_and_counter']['van_count']) 
        jeep_speed = float(cnfg['vehicle_speed_and_counter']['jeep_speed'])
        jeep_count = int(cnfg['vehicle_speed_and_counter']['jeep_count']) 
        car_speed = float(cnfg['vehicle_speed_and_counter']['car_speed'])
        car_count = int(cnfg['vehicle_speed_and_counter']['car_count']) 
        motorbike_speed = float(cnfg['vehicle_speed_and_counter']['motorbike_speed'])
        motorbike_count = int(cnfg['vehicle_speed_and_counter']['motorbike_count']) 
        truck_speed = float(cnfg['vehicle_speed_and_counter']['truck_speed'])
        truck_count = int(cnfg['vehicle_speed_and_counter']['truck_count']) 

        # update speeds 
        for data in distance_arr:
            if(data[1] == 0):
                print('its 3wheel')
                if(data[3]!=data[6] or data[4]!=data[7]):
                    twheel_count+=1
                    twheel_speed+=(calculateDistance(data[3],data[4],data[6],data[7])/(data[5]-data[2]))
                
            elif (data[1]== 1):
                print('its bus')
                if(data[3]!=data[6] or data[4]!=data[7]):
                    bus_count+=1
                    bus_speed+=(calculateDistance(data[3],data[4],data[6],data[7])/(data[5]-data[2]))

            elif (data[1]== 2):
                print('its lorry')
                if(data[3]!=data[6] or data[4]!=data[7]):
                    lorry_count+=1
                    lorry_speed+=(calculateDistance(data[3],data[4],data[6],data[7])/(data[5]-data[2]))

            elif (data[1]== 3):
                print('its van')
                if(data[3]!=data[6] or data[4]!=data[7]):
                    van_count+=1
                    van_speed+=(calculateDistance(data[3],data[4],data[6],data[7])/(data[5]-data[2]))

            elif (data[1]== 4):
                print('its jeep')
                if(data[3]!=data[6] or data[4]!=data[7]):
                    jeep_count+=1
                    jeep_speed+=(calculateDistance(data[3],data[4],data[6],data[7])/(data[5]-data[2]))

            elif (data[1]== 5):
                print('its car')
                if(data[3]!=data[6] or data[4]!=data[7]):
                    car_count+=1
                    car_speed+=(calculateDistance(data[3],data[4],data[6],data[7])/(data[5]-data[2]))

            elif (data[1]== 6):
                print('its bike')
                if(data[3]!=data[6] or data[4]!=data[7]):
                    motorbike_count+=1
                    motorbike_speed+=(calculateDistance(data[3],data[4],data[6],data[7])/(data[5]-data[2]))

            elif (data[1]== 7):
                print('its truck')
                if(data[3]!=data[6] or data[4]!=data[7]):
                    truck_count+=1
                    truck_speed+=(calculateDistance(data[3],data[4],data[6],data[7])/(data[5]-data[2]))

        if twheel_count>0:
            cnfg['vehicle_speed_and_counter']['3wheel_count'] = str(twheel_count)
            cnfg['vehicle_speed_and_counter']['3wheel_speed'] = str(twheel_speed/twheel_count)

        if bus_count>0:
            cnfg['vehicle_speed_and_counter']['bus_count'] = str(bus_count)
            cnfg['vehicle_speed_and_counter']['bus_speed'] = str(bus_speed/bus_count)

        if lorry_count>0:
            cnfg['vehicle_speed_and_counter']['lorry_count'] = str(lorry_count)
            cnfg['vehicle_speed_and_counter']['lorry_speed'] = str(lorry_speed/lorry_count)

        if van_count>0:
            cnfg['vehicle_speed_and_counter']['van_count'] = str(van_count)
            cnfg['vehicle_speed_and_counter']['van_speed'] = str(van_speed/van_count)

        if jeep_count>0:
            cnfg['vehicle_speed_and_counter']['jeep_count'] = str(jeep_count)
            cnfg['vehicle_speed_and_counter']['jeep_speed'] = str(jeep_speed/jeep_count)

        if car_count>0:
            cnfg['vehicle_speed_and_counter']['car_count'] = str(car_count)
            cnfg['vehicle_speed_and_counter']['car_speed'] = str(car_speed/car_count)
        
        if motorbike_count>0:
            cnfg['vehicle_speed_and_counter']['motorbike_count'] = str(motorbike_count)
            cnfg['vehicle_speed_and_counter']['motorbike_speed'] = str(motorbike_speed/motorbike_count)

        if truck_count>0:
            cnfg['vehicle_speed_and_counter']['truck_count'] = str(truck_count)
            cnfg['vehicle_speed_and_counter']['truck_speed'] = str(truck_speed/truck_count)

        with open('configuration/config.ini', 'w', encoding='utf-8') as f:
            cnfg.write(f)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5/weights/basic-medium-50.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    # parser.add_argument('--source', type=str, default='https://youtu.be/e_WBuBqS9h8', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default=['lane2.mp4', 'lane1.mp4', 'lane3.mp4', 'lane4.mp4'], help='source') 
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        for x in range(4):
            current_time=time.time()
            detect(opt, current_time, x)
