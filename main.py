from libraries import *

from model_utils import CRAFT, copyStateDict, RefineNet
from prediction_utils import test_net, demo
from file_utils import extract_frames, get_frames_list, loadImage, generate_words, load_Frame_By_FrameNo_FromVideo, get_FrameNumbers_FromTimeStamp
from regex import *
import mmcv
# start of main function
project_path = os.getcwd() + '/'

# project_path = os.getcwd()+'/text_recognition/'

def GenTextOutput(process_file_path, net, refine_net, frame_timestamps=[], frame_number=30, **passed_kwargs):
    try:

        # --------------------- Reading the Input data ------------------------------
        # checking if its a video or set of frames
        if ".mp4" in process_file_path:
            # pass
            # print("Here")
            # image_list, frame_timestamps = extract_frames(process_file_path, frame_timestamps, frame_number,
            #                                               project_path)
            image_list = get_FrameNumbers_FromTimeStamp( process_file_path, frame_timestamps )            
            
            file_name = process_file_path.split('/')[-1][:-4]

        elif type(process_file_path) is list:
            image_list = process_file_path
#            file_name = process_file_path.split('/')[-1]

        elif os.path.isdir(process_file_path):
            image_list, frame_timestamps = get_frames_list(process_file_path)
            file_name = False

        else:
            return "Please pass correct input"

        # print("image_list::", image_list)

        # --------------------- Processing the Arguments  ------------------------------

        # overwriting existing dict and appending new ones to final dict
        default_kwargs = {'trained_model': project_path + 'Pretrained_Models/craft_mlt_25k.pth',
                          'use_gpu': False, 'refine': False, 'refiner_model': 'weights/craft_refiner_CTW1500.pth',
                          'text_threshold': 0.7, 'low_text': 0.4, 'link_threshold': 0.4, 'canvas_size': 1280,
                          'mag_ratio': 1.5,
                          'poly': False, 'show_time': False, 'test_folder': project_path + '/Frame_vid',
                          'image_folder': project_path + 'Cropped_image', 'workers': 4, 'batch_size': 192,
                          'saved_model': project_path + 'Pretrained_Models/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth',
                          'batch_max_length': 25, 'imgH': 32, 'imgW': 100,
                          'character': '0123456789abcdefghijklmnopqrstuvwxyz',
                          'sensitive': True, 'rgb': False, 'PAD': False, 'Transformation': 'TPS',
                          'FeatureExtraction': 'ResNet',
                          'SequenceModeling': 'BiLSTM', 'Prediction': 'Attn', 'num_fiducial': 20, 'input_channel': 1,
                          'output_channel': 512,
                          'hidden_size': 256, 'debug': False
                          }
        kwargs = default_kwargs.copy()
        kwargs.update(passed_kwargs)
    except Exception as er:
        print("error occured while reading the input data and/or arguments")
        print("error is ", er)
        sys.exit(1)
    # --------------------- Loading the model ------------------------------

    # try:
    #     net = CRAFT()  # intialising the model architechture

    #     print('Loading weights from checkpoint (' + kwargs['trained_model'] + ')')
    #     if kwargs['use_gpu']:
    #         net.load_state_dict(copyStateDict(torch.load(kwargs['trained_model'])))
    #     else:
    #         net.load_state_dict(copyStateDict(torch.load(kwargs['trained_model'], map_location='cpu')))

    #     if kwargs['use_gpu']:
    #         net = net.cuda()
    #         net = torch.nn.DataParallel(net)
    #         cudnn.benchmark = False

    #     net.eval()

    #     # LinkRefiner
    #     refine_net = None
    #     if kwargs['refine']:
    #         refine_net = RefineNet()
    #         print('Loading weights of refiner from checkpoint (' + kwargs['refiner_model'] + ')')
    #         if kwargs['use_gpu']:
    #             refine_net.load_state_dict(copyStateDict(torch.load(kwargs['refiner_model'])))
    #             refine_net = refine_net.cuda()
    #             refine_net = torch.nn.DataParallel(refine_net)
    #         else:
    #             refine_net.load_state_dict(
    #                 copyStateDict(torch.load(kwargs['refiner_model'], map_location=torch.device('cpu'))))
    #             refine_net.eval()
    #             kwargs['poly'] = True
    # except Exception as er:
    #     print("Error occured while Loading the model")
    #     print("Error is ", er)
    #     sys.exit(1)

        # --------------------- Predicting the Text and storing the output to Json ------------------------------

    try:
        t = time.time()
        # writing output to a file
        if kwargs['sensitive']:
          basic=False
        else:
          basic=True
        final_json = {}


        # for each image

        ##SAI's code
        # for k, image_path in enumerate(image_list):
        print('--ACTUAL NO. IMAGE LIST-- ', image_list)
        for k, actualFrameNumber_From_TimeStamp in enumerate( image_list ):
            Json_format = {}
            # loading the image
            # image = loadImage(image_path)
            print(actualFrameNumber_From_TimeStamp, frame_timestamps[k])
            image = load_Frame_By_FrameNo_FromVideo(process_file_path, actualFrameNumber_From_TimeStamp)
            # getting image name from image path
            print("--IMAGE LOADED--")
            # image_name = image_list[k].split('/')[-1].split('.')[0]

            # getting image name from image path
            # image_name = image_list[k].split('/')[-1].split('.')[0]
            # print('--IMAGE NAME--',image_name)

            # --------------------- Detecting the Text Co-ordinates ------------------------------

            # returns the co-ordinates of the text detected in the images and also scores
            bboxes, polys, score_text, det_scores = test_net(net, image, kwargs['text_threshold'],
                                                             kwargs['link_threshold'],
                                                             kwargs['low_text'], kwargs['use_gpu'], kwargs['poly'],
                                                             kwargs, refine_net)

            """
            bboxes= the Bounding box 4-vertices coordinates(x,y)(w,h)
            polys= array of result file
            score_text= heatmap score used for mask image
            det_scores= confidence score for each bounding box
            """

            # variables used to store the output data
            full_text = {}
            Final_output = {}
            # fullTextAnnotation = []
            Json_format = {}
            block_dict = {}
            block_list = []
            # bbox_score={}       # Bbox_score dictionary for datatframe
            Word_list = []  # Word_list dictionary for all bounding boxes vertices
            Pred_txt = ''  # initialization of concat_text
            order_dict={} #for ordering based on co-ordinates
            cnt=0
            # for each detected box(word/words)
            for box_num in range(len(bboxes)):
                bboxes_json = {}  # for the return dictionary from bbox_cord file
                item = bboxes[box_num]  # 2-d array containing x & y coordinate

                # --------------------- genearating the images of each box using Co-ordinates  ------------------------------
                """ Crop_Evaluation file """
                # a image with the  co-ordinates in bboxes is generated in path passed to  "image_folder" parameter

                box_dim = generate_words(k, item, image, img_folder=kwargs['image_folder'], box_number=box_num,
                                         debug=kwargs['debug'])
                if len(box_dim)==4:
                    bboxes_json = {"X": box_dim[0], "Y": box_dim[1], "W": box_dim[2], "H": box_dim[3]}
                else:
                    bboxes_json = {"X":0, "Y":0, "W":0, "H":0}

                # --------------------- Predicting the text in the cropped images  ------------------------------

                """ Demo file """
                if kwargs['sensitive']:
                    import string
                    kwargs['character'] = string.printable[:-6]  # same with ASTER setting (use 94 char).

                cudnn.benchmark = True
                cudnn.deterministic = True
                kwargs['num_gpu'] = torch.cuda.device_count()

                # calling demo function from demo.py file for getting predicted text on cropped images in image_folder
                #print("0",kwargs)
                demo_output = demo(kwargs)
                #print("6",demo_output)
                pred, confidence_score = demo_output
                #pred, confidence_score = demo(kwargs)


                  # return of predicted txt and confidence score
                if pred : 
                    Pred_txt += " " + pred  # concatenation of all predicted word

                # --------------------- Storing the data and writing to Json file   ------------------------------

                vertices_dict = {}  # initializing a empty dict

                vertices_dict["boundingBox"] = bboxes_json
                vertices_dict["confidence"] = confidence_score
                vertices_dict["text"] = pred.replace('\'','').replace('\"','')
                if str(pred) in order_dict.keys():
                  order_dict[str(pred)+"_"+str(cnt)] = [bboxes_json['X'],bboxes_json['Y'],bboxes_json['W'],bboxes_json['H']]
                  cnt=cnt+1
                else:
                  order_dict[str(pred)] = [bboxes_json['X'],bboxes_json['Y'],bboxes_json['W'],bboxes_json['H']]

                # for each word(bounding box) vertices,confidence score and text are appended into Word_list
                Word_list.append(vertices_dict)

            #ordering the text based on-cordinates
            Pred_txt=sort_dict(order_dict)
            #extracting urls and numbers
            nums,urls,text,downloadable,vanity_nums = extract_info(Pred_txt,basic)
            #storing into dict 
            Final_output["ConcatText"] = Pred_txt.replace('\'','').replace('\"','')
            Final_output["fullTextAnnotation"] = Word_list
            Json_format["frame_num"] = k
            if frame_timestamps:
                # timestamp_number = frame_timestamps[k].split('.')[0]
                # print('--TSN--',timestamp_number)s
                Json_format["timestamp"] = int(frame_timestamps[k])
            else:
                Json_format["timestamp"] = 999
            Json_format['urls']=urls
            Json_format['nums']=nums
            Json_format['vanity_numbers']=vanity_nums
            Json_format['text']=text
            Json_format['downloadable']=downloadable
            Json_format["predicted_text"] = Final_output

            final_json[str(k)] = Json_format
            print(Json_format)
        # writing to Json file
        if kwargs['debug']:
            if file_name:
                out_pth = str(file_name) + '_results.json'
            else:
                out_pth = str(k) + '_results.json'
            with open(out_pth, 'w') as f:
                json.dump(final_json, f, indent=4)
        print("elapsed time : {}s".format(time.time() - t))

        return final_json
    except IndexError as er:
        print("error occured while Recognising the text")
        print("error is ", er)
        sys.exit(1)
    finally:
        frame_path = project_path + 'generated_frames/'
        if os.path.isdir(frame_path):
            shutil.rmtree(frame_path)

        if os.path.isdir(project_path + 'Cropped_image'):
            shutil.rmtree(project_path + 'Cropped_image')