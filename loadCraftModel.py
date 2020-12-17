from libraries import *
import time

from model_utils import CRAFT, copyStateDict, RefineNet
from prediction_utils import test_net, demo

project_path = os.getcwd() + '/'

# --------------------- Processing the Arguments  ------------------------------

# overwriting existing dict and appending new ones to final dict


def loadModel_Seperately(**passed_kwargs):
	try : 
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

	try:
        
        
	    net = CRAFT()  # intialising the model architechture
        start = time.time()
	    print('Loading weights from checkpoint (' + kwargs['trained_model'] + ')')
	    if kwargs['use_gpu']:
	        net.load_state_dict(copyStateDict(torch.load(kwargs['trained_model'])))
        end = time.time()
        
	    else:
	        net.load_state_dict(copyStateDict(torch.load(kwargs['trained_model'], map_location='cpu')))
        print('---Model Loaded---')
        print('Time taken for loading the model:',end - star)

	    if kwargs['use_gpu']:
	        net = net.cuda()
	        net = torch.nn.DataParallel(net)
	        cudnn.benchmark = False

	    net.eval()

	    # LinkRefiner
	    refine_net = None
	    if kwargs['refine']:
	        refine_net = RefineNet()
	        print('Loading weights of refiner from checkpoint (' + kwargs['refiner_model'] + ')')
	        if kwargs['use_gpu']:
	            refine_net.load_state_dict(copyStateDict(torch.load(kwargs['refiner_model'])))
	            refine_net = refine_net.cuda()
	            refine_net = torch.nn.DataParallel(refine_net)
	        else:
	            refine_net.load_state_dict(
	                copyStateDict(torch.load(kwargs['refiner_model'], map_location=torch.device('cpu'))))
	            refine_net.eval()
	            kwargs['poly'] = True
	    return net, refine_net
	    
	except Exception as er:
	    print("Error occured while Loading the model")
	    print("Error is ", er)
	    sys.exit(1)
