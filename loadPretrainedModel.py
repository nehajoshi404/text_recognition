def loadModel_pretrained(**kwargs):
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

		opt = default_kwargs.copy()
		opt.update(kwargs)

	except Exception as er:
	    print("error occured while reading the input data and/or arguments")
	    print("error is ", er)
	    sys.exit(1)




    # model configuration 
    if 'CTC' in opt["Prediction"]:
        converter = CTCLabelConverter(opt["character"])
    else:
        converter = AttnLabelConverter(opt["character"])
    opt["num_class"] = len(converter.character)

    if opt['use_gpu']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if opt["rgb"]:
        opt["input_channel"] = 3
    model = Model(opt)
    
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt["saved_model"])
    model.load_state_dict(torch.load(opt["saved_model"], map_location=device))
    print("---Model Loaded ---")

    return model