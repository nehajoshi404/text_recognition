"""
file contains code of:
    1.craft_utils (10-265)
    2.test (268-346)
    3.deep-text-recognition-benchmark/demo (359-453)
"""

# libraries
from libraries import *

#file imports
from file_utils import CTCLabelConverter, AttnLabelConverter,RawDataset, AlignCollate,resize_aspect_ratio,normalizeMeanVariance
from model_utils import Model



#craft_utils file code
# auxilary functions
# unwarp corodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


# end of auxilary functions


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8),
                                                                         connectivity=4)

    det = []
    det_scores = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)
        det_scores.append(np.max(textmap[labels == k]))

    return det, labels, mapper, det_scores


def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 10 or h < 10:
            polys.append(None);
            continue

        # warp image
        tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None);
            continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:, i] != 0)[0]
            if len(region) < 2: continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None);
            continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg  # segment width
        pp = [None] * num_cp  # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0, len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0:
                continue  # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1) / 2)] = (x, cy)
                seg_height[int((seg_num - 1) / 2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh is smaller than character height
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None);
            continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:  # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None);
            continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper, det_scores = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys, det_scores


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys

#test file contains

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, args, refine_net=None):
    """
      arguments:
        net: Model
        image: loaded image
        text_threshold:The Prediction score threshold to be used for considering the word predicted
        link_threshold: same as text_threshold but this score is used for links(urls) (not sure)
        cuda: Gpu option if True will try to use the gpu for computing
        poly:polygen shaped cordinates extracted over detected test from the image
        args: the remaining arguments
        refine_net:(Boolean) (default=None)

      returns:
        boxes, polys, ret_score_text, det_scores

      function:
        Detects the text position  in the image and returns the co-ordinates

      calls:getDetBoxes and adjustResultCoordinates

    """
    t0 = time.time()
    preprocessing_time_start = time.time()
    #print("Inside test_net")


    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, args['canvas_size'], interpolation=cv2.INTER_LINEAR, mag_ratio=args['mag_ratio'])
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    preprocessing_time_end = time.time()
    print('Time required for preprocessing :',preprocessing_time_end - preprocessing_time_start)
    model_execution_start = time.time()
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()
    model_execution_end = time.time()
    print("Time required for model execution = {}".format(model_execution_end - model_execution_start))
    


    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    postprocessing_start = time.time()
    boxes, polys, det_scores = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    postprocessing_end = time.time()
    print("Time required for post processing = {}".format(postprocessing_end - postprocessing_start))
    t1 = time.time() - t1



    """# render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    """
    #commented above to get
    ret_score_text=0
    if args['show_time'] :
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text, det_scores


#deep-text-recognition-benchmark/demo file code

def demo(opt):
    """ model configuration """
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
    # print('model input parameters', opt["imgH"], opt["imgW"], opt["num_fiducial"], opt["input_channel"], opt["output_channel"],
    #     opt["hidden_size"], opt["num_class"], opt["batch_max_length"], opt["Transformation"], opt["FeatureExtraction"],
    #    opt["SequenceModeling"], opt["Prediction"])
    model = torch.nn.DataParallel(model).to(device)

    # load model
    # print('loading pretrained model from %s' % opt["saved_model"])
    model.load_state_dict(torch.load(opt["saved_model"], map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt["imgH"], imgW=opt["imgW"], keep_ratio_with_pad=opt["PAD"])
    demo_data = RawDataset(root=opt["image_folder"], opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt["batch_size"],
        shuffle=False,
        num_workers=int(opt["workers"]),
        collate_fn=AlignCollate_demo, pin_memory=True)
    
    print("Loaded the model")
    # predict
    if (demo_loader != None):
        model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                print('---IPL---',image_path_list)
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([opt["batch_max_length"]] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt["batch_max_length"] + 1).fill_(0).to(device)
    
                if 'CTC' in opt["Prediction"]:
                    preds = model(image, text_for_pred)
    
                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = converter.decode(preds_index, preds_size)
    
                else:
                    preds = model(image, text_for_pred, is_train=False)
    
                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)
    
    
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                Pred_txt = ''
    
                # if (preds_str != '') and (preds_str != None) : 
    
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in opt["Prediction"]:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        # print("pred::",pred)
                        # Pred_txt+=" "+pred
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    # pred_dict={"Text":pred}
                    Confi_dict = f'{confidence_score:0.4f}'

                    return pred, Confi_dict

                    print(f'{img_name:25s}\t{Pred_txt:25s}')
                # else:
                #     return '', {}
    else:
        return '', {}
        
# def demo(opt):
#     """ model configuration """
#     if 'CTC' in opt.Prediction:
#         converter = CTCLabelConverter(opt.character)
#     else:
#         converter = AttnLabelConverter(opt.character)
#     opt.num_class = len(converter.character)

#     if opt.rgb:
#         opt.input_channel = 3
#     model = Model(opt)
#     print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
#           opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
#           opt.SequenceModeling, opt.Prediction)
#     model = torch.nn.DataParallel(model).to(device)

#     # load model
#     print('loading pretrained model from %s' % opt.saved_model)
#     model.load_state_dict(torch.load(opt.saved_model, map_location=device))

#     # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
#     AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
#     demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
#     demo_loader = torch.utils.data.DataLoader(
#         demo_data, batch_size=opt.batch_size,
#         shuffle=False,
#         num_workers=int(opt.workers),
#         collate_fn=AlignCollate_demo, pin_memory=True)

#     # predict
#     model.eval()
#     with torch.no_grad():
#         for image_tensors, image_path_list in demo_loader:
#             batch_size = image_tensors.size(0)
#             image = image_tensors.to(device)
#             # For max length prediction
#             length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
#             text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

#             if 'CTC' in opt.Prediction:
#                 preds = model(image, text_for_pred)

#                 # Select max probabilty (greedy decoding) then decode index to character
#                 preds_size = torch.IntTensor([preds.size(1)] * batch_size)
#                 _, preds_index = preds.max(2)
#                 # preds_index = preds_index.view(-1)
#                 preds_str = converter.decode(preds_index, preds_size)

#             else:
#                 preds = model(image, text_for_pred, is_train=False)

#                 # select max probabilty (greedy decoding) then decode index to character
#                 _, preds_index = preds.max(2)
#                 preds_str = converter.decode(preds_index, length_for_pred)


#             log = open(f'./log_demo_result.txt', 'a')
#             dashed_line = '-' * 80
#             head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
#             print(f'{dashed_line}\n{head}\n{dashed_line}')
#             log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

#             preds_prob = F.softmax(preds, dim=2)
#             preds_max_prob, _ = preds_prob.max(dim=2)
#             for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
#                 if 'Attn' in opt.Prediction:
#                     pred_EOS = pred.find('[s]')
#                     pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
#                     pred_max_prob = pred_max_prob[:pred_EOS]

#                 # calculate confidence score (= multiply of pred_max_prob)
#                 confidence_score = pred_max_prob.cumprod(dim=0)[-1]
#                 print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')

#                 return pred, confidence_score
#                 # log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

#             # log.close()
