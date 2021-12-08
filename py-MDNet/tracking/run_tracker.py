import numpy as np
import os
import sys
import time
import argparse
import yaml, json
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim

sys.path.insert(0, '.')
from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
from data_prov import RegionExtractor
from bbreg import BBRegressor
from gen_config import gen_config

opts = yaml.safe_load(open('tracking/options.yaml','r'))


def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts)
    for i, regions in enumerate(extractor):
        if opts['use_gpu']:
            regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    return feats


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for i in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()


def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=True):

    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    if gt is not None:
        overlap = np.zeros(len(img_list))
        overlap[0] = 1

    # Init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()

    # Init criterion and optimizer 
    criterion = BCELoss()
    model.set_learnable_params(opts['ft_layers'])
    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    # Draw pos/neg samples
    pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
                        target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
                    SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
                    SampleGenerator('whole', image.size)(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
    del init_optimizer, neg_feats
    torch.cuda.empty_cache()

    # Train bbox regressor
    bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'], opts['aspect_bbreg'])(
                        target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
    pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_feats = forward_samples(model, image, neg_examples)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]

    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='auto')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    for i in range(1, len(img_list)):

        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')

        # Estimate target bbox
        samples = sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0
        
        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None,:]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_feats = forward_samples(model, image, pos_examples)
            pos_feats_all.append(pos_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]

            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_feats = forward_samples(model, image, neg_examples)
            neg_feats_all.append(neg_feats)
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(i)), dpi=dpi)

        if gt is None:
            print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
                .format(i, len(img_list), target_score, spf))
        else:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
            print('Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
                .format(i, len(img_list), overlap[i], target_score, spf))

    if gt is not None:
        print('meanIOU: {:.3f}'.format(overlap.mean()))
    fps = len(img_list) / spf_total
    return result, result_bb, fps


if __name__ == "__main__":
    files_list=['Beaker_1','Beaker_2','Beaker_3','Beaker_4','Beaker_5','Beaker_6','Beaker_7','Beaker_8','Beaker_9','Beaker_10','Beaker_11','Beaker_12','Beaker_13','Beaker_14','Beaker_15','BubbleBalloon_1','BubbleBalloon_2','BubbleBalloon_3','BubbleBalloon_4','BubbleBalloon_5','BubbleBalloon_6','BubbleBalloon_7','BubbleBalloon_8','BubbleBalloon_9','BubbleBalloon_10','BubbleBalloon_11','BubbleBalloon_12','BubbleBalloon_13','BubbleBalloon_14','BubbleBalloon_15','Bulb_1','Bulb_2','Bulb_3','Bulb_4','Bulb_5','Bulb_6','Bulb_7','Bulb_8','Bulb_9','Bulb_10','Bulb_11','Bulb_12','Bulb_13','Bulb_14','Bulb_15','Flask_1','Flask_2','Flask_3','Flask_4','Flask_5','Flask_6','Flask_7','Flask_8','Flask_9','Flask_10','Flask_11','Flask_12','Flask_13','Flask_14','Flask_15','GlassBall_1','GlassBall_2','GlassBall_3','GlassBall_4','GlassBall_5','GlassBall_6','GlassBall_7','GlassBall_8','GlassBall_9','GlassBall_10','GlassBall_11','GlassBall_12','GlassBall_13','GlassBall_14','GlassBall_15','GlassBottle_1','GlassBottle_2','GlassBottle_3','GlassBottle_4','GlassBottle_5','GlassBottle_6','GlassBottle_7','GlassBottle_8','GlassBottle_9','GlassBottle_10','GlassBottle_11','GlassBottle_12','GlassBottle_13','GlassBottle_14','GlassBottle_15','GlassCup_1','GlassCup_2','GlassCup_3','GlassCup_4','GlassCup_5','GlassCup_6','GlassCup_7','GlassCup_8','GlassCup_9','GlassCup_10','GlassCup_11','GlassCup_12','GlassCup_13','GlassCup_14','GlassCup_15','GlassJar_1','GlassJar_2','GlassJar_3','GlassJar_4','GlassJar_5','GlassJar_6','GlassJar_7','GlassJar_8','GlassJar_9','GlassJar_10','GlassJar_11','GlassJar_12','GlassJar_13','GlassJar_14','GlassJar_15','GlassSlab_1','GlassSlab_2','GlassSlab_3','GlassSlab_4','GlassSlab_5','GlassSlab_6','GlassSlab_7','GlassSlab_8','GlassSlab_9','GlassSlab_10','GlassSlab_11','GlassSlab_12','GlassSlab_13','GlassSlab_14','GlassSlab_15','JuggleBubble_1','JuggleBubble_2','JuggleBubble_3','JuggleBubble_4','JuggleBubble_5','JuggleBubble_6','JuggleBubble_7','JuggleBubble_8','JuggleBubble_9','JuggleBubble_10','JuggleBubble_11','JuggleBubble_12','JuggleBubble_13','JuggleBubble_14','JuggleBubble_15','MagnifyingGlass_1','MagnifyingGlass_2','MagnifyingGlass_3','MagnifyingGlass_4','MagnifyingGlass_5','MagnifyingGlass_6','MagnifyingGlass_7','MagnifyingGlass_8','MagnifyingGlass_9','MagnifyingGlass_10','MagnifyingGlass_11','MagnifyingGlass_12','MagnifyingGlass_13','MagnifyingGlass_14','MagnifyingGlass_15','ShotGlass_1','ShotGlass_2','ShotGlass_3','ShotGlass_4','ShotGlass_5','ShotGlass_6','ShotGlass_7','ShotGlass_8','ShotGlass_9','ShotGlass_10','ShotGlass_11','ShotGlass_12','ShotGlass_13','ShotGlass_14','ShotGlass_15','TransparentAnimal_1','TransparentAnimal_2','TransparentAnimal_3','TransparentAnimal_4','TransparentAnimal_5','TransparentAnimal_6','TransparentAnimal_7','TransparentAnimal_8','TransparentAnimal_9','TransparentAnimal_10','TransparentAnimal_11','TransparentAnimal_12','TransparentAnimal_13','TransparentAnimal_14','TransparentAnimal_15','WineGlass_1','WineGlass_2','WineGlass_3','WineGlass_4','WineGlass_5','WineGlass_6','WineGlass_7','WineGlass_8','WineGlass_9','WineGlass_10','WineGlass_11','WineGlass_12','WineGlass_13','WineGlass_14','WineGlass_15','WubbleBubble_1','WubbleBubble_2','WubbleBubble_3','WubbleBubble_4','WubbleBubble_5','WubbleBubble_6','WubbleBubble_7','WubbleBubble_8','WubbleBubble_9','WubbleBubble_10','WubbleBubble_11','WubbleBubble_12','WubbleBubble_13','WubbleBubble_14','WubbleBubble_15']
    for file_ in files_list:
    

        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--seq', default=file_, help='input seq')
        parser.add_argument('-j', '--json', default='', help='input json')
        parser.add_argument('-f', '--savefig', action='store_true')
        parser.add_argument('-d', '--display', action='store_true')

        args = parser.parse_args()
        assert args.seq != '' or args.json != ''

        np.random.seed(0)
        torch.manual_seed(0)

    # Generate sequence config
        img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    # Run tracker
        result, result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)

    # Save result
        results = result_bb.round().tolist()
        new_lines = []
        for result in results:
            string_result = str(result)[1:-1] + '\n'
            new_lines.append(string_result)
        new_all_lines = "".join(new_lines)
        new_file = open(result_path, "w")
        new_file.write(new_all_lines)
        new_file.close()

    
    
    #json.dump(res, open(result_path, 'w'), indent=2)
