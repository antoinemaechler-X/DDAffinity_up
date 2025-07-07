import os
import sys
import shutil
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from rde.utils.misc import BlackHole, load_config, seed_all, get_logger, get_new_dir, current_milli_time
from rde.models.protein_mpnn_network_2 import ProteinMPNN_NET
from rde.utils.skempi_mpnn_sequence_grouped import SequenceGroupedSkempiDatasetManager
from rde.utils.transforms import get_transform
from rde.utils.train_mpnn import *
from rde.utils.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--num_cvfolds', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='./logs_skempi_grouped')
    parser.add_argument('--early_stoppingdir', type=str, default='./early_stopping_grouped')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--reset_split', action='store_true', help='Reset sequence grouping cache and recompute split')
    parser.add_argument('--split', type=str, default='data/complex_sequences_grouped.csv', help='Path to the sequence grouping CSV file')
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
        ckpt_dir = None
    else:
        if args.resume:
            log_dir = get_new_dir(args.logdir, tag=args.tag)
        else:
            # Include reset_split info in log directory name
            reset_suffix = "_reset" if args.reset_split else ""
            log_dir = get_new_dir(args.logdir, prefix=f'[grouped-{args.num_cvfolds}fold{reset_suffix}]', tag=args.tag)
            early_stoppingdir = get_new_dir(args.early_stoppingdir, prefix=f'[grouped-{args.num_cvfolds}fold{reset_suffix}]', tag=args.tag)

        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)
    logger.info(f"Using sequence grouping split file: {args.split}")

    # Data
    logger.info('Loading datasets...')
    dataset_mgr = SequenceGroupedSkempiDatasetManager(
        config,
        num_cvfolds=args.num_cvfolds,  # Use the specified number of folds
        num_workers=args.num_workers,
        logger=logger,
        reset_split=args.reset_split,  # Pass the reset_split argument
        split_csv=args.split  # Pass the split file
    )

    # Model, Optimizer & Cross-Validation
    logger.info('Building model...')
    cv_mgr = CrossValidation(
        model_factory=ProteinMPNN_NET,
        config=config,
        early_stoppingdir=early_stoppingdir,
        num_cvfolds=args.num_cvfolds
    ).to(args.device)
    it_first = 0

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']
        cv_mgr.load_state_dict(ckpt['model'], )

    def train_one_epoch(fold, epoch):
        model, optimizer, early_stopping = cv_mgr.get(fold)
        if early_stopping.early_stop == True:
            return

        time_start = current_milli_time()
        mean_loss = torch.zeros(1).to(args.device)
        model.train()
        train_loader = dataset_mgr.get_train_loader(fold)
        train_loader = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_loader):
            batch = recursive_to(data, args.device)

            # Forward pass
            loss_dict, _ = model(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            time_forward_end = current_milli_time()

            # Backward
            loss.backward()

            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
            train_loader.desc = "[epoch {} fold {}] mean loss {}".format(epoch, fold, round(mean_loss.item(), 3))

            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

        time_backward_end = current_milli_time()
        logger.info(f'[epoch {epoch} fold {fold}] mean loss {mean_loss.item():.3f}')

        if epoch >= config.train.early_stopping_epoch and early_stopping.early_stop == False:
            early_stopping(mean_loss.item(), model, fold)

    def validate(epoch):
        scalar_accum = ScalarMetricAccumulator()
        results = []
        with torch.no_grad():
            for fold in range(args.num_cvfolds):
                results_fold = []
                model, optimizer, early_stopping = cv_mgr.get(fold)

                for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc='Validate', dynamic_ncols=True)):
                    batch = recursive_to(batch, args.device)

                    # Forward pass
                    loss_dict, output_dict = model(batch)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    scalar_accum.add(name='loss', value=loss, batchsize=batch['size'], mode='mean')

                    for complex, mutstr, ddg_true, ddg_pred in zip(batch["wt"]['complex'], batch["wt"]['mutstr'], output_dict['ddG_true'], output_dict['ddG_pred']):
                        results.append({
                            'complex': complex,
                            'mutstr': mutstr,
                            'num_muts': len(mutstr.split(',')),
                            'ddG': ddg_true.item(),
                            'ddG_pred': ddg_pred.item()
                        })
                        results_fold.append({
                            'complex': complex,
                            'mutstr': mutstr,
                            'num_muts': len(mutstr.split(',')),
                            'ddG': ddg_true.item(),
                            'ddG_pred': ddg_pred.item()
                        })

                results_fold = pd.DataFrame(results_fold)
                pearson_fold = results_fold[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]

        results = pd.DataFrame(results)
        if ckpt_dir is not None:
            results.to_csv(os.path.join(ckpt_dir, f'results_{epoch}.csv'), index=False)
        pearson_all = results[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
        spearman_all = results[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]

        logger.info(f'[All] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f}')
        avg_loss = scalar_accum.get_average('loss')
        scalar_accum.log(epoch, 'val', logger=logger, writer=writer)
        writer.add_scalar('val/all_pearson', pearson_all, epoch)
        writer.add_scalar('val/all_spearman', spearman_all, epoch)

        return avg_loss

    try:
        for epoch in range(it_first, config.train.max_epochs + 1):
            # training
            for fold in range(args.num_cvfolds):
                mean_loss = train_one_epoch(fold, epoch)

            if epoch % config.train.val_freq == 0:
                avg_loss = validate(epoch)

        if True:
            results = []
            # Saving checkpoint: DDAffinity
            logger.info(f'Saving checkpoint: DDAffinity.pt')
            cv_mgr.save_state_dict(args, config)
            # Loading checkpoint: DDAffinity
            ckpt_path = os.path.join(early_stoppingdir, 'DDAffinity.pt')
            ckpt = torch.load(ckpt_path, map_location=args.device)
            cv_mgr.load_state_dict(ckpt['model'], )

            for fold in range(args.num_cvfolds):
                logger.info(f'Resuming from checkpoint: Fold_{fold}_best_network.pt')
                model, _, _ = cv_mgr.get(fold)
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc='Validate', dynamic_ncols=True)):
                        batch = recursive_to(batch, args.device)

                        # Forward pass
                        loss_dict, output_dict = model(batch)
                        for complex, mutstr, ddg_true, ddg_pred in zip(batch["wt"]['complex'], batch["wt"]['mutstr'], output_dict['ddG_true'], output_dict['ddG_pred']):
                            results.append({
                                'complex': complex,
                                'mutstr': mutstr,
                                'num_muts': len(mutstr.split(',')),
                                'ddG': ddg_true.item(),
                                'ddG_pred': ddg_pred.item()
                            })

            results = pd.DataFrame(results)
            if ckpt_dir is not None:
                results.to_csv(os.path.join(ckpt_dir, f'results_all.csv'), index=False)
            pearson_all = results[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
            spearman_all = results[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]
            logger.info(f'[All] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f}')

    except KeyboardInterrupt:
        logger.info('Terminating...')
